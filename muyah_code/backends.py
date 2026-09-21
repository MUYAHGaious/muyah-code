"""Find and connect to model servers: local (Ollama, LM Studio, vLLM, llama.cpp, colibri, Soup) or remote
(Colab tunnels, OpenRouter, Groq, DeepSeek, OpenAI...)."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

LOCAL_CANDIDATES = [
    ("ollama", "http://localhost:11434/v1"),
    ("lmstudio", "http://localhost:1234/v1"),
    ("local-8000", "http://localhost:8000/v1"),   # vLLM, colibri (coli serve), Soup (soup serve), SGLang
    ("llamacpp", "http://localhost:8080/v1"),
    ("local-5000", "http://localhost:5000/v1"),   # text-generation-webui
    ("sglang", "http://localhost:30000/v1"),
]
KNOWN_HOSTED = {
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
    "together": "https://api.together.xyz/v1",
    "mistral": "https://api.mistral.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}


@dataclass(frozen=True)
class EnginePreset:
    """How to launch an engine locally and how MUYAH-CODE should talk to it."""

    name: str
    port: int
    binary: str
    install: str
    settings: dict            # merged into the profile (deep-merged over the defaults)
    notes: str = ""


ENGINE_PRESETS: dict[str, EnginePreset] = {
    "vllm": EnginePreset(
        "vllm", 8000, "vllm", "pip install vllm   (Linux/WSL with an NVIDIA GPU)",
        {"request_timeout": 600},
        "Serve with --enable-auto-tool-choice --tool-call-parser hermes (Qwen) / llama3_json (Llama) for native tools."),
    "colibri": EnginePreset(
        "colibri", 8000, "coli",
        "git clone https://github.com/JustVugg/colibri && cd colibri && pip install -e .   (then build your model "
        "engine, e.g. make -C c deepseek-v4)",
        # Huge MoE models streamed from disk: first token can take minutes, generation ~1 tok/s on CPU.
        # No request timeout, no extra LLM calls for reflection, modest output budget.
        {"request_timeout": 0, "max_tokens": 4096, "learning": {"reflect": False}, "compact_threshold": 0.75},
        "One request at a time. Tool calling only on GLM/DeepSeek-V4/Kimi engines; others use the text protocol. "
        "Through Cloudflare quick tunnels long prefills hit the ~100s limit (HTTP 524): prefer Pinggy."),
    "soup": EnginePreset(
        "soup", 8000, "soup",
        'pip install "soup-cli[serve]"   (Python 3.10-3.12, in its own virtualenv)',
        {"request_timeout": 600, "max_tokens": 4096},
        "Serves your fine-tuned models (soup serve). max_tokens is capped at 16384 by Soup."),
    "ollama": EnginePreset(
        "ollama", 11434, "ollama", "https://ollama.com/download",
        {"request_timeout": 600},
        "Ollama silently truncates at its default context; MUYAH-CODE starts it with OLLAMA_CONTEXT_LENGTH."),
    "llamacpp": EnginePreset(
        "llamacpp", 8080, "llama-server", "winget install llama.cpp   (or brew install llama.cpp)",
        {"request_timeout": 600},
        "--jinja enables native tool calling with the model's chat template."),
    "lmstudio": EnginePreset(
        "lmstudio", 1234, "lms", "https://lmstudio.ai (then: lms server start)",
        {"request_timeout": 600}, ""),
}


def serve_command(engine: str, model: str, port: int, ctx: int | None, binary: str) -> tuple[list[str], dict]:
    """argv and extra env to start `engine` serving `model` on 127.0.0.1:port."""
    env: dict[str, str] = {}
    if engine == "colibri":
        from pathlib import Path

        argv = [binary, "serve", "--model", model, "--host", "127.0.0.1", "--port", str(port),
                "--model-id", Path(model).name or "colibri"]
        if ctx:
            argv += ["--ctx", str(ctx)]
    elif engine == "soup":
        argv = [binary, "serve", "--model", model, "--port", str(port)]
    elif engine == "vllm":
        parser = "llama3_json" if "llama" in model.lower() else "hermes"
        argv = [binary, "serve", model, "--host", "127.0.0.1", "--port", str(port),
                "--enable-auto-tool-choice", "--tool-call-parser", parser]
        if ctx:
            argv += ["--max-model-len", str(ctx)]
    elif engine == "llamacpp":
        argv = [binary, "-m", model, "--host", "127.0.0.1", "--port", str(port), "--jinja"]
        if ctx:
            argv += ["-c", str(ctx)]
    elif engine == "ollama":
        argv = [binary, "serve"]
        env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
        env["OLLAMA_CONTEXT_LENGTH"] = str(ctx or 32768)
    elif engine == "lmstudio":
        argv = [binary, "server", "start", "--port", str(port)]
    else:
        raise ValueError(f"Unknown engine '{engine}'. Known: {', '.join(ENGINE_PRESETS)}")
    return argv, env


def build_profile(base_url: str, model: str, api_key: str | None, context_window: int | None,
                  engine: str | None) -> dict:
    prof: dict = {}
    if engine:
        prof.update(ENGINE_PRESETS[engine].settings)
        prof["engine"] = engine
    prof.update({"base_url": base_url, "model": model})
    if api_key and api_key != "none":
        prof["api_key"] = api_key
    if context_window:
        prof["context_window"] = context_window
    return prof


def tunnel_warning(url: str, engine: str | None) -> str | None:
    if "trycloudflare.com" in url and engine in ("colibri", None):
        return ("This is a Cloudflare quick tunnel: requests that stay silent for ~100s are cut (HTTP 524). "
                "Big models with long prompts can exceed that before the first token. If you see 524 errors, "
                "use the Pinggy URL from the server notebook instead.")
    return None


@dataclass
class Endpoint:
    name: str
    base_url: str
    models: list[str] = field(default_factory=list)
    error: str | None = None
    server: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


def normalize_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not re.match(r"^https?://", url):
        url = ("http://" if url.startswith(("localhost", "127.", "0.0.0.0")) else "https://") + url
    for suffix in ("/chat/completions", "/completions", "/models"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


def probe(base_url: str, api_key: str = "none", timeout: float = 5.0, name: str = "") -> Endpoint:
    """GET {base}/models. Tries adding /v1 when the bare URL does not answer like an OpenAI server."""
    base = normalize_base_url(base_url)
    candidates = [base] if base.endswith("/v1") or "/v1" in base else [base + "/v1", base]
    last_error = "no response"
    for cand in candidates:
        try:
            r = httpx.get(cand + "/models", headers={"Authorization": f"Bearer {api_key or 'none'}"},
                          timeout=timeout, follow_redirects=True)
        except httpx.HTTPError as e:
            last_error = f"{e.__class__.__name__}"
            continue
        if r.status_code == 401:
            return Endpoint(name, cand, error="401 unauthorized: an API key is required")
        if r.status_code >= 400:
            last_error = f"HTTP {r.status_code}"
            continue
        try:
            data = r.json()
        except ValueError:
            last_error = "not JSON (is this an OpenAI-compatible /v1 endpoint?)"
            continue
        items = data.get("data") if isinstance(data, dict) else data
        models = [m.get("id") for m in items or [] if isinstance(m, dict) and m.get("id")]
        return Endpoint(name, cand, models=models, server=r.headers.get("server", ""))
    return Endpoint(name, candidates[0], error=last_error)


def scan_local(timeout: float = 1.5) -> list[Endpoint]:
    with ThreadPoolExecutor(max_workers=len(LOCAL_CANDIDATES)) as pool:
        futures = [pool.submit(probe, url, "none", timeout, name) for name, url in LOCAL_CANDIDATES]
        return [f.result() for f in futures]


def check_health(base_url: str, timeout: float = 5.0) -> str | None:
    """colibri, Soup, vLLM and llama.cpp expose /health at the server root."""
    root = normalize_base_url(base_url)
    root = root[:-3] if root.endswith("/v1") else root
    try:
        r = httpx.get(root + "/health", timeout=timeout)
        return f"HTTP {r.status_code}"
    except httpx.HTTPError as e:
        return f"unreachable ({e.__class__.__name__})"


def smoke_test(base_url: str, model: str, api_key: str = "none", timeout: float = 120.0) -> tuple[bool, str]:
    """One tiny completion (PONG test) to prove the endpoint really serves the model."""
    try:
        r = httpx.post(normalize_base_url(base_url) + "/chat/completions",
                       headers={"Authorization": f"Bearer {api_key or 'none'}"},
                       json={"model": model, "messages": [{"role": "user", "content": "Reply with PONG only."}],
                             "max_tokens": 8, "temperature": 0}, timeout=timeout)
    except httpx.HTTPError as e:
        return False, f"{e.__class__.__name__}: {e}"
    if r.status_code >= 400:
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    try:
        return True, (r.json()["choices"][0]["message"].get("content") or "").strip()[:60]
    except (ValueError, KeyError, IndexError):
        return False, f"unexpected response: {r.text[:200]}"


def tool_calling_test(base_url: str, model: str, api_key: str = "none", timeout: float = 120.0) -> str:
    """Returns 'native', 'rejected' (server refuses tools -> text protocol), or 'ignored'."""
    tool = {"type": "function", "function": {"name": "get_time", "description": "Get the current time",
                                             "parameters": {"type": "object", "properties": {}}}}
    try:
        r = httpx.post(normalize_base_url(base_url) + "/chat/completions",
                       headers={"Authorization": f"Bearer {api_key or 'none'}"},
                       json={"model": model, "messages": [{"role": "user", "content": "What time is it? Use the tool."}],
                             "tools": [tool], "tool_choice": "auto", "max_tokens": 64, "temperature": 0},
                       timeout=timeout)
    except httpx.HTTPError as e:
        return f"error: {e.__class__.__name__}"
    if r.status_code >= 400:
        return "rejected"
    try:
        msg = r.json()["choices"][0]["message"]
    except (ValueError, KeyError, IndexError):
        return "error: bad response"
    return "native" if msg.get("tool_calls") else "ignored"


def profile_name_for(url: str) -> str:
    host = re.sub(r"^https?://", "", url).split("/")[0].split(":")[0]
    for name, known in KNOWN_HOSTED.items():
        if known.split("/")[2] == host:
            return name
    if "trycloudflare" in host or "pinggy" in host or "ngrok" in host:
        return "colab"
    if host in ("localhost", "127.0.0.1", "0.0.0.0"):
        port = re.search(r":(\d+)", url)
        return {"11434": "ollama", "1234": "lmstudio", "8080": "llamacpp"}.get(port.group(1) if port else "", "local")
    return re.sub(r"[^a-z0-9]+", "-", host.lower()).strip("-")[:30] or "custom"
