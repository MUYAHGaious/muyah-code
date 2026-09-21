"""Provider catalog + API-key store: pick a provider, paste a key, done.

Keys live in ~/.muyah/credentials.json (never in settings.json, so settings can be shared or committed).
A profile references its key by provider id ("credential": "anthropic"); the key is resolved at load time.
Environment variables the provider's own tools use (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...) are detected too.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    base_url: str
    default_model: str
    env_vars: tuple[str, ...] = ()
    key_url: str = ""               # where to create a key
    key_hint: str = ""              # typical key prefix, for a friendly sanity check
    api: str = "openai"             # "openai" (OpenAI-compatible) or "anthropic" (native Messages API)
    local: bool = False             # no key; server runs on this machine
    context_window: int = 0         # when the provider does not report it
    prefer: tuple[str, ...] = ()    # substrings that rank models higher in the picker
    billing_url: str = ""           # where to add credits when the account has no balance
    pin_prefer: bool = False        # True: `prefer` order wins over version (use only for a curated, current list)
    extra: dict = field(default_factory=dict)  # extra profile settings


PROVIDERS: list[Provider] = [
    Provider("anthropic", "Anthropic (Claude)", "https://api.anthropic.com", "claude-opus-5",
             ("ANTHROPIC_API_KEY",), "https://console.anthropic.com/settings/keys", "sk-ant-", api="anthropic",
             billing_url="https://console.anthropic.com/settings/billing", pin_prefer=True,
             prefer=("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"),
             # room for adaptive thinking + long edits; the reply streams, so no timeout risk
             extra={"max_tokens": 32000, "request_timeout": 900}),
    Provider("openai", "OpenAI", "https://api.openai.com/v1", "gpt-4.1", ("OPENAI_API_KEY",),
             "https://platform.openai.com/api-keys", "sk-", billing_url="https://platform.openai.com/settings/organization/billing", prefer=("gpt-5", "gpt-4.1", "o4", "o3")),
    Provider("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
             "", ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "https://aistudio.google.com/apikey",
             prefer=("gemini",), pin_prefer=True),  # Gemini before Gemma; newest version + pro tier wins
    Provider("openrouter", "OpenRouter (hundreds of models)", "https://openrouter.ai/api/v1", "qwen/qwen3-coder",
             ("OPENROUTER_API_KEY",), "https://openrouter.ai/keys", "sk-or-",
             billing_url="https://openrouter.ai/settings/credits", pin_prefer=True,
             prefer=("qwen3-coder", "qwen", "deepseek", "kimi", "glm", "gemini", "gpt", "claude")),
    Provider("groq", "Groq", "https://api.groq.com/openai/v1", "moonshotai/kimi-k2-instruct",
             ("GROQ_API_KEY",), "https://console.groq.com/keys", "gsk_",
             prefer=("kimi", "qwen", "gpt-oss", "llama"), pin_prefer=True),
    Provider("deepseek", "DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat", ("DEEPSEEK_API_KEY",),
             "https://platform.deepseek.com/api_keys", "sk-", context_window=131072, billing_url="https://platform.deepseek.com/top_up", prefer=("deepseek-chat",)),
    Provider("mistral", "Mistral", "https://api.mistral.ai/v1", "devstral-medium-latest", ("MISTRAL_API_KEY",),
             "https://console.mistral.ai/api-keys", prefer=("devstral", "codestral", "mistral-large")),
    Provider("xai", "xAI (Grok)", "https://api.x.ai/v1", "grok-code-fast-1", ("XAI_API_KEY",),
             "https://console.x.ai", "xai-", prefer=("grok-code", "grok-4")),
    Provider("together", "Together AI", "https://api.together.xyz/v1", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
             ("TOGETHER_API_KEY",), "https://api.together.ai/settings/api-keys",
             prefer=("qwen3-coder", "qwen", "deepseek", "kimi", "glm"), pin_prefer=True),
    Provider("fireworks", "Fireworks AI", "https://api.fireworks.ai/inference/v1",
             "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct", ("FIREWORKS_API_KEY",),
             "https://app.fireworks.ai/settings/users/api-keys", "fw_",
             prefer=("qwen3-coder", "qwen", "deepseek", "kimi", "glm"), pin_prefer=True),
    Provider("cerebras", "Cerebras", "https://api.cerebras.ai/v1", "qwen-3-coder-480b", ("CEREBRAS_API_KEY",),
             "https://cloud.cerebras.ai", "csk-", prefer=("qwen-3-coder", "gpt-oss", "llama")),
    Provider("moonshot", "Moonshot (Kimi)", "https://api.moonshot.ai/v1", "kimi-k2-0905-preview",
             ("MOONSHOT_API_KEY",), "https://platform.moonshot.ai/console/api-keys", "sk-", prefer=("kimi-k2",)),
    Provider("zai", "Z.ai (GLM)", "https://api.z.ai/api/paas/v4", "glm-4.6", ("ZAI_API_KEY",),
             "https://z.ai/manage-apikey/apikey-list", prefer=("glm-4.6", "glm-4.5")),
    Provider("qwen", "Alibaba Qwen (DashScope)", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
             "qwen3-coder-plus", ("DASHSCOPE_API_KEY",), "https://modelstudio.console.alibabacloud.com", "sk-",
             prefer=("qwen3-coder",)),
    Provider("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1", "qwen/qwen3-coder-480b-a35b-instruct",
             ("NVIDIA_API_KEY",), "https://build.nvidia.com", "nvapi-",
             prefer=("qwen3-coder", "qwen", "deepseek", "kimi"), pin_prefer=True),
    Provider("huggingface", "Hugging Face Inference", "https://router.huggingface.co/v1",
             "Qwen/Qwen3-Coder-480B-A35B-Instruct", ("HF_TOKEN",), "https://huggingface.co/settings/tokens", "hf_",
             prefer=("qwen3-coder", "qwen", "deepseek", "kimi", "glm"), pin_prefer=True),
    Provider("ollama", "Ollama (local)", "http://localhost:11434/v1", "qwen2.5-coder:14b", local=True,
             prefer=("coder", "qwen", "deepseek")),
    Provider("lmstudio", "LM Studio (local)", "http://localhost:1234/v1", "", local=True, prefer=("coder",)),
    Provider("llamacpp", "llama.cpp server (local)", "http://localhost:8080/v1", "", local=True),
    Provider("vllm", "vLLM / colibri / Soup (local :8000)", "http://localhost:8000/v1", "", local=True),
]
BY_ID = {p.id: p for p in PROVIDERS}


def get_provider(key: str) -> Provider | None:
    key = key.strip().lower()
    if key.isdigit() and 1 <= int(key) <= len(PROVIDERS):
        return PROVIDERS[int(key) - 1]
    if key in BY_ID:
        return BY_ID[key]
    for p in PROVIDERS:  # fuzzy: "claude", "google", "grok", "kimi"...
        if key and (key in p.name.lower() or key in p.id):
            return p
    return None


# ---------------------------------------------------------------------------------------------- credentials

def _cred_path(home: Path) -> Path:
    return home / "credentials.json"


def load_credentials(home: Path) -> dict[str, str]:
    try:
        data = json.loads(_cred_path(home).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, str)} if isinstance(data, dict) else {}


def save_credential(home: Path, provider_id: str, key: str) -> Path:
    creds = load_credentials(home)
    creds[provider_id] = key.strip()
    path = _cred_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(creds, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 600 on POSIX; best effort on Windows
    except OSError:
        pass
    os.replace(tmp, path)
    return path


def remove_credential(home: Path, provider_id: str) -> bool:
    creds = load_credentials(home)
    if provider_id not in creds:
        return False
    del creds[provider_id]
    _cred_path(home).write_text(json.dumps(creds, indent=2) + "\n", encoding="utf-8")
    return True


def env_key(provider: Provider) -> tuple[str, str] | None:
    for var in provider.env_vars:
        if os.environ.get(var):
            return var, os.environ[var]
    return None


def resolve_key(home: Path, provider_id: str) -> str | None:
    """Saved key first (the user chose it explicitly), then the provider's usual env vars."""
    saved = load_credentials(home).get(provider_id)
    if saved:
        return saved
    p = BY_ID.get(provider_id)
    found = env_key(p) if p else None
    return found[1] if found else None


def key_status(home: Path, provider: Provider) -> str:
    if provider.local:
        return "local"
    if load_credentials(home).get(provider.id):
        return "key saved"
    found = env_key(provider)
    return f"${found[0]}" if found else ""


def mask(key: str) -> str:
    return key[:6] + "…" + key[-4:] if len(key) > 12 else "***"


def looks_wrong(provider: Provider, key: str) -> str | None:
    key = key.strip()
    if not key:
        return "the key is empty"
    if re.search(r"\s", key):
        return "the key contains spaces or line breaks (paste only the key)"
    if provider.key_hint and not key.startswith(provider.key_hint):
        return f"{provider.name} keys usually start with '{provider.key_hint}'"
    return None


# ---------------------------------------------------------------------------------------------- models

def clean_model_id(model_id: str) -> str:
    return model_id[len("models/"):] if model_id.startswith("models/") else model_id  # Gemini lists models/...


# Models that cannot hold a coding conversation (speech, images, embeddings, ...): hidden from the picker.
NON_CHAT = ("embed", "whisper", "tts", "dall-e", "image", "audio", "moderation", "transcribe", "realtime", "aqa",
            "lyria", "veo", "imagen", "native-audio", "-live", "robotics", "guard", "rerank", "speech", "babbage",
            "davinci", "computer-use", "omni-moderation")
# Tier words: stronger models first when versions tie.
TIER = [("opus", 0), ("-pro", 0), ("large", 0), ("max", 0), ("coder", 1), ("sonnet", 1), ("medium", 2),
        ("flash", 2), ("haiku", 3), ("mini", 4), ("small", 4), ("lite", 5), ("nano", 6)]
_VERSION = re.compile(r"(?<![\d.])(\d{1,2})(?:[.-](\d{1,2}))?(?![\d]|[bBkKmM]\b|[bB]-)")


def model_version(model_id: str) -> float:
    """Best-effort generation number: gemini-3.1-pro -> 3.1, claude-opus-4-6 -> 4.6, gpt-5 -> 5, qwen3 -> 3.
    Parameter sizes (480b, 70b) and dates (20251001) are ignored."""
    m = _VERSION.search(model_id.lower())
    if not m:
        return 0.0
    return float(f"{m.group(1)}.{m.group(2)}") if m.group(2) else float(m.group(1))


def rank_models(provider: Provider, models: list[str]) -> list[str]:
    """Order a provider's model list for the picker. Newest generation first, stronger tier first, and
    non-chat models removed. Provider `prefer` hints only break ties, so the order never goes stale
    when a provider retires a model."""
    models = list(dict.fromkeys(clean_model_id(m) for m in models if m))
    chat = [m for m in models if not any(s in m.lower() for s in NON_CHAT)] or models

    def score(m: str) -> tuple:
        low = m.lower()
        pref = next((i for i, p in enumerate(provider.prefer) if p in low), len(provider.prefer))
        tier = next((t for word, t in TIER if word in low), 3)
        dated = 1 if re.search(r"\d{6,}", low) else 0  # prefer aliases over dated snapshots
        return (pref if provider.pin_prefer else 0, -model_version(m), tier, pref, dated, len(m), m)

    return sorted(chat, key=score)


def build_provider_profile(provider: Provider, model: str, context_window: int | None = None) -> dict:
    prof: dict = {"base_url": provider.base_url, "model": model, "provider": provider.id}
    if provider.api != "openai":
        prof["api"] = provider.api
    if not provider.local:
        prof["credential"] = provider.id
        prof["api_key"] = "none"  # the real key is resolved from credentials at load time
    window = context_window or provider.context_window
    if window:
        prof["context_window"] = window
    prof.update(provider.extra)
    return prof
