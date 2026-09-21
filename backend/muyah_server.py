"""
MUYAH-CODE model server for Google Colab or any Linux GPU machine (RunPod, Lambda, a home server...).

Starts an OpenAI-compatible server, publishes it through a tunnel, verifies it end to end, and prints the exact
`muyah connect ...` command to run on your computer.

Engines
  vllm     fast GPU serving of HF models (default). Native tool calling via --tool-call-parser.
  colibri  huge MoE models (GLM-5.x, DeepSeek-V4, Kimi...) streamed from NVMe/RAM (github.com/JustVugg/colibri).

Tunnels (in order of preference)
  pinggy      SSH tunnel, no request timeout -> best for slow prefill on big models
  cloudflare  quick tunnel; drops requests that stay silent ~100s (HTTP 524) -> fine for fast models

Configure with environment variables (or edit SETTINGS below), then run:  python muyah_server.py
"""

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

SETTINGS = {
    "ENGINE": os.environ.get("MUYAH_ENGINE", "vllm"),
    # vllm: HF model id. colibri: HF repo to download (or a local dir).
    "MODEL": os.environ.get("MUYAH_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"),
    "CTX": int(os.environ.get("MUYAH_CTX", "32768")),
    "PORT": int(os.environ.get("MUYAH_PORT", "8000")),
    "TUNNELS": os.environ.get("MUYAH_TUNNELS", "pinggy,cloudflare"),
    # colibri only
    "COLIBRI_DIR": os.environ.get("MUYAH_COLIBRI_DIR", "/workspace/colibri"),
    "COLIBRI_TARGET": os.environ.get("MUYAH_COLIBRI_TARGET", ""),  # e.g. deepseek-v4 (make target)
    "COLIBRI_MAKE_ARGS": os.environ.get("MUYAH_COLIBRI_MAKE_ARGS", "CUDA=1"),
    "COLIBRI_RAM_GB": os.environ.get("MUYAH_COLIBRI_RAM_GB", ""),
    "MODELS_DIR": os.environ.get("MUYAH_MODELS_DIR", "/workspace/models"),
    # optional API key required from clients (recommended for public tunnels)
    "API_KEY": os.environ.get("MUYAH_SERVER_API_KEY", ""),
}
LOG_DIR = os.environ.get("MUYAH_LOG_DIR", "/tmp/muyah-server")


def sh(cmd, check=True, **kw):
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=check, **kw)


def log_path(name):
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, name + ".log")


def tail(path, n=5):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def http_json(url, payload=None, timeout=10, api_key=""):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key or 'none'}"}
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def daemon(argv, name, env=None):
    """Start detached from the notebook kernel so interrupting a cell does not kill the server."""
    out = open(log_path(name), "w")  # noqa: SIM115 - lives as long as the daemon
    return subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})},
                            start_new_session=True)


# ---------------------------------------------------------------------------------------------- engines

def start_vllm(s):
    if shutil.which("vllm") is None:
        sh([sys.executable, "-m", "pip", "install", "-q", "-U", "vllm"])
    parser = "llama3_json" if "llama" in s["MODEL"].lower() else "hermes"
    argv = ["vllm", "serve", s["MODEL"], "--host", "0.0.0.0", "--port", str(s["PORT"]),
            "--max-model-len", str(s["CTX"]), "--enable-auto-tool-choice", "--tool-call-parser", parser,
            "--enable-prefix-caching", "--trust-remote-code"]
    if s["API_KEY"]:
        argv += ["--api-key", s["API_KEY"]]
    env = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
    return daemon(argv, "server", env), s["MODEL"]


def start_colibri(s):
    cdir = s["COLIBRI_DIR"]
    if not os.path.isdir(os.path.join(cdir, ".git")):
        os.makedirs(os.path.dirname(cdir) or ".", exist_ok=True)
        sh(["git", "clone", "--depth", "1", "https://github.com/JustVugg/colibri.git", cdir])
    if s["COLIBRI_TARGET"]:
        sh(["make", "-C", os.path.join(cdir, "c"), s["COLIBRI_TARGET"], *shlex.split(s["COLIBRI_MAKE_ARGS"])])
    model = s["MODEL"]
    if not os.path.isdir(model):
        local = os.path.join(s["MODELS_DIR"], model.split("/")[-1])
        if not os.path.exists(os.path.join(local, "config.json")):
            free = shutil.disk_usage(s["MODELS_DIR"] if os.path.isdir(s["MODELS_DIR"]) else "/").free / 1024 ** 3
            print(f"Downloading {model} ({free:.0f} GB free)...", flush=True)
            sh([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub"])
            sh(["hf", "download", model, "--local-dir", local])
        model = local
    model_id = os.path.basename(model.rstrip("/")) + "-colibri"
    argv = [os.path.join(cdir, "c", "coli"), "serve", "--model", model, "--ctx", str(s["CTX"]),
            "--host", "0.0.0.0", "--port", str(s["PORT"]), "--allowed-host", "*", "--model-id", model_id]
    if s["COLIBRI_RAM_GB"]:
        argv += ["--ram", s["COLIBRI_RAM_GB"]]
    env = {"COLI_ALLOW_INSECURE_BIND": "1", "COLI_ALLOWED_HOSTS": "*"}
    if s["API_KEY"]:
        env["COLI_API_KEY"] = s["API_KEY"]
    return daemon(argv, "server", env), model_id


def wait_ready(proc, s, timeout=3600):
    url = f"http://127.0.0.1:{s['PORT']}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited with code {proc.returncode}:\n{tail(log_path('server'), 30)}")
        try:
            data = http_json(url, timeout=3, api_key=s["API_KEY"])
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            pass
        elapsed = int(time.time() - start)
        if elapsed % 30 == 0:
            print(f"[{elapsed}s] loading... {tail(log_path('server'), 1).strip()[-120:]}", flush=True)
        time.sleep(2)
    raise RuntimeError("Server not ready in time; see " + log_path("server"))


# ---------------------------------------------------------------------------------------------- tunnels

def tunnel_pinggy(port):
    if shutil.which("ssh") is None:
        return None
    subprocess.run("pkill -f 'a.pinggy.io' || true", shell=True)
    daemon(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30", "-p", "443",
            f"-R0:localhost:{port}", "a.pinggy.io"], "pinggy")
    for _ in range(30):
        time.sleep(1.5)
        urls = re.findall(r"https://[-a-zA-Z0-9]+\.a\.(?:free\.)?pinggy\.(?:link|online)", tail(log_path("pinggy"), 50))
        if urls:
            return urls[-1]
    return None


def tunnel_cloudflare(port):
    if shutil.which("cloudflared") is None:
        sh("curl -L --fail -s https://github.com/cloudflare/cloudflared/releases/latest/download/"
           "cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
           check=False)
    if shutil.which("cloudflared") is None:
        return None
    subprocess.run("pkill -f 'cloudflared tunnel' || true", shell=True)
    daemon(["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}", "--http-host-header",
            f"localhost:{port}", "--no-autoupdate"], "cloudflared")
    for _ in range(30):
        time.sleep(1.5)
        urls = re.findall(r"https://[-a-zA-Z0-9]+\.trycloudflare\.com", tail(log_path("cloudflared"), 80))
        if urls:
            return urls[-1]
    return None


def verify(base, model, api_key):
    try:
        r = http_json(base + "/chat/completions", {"model": model, "max_tokens": 8, "temperature": 0,
                                                   "messages": [{"role": "user", "content": "Reply PONG only."}]},
                      timeout=600, api_key=api_key)
        return True, (r["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        return False, str(e)


def main():
    s = SETTINGS
    print(f"=== MUYAH server: engine={s['ENGINE']} model={s['MODEL']} ctx={s['CTX']} ===", flush=True)
    for pat in ("vllm serve", "coli serve"):
        subprocess.run(f"pkill -f '{pat}' || true", shell=True)
    start = {"vllm": start_vllm, "colibri": start_colibri}.get(s["ENGINE"])
    if start is None:
        raise SystemExit(f"Unknown ENGINE {s['ENGINE']!r}; use vllm or colibri")
    proc, model_id = start(s)
    served = wait_ready(proc, s)
    model_id = served[0] if served else model_id
    print(f"Server ready locally. Models: {served}", flush=True)

    urls = {}
    for name in [t.strip() for t in s["TUNNELS"].split(",") if t.strip()]:
        fn = {"pinggy": tunnel_pinggy, "cloudflare": tunnel_cloudflare}.get(name)
        url = fn(s["PORT"]) if fn else None
        if url:
            urls[name] = url + "/v1"
            print(f"{name} tunnel: {url}/v1", flush=True)
    if not urls:
        print("No public tunnel could be opened; the server is only reachable inside this machine.")
        return

    best = urls.get("pinggy") or next(iter(urls.values()))
    ok, out = verify(best, model_id, s["API_KEY"])
    print(("PONG test through the tunnel OK: " + out) if ok else ("Tunnel test warning: " + out), flush=True)
    key = f" --api-key {s['API_KEY']}" if s["API_KEY"] else ""
    engine = f" --engine {s['ENGINE']}"
    print("\n" + "=" * 72)
    print("READY. On your computer run:\n")
    print(f"    muyah connect {best} --model {model_id}{engine} --context-window {s['CTX']}{key}\n")
    for name, url in urls.items():
        if url != best:
            note = "  (may drop long silent requests: HTTP 524)" if name == "cloudflare" else ""
            print(f"    alternative ({name}): muyah connect {url} --model {model_id}{engine}{key}{note}")
    print("=" * 72, flush=True)


def monitor():
    """Tail the server log live (Ctrl+C / cell stop only stops the monitor)."""
    path = log_path("server")
    with open(path, encoding="utf-8", errors="replace") as f:
        f.seek(0, 2)
        try:
            while True:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(0.3)
        except KeyboardInterrupt:
            print("monitor stopped (server keeps running)")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor()
    else:
        main()
