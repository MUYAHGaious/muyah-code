"""
Google Colab Server Script
==========================
Run this code in a single Google Colab cell (with A100 or L4 GPU + High-RAM enabled).
It will:
1. Install vLLM and Cloudflared
2. Download & serve Qwen/Qwen2.5-Coder-32B-Instruct-AWQ via vLLM
3. Expose the port via a secure Cloudflare Tunnel
4. Output the public OpenAI-compatible URL
"""

import subprocess
import time
import re
import sys
import os

print("--> [1/4] Installing vLLM and Cloudflared...")
subprocess.run("pip install -q -U vllm", shell=True, check=True)
# Fix Colab PyTorch/TorchAudio CUDA version mismatch
subprocess.run("pip uninstall -y torchaudio", shell=True, check=True)
subprocess.run("pip install -q -U --force-reinstall torchaudio --index-url https://download.pytorch.org/whl/cu130", shell=True, check=True)
subprocess.run("wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb", shell=True, check=True)
subprocess.run("dpkg -i cloudflared-linux-amd64.deb", shell=True, check=True)

# Fix FlashInfer on newer Blackwell/Ada GPUs (RTX PRO 6000 / sm120)
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_USE_FLASHINFER"] = "0"

# Select Model
# Qwen2.5-Coder-32B-Instruct-AWQ fits nicely in 40GB A100 or 24GB L4 and generates blazing-fast code.
# Or use: "jiangchengchengNLP/llama3.3-70B-instruct-abliterated-awq" (requires A100-80GB / PRO 6000)
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
MAX_MODEL_LEN = 16384  # Up to 32768 depending on memory needs

# Auto-detect parser: Llama 3 models need llama3_json; Qwen / Mistral / Hermes use hermes
tool_parser = "llama3_json" if "llama" in MODEL_NAME.lower() else "hermes"
print(f"--> [2/4] Starting vLLM with {MODEL_NAME} (tool-parser: {tool_parser})...")

vllm_cmd = [
    "vllm", "serve",
    MODEL_NAME,
    "--port", "8000",
    "--host", "0.0.0.0",
    "--max-model-len", str(MAX_MODEL_LEN),
    "--enable-auto-tool-choice",
    "--tool-call-parser", tool_parser,
    "--trust-remote-code"
]
vllm_proc = subprocess.Popen(vllm_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ})

print("--> [3/4] Initializing Cloudflare Tunnel...")
tunnel_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

tunnel_url = None
while True:
    line = tunnel_proc.stdout.readline()
    if not line:
        break
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        tunnel_url = match.group(0)
        break

print("\n" + "=" * 65)
print("  🚀 YOUR LLM BACKEND IS LIVE!")
print(f"  API Base URL:  {tunnel_url}/v1")
print(f"  Model Name:    {MODEL_NAME}")
print("=" * 65 + "\n")
print("Enter this URL into your local Claude Code CLI or Aider.\n")
print("Streaming vLLM loading logs (Wait until you see 'Application startup complete'):")

while True:
    v_line = vllm_proc.stdout.readline()
    if v_line:
        print(v_line, end="")
    time.sleep(0.01)
