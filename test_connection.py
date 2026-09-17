"""
Test Colab Tunnel Connection
============================
Quick sanity check to verify your Google Colab vLLM server is reachable.
"""

import sys
import json
from pathlib import Path
from openai import OpenAI

CONFIG_FILE = Path(".agent_config.json")

def test_connection():
    base_url = "http://localhost:8000/v1"
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
                base_url = cfg.get("base_url", base_url)
        except Exception:
            pass

    if len(sys.argv) > 1:
        base_url = sys.argv[1].rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"

    print(f"Testing connection to: {base_url}")

    client = OpenAI(base_url=base_url, api_key="dummy")
    try:
        models = client.models.list()
        print("[+] Connected successfully!")
        print("Available models:")
        for m in models.data:
            print(f"  - {m.id}")
    except Exception as e:
        print(f"[-] Connection failed: {e}")
        print("\nMake sure:")
        print("1. Colab cell is actively running")
        print("2. The tunnel URL is correct (e.g. https://xxxx.trycloudflare.com/v1)")

if __name__ == "__main__":
    test_connection()
