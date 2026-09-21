"""What a model call costs, in dollars. Never guessed.

Prices come from, in order:
  1. your own `pricing` setting: {"model-id": {"input": 0.5, "output": 1.5, "cache_read": 0.05,
     "cache_write": 0.6}} in dollars per million tokens,
  2. OpenRouter's live price list, for OpenRouter models,
  3. LiteLLM's public price table: a copy ships with MUYAH-CODE (muyah_code/data/prices.json) and is
     refreshed at most once a day into ~/.muyah/cache/prices.json, in the background.

Models on your own machine or your own server (Ollama, LM Studio, vLLM, colibri, Soup, a Colab tunnel)
cost $0. A model found nowhere is "unpriced": /usage says so instead of inventing a number.

The table also says which models accept images (used to decide whether screenshots are sent).
"""

from __future__ import annotations

import ipaddress
import json
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
REFRESH_SECONDS = 24 * 3600
BUNDLED = Path(__file__).parent / "data" / "prices.json"

# LiteLLM provider names kept in the trimmed table, and our provider ids -> their key prefix
KEEP_PROVIDERS = {"openai", "anthropic", "gemini", "groq", "deepseek", "mistral", "xai", "together_ai",
                  "fireworks_ai", "cerebras", "moonshot", "zai", "dashscope", "nvidia_nim", "openrouter",
                  "huggingface"}
PREFIX = {"gemini": "gemini", "groq": "groq", "deepseek": "deepseek", "mistral": "mistral", "xai": "xai",
          "together": "together_ai", "fireworks": "fireworks_ai", "cerebras": "cerebras", "moonshot": "moonshot",
          "zai": "zai", "qwen": "dashscope", "nvidia": "nvidia_nim", "openrouter": "openrouter",
          "huggingface": "huggingface", "openai": "", "anthropic": ""}
TUNNEL_HOSTS = (".trycloudflare.com", ".ngrok-free.app", ".ngrok.io", ".ngrok.app", ".loca.lt")


@dataclass(frozen=True)
class Price:
    """Dollars per token. cache_* are None when the provider has no separate cache price."""
    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None
    source: str = "litellm"

    def cost(self, tokens_in: int, tokens_out: int, cache_read: int = 0, cache_write: int = 0) -> float:
        """tokens_in counts every input token, cached ones included (see usage.normalize)."""
        cache_read = min(cache_read, tokens_in)
        cache_write = min(cache_write, tokens_in - cache_read)
        fresh = tokens_in - cache_read - cache_write
        return (fresh * self.input + tokens_out * self.output
                + cache_read * (self.input if self.cache_read is None else self.cache_read)
                + cache_write * (self.input if self.cache_write is None else self.cache_write))


FREE = Price(0.0, 0.0, 0.0, 0.0, source="self-hosted")


def _per_m(v) -> float | None:
    return None if v is None else round(float(v) * 1e6, 6)


def trim_litellm(raw: dict) -> dict:
    """LiteLLM's table -> {"models": {key: [in, out, cache_read, cache_write, max_input, vision]}}, $/1M."""
    models = {}
    for key, v in raw.items():
        if not isinstance(v, dict) or v.get("litellm_provider") not in KEEP_PROVIDERS:
            continue
        if v.get("mode") not in (None, "chat", "responses") or v.get("input_cost_per_token") is None:
            continue
        models[key.lower()] = [_per_m(v.get("input_cost_per_token")), _per_m(v.get("output_cost_per_token") or 0),
                               _per_m(v.get("cache_read_input_token_cost")),
                               _per_m(v.get("cache_creation_input_token_cost")),
                               int(v.get("max_input_tokens") or 0), 1 if v.get("supports_vision") else 0]
    return {"source": LITELLM_URL, "fetched": time.strftime("%Y-%m-%d"), "models": models}


def is_self_hosted(base_url: str, provider_local: bool = False) -> bool:
    """Your machine, your network, or your own tunnel (e.g. a Colab GPU): no per-token bill."""
    if provider_local:
        return True
    host = (urlparse(base_url or "").hostname or "").lower()
    if not host or host == "localhost" or host.endswith(".local") or host.endswith(TUNNEL_HOSTS):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


class Pricing:
    def __init__(self, home: Path, overrides: dict | None = None, refresh: bool = True):
        self.home = home
        self.overrides = overrides or {}
        self.refresh_enabled = refresh
        self._models: dict[str, list] = {}
        self._openrouter: dict[str, Price] = {}
        self._lock = threading.Lock()
        self._refreshing = False
        self._or_started = False
        self.fetched = ""
        self.update_error = ""        # shown by /usage --all
        self._load()

    # ------------------------------------------------------------------ data

    @property
    def cache_path(self) -> Path:
        return self.home / "cache" / "prices.json"

    def _load(self) -> None:
        for path in (BUNDLED, self.cache_path):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data.get("models"), dict):
                self._models.update(data["models"])
                self.fetched = str(data.get("fetched") or self.fetched)

    def refresh_in_background(self) -> None:
        """Download today's table if the local copy is older than a day. Never blocks, never raises."""
        if not self.refresh_enabled or self._refreshing:
            return
        try:
            if time.time() - self.cache_path.stat().st_mtime < REFRESH_SECONDS:
                return
        except OSError:
            pass
        self._refreshing = True
        threading.Thread(target=self._refresh, name="muyah-prices", daemon=True).start()

    def _refresh(self) -> None:
        try:
            with urllib.request.urlopen(LITELLM_URL, timeout=30) as resp:
                table = trim_litellm(json.loads(resp.read().decode("utf-8")))
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(table, separators=(",", ":")), encoding="utf-8")
            tmp.replace(self.cache_path)
            with self._lock:
                self._models.update(table["models"])
                self.fetched = table["fetched"]
        except (OSError, ValueError, KeyError) as e:   # offline or the list moved: the bundled copy keeps working
            self.update_error = f"price list update failed: {e}"
        finally:
            self._refreshing = False

    def load_openrouter(self, fetch=None) -> None:
        """OpenRouter's own live prices (their table changes often and covers every model they route)."""
        if self._openrouter:
            return
        try:
            if fetch is None:
                with urllib.request.urlopen(OPENROUTER_URL, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            else:
                data = fetch()
        except (OSError, ValueError) as e:
            self.update_error = f"OpenRouter prices unavailable ({e}); using the bundled table"
            return
        for m in data.get("data") or []:
            p = m.get("pricing") or {}
            try:
                price = Price(float(p.get("prompt") or 0), float(p.get("completion") or 0),
                              float(p["input_cache_read"]) if p.get("input_cache_read") else None,
                              float(p["input_cache_write"]) if p.get("input_cache_write") else None, "openrouter")
            except (TypeError, ValueError):
                continue
            self._openrouter[str(m.get("id", "")).lower()] = price

    # ------------------------------------------------------------------ lookup

    def _entry(self, model: str, provider: str = "") -> list | None:
        m = (model or "").lower().strip()
        prefix = PREFIX.get(provider or "")
        last = m.rsplit("/", 1)[-1]
        candidates = []
        if prefix:
            candidates += [f"{prefix}/{m}", f"{prefix}/{last}"]
        candidates += [m, last] if prefix is not None else [m]
        if provider == "anthropic" or m.startswith("claude"):
            candidates.append(m.split("@")[0])
        with self._lock:
            for key in candidates:
                if key in self._models:
                    return self._models[key]
        return None

    def price(self, model: str, provider: str = "", base_url: str = "", local: bool = False) -> Price | None:
        own = self.overrides.get(model)       # first: a local proxy (e.g. LiteLLM) can forward to a paid API
        if isinstance(own, dict):
            return Price(float(own.get("input") or 0) / 1e6, float(own.get("output") or 0) / 1e6,
                         None if own.get("cache_read") is None else float(own["cache_read"]) / 1e6,
                         None if own.get("cache_write") is None else float(own["cache_write"]) / 1e6, "settings")
        if is_self_hosted(base_url, local):
            return FREE
        if provider == "openrouter":
            if not self._openrouter and not self._or_started and self.refresh_enabled:
                self._or_started = True    # first call: fetch in the background, use LiteLLM's copy meanwhile
                threading.Thread(target=self.load_openrouter, name="muyah-or-prices", daemon=True).start()
            found = self._openrouter.get((model or "").lower())
            if found is not None:
                return found
        entry = self._entry(model, provider)
        if entry is None:
            return None
        per = [None if x is None else x / 1e6 for x in entry[:4]]
        return Price(per[0] or 0.0, per[1] or 0.0, per[2], per[3])

    def supports_vision(self, model: str, provider: str = "") -> bool | None:
        entry = self._entry(model, provider)
        return None if entry is None else bool(entry[5])


def money(value: float | None) -> str:
    if value is None:
        return "unpriced"
    if value == 0:
        return "$0"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"
