"""Several models working together: roles, escalation and fallback providers.

Set models per role in settings (anything not set uses your main model):

    "models": {
      "explore":   "groq:llama-3.1-8b-instant",   # the explore sub-agent: many cheap reads
      "edit":      "deepseek-chat",                # the editor sub-agent applies described changes
      "summarize": "gemini:gemini-2.5-flash",      # compaction, learning, web page summaries
      "strong":    "anthropic:claude-opus-5"       # escalation target when the main model is stuck
    },
    "fallback": ["openrouter", "groq:llama-3.3-70b-versatile"]

A model spec is one of:
  * a saved profile name (`/profile` lists them),
  * "provider:model" with a provider from `/provider` (its saved key is used; your main key never leaks
    to another provider),
  * a bare model id, served by the same endpoint as your main model.

Escalation goes one step up the ladder: a role model -> main -> strong. Fallback is used for one request when
the provider fails after its own retries (rate limit, server error, network): the next entry in "fallback"
answers instead, and the terminal says so.
"""

from __future__ import annotations

import copy
import threading

from muyah_code.config import Config, ConfigError, load_config, resolve_credential

ROLES = ("main", "explore", "edit", "summarize", "verify", "btw", "strong")
ROLE_HELP = {
    "explore": "the explore sub-agent (reading and searching)",
    "edit": "the editor sub-agent (applies changes the main model describes)",
    "summarize": "compaction, learning and web page summaries",
    "verify": "/verify e2e",
    "btw": "/btw side questions",
    "strong": "where the main model escalates when it is stuck",
}
FALLBACK_KINDS = {"rate_limit", "server", "network", "timeout"}


class ModelPool:
    def __init__(self, cfg: Config, main, watch=None):
        """main: the main client; watch(client) wires usage accounting into every client it makes."""
        self.cfg = cfg
        self.main = main
        self.watch = watch
        self._clients: dict[str, object] = {}
        self._windows: dict[int, tuple[int, str]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ specs

    def spec(self, role: str) -> str:
        if role == "main":
            return ""
        return str(self.cfg.get(f"models.{role}") or "").strip()

    def configured(self) -> dict[str, str]:
        return {r: s for r in ROLES if (s := self.spec(r))}

    def resolve(self, name: str | None) -> str:
        """A role name or a spec -> a spec ("" = the main model)."""
        name = (name or "").strip()
        if not name or name == "main":
            return ""
        return self.spec(name) if name in ROLES else name

    def fallbacks(self) -> list[str]:
        value = self.cfg.get("fallback") or []
        return [value] if isinstance(value, str) else [str(v) for v in value if v]

    # ------------------------------------------------------------------ clients

    def set_main(self, client) -> None:
        with self._lock:
            self.main = client
            self._clients.clear()      # bare-model specs follow the main endpoint

    def get(self, role_or_spec: str | None):
        spec = self.resolve(role_or_spec)
        if not spec:
            return self.main
        with self._lock:
            client = self._clients.get(spec)
        if client is None:
            client = self._build(spec)
            with self._lock:
                self._clients.setdefault(spec, client)
                client = self._clients[spec]
        return client

    def config_for(self, spec: str) -> Config:
        """The settings a spec stands for (a copy: the session's own settings are never changed)."""
        from muyah_code.providers import BY_ID, build_provider_profile

        profiles = self.cfg.profiles()
        if spec in profiles:
            return load_config(cwd=self.cfg.project_root, overrides={"profile": spec})
        provider_id, sep, model = spec.partition(":")
        if sep and provider_id in BY_ID:
            if provider_id in profiles:
                fresh = load_config(cwd=self.cfg.project_root, overrides={"profile": provider_id})
                fresh.set("model", model or fresh.get("model"))
                return fresh
            p = BY_ID[provider_id]
            fresh = Config(copy.deepcopy(self.cfg.data), self.cfg.project_root, self.cfg.home, list(self.cfg.sources))
            for key in ("api", "engine", "context_window", "effort", "extra_headers"):
                fresh.data.pop(key, None)
            fresh.data.update(build_provider_profile(p, model or p.default_model))
            fresh.data.setdefault("api", "openai")
            if p.local:
                fresh.set("api_key", "none")
            resolve_credential(fresh)
            if not p.local and fresh.get("api_key") in (None, "", "none"):
                raise ConfigError(f"No API key for {p.name}: run /provider {p.id} once to save one.")
            return fresh
        # a bare model id on the main endpoint
        fresh = Config(copy.deepcopy(self.cfg.data), self.cfg.project_root, self.cfg.home, list(self.cfg.sources))
        fresh.set("model", spec)
        fresh.set("context_window", 0)
        return fresh

    def _build(self, spec: str):
        from muyah_code.llm.client import make_client

        cfg = self.config_for(spec)
        client = make_client(cfg)
        client.spec = spec
        client.role_cfg = cfg
        if self.watch is not None:
            self.watch(client, cfg.get("provider") or "")
        return client

    def window_for(self, client) -> tuple[int, str]:
        """The context window of a pool client (probed once)."""
        from muyah_code.app import resolve_context_window

        key = id(client)
        if key not in self._windows:
            cfg = getattr(client, "role_cfg", None) or self.cfg
            self._windows[key] = resolve_context_window(cfg, client)
        return self._windows[key]

    # ------------------------------------------------------------------ ladder

    def step_up(self, client):
        """The next bigger model for escalation, or None: a role model -> main -> strong."""
        if client is not self.main:
            return self.main
        strong = self.spec("strong")
        if strong:
            target = self.get("strong")
            return None if target is client else target
        return None

    def label(self, client) -> str:
        spec = getattr(client, "spec", "")
        role = next((r for r in ROLES if r != "main" and spec and self.spec(r) == spec), "")
        name = getattr(client, "model", "?")
        if client is self.main:
            return f"main ({name})"
        return f"{role} ({name})" if role else name


class RoleClient:
    """Stands in for a role's model and builds its client on first use, so a missing key for a role never
    breaks startup: the main model is used instead, and you are told once."""

    def __init__(self, pool: ModelPool, role: str, warn=None):
        self.pool = pool
        self.role = role
        self.warn = warn
        self._warned = False

    def client(self):
        try:
            return self.pool.get(self.role)
        except (ConfigError, ValueError) as e:
            if not self._warned and self.warn is not None:
                self._warned = True
                self.warn(f"The {self.role} model is not usable ({e}); using the main model for it.")
            return self.pool.main

    def chat(self, messages, **kwargs):
        return self.client().chat(messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self.client(), name)


CLAUDE_ALIASES = {"opus": "claude-opus-5", "sonnet": "claude-sonnet-5", "haiku": "claude-haiku-4-5"}


def agent_model(value, main_api: str) -> str | None:
    """A sub-agent definition's `model:` -> a role or spec. Claude Code's aliases (opus/sonnet/haiku) only mean
    something on Claude; `inherit` means the main model."""
    v = str(value or "").strip()
    if not v or v.lower() == "inherit":
        return None
    if v.lower() in CLAUDE_ALIASES:
        return CLAUDE_ALIASES[v.lower()] if main_api == "anthropic" else None
    return v


def describe_roles(pool: ModelPool) -> list[tuple[str, str, str]]:
    """(role, model, what it is for) for /models roles."""
    rows = [("main", getattr(pool.main, "model", "?"), "everything not listed below")]
    for role in ROLES[1:]:
        spec = pool.spec(role)
        rows.append((role, spec or "(main)", ROLE_HELP[role]))
    return rows


__all__ = ["FALLBACK_KINDS", "ModelPool", "ROLES", "describe_roles"]
