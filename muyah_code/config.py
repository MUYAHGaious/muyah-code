"""Layered configuration.

Precedence (lowest -> highest):
    built-in defaults
    ~/.muyah/settings.json                 (user)
    <project>/.muyah/settings.json         (project, commit to git)
    <project>/.muyah/settings.local.json   (project, personal, git-ignored)
    --settings <file>                      (explicit)
    active profile overlay                 ("profile": "<name>" picks from "profiles")
    MUYAH_* environment variables
    CLI flags

Dicts are deep-merged; the permission rule lists (allow/ask/deny) are concatenated
so that a project can add rules without erasing the user's.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "base_url": "http://localhost:8000/v1",
    "api_key": "none",
    "model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
    "temperature": 0.2,
    "max_tokens": 4096,
    # 0 = probe the server (/v1/models max_model_len), falling back to 16384
    "context_window": 0,
    "request_timeout": 300,
    "stream": True,
    # auto: try native function calling, fall back to text protocol if the server rejects it
    "tool_mode": "auto",
    "max_steps": 60,
    "bash_timeout": 120,
    # auto | bash | powershell | cmd
    "shell": "auto",
    "compact_threshold": 0.8,
    "max_tool_output_chars": 24000,
    "permissions": {"mode": "default", "allow": [], "ask": [], "deny": []},
    "hooks": {},
    "profiles": {},
    "profile": None,
    "learning": {"enabled": True, "max_lessons_in_prompt": 5, "reflect": True},
    "compat": {"claude_skills": True, "claude_md": True},
    "extra_headers": {},
    "theme": "teal",
    # engine preset applied by `muyah connect --engine` / `muyah serve` (see backends.ENGINE_PRESETS)
    "engine": None,
    # "openai" = any OpenAI-compatible server; "anthropic" = native Claude Messages API
    "api": "openai",
    # provider id from the /provider catalog, and which saved key (~/.muyah/credentials.json) to use
    "provider": None,
    "credential": None,
    # Claude only: effort level (None = xhigh on Opus 5 / Sonnet 5) and server-side refusal fallbacks
    "effort": None,
    "fallbacks": True,
    # /verify: auto = off | quick | full (after a turn that changed files); test/lint = your own commands
    "verify": {"auto": "off", "test": None, "lint": None, "timeout": 900},
    # dollars: 0 = no limit. At 80% you are warned, at 100% the agent asks before going on.
    "budget": {"session_usd": 0, "daily_usd": 0},
    # auto | full | lean: lean = short prompt + 6 core tools for small models / windows under 32k (agent/lean.py)
    "prompt_profile": "auto",
    # model per role (see llm/pool.py): main, explore, edit, summarize, verify, btw, strong
    "models": {},
    # profiles or provider:model specs that answer when the main provider fails (rate limit, 5xx, network)
    "fallback": [],
    # update: refresh the public price list once a day; models: your own prices ($ per 1M tokens)
    "pricing": {"update": True, "models": {}},
}

LIST_MERGE_KEYS = {"allow", "ask", "deny"}
ENV_MAP = {
    "MUYAH_BASE_URL": "base_url",
    "MUYAH_API_KEY": "api_key",
    "MUYAH_MODEL": "model",
    "MUYAH_PROFILE": "profile",
}
LEGACY_CONFIG = Path.home() / ".colab_code_agent_config.json"


def muyah_home() -> Path:
    return Path(os.environ.get("MUYAH_HOME") or (Path.home() / ".muyah"))


def find_project_root(start: Path) -> Path:
    """Nearest ancestor containing .git or a project .muyah folder; otherwise the start directory.

    The global MUYAH home (~/.muyah) is not a project marker: otherwise every folder under your home
    directory would be one project rooted at the home directory."""
    start = start.resolve()
    global_homes = {muyah_home().resolve(), (Path.home() / ".muyah").resolve()}
    for p in (start, *start.parents):
        if (p / ".git").exists():
            return p
        marker = p / ".muyah"
        if marker.is_dir() and marker.resolve() not in global_homes:
            return p
    return start


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        elif k in LIST_MERGE_KEYS and isinstance(v, list) and isinstance(out.get(k), list):
            out[k] = out[k] + [x for x in v if x not in out[k]]
        else:
            out[k] = copy.deepcopy(v)
    return out


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"Invalid settings file {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"Settings file {path} must contain a JSON object")
    return data


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class ConfigError(Exception):
    pass


def migrate_legacy(home: Path) -> bool:
    """Import the old colab-code config the first time MUYAH-CODE runs."""
    user_file = home / "settings.json"
    if user_file.exists() or not LEGACY_CONFIG.exists():
        return False
    try:
        old = json.loads(LEGACY_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    new: dict[str, Any] = {}
    for key in ("base_url", "model", "api_key"):
        if old.get(key):
            new[key] = old[key]
    if old.get("command_timeout"):
        new["bash_timeout"] = int(old["command_timeout"])
    if new.get("base_url"):
        new["profiles"] = {"colab": {k: new[k] for k in ("base_url", "model", "api_key") if k in new}}
    write_json(user_file, new)
    return True


@dataclass
class Config:
    data: dict
    project_root: Path
    home: Path
    sources: list[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def set(self, key: str, value: Any) -> None:
        """Set a value for this process only."""
        parts = key.split(".")
        cur = self.data
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    @property
    def user_settings_path(self) -> Path:
        return self.home / "settings.json"

    @property
    def project_settings_path(self) -> Path:
        return self.project_root / ".muyah" / "settings.json"

    @property
    def local_settings_path(self) -> Path:
        return self.project_root / ".muyah" / "settings.local.json"

    def persist(self, key: str, value: Any, scope: str = "user") -> Path:
        """Write a value to a settings file (and apply it to this process)."""
        path = {
            "user": self.user_settings_path,
            "project": self.project_settings_path,
            "local": self.local_settings_path,
        }[scope]
        data = read_json(path)
        parts = key.split(".")
        cur = data
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
        write_json(path, data)
        self.set(key, value)
        return path

    def append_rule(self, kind: str, rule: str, scope: str = "local") -> Path:
        path = self.local_settings_path if scope == "local" else self.project_settings_path
        data = read_json(path)
        rules = data.setdefault("permissions", {}).setdefault(kind, [])
        if rule not in rules:
            rules.append(rule)
        write_json(path, data)
        return path

    def profiles(self) -> dict:
        return self.data.get("profiles") or {}

    def apply_profile(self, name: str) -> None:
        prof = self.profiles().get(name)
        if prof is None:
            raise ConfigError(f"Unknown profile '{name}'. Known: {', '.join(self.profiles()) or 'none'}")
        self.data = deep_merge(self.data, prof)
        self.data["profile"] = name


def resolve_credential(cfg: Config) -> None:
    """Fill api_key from the saved key (or the provider's env var) named by "credential"."""
    cred = cfg.get("credential")
    if not cred or cfg.get("api_key") not in (None, "", "none", "dummy"):
        return
    from muyah_code.providers import resolve_key

    key = resolve_key(cfg.home, cred)
    if key:
        cfg.set("api_key", key)


def load_config(
    cwd: Path | None = None,
    settings_file: str | None = None,
    overrides: dict | None = None,
) -> Config:
    cwd = (cwd or Path.cwd()).resolve()
    home = muyah_home()
    home.mkdir(parents=True, exist_ok=True)
    migrate_legacy(home)
    root = find_project_root(cwd)

    data = copy.deepcopy(DEFAULTS)
    sources = ["defaults"]
    layers = [home / "settings.json", root / ".muyah" / "settings.json", root / ".muyah" / "settings.local.json"]
    if settings_file:
        layers.append(Path(settings_file).expanduser().resolve())
    for layer in layers:
        layer_data = read_json(layer)
        if layer_data:
            data = deep_merge(data, layer_data)
            sources.append(str(layer))

    cfg = Config(data=data, project_root=root, home=home, sources=sources)

    env_over = {v: os.environ[k] for k, v in ENV_MAP.items() if os.environ.get(k)}
    overrides = dict(overrides or {})
    profile = overrides.pop("profile", None) or env_over.pop("profile", None) or data.get("profile")
    if profile:
        cfg.apply_profile(profile)
        sources.append(f"profile:{profile}")
    resolve_credential(cfg)
    for key, value in {**env_over, **{k: v for k, v in overrides.items() if v is not None}}.items():
        cfg.set(key, value)
    return cfg
