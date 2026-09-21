"""Interactive provider setup used by `/provider` (REPL) and `muyah login` (CLI).

    pick a provider (↑/↓) -> paste a key (or reuse a saved / environment key) -> key is verified by listing
    models -> pick a model (↑/↓, best first) -> one tiny test reply -> profile saved and activated
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from rich.console import Console
from rich.markup import escape

from muyah_code.backends import probe, smoke_test
from muyah_code.config import Config
from muyah_code.llm.models import is_billing_error
from muyah_code.providers import (
    PROVIDERS,
    Provider,
    build_provider_profile,
    env_key,
    get_provider,
    key_status,
    load_credentials,
    looks_wrong,
    mask,
    rank_models,
    save_credential,
)

MAX_KEY_ATTEMPTS = 3
MAX_MODEL_ATTEMPTS = 3
SHOWN_MODELS = 12
OTHER = "__other__"
UNAVAILABLE = ("404", "not found", "no longer available", "does not exist", "not available", "not supported",
               "decommissioned", "deprecated", "model_not_found", "unknown model", "invalid model")


class Prompts(Protocol):
    def select(self, message: str, options: list[tuple[Any, str]], default: Any = None) -> Any | None: ...
    def ask(self, message: str, password: bool = False, default: str = "") -> str: ...
    def confirm(self, message: str, default: bool = True) -> bool: ...


def error_message(text: str) -> str:
    """Pull the human sentence out of a raw provider error payload (JSON or Python repr)."""
    for pattern in (r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', r"'message'\s*:\s*'((?:[^'\\]|\\.)*)'"):
        m = re.search(pattern, text)
        if m:
            return " ".join(m.group(1).split())[:300]
    return " ".join(text.split())[:300]


def is_model_unavailable(text: str) -> bool:
    low = text.lower()
    return any(u in low for u in UNAVAILABLE)


def suggested_model(text: str) -> str | None:
    """Providers often name the replacement: 'Please update your code to use models/gemini-3.1-pro-preview'."""
    m = re.search(r"\buse (?:the )?(?:model )?(?:models/)?([A-Za-z][\w.\-/:]*\d[\w.\-/:]*)", text)
    return m.group(1).rstrip(".,;") if m else None


def provider_options(cfg: Config) -> list[tuple[str, str]]:
    active = cfg.get("provider")
    options = []
    for p in PROVIDERS:
        status = key_status(cfg.home, p)
        if p.id == active:
            status = (status + ", " if status else "") + "active"
        options.append((p.id, f"{p.name}" + (f"  ({status})" if status else "")))
    return options


def _list_models(p: Provider, key: str) -> tuple[list[str] | None, str | None]:
    """Returns (models, error). models=None + error means the key/endpoint failed."""
    if p.api == "anthropic":
        from muyah_code.llm.anthropic_client import AnthropicClient

        try:
            return [m["id"] for m in AnthropicClient(model=p.default_model, api_key=key).list_models()], None
        except Exception as e:  # SDK error classes vary; report the message
            status = getattr(e, "status_code", None)
            return None, "the key was rejected (401)" if status == 401 else f"{e.__class__.__name__}: {e}"
    ep = probe(p.base_url, key, timeout=20)
    if ep.ok:
        return ep.models, None
    if ep.error and ep.error.startswith("401"):
        return None, "the key was rejected (401)"
    return [], ep.error  # some providers don't expose /models: continue with a typed model id


def _test(p: Provider, key: str, model: str) -> tuple[bool, str]:
    if p.api == "anthropic":
        from muyah_code.llm.anthropic_client import AnthropicClient

        try:
            reply = AnthropicClient(model=model, api_key=key, fallbacks=False).chat(
                [{"role": "user", "content": "Reply with PONG only."}], max_tokens=1024)
            return True, reply.content.strip()[:60]
        except Exception as e:
            return False, str(e)
    return smoke_test(p.base_url, model, key or "none", timeout=120)


def _choose_model(p: Provider, models: list[str], ui: Prompts, preset: str | None,
                  suggested: str | None = None) -> str | None:
    if preset:
        return preset
    ranked = rank_models(p, models)
    # The live list beats any hard-coded default: providers retire models all the time.
    default = suggested or (ranked[0] if ranked else p.default_model)
    if not ranked:
        return ui.ask("Model id: ", default=default or "").strip() or None
    shown = ranked[:SHOWN_MODELS]
    if default and default not in shown:
        shown = [default] + shown[: SHOWN_MODELS - 1]
    options = [(m, m + ("   (suggested)" if m == suggested else "")) for m in shown]
    options.append((OTHER, f"Other… ({len(ranked)} available, type an id)"))
    picked = ui.select("Choose a model", options, default=default)
    if picked == OTHER:
        return ui.ask("Model id: ").strip() or None
    return picked


def setup_provider(cfg: Config, console: Console, ui: Prompts, choice: str | None = None, key: str | None = None,
                   model: str | None = None, test: bool = True) -> str | None:
    """Run the whole flow. Returns the activated profile name, or None if cancelled/failed."""
    if not choice:
        choice = ui.select("Choose a provider", provider_options(cfg), default=cfg.get("provider"))
        if not choice:
            return None
    p = get_provider(choice)
    if p is None:
        console.print(f"[red]Unknown provider '{escape(choice)}'.[/] Run /provider to pick from the list.")
        return None
    console.print(f"[bold]{escape(p.name)}[/]")

    source = "pasted"
    if p.local:
        key = "none"
        ep = probe(p.base_url, timeout=3)
        if not ep.ok:
            hint = {"ollama": "muyah serve ollama -m qwen2.5-coder:14b", "llamacpp": "muyah serve llamacpp -m model.gguf",
                    "vllm": "muyah serve vllm -m <hf-model>", "lmstudio": "open LM Studio and start its server"}
            console.print(f"[yellow]{p.name} is not running at {p.base_url}.[/] Start it ({hint.get(p.id, '')}) "
                          "and try again.")
            return None
        models: list[str] | None = ep.models
    else:
        if not key:
            saved = load_credentials(cfg.home).get(p.id)
            env = env_key(p)
            if saved and ui.confirm(f"Use your saved {p.name} key {mask(saved)}?"):
                key, source = saved, "saved"
            elif env and ui.confirm(f"Use ${env[0]} ({mask(env[1])})?"):
                key, source = env[1], "env"
        models = None
        for attempt in range(MAX_KEY_ATTEMPTS):
            if not key:
                if p.key_url:
                    console.print(f"[dim]Get a key at {p.key_url}[/]")
                key = ui.ask(f"Paste your {p.name} API key: ", password=True).strip()
                source = "pasted"
                if not key:
                    console.print("Cancelled.")
                    return None
                problem = looks_wrong(p, key)
                if problem and not ui.confirm(f"{problem[0].upper() + problem[1:]}. Use it anyway?", default=False):
                    key = None
                    continue
            with console.status("Checking the key..."):
                models, err = _list_models(p, key)
            if models is not None:
                if err:
                    console.print(f"[dim]Could not list models ({escape(err)}); you can type a model id.[/]")
                break
            console.print(f"[red]{escape(err or 'key check failed')}[/]")
            key = None
            if attempt == MAX_KEY_ATTEMPTS - 1:
                return None
        console.print(f"[green]Key OK[/] ({mask(key)}, {source}).")

    available = list(models or [])
    chosen = _choose_model(p, available, ui, model)
    for attempt in range(MAX_MODEL_ATTEMPTS if test else 0):
        if not chosen:
            break
        with console.status(f"Testing {chosen}..."):
            ok, out = _test(p, key or "none", chosen)
        if ok:
            console.print(f"[green]Working.[/] {escape(chosen)} replied {out!r}")
            break
        if not is_billing_error(out) and is_model_unavailable(out) and attempt < MAX_MODEL_ATTEMPTS - 1:
            # Listed but not usable (retired, not enabled for this account...): let the user pick again.
            suggestion = suggested_model(out)
            console.print(f"[yellow]{escape(chosen)} is not available:[/] {escape(error_message(out))}")
            available = [m for m in available if m.split("/")[-1] != chosen]
            chosen = _choose_model(p, available, ui, None, suggested=suggestion)
            continue
        if is_billing_error(out):
            # The key is valid (it listed models) but the account cannot pay: no other model will work either.
            if not p.local and source == "pasted":
                save_credential(cfg.home, p.id, key)
            where = f" ({p.billing_url})" if p.billing_url else ""
            console.print(
                f"[yellow]Your {escape(p.name)} key works, but the account has no credits or quota.[/]\n"
                f"Add credits / a payment method in your {escape(p.name)} console{where}, then run "
                f"`/provider {p.id}` again - the key is saved, so you won't need to paste it."
                f"\n[dim]Provider said: {escape(error_message(out))}[/]")
            return None
        console.print(f"[red]Test reply failed:[/] {escape(error_message(out))}\n"
                      "Nothing was saved. Run /provider again and pick another model.")
        return None
    if not chosen:
        console.print("Cancelled.")
        return None

    if not p.local and source == "pasted":
        path = save_credential(cfg.home, p.id, key)
        console.print(f"[dim]Key saved to {path} (only your user account can read it; never in settings.json).[/]")
    elif source == "env":
        console.print(f"[dim]Using the key from ${env_key(p)[0]} (not copied to disk).[/]")
    cfg.persist(f"profiles.{p.id}", build_provider_profile(p, chosen), scope="user")
    cfg.persist("profile", p.id, scope="user")
    console.print(f"[bold green]Ready:[/] {escape(p.name)} · {escape(chosen)}")
    return p.id
