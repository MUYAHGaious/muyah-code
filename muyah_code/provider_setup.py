"""Interactive provider setup used by `/provider` (REPL) and `muyah login` (CLI).

    pick a provider -> paste a key (or reuse a saved / environment key) -> key is verified by listing models
    -> pick a model (sensible default preselected) -> one tiny test reply -> profile saved and activated
"""

from __future__ import annotations

import re
from collections.abc import Callable

from rich.console import Console
from rich.markup import escape
from rich.table import Table

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

Ask = Callable[..., str]  # ask(prompt, password=False) -> str
MAX_KEY_ATTEMPTS = 3
SHOWN_MODELS = 12


MAX_MODEL_ATTEMPTS = 3
UNAVAILABLE = ("404", "not found", "no longer available", "does not exist", "not available", "not supported",
               "decommissioned", "deprecated", "model_not_found", "unknown model", "invalid model")


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


def provider_table(cfg: Config) -> Table:
    t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t.add_column("#", justify="right")
    t.add_column("provider")
    t.add_column("status")
    active = cfg.get("provider")
    for i, p in enumerate(PROVIDERS, 1):
        status = key_status(cfg.home, p)
        if p.id == active:
            status = (status + " · " if status else "") + "[bold]active[/]"
        t.add_row(str(i), p.name, status)
    return t


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
    return [], ep.error  # some providers don't expose /models: continue with the default model


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


def _choose_model(p: Provider, models: list[str], ask: Ask, console: Console, preset: str | None,
                  suggested: str | None = None) -> str:
    if preset:
        return preset
    ranked = rank_models(p, models)
    # The live list beats any hard-coded default: providers retire models all the time.
    default = suggested or (ranked[0] if ranked else p.default_model)
    if not ranked:
        answer = ask(f"Model id [{default or 'required'}]: ").strip()
        return answer or default
    shown = ranked[:SHOWN_MODELS]
    if default and default not in shown:
        shown = [default] + shown[: SHOWN_MODELS - 1]
    console.print(f"[bold]Models[/] ({len(ranked)} available{', showing the best matches' if len(ranked) > len(shown) else ''}):")
    for i, m in enumerate(shown, 1):
        mark = "  [dim]<- default[/]" if m == default else ""
        console.print(f"  {i:>2}. {escape(m)}{mark}")
    answer = ask("Model number or id [Enter = default]: ").strip()
    if not answer:
        return default
    if answer.isdigit() and 1 <= int(answer) <= len(shown):
        return shown[int(answer) - 1]
    return answer


def setup_provider(cfg: Config, console: Console, ask: Ask, choice: str | None = None, key: str | None = None,
                   model: str | None = None, test: bool = True) -> str | None:
    """Run the whole flow. Returns the activated profile name, or None if cancelled/failed."""
    if not choice:
        console.print(provider_table(cfg))
        choice = ask("Provider (number or name): ").strip()
        if not choice:
            return None
    p = get_provider(choice)
    if p is None:
        console.print(f"[red]Unknown provider '{escape(choice)}'.[/] Try a number from the list.")
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
            if saved and ask(f"Use your saved {p.name} key {mask(saved)}? [Y/n] ").strip().lower() in ("", "y", "yes"):
                key, source = saved, "saved"
            elif env and ask(f"Found ${env[0]} ({mask(env[1])}). Use it? [Y/n] ").strip().lower() in ("", "y", "yes"):
                key, source = env[1], "env"
        models = None
        for attempt in range(MAX_KEY_ATTEMPTS):
            if not key:
                if p.key_url:
                    console.print(f"[dim]Get a key at {p.key_url}[/]")
                key = ask(f"Paste your {p.name} API key: ", password=True).strip()
                source = "pasted"
                if not key:
                    console.print("Cancelled.")
                    return None
                problem = looks_wrong(p, key)
                if problem and ask(f"Hmm, {problem}. Use it anyway? [y/N] ").strip().lower() not in ("y", "yes"):
                    key = None
                    continue
            with console.status("Checking the key..."):
                models, err = _list_models(p, key)
            if models is not None:
                if err:
                    console.print(f"[dim]Could not list models ({escape(err)}); you can still type a model id.[/]")
                break
            console.print(f"[red]{escape(err or 'key check failed')}[/]")
            key = None
            if attempt == MAX_KEY_ATTEMPTS - 1:
                return None
        console.print(f"[green]Key OK[/] ({mask(key)}, {source}).")

    available = list(models or [])
    chosen = _choose_model(p, available, ask, console, model)
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
            chosen = _choose_model(p, available, ask, console, None, suggested=suggestion)
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
        console.print("[red]No model selected.[/]")
        return None

    if not p.local and source == "pasted":
        path = save_credential(cfg.home, p.id, key)
        console.print(f"[dim]Key saved to {path} (only your user account can read it; never in settings.json).[/]")
    elif source == "env":
        console.print(f"[dim]Using the key from ${env_key(p)[0]} (not copied to disk).[/]")
    cfg.persist(f"profiles.{p.id}", build_provider_profile(p, chosen), scope="user")
    cfg.persist("profile", p.id, scope="user")
    console.print(f"[bold green]Ready:[/] {escape(p.name)} · {escape(chosen)} (profile '{p.id}', now the default).")
    return p.id
