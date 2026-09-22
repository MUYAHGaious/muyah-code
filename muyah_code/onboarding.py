"""First run: nobody should be stuck on "which model?" before they can type.

When nothing is set up (no provider, no profile, no endpoint of your own), `muyah` asks once:

  * a model already running on this computer (Ollama, LM Studio, llama.cpp, vLLM): free, private, no key;
  * a free key in about a minute (Groq, Google Gemini, OpenRouter): the key page opens in your browser;
  * "Try it now": the free trial (Pollinations, no key), clearly labelled slow and public;
  * a key you already have, from any of the providers.

Whatever you pick is saved as your default; `/provider` (or `muyah login`) changes it any time. While the free
trial is in use, MUYAH-CODE keeps pointing to `/provider` (status line, start screen, after each answer).
"""

from __future__ import annotations

import os
import webbrowser

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from muyah_code.backends import probe, scan_local
from muyah_code.config import DEFAULTS, Config
from muyah_code.providers import TRIAL, get_provider

FREE_KEYS = (
    ("groq", "Groq", "Very fast, generous free tier (Kimi K2, Qwen, Llama). Sign in with Google or GitHub."),
    ("gemini", "Google Gemini", "Free tier with a Google account; large context window."),
    ("openrouter", "OpenRouter", "One key for hundreds of models, including free ones."),
)
TRIAL_NOTE = "Free trial model (slow, public): type /provider to use your own key or a local model."
ENV_SETUP = ("MUYAH_BASE_URL", "MUYAH_MODEL", "MUYAH_PROFILE", "MUYAH_API_KEY")
LOCAL_PROVIDER = {"ollama": "ollama", "lmstudio": "lmstudio", "llamacpp": "llamacpp", "local-8000": "vllm"}


def needs_setup(cfg: Config, explicit: bool = False) -> bool:
    """True when nothing tells MUYAH-CODE which model to use (and nothing answers at the default address)."""
    if explicit or any(os.environ.get(v) for v in ENV_SETUP):
        return False
    if cfg.get("profile") or cfg.get("provider"):
        return False
    if (cfg.get("base_url") or "") != DEFAULTS["base_url"]:
        return False
    return not probe(DEFAULTS["base_url"], "none", 0.6).ok      # a server of your own at the default address


def is_trial(cfg: Config) -> bool:
    return cfg.get("provider") == TRIAL


def first_run(cfg: Config, console: Console, prompter) -> str | None:
    """Ask once how MUYAH-CODE should think; set it up. Returns the profile it saved, or None if skipped."""
    from muyah_code.provider_setup import setup_provider

    console.print(Panel("[bold]Welcome to MUYAH-CODE.[/] Pick the AI it should use. You can change it any time "
                        "with [bold]/provider[/].", border_style="bright_black", padding=(0, 1)))
    with console.status("Looking for a model on this computer..."):
        found = [e for e in scan_local(1.0) if e.ok and e.models and e.name in LOCAL_PROVIDER]
    choices: list[tuple[str, str, str]] = []
    for e in found:
        provider = get_provider(LOCAL_PROVIDER[e.name])
        choices.append((f"local:{e.name}", f"{provider.name}: {e.models[0]} (on this computer)",
                        "Free, private and unlimited. No key; it is already running."))
    choices += [
        ("free-key", "Get a free key (about a minute)", "Groq, Google Gemini or OpenRouter: free, no card. The best "
                                                       "free option: fast and private to your account."),
        ("trial", "Try it now, no key", "A free public service: slow (one request every 15 s), and your prompts go "
                                       "to Pollinations. Good for a first look."),
        ("have-key", "I have a key (Claude, OpenAI, DeepSeek...)", "Choose from 20 providers and paste your key."),
    ]
    first = choices[0]
    choices[0] = (first[0], first[1] + " (Recommended)", first[2])
    picked = _ask(prompter, "How should MUYAH-CODE think?", [(label, desc) for _, label, desc in choices], "Setup")
    ident = next((c[0] for c in choices if c[1] == picked), None)
    if ident is None:
        return None
    if ident.startswith("local:"):
        name = ident.split(":", 1)[1]
        model = next(e for e in found if e.name == name).models[0]
        return setup_provider(cfg, console, prompter, LOCAL_PROVIDER[name], model=model)
    if ident == "trial":
        return setup_provider(cfg, console, prompter, TRIAL, model=get_provider(TRIAL).default_model)
    if ident == "have-key":
        return setup_provider(cfg, console, prompter)
    label = _ask(prompter, "Which free key?", [(name, desc) for _, name, desc in FREE_KEYS], "Free key")
    pid = next((p for p, name, _ in FREE_KEYS if name == label), None)
    if pid is None:
        return None
    provider = get_provider(pid)
    console.print(f"Opening [bold]{escape(provider.key_url)}[/]: create a key, copy it, and paste it here.")
    try:
        webbrowser.open(provider.key_url)
    except webbrowser.Error:
        pass
    return setup_provider(cfg, console, prompter, pid)


def _ask(prompter, question: str, options: list[tuple[str, str]], header: str) -> str | None:
    ask = getattr(prompter, "question", None)
    if ask is not None:
        picked = ask(question, options, header=header)
        return picked if isinstance(picked, str) and picked != "__other__" else None
    return prompter.select(question, [(label, label) for label, _ in options])
