"""`muyah doctor`: check the setup end to end and say exactly what to fix."""

from __future__ import annotations

import platform
import shutil
import sys

from rich.console import Console
from rich.table import Table

from muyah_code import __version__
from muyah_code.backends import probe, smoke_test, tool_calling_test
from muyah_code.config import Config
from muyah_code.hooks import HookRunner
from muyah_code.llm.client import make_client
from muyah_code.llm.models import known_context_window
from muyah_code.permissions import PermissionManager
from muyah_code.skills.loader import SkillRegistry
from muyah_code.subagents import load_agent_defs
from muyah_code.tools.shell import detect_shell

OK, WARN, FAIL = "[green]ok[/]", "[yellow]warn[/]", "[red]fail[/]"


def _doctor_anthropic(cfg: Config, add, deep: bool) -> None:
    """Native Claude: verify the key by listing models, then optionally a real reply."""
    llm = make_client(cfg)
    if not cfg.get("api_key") or cfg.get("api_key") in ("none", "dummy"):
        add("Claude API key", FAIL, "no key found. Run `muyah login anthropic` or /provider.")
        return
    try:
        ids = [m["id"] for m in llm.list_models()]
    except Exception as e:
        add("Claude API", FAIL, f"{e.__class__.__name__}: {e}. Run `muyah login anthropic` to set a new key.")
        return
    add("Claude API", OK, f"key valid, {len(ids)} models")
    add("Model", OK if llm.model in ids else WARN,
        llm.model if llm.model in ids else f"'{llm.model}' not listed for this key; available: {', '.join(ids[:6])}")
    window = llm.probe_context_window() or known_context_window(llm.model)
    add("Context window", OK, f"{window:,} tokens" if window else "unknown")
    if deep:
        try:
            reply = llm.chat([{"role": "user", "content": "Reply with PONG only."}], max_tokens=1024)
            add("Completion", OK, f"replied: {reply.content[:40]!r}")
        except Exception as e:
            add("Completion", FAIL, str(e))


def run_doctor(cfg: Config, console: Console, deep: bool = False) -> int:
    rows: list[tuple[str, str, str]] = []
    failures = 0

    def add(check: str, status: str, detail: str) -> None:
        nonlocal failures
        if status == FAIL:
            failures += 1
        rows.append((check, status, detail))

    add("MUYAH-CODE", OK, f"v{__version__} on Python {platform.python_version()} ({sys.platform})")
    add("Settings", OK, " -> ".join(s.replace(str(cfg.home), "~/.muyah") for s in cfg.sources))
    add("Project root", OK, str(cfg.project_root))

    base, model, key = cfg["base_url"], cfg["model"], cfg.get("api_key", "none")
    if (cfg.get("api") or "openai") == "anthropic":
        _doctor_anthropic(cfg, add, deep)
        ep = None
    else:
        ep = probe(base, key, timeout=10)
    if ep is None:
        pass
    elif not ep.ok:
        add("Endpoint", FAIL, f"{base}: {ep.error}. Start your server/tunnel, or run `muyah connect <url>` / "
                              "`muyah connect --scan`.")
    else:
        add("Endpoint", OK, f"{ep.base_url} ({len(ep.models)} models{', ' + ep.server if ep.server else ''})")
        if ep.base_url != base.rstrip("/"):
            add("Endpoint URL", WARN, f"works as {ep.base_url}; update base_url to that")
        if ep.models and model not in ep.models:
            add("Model", FAIL, f"'{model}' is not served. Available: {', '.join(ep.models[:8])}. "
                               f"Fix: muyah config set model {ep.models[0]}")
        else:
            add("Model", OK, model)
        llm = make_client(cfg)
        probed = llm.probe_context_window()
        configured = int(cfg.get("context_window") or 0)
        table = known_context_window(model)
        if configured:
            add("Context window", OK, f"{configured:,} tokens (configured)")
        elif probed:
            add("Context window", OK, f"{probed:,} tokens (reported by server)")
        elif table:
            add("Context window", WARN, f"{table:,} tokens (guessed from model name; set context_window to be exact)")
        else:
            add("Context window", WARN, "unknown; using 16,384. Set context_window in settings.")
        if ":11434" in ep.base_url:
            add("Ollama", WARN, "Ollama's default context is small (2-4k) and it silently truncates. Start it with "
                                "OLLAMA_CONTEXT_LENGTH=32768 (or larger) and set context_window to match.")
        if deep and (not ep.models or model in ep.models):
            ok, out = smoke_test(ep.base_url, model, key)
            add("Completion", OK if ok else FAIL, f"replied: {out!r}" if ok else out)
            mode = tool_calling_test(ep.base_url, model, key)
            detail = {"native": "native function calling works",
                      "rejected": "server rejects tools -> MUYAH-CODE will use its text tool protocol "
                                  "(for vLLM add --enable-auto-tool-choice --tool-call-parser hermes)",
                      "ignored": "model did not call the tool; text protocol may be more reliable "
                                 "(set tool_mode: text if tools misbehave)"}.get(mode, mode)
            add("Tool calling", OK if mode == "native" else WARN, detail)

    shell = detect_shell(cfg.get("shell", "auto"))
    add("Shell", OK, f"{shell.kind}: {shell.executable}")
    add("git", OK if shutil.which("git") else WARN, shutil.which("git") or "not found (git features limited)")

    perms = PermissionManager.from_config(cfg)
    add("Permissions", FAIL if perms.errors else OK,
        "; ".join(perms.errors) or f"mode {perms.mode}, {len(perms.allow)} allow / {len(perms.ask)} ask / "
                                    f"{len(perms.deny)} deny rules")
    hooks = HookRunner(cfg.get("hooks") or {}, cfg.project_root)
    add("Hooks", FAIL if hooks.errors else OK, "; ".join(hooks.errors) or f"{sum(len(v or []) for v in hooks.config.values())} hook groups")
    skills = SkillRegistry.load(cfg.project_root, cfg.home, bool(cfg.get("compat.claude_skills", True)))
    add("Skills", WARN if skills.errors else OK,
        "; ".join(skills.errors[:3]) or f"{len(skills.names())} loaded: {', '.join(skills.names()[:12])}")
    defs, errs = load_agent_defs(cfg.project_root, cfg.home)
    add("Sub-agents", WARN if errs else OK, "; ".join(errs[:3]) or ", ".join(defs))

    from muyah_code.mcp.client import load_server_configs

    mcp = load_server_configs(cfg.project_root, cfg.home)
    add("MCP servers", OK, ", ".join(mcp) if mcp else "none configured")

    t = Table(title="MUYAH-CODE doctor", show_lines=False, header_style="bold")
    t.add_column("Check")
    t.add_column("Status")
    t.add_column("Details", overflow="fold")
    for r in rows:
        t.add_row(*r)
    console.print(t)
    if not deep:
        console.print("[dim]Run `muyah doctor --deep` to also test a real completion and tool calling.[/]")
    return 1 if failures else 0
