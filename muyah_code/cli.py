"""Command-line entry point.

    muyah                         interactive session
    muyah "fix the failing test"  interactive, starting with a prompt
    muyah -p "..."                headless: print the answer and exit (--output-format text|json|stream-json)
    muyah -c / --resume [id]      continue the last / a specific session
    muyah login [provider]        pick a provider (Anthropic, OpenAI, Gemini, OpenRouter...), paste the key, done
    muyah connect <url> | --scan  set up a backend profile (local servers, Colab tunnels, custom URLs)
    muyah serve <engine> -m ...   run colibri / soup / ollama / vllm / llama.cpp locally and connect
    muyah doctor [--deep]         check the setup
    muyah eval [tasks...]         run the benchmark and track the pass rate
    muyah config get|set|unset    edit ~/.muyah/settings.json
    muyah sessions                list sessions for this project
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from muyah_code import __version__

SUBCOMMANDS = {"login", "logout", "connect", "serve", "doctor", "eval", "config", "sessions"}


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def _console():
    from rich.console import Console

    return Console(highlight=False)


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        return _subcommand(argv[0], argv[1:])
    return _main(argv)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="muyah", description="MUYAH-CODE: agentic coding in your terminal, on any "
                                "OpenAI-compatible model.", epilog="Subcommands: login, logout, connect, serve, "
                                "doctor, eval, config, sessions (run `muyah <subcommand> -h`).")
    p.add_argument("prompt", nargs="*", help="Initial prompt")
    p.add_argument("-p", "--print", dest="headless", action="store_true", help="Headless: answer and exit")
    p.add_argument("--output-format", choices=["text", "json", "stream-json"], default="text")
    p.add_argument("-v", "--verbose", action="store_true", help="Show tool activity in headless mode")
    p.add_argument("-c", "--continue", dest="continue_last", action="store_true", help="Continue the last session")
    p.add_argument("-r", "--resume", metavar="ID", help="Resume a session by id (prefix ok)")
    p.add_argument("-m", "--model", help="Model id")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    p.add_argument("--api-key", help="API key for the endpoint")
    p.add_argument("--profile", help="Backend profile from settings")
    p.add_argument("--context-window", type=int, help="Override the model context window (tokens)")
    p.add_argument("--mode", "--permission-mode", dest="mode",
                   help="default | acceptEdits | plan | bypassPermissions")
    p.add_argument("--allowedTools", "--allowed-tools", dest="allowed", action="append", default=[],
                   help='Allow rules, e.g. "Bash(git *)" "Edit" (repeatable or comma-separated)')
    p.add_argument("--disallowedTools", "--disallowed-tools", dest="disallowed", action="append", default=[],
                   help="Deny rules")
    p.add_argument("--max-steps", type=int, help="Max model steps per turn")
    p.add_argument("--tool-mode", choices=["auto", "native", "text"], help="Tool calling protocol")
    p.add_argument("--settings", help="Extra settings JSON file")
    p.add_argument("--cwd", help="Working directory")
    p.add_argument("--no-mcp", action="store_true", help="Do not start MCP servers")
    p.add_argument("--no-learn", action="store_true", help="Disable lessons for this run")
    p.add_argument("--show-reasoning", action="store_true", help="Stream the model's reasoning text")
    p.add_argument("--version", action="version", version=f"MUYAH-CODE {__version__}")
    return p


def _split_rules(values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        depth, cur = 0, ""
        for ch in v:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch in ", " and depth == 0:
                if cur.strip():
                    out.append(cur.strip())
                cur = ""
            else:
                cur += ch
        if cur.strip():
            out.append(cur.strip())
    return out


def _main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    from muyah_code.config import ConfigError, load_config

    cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    if not cwd.is_dir():
        print(f"error: --cwd {cwd} is not a directory", file=sys.stderr)
        return 2
    overrides = {"model": args.model, "base_url": args.base_url, "api_key": args.api_key, "profile": args.profile,
                 "context_window": args.context_window, "max_steps": args.max_steps, "tool_mode": args.tool_mode}
    try:
        cfg = load_config(cwd=cwd, settings_file=args.settings, overrides=overrides)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.no_learn:
        cfg.set("learning.enabled", False)

    prompt = " ".join(args.prompt).strip()
    if not sys.stdin.isatty() and args.headless:
        piped = sys.stdin.read()
        if piped.strip():
            prompt = f"{prompt}\n\n<stdin>\n{piped}\n</stdin>" if prompt else piped

    if args.headless:
        return _headless(cfg, cwd, args, prompt)
    return _interactive(cfg, cwd, args, prompt or None)


def _build_app(cfg, cwd, args, ui, headless: bool):
    from muyah_code.app import App

    return App(cfg, ui, cwd=cwd, mode=args.mode, headless=headless, allowed_tools=_split_rules(args.allowed),
               disallowed_tools=_split_rules(args.disallowed), resume=args.resume,
               continue_last=args.continue_last, enable_mcp=not args.no_mcp)


def _headless(cfg, cwd, args, prompt: str) -> int:
    from muyah_code.ui.headless import HeadlessUI

    if not prompt:
        print("error: -p needs a prompt (argument or stdin)", file=sys.stderr)
        return 2
    ui = HeadlessUI(args.output_format, verbose=args.verbose)
    try:
        app = _build_app(cfg, cwd, args, ui, headless=True)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        result = app.run_prompt(prompt)
    finally:
        app.shutdown()
    payload = {
        "type": "result", "session_id": app.session_id, "status": result.status, "result": result.text,
        "num_turns": result.steps, "tool_calls": result.tool_calls, "duration_s": round(result.duration, 2),
        "usage": app.llm.total_usage, "error": result.error,
    }
    if args.output_format == "text":
        if result.text:
            print(result.text)
    elif args.output_format == "json":
        print(json.dumps(payload, ensure_ascii=False))
    else:
        ui._event(**payload)
    return 0 if result.status == "ok" else 1


def _interactive(cfg, cwd, args, prompt: str | None) -> int:
    from muyah_code.ui.repl import run_repl
    from muyah_code.ui.terminal import TerminalUI

    from muyah_code.ui.theme import set_theme

    try:
        set_theme(cfg.get("theme") or "teal")
    except ValueError:
        set_theme("teal")
    console = _console()
    ui = TerminalUI(console, show_reasoning=args.show_reasoning)
    try:
        with console.status("Starting MUYAH-CODE..."):
            app = _build_app(cfg, cwd, args, ui, headless=False)
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]error:[/] {e}")
        return 2
    try:
        return run_repl(app, ui, prompt)
    finally:
        app.shutdown()


# ---------------------------------------------------------------------- subcommands

def _subcommand(name: str, argv: list[str]) -> int:
    from muyah_code.config import load_config

    console = _console()
    if name == "doctor":
        p = argparse.ArgumentParser(prog="muyah doctor")
        p.add_argument("--deep", action="store_true", help="Also run a completion and a tool-calling test")
        a = p.parse_args(argv)
        from muyah_code.doctor import run_doctor

        return run_doctor(load_config(), console, deep=a.deep)

    if name == "eval":
        p = argparse.ArgumentParser(prog="muyah eval")
        p.add_argument("tasks", nargs="*", help="Task names (default: all)")
        p.add_argument("--dir", default=None, help="Tasks directory (default: ./evals/tasks if present, else the bundled set)")
        p.add_argument("-m", "--model")
        p.add_argument("--profile")
        p.add_argument("--no-learn", action="store_true", help="Evaluate without lessons (baseline)")
        p.add_argument("-v", "--verbose", action="store_true")
        a = p.parse_args(argv)
        from muyah_code.learning.eval import run_eval

        tasks_dir = Path(a.dir) if a.dir else _default_tasks_dir()
        return run_eval(tasks_dir, a.tasks, {"model": a.model, "profile": a.profile}, not a.no_learn, a.verbose,
                        console)

    if name == "login":
        p = argparse.ArgumentParser(prog="muyah login", description="Set up an AI provider with your API key.")
        p.add_argument("provider", nargs="?", help="anthropic, openai, gemini, openrouter, groq, deepseek, ... or a number")
        p.add_argument("--key", help="API key (otherwise you are asked; input is hidden)")
        p.add_argument("-m", "--model", help="Model id (otherwise pick from the list)")
        p.add_argument("--no-test", action="store_true", help="Skip the test reply")
        a = p.parse_args(argv)
        from muyah_code.provider_setup import setup_provider
        from muyah_code.ui.select import Prompter

        prompter = Prompter()
        return 0 if setup_provider(load_config(), console, prompter, a.provider, a.key, a.model, not a.no_test) else 1

    if name == "logout":
        from muyah_code.providers import get_provider, remove_credential

        prov = get_provider(argv[0]) if argv else None
        if prov is None:
            console.print("usage: muyah logout <provider>")
            return 2
        removed = remove_credential(load_config().home, prov.id)
        console.print(f"Removed the saved {prov.name} key." if removed else f"No saved key for {prov.name}.")
        return 0

    if name == "connect":
        return _connect(argv, console)

    if name == "serve":
        from muyah_code.serve import main as serve_main

        return serve_main(load_config(), argv, console)

    if name == "config":
        return _config_cmd(argv, console)

    if name == "sessions":
        from muyah_code.session import Session, sessions_dir

        cfg = load_config()
        for s in Session.list_sessions(sessions_dir(cfg.home, cfg.project_root)):
            console.print(f"{s.id}  ({s.messages} msgs)  {s.title}")
        return 0
    return 2


def _default_tasks_dir() -> Path:
    local = Path.cwd() / "evals" / "tasks"
    if local.is_dir():
        return local
    return Path(__file__).resolve().parent / "evals" / "tasks"


def _connect(argv: list[str], console) -> int:
    from muyah_code.backends import (
        ENGINE_PRESETS,
        build_profile,
        normalize_base_url,
        probe,
        profile_name_for,
        scan_local,
        smoke_test,
        tunnel_warning,
    )
    from muyah_code.config import load_config

    p = argparse.ArgumentParser(prog="muyah connect", description="Create/activate a backend profile.")
    p.add_argument("url", nargs="?", help="Endpoint URL, e.g. https://xyz.a.pinggy.link/v1 or http://localhost:11434")
    p.add_argument("--scan", action="store_true", help="Look for local servers (Ollama, LM Studio, vLLM...)")
    p.add_argument("--name", help="Profile name (default: derived from the URL)")
    p.add_argument("-m", "--model", help="Model id (default: first served model)")
    p.add_argument("--api-key", default=None, help="API key (or set it later in settings)")
    p.add_argument("--context-window", type=int, help="Context window, if the server does not report it")
    p.add_argument("--engine", choices=list(ENGINE_PRESETS),
                   help="Engine behind the URL: applies its tuned preset (e.g. colibri: no timeout, no reflection)")
    p.add_argument("--no-test", action="store_true", help="Skip the PONG completion test")
    a = p.parse_args(argv)
    cfg = load_config()

    if a.scan or not a.url:
        found = [e for e in scan_local() if e.ok]
        if not found:
            console.print("No local OpenAI-compatible servers found on the usual ports (11434 Ollama, 1234 LM Studio,"
                          " 8000 vLLM/colibri/Soup, 8080 llama.cpp). Start one, or pass a URL.")
            return 1
        for e in found:
            console.print(f"  [green]found[/] {e.name}: {e.base_url}  models: {', '.join(e.models[:6]) or '?'}")
        if not a.url:
            a.url = found[0].base_url
            console.print(f"Using {a.url}")

    key = a.api_key if a.api_key is not None else (os.environ.get("MUYAH_API_KEY") or "none")
    ep = probe(normalize_base_url(a.url), key, timeout=15)
    if not ep.ok:
        console.print(f"[red]Could not reach {ep.base_url}: {ep.error}[/]")
        return 1
    model = a.model or (ep.models[0] if ep.models else None)
    if not model:
        console.print("[red]The server lists no models; pass --model.[/]")
        return 1
    if ep.models and model not in ep.models:
        console.print(f"[yellow]'{model}' is not in the served list: {', '.join(ep.models[:8])}[/]")
    warning = tunnel_warning(ep.base_url, a.engine)
    if warning:
        console.print(f"[yellow]{warning}[/]")
    if not a.no_test:
        with console.status("Testing a completion (large models may take a while)..."):
            ok, out = smoke_test(ep.base_url, model, key, timeout=900 if a.engine == "colibri" else 180)
        if not ok:
            console.print(f"[red]Completion test failed: {out}[/]")
            return 1
        console.print(f"[green]Completion OK[/] (model replied {out!r})")
    name = a.name or (a.engine if a.engine and profile_name_for(ep.base_url) == "colab" else
                      profile_name_for(ep.base_url))
    prof = build_profile(ep.base_url, model, key, a.context_window, a.engine)
    cfg.persist(f"profiles.{name}", prof, scope="user")
    cfg.persist("profile", name, scope="user")
    console.print(f"Saved profile [bold]{name}[/] and made it active ({cfg.user_settings_path}).")
    console.print("Start coding with: [bold]muyah[/]")
    return 0


def _config_cmd(argv: list[str], console) -> int:
    from muyah_code.config import load_config, read_json, write_json

    p = argparse.ArgumentParser(prog="muyah config")
    p.add_argument("action", choices=["get", "set", "unset", "path", "show"])
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")
    p.add_argument("--project", action="store_true", help="Write to .muyah/settings.json instead of user settings")
    a = p.parse_args(argv)
    cfg = load_config()
    scope = "project" if a.project else "user"
    path = cfg.project_settings_path if a.project else cfg.user_settings_path
    if a.action == "path":
        console.print(str(path))
    elif a.action == "show":
        console.print_json(json.dumps(read_json(path)))
    elif a.action == "get":
        if not a.key:
            console.print("usage: muyah config get <key>")
            return 2
        console.print(json.dumps(cfg.get(a.key), default=str))
    elif a.action == "set":
        if not a.key or a.value is None:
            console.print("usage: muyah config set <key> <value>   (value is parsed as JSON when possible)")
            return 2
        try:
            value = json.loads(a.value)
        except json.JSONDecodeError:
            value = a.value
        cfg.persist(a.key, value, scope=scope)
        console.print(f"{a.key} = {json.dumps(value)}  ({path})")
    elif a.action == "unset":
        data = read_json(path)
        parts = (a.key or "").split(".")
        cur = data
        for part in parts[:-1]:
            cur = cur.get(part, {}) if isinstance(cur, dict) else {}
        if isinstance(cur, dict) and parts[-1] in cur:
            del cur[parts[-1]]
            write_json(path, data)
            console.print(f"removed {a.key}")
        else:
            console.print(f"{a.key} is not set in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
