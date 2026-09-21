"""Slash commands for the interactive REPL."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape
from rich.table import Table

from muyah_code.memory import INIT_TEMPLATE
from muyah_code.permissions import MODES
from muyah_code.session import Session, sessions_dir
from muyah_code.ui.usage_view import render_usage

EXIT = object()


def mask_secret(value):
    if not value or value in ("none", "dummy") or not isinstance(value, str):
        return value
    return value[:4] + "..." + value[-2:] if len(value) > 8 else "***"


@dataclass
class Command:
    name: str
    help: str
    handler: Callable
    usage: str = ""


INIT_PROMPT = """Analyze this codebase and create (or improve) a MUYAH.md file at the project root. It gives future \
sessions the essential project knowledge. Explore first: README, build/config files (pyproject.toml, package.json, \
Makefile...), the source layout, tests, and existing AGENTS.md/CLAUDE.md. Then write a concise MUYAH.md (under ~80 \
lines) with: a 1-3 sentence overview; exact commands to install, test (including a single test), lint and run; \
architecture notes a newcomer would not guess from file names; conventions and gotchas. Only include facts you \
verified. Use this shape:

""" + INIT_TEMPLATE


class CommandRouter:
    def __init__(self, repl):
        self.repl = repl
        self.commands: dict[str, Command] = {}
        for c in [
            Command("help", "Show commands", self.help),
            Command("exit", "Quit (also /quit, Ctrl+D)", lambda a: EXIT),
            Command("quit", "Quit", lambda a: EXIT),
            Command("clear", "Start a fresh conversation (history is kept in the session file)", self.clear),
            Command("compact", "Summarize the conversation to free context", self.compact, "[focus]"),
            Command("context", "Show context window usage", self.context),
            Command("model", "Show or switch the model", self.model, "[name]"),
            Command("models", "List models served by the endpoint", self.models),
            Command("provider", "Pick an AI provider, paste your API key, choose a model (also /login)",
                    self.provider, "[name] [model]"),
            Command("login", "Same as /provider", self.provider, "[name] [model]"),
            Command("logout", "Remove a provider's saved API key", self.logout, "<provider>"),
            Command("profile", "Show or switch backend profile", self.profile, "[name]"),
            Command("connect", "Point at a new endpoint (URL) and pick a model", self.connect, "<url> [model]"),
            Command("mode", "Permission mode: plan | edit | manual | auto (or bypassPermissions)", self.mode, "[mode]"),
            Command("plan", "Toggle plan mode (read-only exploration, then a plan)", self.plan),
            Command("undo", "Undo the last turn: its file changes and its messages", self.undo),
            Command("rewind", "Go back to before any earlier turn: code, conversation, or both (Esc Esc)",
                    self.rewind),
            Command("todos", "Show the current todo list", self.todos),
            Command("skills", "List skills (run one with /<skill-name> [args])", self.skills),
            Command("agents", "List sub-agent types", self.agents),
            Command("lessons", "List, show or delete learned lessons", self.lessons, "[rm <id> | show <id>]"),
            Command("good", "Tell MUYAH-CODE the last turn went well", self.good, "[note]"),
            Command("bad", "Tell MUYAH-CODE the last turn went badly (it will learn why)", self.bad, "[what was wrong]"),
            Command("learn", "Turn learning on/off for this session", self.learn, "on|off"),
            Command("memory", "Show instruction files; '/memory add <text>' appends to MUYAH.md", self.memory,
                    "[add <text>]"),
            Command("init", "Generate a MUYAH.md for this project", self.init),
            Command("btw", "Ask a side question: answered now, not added to the conversation (works mid-turn)",
                    self.btw, "<question>"),
            Command("verify", "Check the last changes: quick (changed files) | full (all tests, lint) | e2e (run it)",
                    self.verify, "[quick|full|e2e]"),
            Command("permissions", "Show permission rules", self.permissions),
            Command("tools", "List available tools", self.tools),
            Command("mcp", "Show MCP server status", self.mcp),
            Command("resume", "Resume an earlier conversation (pick from a list)", self.resume, "[id]"),
            Command("sessions", "List recent sessions for this project", self.sessions),
            Command("export", "Export the conversation to a markdown file", self.export, "[file]"),
            Command("usage", "Cost, tokens and where they went; context; provider limits (--all: everything)",
                    self.usage, "[--all]"),
            Command("cost", "Same as /usage", self.usage),
            Command("config", "Show effective configuration", self.config),
            Command("status", "Model, endpoint, context, mode and what's loaded", self.status),
            Command("doctor", "Check the setup", self.doctor),
            Command("mic", "Talk instead of typing (Windows voice typing; also Ctrl+Space)", self.mic),
            Command("viz", "Watch the agent work, live, in your browser ('/viz stop' to end)", self.viz, "[stop]"),
            Command("theme", "Switch color theme (saved): teal | muyah | ocean | forest | mono | light", self.theme,
                    "[name]"),
        ]:
            self.commands[c.name] = c

    @property
    def app(self):
        return self.repl.app

    @property
    def console(self):
        return self.repl.console

    def menu(self) -> list[tuple[str, str]]:
        """(name, one-line help) for the "/" menu: commands first, then skills."""
        items = [(c.name, c.help) for c in self.commands.values() if c.name not in ("quit", "login")]
        for s in self.app.skills.all():
            if s.user_invocable:
                desc = s.description if len(s.description) <= 70 else s.description[:67] + "..."
                items.append((s.name, f"skill · {desc}"))
        return items

    def names(self) -> list[str]:
        return list(self.commands) + [s.name for s in self.app.skills.all() if s.user_invocable]

    def dispatch(self, line: str):
        name, _, arg = line[1:].partition(" ")
        name, arg = name.strip(), arg.strip()
        cmd = self.commands.get(name)
        if cmd:
            return cmd.handler(arg)
        skill = self.app.skills.get(name)
        if skill and skill.user_invocable:
            prompt = (f"Follow the '{skill.name}' skill below for this task.\n\n{skill.render(arg)}"
                      + (f"\n\nTask: {arg}" if arg else ""))
            return ("prompt", prompt)
        self.console.print(f"[red]Unknown command /{escape(name)}[/]. Type /help.")
        return None

    # ------------------------------------------------------------------ handlers

    def help(self, arg):
        t = Table(show_header=False, box=None, padding=(0, 2))
        for c in self.commands.values():
            if c.name in ("quit", "login"):
                continue
            t.add_row(f"[bold]/{c.name}[/] {escape(c.usage)}", c.help)
        self.console.print(t)
        self.console.print("[dim]Tips: @path adds a file to your message · '#text' saves a note to MUYAH.md · "
                           "Shift+Tab cycles modes · Alt+Enter or Ctrl+J for a new line · Ctrl+C interrupts.[/]")

    def clear(self, arg):
        self.app.agent.clear()
        self.console.print("[dim]Conversation cleared.[/]")

    def compact(self, arg):
        with self.console.status("Compacting..."):
            self.console.print(f"[dim]{escape(self.app.compact(arg))}[/]")

    def context(self, arg):
        used, usable = self.app.context_usage()
        pct = 100 * used / max(1, usable)
        c = self.app.context
        self.console.print(
            f"Context: [bold]{used:,}[/] / {usable:,} usable tokens ({pct:.0f}%) · window {c.window:,} "
            f"({self.app.window_source}) · output reserve {c.max_output:,} · auto-compact at "
            f"{int(c.threshold * 100)}% · estimate calibration x{c.ratio:.2f}"
            f"{'' if c.calibrated else ' (not yet calibrated)'} · {len(self.app.agent.messages)} messages")

    def model(self, arg):
        if not arg:
            self.console.print(f"Model: [bold]{escape(self.app.llm.model)}[/] at {escape(self.app.llm.base_url)} "
                               f"(window {self.app.window:,}, {self.app.window_source})")
            return
        with self.console.status("Switching model..."):
            self.console.print(escape(self.app.set_model(arg)))

    def models(self, arg):
        try:
            ids = [m.get("id") for m in self.app.llm.list_models()]
        except Exception as e:
            self.console.print(f"[red]Could not list models: {escape(str(e))}[/]")
            return
        for i in ids:
            mark = " [green](current)[/]" if i == self.app.llm.model else ""
            self.console.print(f"  {escape(str(i))}{mark}")

    def profile(self, arg):
        profiles = self.app.cfg.profiles()
        if not arg:
            if not profiles:
                self.console.print("No profiles yet. Add one with /provider.")
                return
            active = self.app.cfg.get("profile")
            options = [(name, f"{name}  ·  {p.get('model', '')}" + ("  (active)" if name == active else ""))
                       for name, p in profiles.items()]
            arg = self.repl.prompter.select("Switch to", options, default=active)
            if not arg or arg == active:
                return
        if arg not in profiles:
            self.console.print(f"[red]Unknown profile {escape(arg)}[/]")
            return
        with self.console.status("Switching..."):
            msg = self.app.switch_profile(arg)
        self.console.print(escape(msg))

    def provider(self, arg):
        """Pick a provider, paste a key, choose a model - then switch this session to it."""
        from muyah_code.provider_setup import setup_provider

        parts = arg.split()
        name = setup_provider(self.app.cfg, self.console, self.repl.prompter, choice=parts[0] if parts else None,
                              model=parts[1] if len(parts) > 1 else None)
        if name:
            with self.console.status("Switching this session..."):
                msg = self.app.switch_profile(name)
            self.console.print(f"[dim]{escape(msg)}[/]")

    def logout(self, arg):
        from muyah_code.providers import get_provider, remove_credential

        p = get_provider(arg) if arg else None
        if p is None:
            self.console.print("Usage: /logout <provider>   (removes its saved API key)")
            return
        removed = remove_credential(self.app.cfg.home, p.id)
        self.console.print(f"Removed the saved {p.name} key." if removed else f"No saved key for {p.name}.")

    def connect(self, arg):
        from muyah_code.backends import probe

        parts = arg.split()
        if not parts:
            self.console.print("Usage: /connect <url> [model]")
            return
        ep = probe(parts[0], self.app.cfg.get("api_key", "none"))
        if not ep.ok:
            self.console.print(f"[red]{escape(ep.base_url)}: {escape(ep.error or '')}[/]")
            return
        model = parts[1] if len(parts) > 1 else (ep.models[0] if ep.models else self.app.llm.model)
        self.console.print(escape(self.app.set_endpoint(ep.base_url, None, model)))

    def mode(self, arg):
        if not arg:
            self.console.print(f"Mode: [bold]{self.app.permissions.mode}[/]  (options: {', '.join(MODES)})")
            return
        try:
            self.console.print(f"Mode: [bold]{self.app.set_mode(arg)}[/]")
        except ValueError as e:
            self.console.print(f"[red]{escape(str(e))}[/]")

    def plan(self, arg):
        new = "default" if self.app.permissions.mode == "plan" else "plan"
        self.console.print(f"Mode: [bold]{self.app.set_mode(new)}[/]")

    def undo(self, arg):
        self.console.print(escape(self.app.undo()))

    def rewind(self, arg):
        """Pick a turn, pick what to restore, see what changes, confirm."""
        from muyah_code.app import describe_rewind
        from muyah_code.ui.history import ago

        rw = self.app.rewind
        agent = self.app.agent
        turns = rw.turns()
        if not turns and rw.last_safety() is None:
            self.console.print("[dim]Nothing to rewind yet: no turns in this session.[/]")
            return
        options = []
        if rw.last_safety() is not None:
            options.append(("__undo__", "↩ Undo the last rewind"))
        for p in reversed(turns[-30:]):
            convo = "" if rw.can_rewind_conversation(p, agent) else "  (code only)"
            options.append((p.turn, f"Turn {p.turn} · {ago(p.ts)} · {p.prompt[:70]}{convo}"))
        picked = self.repl.prompter.select("Rewind to before which turn?", options,
                                           default=options[0][0] if options else None)
        if picked is None:
            return
        if picked == "__undo__":
            result = rw.undo_rewind(agent)
            if result is None:
                self.console.print("[dim]Nothing to undo.[/]")
                return
            files = len(result["code"])
            self.console.print(f"Back to how things were before the last rewind ({files} file"
                               f"{'s' if files != 1 else ''} restored"
                               f"{', conversation restored' if result['conversation'] else ''}).")
            return
        point = next(p for p in turns if p.turn == picked)
        modes = [("both", "Code and conversation"), ("conversation", "Conversation only (files stay as they are)"),
                 ("code", "Code only (keep the conversation)")]
        if not rw.can_rewind_conversation(point, agent):
            modes = [("code", "Code only (the conversation was cleared or compacted since then)")]
        mode = self.repl.prompter.select(f"Rewind to before turn {point.turn}: what should go back?", modes,
                                         default=modes[0][0])
        if mode is None:
            return
        code, conversation = mode in ("both", "code"), mode in ("both", "conversation")
        safety_sha = None
        if code:
            safety_sha, changes = rw.preview(point)
            if safety_sha is None and not rw.covers_commands:
                self.console.print(f"[yellow]Only files edited with Write/Edit can be restored "
                                   f"({escape(rw.disabled_reason)}).[/]")
            elif not changes:
                self.console.print("[dim]No files changed since then.[/]")
            else:
                verbs = {"M": "restore", "A": "bring back", "D": "remove"}
                self.console.print(f"This will change {len(changes)} file{'s' if len(changes) != 1 else ''}:")
                for status, path in changes[:15]:
                    self.console.print(f"  [dim]{verbs[status]}[/] {escape(path)}")
                if len(changes) > 15:
                    self.console.print(f"  [dim]… and {len(changes) - 15} more[/]")
            self.console.print("[dim]Things outside this folder (databases, installed packages, network calls) "
                               "cannot be rewound.[/]")
        ok = self.repl.prompter.select("Go ahead?", [("yes", "Yes, rewind"), ("no", "No")], default="yes")
        if ok != "yes":
            return
        result = rw.restore(point, code=code, conversation=conversation, agent=agent, safety_sha=safety_sha)
        self.console.print(escape(describe_rewind(point, result, rw)))
        if result.get("conversation"):
            self.repl._resume_text = point.prompt   # your prompt is back in the input, to edit or resend

    def todos(self, arg):
        if not self.app.ctx.todos:
            self.console.print("[dim]No todos.[/]")
        else:
            self.repl.ui.on_todos(self.app.ctx.todos)

    def skills(self, arg):
        for s in self.app.skills.all():
            flag = "" if not s.disable_model_invocation else " [dim](user only)[/]"
            self.console.print(f"  [bold]/{escape(s.name)}[/] [dim]({s.source})[/]{flag} {escape(s.description[:110])}")
        for e in self.app.skills.errors:
            self.console.print(f"  [yellow]{escape(e)}[/]")

    def agents(self, arg):
        for d in self.app.agent_defs.values():
            self.console.print(f"  [bold]{escape(d.name)}[/] [dim]({d.source})[/] {escape(d.description[:110])}")

    def lessons(self, arg):
        store = self.app.lessons
        sub, _, rest = arg.partition(" ")
        if sub == "rm" and rest:
            self.console.print("Deleted." if store.remove(rest.strip()) else "[red]No such lesson.[/]")
            return
        if sub == "show" and rest:
            lesson = store.get(rest.strip())
            if lesson:
                self.console.print_json(json.dumps(lesson.__dict__, default=str))
            else:
                self.console.print("[red]No such lesson.[/]")
            return
        if not store.lessons:
            self.console.print("[dim]No lessons yet. They are learned from failures, fixes and your /good /bad "
                               "feedback.[/]")
            return
        t = Table(header_style="bold")
        for col in ("id", "scope", "score", "used", "lesson"):
            t.add_column(col, overflow="fold")
        for x in sorted(store.lessons, key=lambda z: z.score, reverse=True):
            t.add_row(x.id, x.scope, f"{x.score:.2f}", str(x.uses), escape(x.render()[2:]))
        self.console.print(t)

    def good(self, arg):
        self.console.print(escape(self.app.feedback(True, arg)))

    def bad(self, arg):
        with self.console.status("Learning from the feedback..."):
            msg = self.app.feedback(False, arg)
        self.console.print(escape(msg))

    def learn(self, arg):
        if arg in ("on", "off"):
            self.app.learning_enabled = arg == "on"
        self.console.print(f"Learning: [bold]{'on' if self.app.learning_enabled else 'off'}[/]")

    def memory(self, arg):
        if arg.startswith("add "):
            self.console.print(escape(self.repl.add_memory(arg[4:])))
            return
        if not self.app.instructions:
            self.console.print("No instruction files loaded. Create one with /init or '#note'.")
        for f in self.app.instructions:
            self.console.print(f"  [bold]{escape(str(f.path))}[/] ({len(f.text):,} chars)")
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
        target = self.app.root / "MUYAH.md"
        if arg == "edit":
            if editor:
                subprocess.call([editor, str(target)])
            else:
                self.console.print(f"Set $EDITOR, or open {escape(str(target))} yourself.")

    def init(self, arg):
        from muyah_code.verify import detect

        found = detect(self.app.root)
        if not found:
            return ("prompt", INIT_PROMPT)
        listing = "\n".join(f"- {c.kind}: {c.command}" for c in found)
        return ("prompt", INIT_PROMPT + "\n\nMUYAH-CODE detected these check commands from the config files "
                "(confirm they work before listing them):\n" + listing)

    def btw(self, arg):
        if not arg.strip():
            self.console.print("Usage: /btw <question>  (answered on the side; the conversation is not changed)")
            return None
        with self.console.status("[dim]Answering on the side…[/]", spinner="dots"):
            answer = self.app.btw(arg.strip())
        self.repl.ui.side_answer(arg.strip(), answer)
        return None

    def verify(self, arg):
        from muyah_code import verify as v

        depth = (arg or "quick").split()[0].lower()
        if depth not in v.DEPTHS:
            self.console.print(f"[red]Unknown depth '{escape(depth)}'[/]. Use /verify quick | full | e2e.")
            return None
        app = self.app
        changed = [rel for status, rel in app.rewind.turn_changes() if status != "D"]
        if depth == "e2e":
            return ("prompt", v.e2e_prompt(changed))
        if depth == "quick":
            if not changed:
                self.console.print("[dim]The last turn changed no files. /verify full checks the whole project.[/]")
                return None
            checks, notes = v.quick_checks(app.cfg, app.root, changed)
        else:
            checks, from_settings = v.configured(app.cfg, app.root)
            notes = [] if checks else ["no test or lint commands found"]
        for note in notes:
            self.console.print(f"[dim]{escape(note)}[/]")
        if not checks:
            self.console.print("[yellow]Nothing to run.[/] Set verify.test / verify.lint in .muyah/settings.json "
                               "(a command or a list), or try /verify e2e.")
            return None
        results = self._run_checks(depth, checks)
        app.verify_note = v.summary_for_model(depth, results)
        return None

    def _run_checks(self, depth, checks):
        from rich.panel import Panel
        from rich.text import Text

        from muyah_code import verify as v
        from muyah_code.ui.theme import theme

        t, app = theme(), self.app
        timeout = float(app.cfg.get("verify.timeout", 900) or 900)
        self.console.print(Text(f"◆ Verify {depth}", style=f"bold {t.accent}"))
        status = None

        def start(check):
            nonlocal status
            status = self.console.status(Text(f"$ {check.command}", style=t.dim), spinner="dots")
            status.start()

        def done(r):
            status.stop()
            line = Text("  ")
            line.append("✓ " if r.ok else "✗ ", style=t.ok if r.ok else t.err)
            line.append(r.check.command, style="bold")
            result = "timed out" if r.timed_out else ("passed" if r.ok else f"exit {r.exit_code}")
            line.append(f"  {r.check.kind} · {result} · {r.seconds:.1f}s", style=t.dim)
            self.console.print(line)
            if not r.ok and r.output.strip():
                self.console.print(Panel(Text(r.tail()), border_style=t.err, expand=False, padding=(0, 1)))

        try:
            results = v.run_checks(checks, app.cwd, app.shell, timeout, on_start=start, on_done=done)
        finally:
            if status is not None:
                status.stop()
        failed = [r for r in results if not r.ok]
        if failed:
            self.console.print(Text(f"  {len(failed)} of {len(results)} failed. The next prompt tells the model "
                                    "(say \"fix it\").", style=t.warn))
        else:
            self.console.print(Text(f"  All {len(results)} passed.", style=t.ok))
        return results

    def permissions(self, arg):
        p = self.app.permissions
        self.console.print(f"Mode: [bold]{p.mode}[/]")
        for kind in ("allow", "ask", "deny"):
            rules = getattr(p, kind)
            self.console.print(f"  {kind}: " + (", ".join(escape(str(r)) for r in rules) or "[dim](none)[/]"))

    def tools(self, arg):
        for t in self.app.registry.tools():
            self.console.print(f"  [bold]{escape(t.name)}[/] [dim]{t.kind}[/]")

    def mcp(self, arg):
        if not self.app.mcp:
            self.console.print("No MCP servers configured (.mcp.json or ~/.muyah/mcp.json).")
            return
        for name, status in self.app.mcp.status.items():
            self.console.print(f"  [bold]{escape(name)}[/]: {escape(status)}")

    def resume(self, arg):
        sdir = sessions_dir(self.app.home, self.app.root)
        if not arg:
            from muyah_code.ui.history import pick_session

            arg = pick_session(sdir, self.repl.prompter)
            if not arg:
                self.console.print("[dim]No earlier conversations in this folder.[/]")
                return
        try:
            session, messages, _ = Session.resume(sdir, arg)
        except FileNotFoundError as e:
            self.console.print(f"[red]{escape(str(e))}[/]")
            return
        self.app.session = session
        self.app.agent.session = session
        self.app.agent.load_history(messages)
        from muyah_code.ui.history import print_history

        print_history(self.console, messages)
        self.console.print(f"[dim]Resumed {session.id} ({len(messages)} messages).[/]")

    def sessions(self, arg):
        infos = Session.list_sessions(sessions_dir(self.app.home, self.app.root))
        if not infos:
            self.console.print("[dim]No sessions yet.[/]")
        for s in infos:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.modified))
            self.console.print(f"  [bold]{s.id}[/] {when} ({s.messages} msgs) {escape(s.title)}")
        if infos:
            self.console.print("[dim]Resume with /resume <id> or `muyah --resume <id>`.[/]")

    def export(self, arg):
        path = Path(arg or f"muyah-session-{self.app.session_id or 'export'}.md")
        lines = [f"# MUYAH-CODE session {self.app.session_id}\n"]
        for m in self.app.agent.messages[1:]:
            from muyah_code.llm.content import text_of

            role = m.get("role")
            content = text_of(m.get("content") or "")
            if role == "tool" or m.get("_images"):
                lines.append(f"<details><summary>tool result</summary>\n\n```\n{content[:4000]}\n```\n</details>\n")
            elif m.get("tool_calls"):
                calls = ", ".join(tc["function"]["name"] for tc in m["tool_calls"])
                lines.append(f"**assistant** ({calls}):\n\n{content}\n")
            else:
                lines.append(f"**{role}**:\n\n{content}\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        self.console.print(f"Exported to {escape(str(path.resolve()))}")

    def usage(self, arg):
        render_usage(self.console, self.app.home, llm=self.app.llm, app=self.app, full="--all" in arg)

    def config(self, arg):
        data = copy.deepcopy(self.app.cfg.data)
        for holder in [data, *[p for p in (data.get("profiles") or {}).values() if isinstance(p, dict)]]:
            holder["api_key"] = mask_secret(holder.get("api_key"))
        self.console.print_json(json.dumps(data, default=str))
        self.console.print("[dim]Sources: " + escape(" -> ".join(self.app.cfg.sources)) + "[/]")

    def status(self, arg):
        a = self.app
        used, usable = a.context_usage()
        rows = [
            ("model", a.llm.model),
            ("provider", a.cfg.get("provider") or a.cfg.get("profile") or "custom endpoint"),
            ("endpoint", a.llm.base_url),
            ("context", f"{a.window:,} tokens ({a.window_source}) · {100 * used // max(1, usable)}% used"),
            ("mode", a.permissions.mode),
            ("folder", str(a.cwd)),
            ("session", a.session_id or "(not saved)"),
            ("loaded", f"{len(a.instructions)} instruction files · {len(a.skills.names())} skills · "
                       f"{len(a.lessons.lessons)} lessons · {len(a.mcp_tools)} MCP tools"),
            ("prompt", a.prompt_profile + (" (short prompt, 6 core tools, find_tools for the rest)"
                                           if a.prompt_profile == "lean" else "")),
        ]
        roles = a.models.configured()
        if roles:
            rows.append(("models", " · ".join(f"{role}: {spec}" for role, spec in roles.items())))
        if a.models.fallbacks():
            rows.append(("fallback", " → ".join(a.models.fallbacks())))
        t = Table(show_header=False, box=None, padding=(0, 2))
        for k, v in rows:
            t.add_row(f"[dim]{k}[/]", escape(str(v)))
        self.console.print(t)

    def doctor(self, arg):
        from muyah_code.doctor import run_doctor

        run_doctor(self.app.cfg, self.console, deep=arg == "--deep")

    def mic(self, arg):
        from muyah_code.ui.voice import start_dictation

        started, message = start_dictation()
        self.console.print(f"[dim]{escape(message)}[/]" if started else f"[yellow]{escape(message)}[/]")

    def viz(self, arg):
        import webbrowser

        from muyah_code.viz import VizServer

        server = self.app.viz
        if arg == "stop":
            if server is None:
                self.console.print("[dim]The visualization is not running.[/]")
            else:
                server.stop()
                self.app.viz = None
                self.console.print("[dim]Visualization stopped.[/]")
            return
        if server is None or not server.running:
            server = VizServer(bus=self.app.events, title=self.app.session_id,
                               recording=self.app.events.record_to)
            server.start()
            self.app.viz = server
        self.console.print(f"Live view: [link={server.url}]{server.url}[/link]")
        self.console.print("[dim]Only this computer can open it. '/viz stop' ends it.[/]")
        try:
            webbrowser.open(server.url)
        except webbrowser.Error:
            self.console.print("[dim]Could not open a browser; open the link above.[/]")

    def theme(self, arg):
        from muyah_code.ui.terminal import banner_text
        from muyah_code.ui.theme import THEMES, set_theme, theme

        if not arg:
            self.console.print(f"Theme: [bold]{theme().name}[/]  (available: {', '.join(THEMES)})")
            return
        try:
            set_theme(arg)
        except ValueError as e:
            self.console.print(f"[red]{escape(str(e))}[/]")
            return
        self.app.cfg.persist("theme", arg, scope="user")
        self.console.print(banner_text())
        self.console.print(f"Theme set to [bold]{arg}[/] (saved).")
