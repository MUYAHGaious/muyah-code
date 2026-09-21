"""Interactive REPL: prompt_toolkit input, slash commands, @file completion, Shift+Tab mode cycling."""

from __future__ import annotations

import html

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from muyah_code import __version__
from muyah_code.tools.search import list_files
from muyah_code.ui.commands import EXIT, CommandRouter
from muyah_code.ui.terminal import TerminalUI, banner_text
from muyah_code.ui.theme import theme

MODE_LABEL = {
    "default": "",
    "acceptEdits": "⏵⏵ accept edits on",
    "plan": "⏸ plan mode on",
    "bypassPermissions": "⚠ bypass permissions on",
}


class _Completer(Completer):
    def __init__(self, repl: Repl):
        self.repl = repl
        self._files: list[str] | None = None

    def files(self) -> list[str]:
        if self._files is None:
            root = self.repl.app.cwd
            try:
                self._files = sorted(p.relative_to(root).as_posix() for p in list_files(root))[:20000]
            except (OSError, ValueError):
                self._files = []
        return self._files

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/") and " " not in text:
            for name in self.repl.router.names():
                if name.startswith(text[1:]):
                    yield Completion("/" + name, start_position=-len(text))
            return
        word = text.split()[-1] if text.split() and not text.endswith(" ") else ""
        if word.startswith("@"):
            q = word[1:].lower()
            n = 0
            for f in self.files():
                if q in f.lower():
                    yield Completion("@" + f, start_position=-len(word))
                    n += 1
                    if n >= 50:
                        break


class Repl:
    def __init__(self, app, ui: TerminalUI, **session_kwargs):
        self.app = app
        self.ui = ui
        self.console = ui.console
        self.router = CommandRouter(self)
        kb = KeyBindings()

        @kb.add("s-tab")
        def _(event):
            self.app.set_mode(self.app.permissions.cycle_mode())
            event.app.invalidate()

        @kb.add("escape", "enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add("c-j")
        def _(event):
            event.current_buffer.insert_text("\n")

        self._io = {k: v for k, v in session_kwargs.items() if k in ("input", "output")}
        history = FileHistory(str(app.home / "history"))
        self.session = PromptSession(history=history, completer=_Completer(self), key_bindings=kb,
                                     complete_while_typing=True, bottom_toolbar=self._toolbar, **session_kwargs)

    def ask(self, message: str, password: bool = False) -> str:
        """One-line question (hidden input for secrets such as API keys)."""
        try:
            return PromptSession(**self._io).prompt([(f"fg:{theme().accent}", "? "), ("", message)],
                                                    is_password=password)
        except (EOFError, KeyboardInterrupt):
            return ""

    def _ctx_pct(self) -> int:
        used, usable = self.app.context_usage()
        return min(100, int(100 * used / max(1, usable)))

    def _files_changed(self) -> int:
        turns = self.app.checkpoints.turns
        return len(turns[-1]["changes"]) if turns else 0

    def _toolbar(self):
        pct = self._ctx_pct()
        mode = MODE_LABEL.get(self.app.permissions.mode, "")
        left = f"<b>{mode}</b> (shift+tab to cycle) · " if mode else "shift+tab: modes · "
        return HTML(f" {left}{html.escape(self.app.llm.model)} · context {pct}% · /help")

    def banner(self) -> None:
        a = self.app
        t = theme()
        if self.console.width >= 60:
            self.console.print(banner_text())
            self.console.print()
        body = Text()
        body.append("MUYAH-CODE", style=f"bold {t.accent}")
        body.append(f"  v{__version__}", style=t.dim)
        if a.cfg.get("profile"):
            body.append(f"  ·  profile {a.cfg.get('profile')}", style=t.dim)
        body.append("\n")
        rows = [("model", a.llm.model), ("endpoint", a.llm.base_url),
                ("context", f"{a.window:,} tokens ({a.window_source})"), ("mode", a.permissions.mode),
                ("cwd", str(a.cwd))]
        for label, value in rows:
            body.append(f"{label:<9}", style=t.dim)
            body.append(f"{value}\n", style="default" if label == "model" else t.dim)
        extras = []
        if a.instructions:
            extras.append(f"{len(a.instructions)} instruction file(s)")
        extras.append(f"{len(a.skills.names())} skills")
        if a.lessons.lessons:
            extras.append(f"{len(a.lessons.lessons)} lessons learned")
        if a.mcp:
            extras.append(f"{len(a.mcp_tools)} MCP tools")
        body.append(" · ".join(extras) + "\n", style=t.dim)
        body.append("/help · @file to attach · #note to remember · shift+tab modes · ctrl+c interrupt",
                    style=t.dim)
        self.console.print(Panel(body, border_style=t.accent, expand=False, padding=(0, 1)))
        if a.mcp:
            for name, status in a.mcp.status.items():
                if status.startswith("failed"):
                    self.ui.warn(f"MCP server {name}: {status}")

    def add_memory(self, note: str) -> str:
        path = self.app.root / "MUYAH.md"
        existing = path.read_text(encoding="utf-8") if path.exists() else "# MUYAH.md\n"
        if "## Notes" not in existing:
            existing = existing.rstrip() + "\n\n## Notes\n"
        path.write_text(existing.rstrip() + f"\n- {note.strip()}\n", encoding="utf-8")
        from muyah_code.memory import load_instructions

        self.app.instructions = load_instructions(self.app.root, self.app.cwd, self.app.home,
                                                  bool(self.app.cfg.get("compat.claude_md", True)))
        self.app.agent.invalidate_system_prompt()
        return f"Saved to {path}"

    def run(self, initial_prompt: str | None = None) -> int:
        self.banner()
        pending = initial_prompt
        while True:
            if pending is None:
                try:
                    line = self.session.prompt(HTML(f'<style fg="{theme().accent}">❯</style> '))
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break
            else:
                line, pending = pending, None
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") and not line.startswith("#!") and len(line) > 1:
                self.console.print(f"[dim]{escape(self.add_memory(line[1:]))}[/]")
                continue
            if line.startswith("/"):
                try:
                    out = self.router.dispatch(line)
                except KeyboardInterrupt:
                    continue
                except Exception as e:  # a failing command must not kill the session
                    self.ui.error(f"/{line[1:].split()[0]} failed: {e}")
                    continue
                if out is EXIT:
                    break
                if isinstance(out, tuple) and out[0] == "prompt":
                    line = out[1]
                else:
                    continue
            self.console.print()
            self.ui.context_pct = self._ctx_pct()
            result = self.app.run_prompt(line)
            self.ui.turn_footer(result.status, result.duration, result.tool_calls, self._files_changed(),
                                self._ctx_pct())
            self.console.print()
        self.console.print("[dim]Bye. Session saved as " + escape(self.app.session_id) + "[/]")
        return 0


def run_repl(app, ui: TerminalUI, initial_prompt: str | None = None) -> int:
    return Repl(app, ui).run(initial_prompt)


__all__ = ["Repl", "run_repl"]
