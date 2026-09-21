"""Interactive REPL: prompt_toolkit input, slash-command menu, @file completion, Shift+Tab mode cycling.

Keys: Enter send · Alt+Enter / Ctrl+J newline · Shift+Tab mode · Ctrl+C clear line (twice on an empty line
to exit) · Ctrl+D exit · typing "/" opens the command menu (↑/↓ + Enter).
"""

from __future__ import annotations

import time

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from rich.markup import escape
from rich.text import Text

from muyah_code import __version__
from muyah_code.tools.search import list_files
from muyah_code.ui.commands import EXIT, CommandRouter
from muyah_code.ui.select import Prompter
from muyah_code.ui.terminal import TerminalUI
from muyah_code.ui.theme import theme

MODE_LABEL = {
    "default": "",
    "acceptEdits": "⏵⏵ accept edits on",
    "plan": "⏸ plan mode on",
    "bypassPermissions": "⚠ bypass permissions on",
}
EXIT_WINDOW = 2.0  # seconds between two Ctrl+C presses to exit
MENU_ROWS = 8      # rows the "/" and "@" menus may use


class _CompactPromptSession(PromptSession):
    """prompt_toolkit stretches the input area over every row below the cursor, which pushes the bottom
    toolbar to the bottom of the terminal. Cap the input at its own lines (plus room for an open
    completion menu) so the status line sits directly under the input, like Claude Code."""

    def _get_default_buffer_control_height(self) -> Dimension:
        buff = self.default_buffer
        lines = max(1, buff.document.line_count)
        if not get_app().is_done and buff.complete_state is not None:
            rows = min(MENU_ROWS, len(buff.complete_state.completions)) + 1
            return Dimension.exact(lines + rows)
        return Dimension.exact(lines)


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
            for name, help_text in self.repl.router.menu():
                if name.startswith(text[1:]):
                    yield Completion("/" + name, start_position=-len(text), display="/" + name,
                                     display_meta=help_text)
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
    def __init__(self, app, ui: TerminalUI, prompter: Prompter | None = None, **session_kwargs):
        self.app = app
        self.ui = ui
        self.console = ui.console
        self.prompter = prompter or Prompter()
        self.ui.prompter = self.prompter
        self.router = CommandRouter(self)
        self._last_interrupt = 0.0
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

        @kb.add("c-c")
        def _(event):
            buf = event.current_buffer
            if buf.text:
                buf.reset()  # first Ctrl+C clears what you typed
            else:
                event.app.exit(exception=KeyboardInterrupt())

        history = FileHistory(str(app.home / "history"))
        t = theme()
        style = Style.from_dict({
            "completion-menu.completion": "bg:default fg:default",
            "completion-menu.completion.current": f"bg:default fg:{t.accent} bold",
            "completion-menu.meta.completion": "bg:default fg:ansibrightblack",
            "completion-menu.meta.completion.current": "bg:default fg:ansibrightblack",
            "bottom-toolbar": "noreverse fg:ansibrightblack",
        })
        self.session = _CompactPromptSession(history=history, completer=_Completer(self), key_bindings=kb,
                                     complete_while_typing=True, bottom_toolbar=self._toolbar, style=style,
                                     reserve_space_for_menu=0, refresh_interval=0.5, **session_kwargs)

    def ask(self, message: str, password: bool = False) -> str:
        return self.prompter.ask(message, password=password)

    def _ctx_pct(self) -> int:
        used, usable = self.app.context_usage()
        return min(100, int(100 * used / max(1, usable)))

    def _files_changed(self) -> int:
        turns = self.app.checkpoints.turns
        return len(turns[-1]["changes"]) if turns else 0

    @staticmethod
    def _cols() -> int:
        try:
            return get_app().output.get_size().columns
        except Exception:
            return 80

    def _toolbar(self):
        """Under the input: a closing rule, then one status line (mode · context · hint)."""
        t = theme()
        rule = [("fg:ansibrightblack", "─" * self._cols() + "\n")]
        if time.monotonic() - self._last_interrupt < EXIT_WINDOW:
            return rule + [(f"fg:{t.accent}", "  Press Ctrl+C again to exit")]
        mode = MODE_LABEL.get(self.app.permissions.mode, "")
        if mode:
            status = [("", "  "), (f"fg:{t.accent} bold", mode), ("fg:ansibrightblack", " (shift+tab to cycle)")]
        else:
            status = [("fg:ansibrightblack", "  / for commands · shift+tab for modes")]
        status.append(("fg:ansibrightblack", f" · ctx {self._ctx_pct()}%"))
        return rule + status

    def _prompt_message(self):
        """A rule above the ❯ prompt (the input sits between two rules, like Claude Code)."""
        return [("fg:ansibrightblack", "─" * self._cols() + "\n"), (f"fg:{theme().accent}", "❯ ")]

    def _provider_label(self) -> str:
        from muyah_code.providers import BY_ID

        a = self.app
        prov = BY_ID.get(a.cfg.get("provider") or "")
        if prov:
            return prov.name
        profile = a.cfg.get("profile")
        return f"profile {profile}" if profile else a.llm.base_url.split("//")[-1].split("/")[0]

    def header(self, animate: bool = True) -> None:
        """Mascot + name/version, model/provider and folder - nothing else."""
        from muyah_code.ui.logo import play_intro

        a = self.app
        t = theme()
        cwd = str(a.cwd)
        room = max(20, self.console.width - 20)
        if len(cwd) > room:  # keep the end of the path (the folder you are in), never wrap
            cwd = "…" + cwd[-(room - 1):]
        title = Text()
        title.append("MUYAH-CODE", style="bold")
        title.append(f" v{__version__}", style=t.dim)
        lines = [title, Text(f"{a.llm.model} · {self._provider_label()}", style=t.dim), Text(cwd, style=t.dim)]
        play_intro(self.console, lines, animate=animate and self.ui.animate)
        if a.mcp:
            for name, status in a.mcp.status.items():
                if status.startswith("failed"):
                    self.ui.warn(f"MCP server {name}: {status}")
        self.console.print()

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

    def _read_line(self) -> str | None:
        """Returns the typed line, or None to exit."""
        while True:
            try:
                return self.session.prompt(self._prompt_message)
            except KeyboardInterrupt:  # Ctrl+C on an empty line
                now = time.monotonic()
                if now - self._last_interrupt < EXIT_WINDOW:
                    return None
                self._last_interrupt = now
            except EOFError:  # Ctrl+D
                return None

    def run(self, initial_prompt: str | None = None) -> int:
        self.header()
        pending = initial_prompt
        while True:
            if pending is None:
                line = self._read_line()
                if line is None:
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
        self.console.print(f"[dim]Bye. Resume this session with: muyah -r {escape(self.app.session_id)}[/]")
        return 0


def run_repl(app, ui: TerminalUI, initial_prompt: str | None = None) -> int:
    return Repl(app, ui).run(initial_prompt)


__all__ = ["Repl", "run_repl"]
