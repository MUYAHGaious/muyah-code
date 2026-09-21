"""Interactive REPL: prompt_toolkit input, slash-command menu, @file completion, Shift+Tab mode cycling.

Keys: Enter send · Alt+Enter / Ctrl+J newline · Shift+Tab mode · Ctrl+C clear line (twice on an empty line
to exit) · Ctrl+D exit · typing "/" opens the command menu (↑/↓ + Enter).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from rich.markup import escape
from rich.text import Text

from muyah_code import __version__
from muyah_code.tools.search import list_files
from muyah_code.ui import blocks
from muyah_code.ui.commands import EXIT, CommandRouter
from muyah_code.ui.select import Prompter
from muyah_code.ui.terminal import TerminalUI
from muyah_code.ui.theme import theme
from muyah_code.ui.voice import windows_dictation_available

MODE_LABEL = {
    "plan": ("⏸ plan", "read-only: explores and proposes a plan"),
    "acceptEdits": ("⏵ edit", "accepts file edits, asks before commands"),
    "default": ("● manual", "asks before edits and commands"),
    "auto": ("⏵⏵ auto", "runs on its own, asks only for risky actions"),
    "bypassPermissions": ("⚠ bypass", "no permission checks at all"),
}
# each mode has its own color (status line and the ❯ prompt): plan green, edit violet, auto yellow, bypass red
MODE_COLOR = {"plan": "#4ade80", "acceptEdits": "#c4b5fd", "auto": "#facc15", "bypassPermissions": "#f87171"}
EXIT_WINDOW = 2.0  # seconds between two Ctrl+C presses to exit
RESIZE_POLL = 0.1     # how often the prompt checks the terminal width
RESIZE_SETTLE = 0.3   # redraw once the width has stayed the same this long (user stopped dragging)
_RESIZED = object()   # prompt result meaning "the terminal was resized; redraw and ask again"
MENU_ROWS = 8      # rows the "/" and "@" menus may use
HISTORY_LIMIT = 1000   # ↑ goes back through at most this many earlier prompts (all projects)


class RecentHistory(FileHistory):
    """Your earlier prompts for ↑/↓, the newest HISTORY_LIMIT only (the file itself keeps growing)."""

    def load_history_strings(self):
        for i, item in enumerate(super().load_history_strings()):   # newest first
            if i >= HISTORY_LIMIT:
                return
            yield item

    def recent(self, n: int = HISTORY_LIMIT) -> list[str]:
        """Oldest to newest, for ↑ while it works."""
        return list(reversed(list(self.load_history_strings())[:n]))
PASTE_LINES = 8    # a paste longer than this (or PASTE_CHARS) shows as "[Pasted text #1 +245 lines]"
PASTE_CHARS = 1200


class _CompactPromptSession(PromptSession):
    """prompt_toolkit stretches the input area over every row below the cursor, which pushes the bottom
    toolbar to the bottom of the terminal. Cap the input at its own lines (plus room for an open
    completion menu) so the status line sits directly under the input, like Claude Code."""

    def _get_default_buffer_control_height(self) -> Dimension:
        buff = self.default_buffer
        try:
            rows = get_app().output.get_size().rows
        except Exception:
            rows = 24
        # never taller than the terminal: prompt_toolkit would show "Window too small"; the box scrolls instead
        lines = max(1, min(buff.document.line_count, max(3, rows - 6)))
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
        self.ui.cwd = app.cwd
        self.ui.events = app.events
        self.ui.on_mode_cycle = lambda: self.app.set_mode(self.app.permissions.cycle_mode())
        self.ui.input_status = lambda: (*MODE_LABEL.get(self.app.permissions.mode, (self.app.permissions.mode, "")),
                                        self._mode_color())
        self.ui.model_name = getattr(app.llm, "model", "")
        self.router = CommandRouter(self)
        self._last_interrupt = 0.0
        self._layout_width: int | None = None   # width the transcript was last laid out at
        self._resume_text = ""                  # typed text to restore after a resize redraw
        self._pastes: dict[str, str] = {}       # "[Pasted text #1 +245 lines]" -> the text (sent on Enter)
        kb = KeyBindings()

        @kb.add("s-tab")
        def _(event):
            self.app.set_mode(self.app.permissions.cycle_mode())
            event.app.invalidate()

        @kb.add(Keys.BracketedPaste)
        def _(event):
            event.current_buffer.insert_text(self._paste(event.data))

        @kb.add("c-space")
        def _(event):
            from prompt_toolkit.application import run_in_terminal

            from muyah_code.ui.voice import start_dictation

            started, message = start_dictation()
            if not started:
                run_in_terminal(lambda: self.console.print(f"[dim]{escape(message)}[/]"))

        @kb.add("escape", "escape")
        def _(event):
            if not event.current_buffer.text:
                event.app.exit(result="/rewind")

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

        history = RecentHistory(str(app.home / "history"))
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
                                     reserve_space_for_menu=0, erase_when_done=True, **session_kwargs)

    def ask(self, message: str, password: bool = False) -> str:
        return self.prompter.ask(message, password=password)

    def _ctx_pct(self) -> int:
        used, usable = self.app.context_usage()
        return min(100, int(100 * used / max(1, usable)))

    _turn_started = 0.0
    _notifier = None
    _carry: list = []          # queued messages still waiting after the one now running

    @property
    def notifier(self):
        if self._notifier is None:
            import sys

            from muyah_code.notify import Notifier

            def write(seq: str) -> None:
                out = getattr(self.console, "file", None) or sys.stdout
                out.write(seq)
                out.flush()

            self._notifier = Notifier(self.app.cfg, write, hooks=self.app.hooks)
        return self._notifier

    def _attention(self, message: str) -> None:
        if self.notifier.should(self.ui.focused, time.monotonic() - self._turn_started):
            self.notifier.notify("attention", message)

    def _notify_done(self, result) -> None:
        if result.status == "interrupted" or not self.notifier.should(self.ui.focused, result.duration):
            return
        first = " ".join((result.text or "").split())[:100]
        if result.status == "ok":
            self.notifier.notify("done", f"Done in {result.duration:.0f}s" + (f": {first}" if first else ""))
        else:
            what = "budget reached" if result.status == "budget" else f"stopped ({result.status})"
            self.notifier.notify("budget" if result.status == "budget" else "attention",
                                 f"{what}: {result.error or first}"[:160])

    def _btw_async(self, question: str) -> None:
        """/btw typed while it works: answered in parallel, shown as soon as it is ready."""
        import threading

        def run():
            try:
                answer = self.app.btw(question)
            except Exception as e:   # a failed side question must never disturb the running turn
                answer = f"(The side question failed: {e})"
            self.ui.side_answer(question, answer)

        threading.Thread(target=run, name="muyah-btw", daemon=True).start()

    def _spent(self) -> str:
        """"$0.42" (or "$0.42 of $5.00" with a session budget) once this session has cost something."""
        ledger, budget = getattr(self.app, "ledger", None), getattr(self.app, "budget", None)
        if ledger is None or not ledger.cost:
            return ""
        from muyah_code.pricing import money

        limit = budget.session_usd if budget is not None else 0
        return money(ledger.cost) + (f" of ${limit:.2f}" if limit else "")

    def _turn_changes(self) -> tuple[list[tuple[str, str]], list[str]]:
        """What the last turn changed (including by commands) and any tests it weakened."""
        from muyah_code.verify import weakened_tests

        rewind = self.app.rewind
        try:
            changes = rewind.turn_changes()
            return changes, weakened_tests(changes, rewind.text_before_turn, self.app.root)
        except Exception as e:  # the footer must never break the session
            self.ui.error(f"could not list this turn's changes: {e}")
            return [], []

    @staticmethod
    def _cols() -> int:
        try:
            return get_app().output.get_size().columns
        except Exception:
            return 80

    def _toolbar(self):
        """Under the input: a closing rule, then one status line (mode · context · hint)."""
        t = theme()
        # one column short of the edge: an exactly full-width line gets re-wrapped by terminals on resize
        rule = [("fg:ansibrightblack", "─" * max(1, self._cols() - 1) + "\n")]
        if time.monotonic() - self._last_interrupt < EXIT_WINDOW:
            return rule + [(f"fg:{t.accent}", "  Press Ctrl+C again to exit")]
        label, what = MODE_LABEL.get(self.app.permissions.mode, (self.app.permissions.mode, ""))
        style = f"fg:{self._mode_color()} bold"
        status = [("", "  "), (style, label), ("fg:ansibrightblack", f" · {what} (shift+tab)")]
        if windows_dictation_available():
            status.append(("fg:ansibrightblack", " · ctrl+space: speak"))
        viz = getattr(self.app, "viz", None)
        if viz is not None and viz.running:
            status.append((f"fg:{t.accent}", " · ● live view on (/viz reopens it)"))
        # context use (and cost) at the bottom right, like Claude Code
        spent = self._spent()
        right = f"{spent} · " if spent else ""
        right += f"ctx {self._ctx_pct()}%"
        used = sum(len(text) for _, text in status)
        gap = self._cols() - 1 - used - len(right) - 1
        if gap < 2:           # too narrow: keep it on the same line, right after the rest
            return rule + status + [("fg:ansibrightblack", " · " + right)]
        return rule + status + [("", " " * gap), ("fg:ansibrightblack", right)]

    def _prompt_message(self):
        """A rule above the ❯ prompt (the input sits between two rules, like Claude Code)."""
        return [("fg:ansibrightblack", "─" * max(1, self._cols() - 1) + "\n"), (f"fg:{self._mode_color()} bold", "❯ ")]

    def _mode_color(self) -> str:
        return MODE_COLOR.get(self.app.permissions.mode, theme().accent)

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

    def _watch_resize(self) -> None:
        """Runs inside the prompt: when the width changes and then stays still for a moment (the user
        stopped dragging), leave the prompt so the whole conversation can be redrawn at the new width."""
        app = get_app()

        async def watch():
            pending, since = None, 0.0
            while True:
                await asyncio.sleep(RESIZE_POLL)
                cols = self._cols()
                if cols == self._layout_width:
                    pending = None
                    continue
                if cols != pending:
                    pending, since = cols, time.monotonic()
                elif time.monotonic() - since >= RESIZE_SETTLE:
                    self._resume_text = app.current_buffer.text
                    app.exit(result=_RESIZED)
                    return

        app.create_background_task(watch())

    def _redraw(self) -> None:
        self._layout_width = self._cols_now()
        self.ui.rerender()

    def _cols_now(self) -> int:
        try:
            return self.session.app.output.get_size().columns
        except Exception:
            return self.console.width

    def _read_line(self) -> str | None:
        """Returns the typed line, or None to exit."""
        while True:
            if self._layout_width is None:
                self._layout_width = self._cols_now()
            elif self._cols_now() != self._layout_width:  # resized while the agent was working
                self._redraw()
            default, self._resume_text = self._resume_text, ""
            try:
                line = self.session.prompt(self._prompt_message, default=default, pre_run=self._watch_resize)
            except KeyboardInterrupt:  # Ctrl+C on an empty line
                now = time.monotonic()
                if now - self._last_interrupt < EXIT_WINDOW:
                    return None
                self._last_interrupt = now
                continue
            except EOFError:  # Ctrl+D
                return None
            if line is _RESIZED:
                self._redraw()
                continue
            if line == "/rewind" and not self.session.default_buffer.text:
                return line
            # the input area is erased when you press Enter; your message is shown as a highlighted block
            if line.strip():
                self.console.print(blocks.user_prompt(line))   # big pastes stay collapsed on screen
                self.ui.mark_prompt()
            return self._expand_pastes(line)

    def _paste(self, data: str) -> str:
        """A big paste goes into the prompt as a short placeholder (like Claude Code); the real text is sent."""
        text = data.replace("\r\n", "\n").replace("\r", "\n")
        image = self._pasted_image_path(text)
        if image:
            return image
        lines = text.count("\n") + 1
        if lines <= PASTE_LINES and len(text) <= PASTE_CHARS:
            return text
        key = f"[Pasted text #{len(self._pastes) + 1} +{lines} lines]"
        self._pastes[key] = text
        return key

    def _pasted_image_path(self, text: str) -> str:
        """Dragging an image into the terminal pastes its path: turn it into an @mention (it is attached)."""
        from muyah_code.llm.content import is_image_path

        raw = text.strip().strip('"').strip("'")
        if not raw or "\n" in raw or len(raw) > 400:
            return ""
        path = Path(raw).expanduser()
        try:
            if not (path.is_file() and is_image_path(path)):
                return ""
            rel = path.resolve().relative_to(self.app.cwd)
            shown = rel.as_posix()
        except (OSError, ValueError):
            shown = path.as_posix()
        return "" if " " in shown else f"@{shown} "   # mentions cannot contain spaces: leave those as typed

    def _expand_pastes(self, line: str) -> str:
        for key, text in self._pastes.items():
            line = line.replace(key, text)
        self._pastes.clear()
        return line

    def offer_viz(self) -> None:
        """At startup: open the live view? Asked until you pick Always or Never (setting viz.autostart)."""
        choice = self.app.cfg.get("viz.autostart")
        if choice is False:
            return
        if choice is not True:
            picked = self.prompter.select("Open the live view in your browser? It shows what MUYAH-CODE is doing", [
                ("yes", "Yes, open it"),
                ("always", "Always open it (don't ask again)"),
                ("no", "Not now"),
                ("never", "Never ask again (/viz still opens it)"),
            ], default="no")
            if picked in ("always", "never"):
                self.app.cfg.persist("viz.autostart", picked == "always")
            if picked not in ("yes", "always"):
                return
        self.router.viz("")

    def run(self, initial_prompt: str | None = None) -> int:
        self.header()
        if self.console.is_terminal and _interactive_stdin():
            self.offer_viz()
        if getattr(self.app, "resumed", False):
            from muyah_code.ui.history import print_history

            print_history(self.console, self.app.agent.messages[1:])
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
            self.ui.context_pct = self._ctx_pct()
            self.ui.on_btw = self._btw_async
            if self.app.cfg.get("tips", True):
                from muyah_code.ui.tips import build_tips

                self.ui.tips.reset(build_tips(self.app, self.router.commands))   # rebuilt: new skills/tools show up
            self.ui.history = self.session.history.recent if hasattr(self.session.history, "recent") else None
            self.ui.on_attention = self._attention
            self._turn_started = time.monotonic()
            self.ui.begin_typing(carried=self._carry)
            self._carry = []
            cost_before = self.app.ledger.cost
            try:
                result = self.app.run_prompt(line)
            finally:
                queued, draft = self.ui.end_typing()
            changes: list = []
            try:
                changes, weakened = self._turn_changes()
                self.ui.turn_footer(result.status, result.duration, result.tool_calls, len(changes),
                                    self._ctx_pct(), warnings=weakened, cost=self.app.ledger.cost - cost_before)
                self.console.print()
            except KeyboardInterrupt:  # an Esc that arrived just as the turn ended
                pass
            self._notify_done(result)
            auto = str(self.app.cfg.get("verify.auto", "off") or "off").lower()
            if auto in ("quick", "full") and changes and result.status == "ok" and not queued:
                try:
                    self.router.verify(auto)
                    self.console.print()
                except KeyboardInterrupt:
                    self.console.print("[dim]Verification stopped.[/]")
            if queued:
                # typed while it worked and not delivered yet (Esc, or the turn ended first): they run one at a
                # time, oldest first; the rest stay queued, and each Esc stops the current one and moves on
                pending, self._carry = queued[0], queued[1:]
                self.console.print(blocks.user_prompt(pending))
                self.ui.mark_prompt()
                continue
            if draft:
                self._resume_text = draft
        self.console.print(f"[dim]Bye. Resume this session with: muyah -r {escape(self.app.session_id)}[/]")
        return 0


def _interactive_stdin() -> bool:
    import sys

    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (ValueError, OSError):
        return False


def run_repl(app, ui: TerminalUI, initial_prompt: str | None = None) -> int:
    return Repl(app, ui).run(initial_prompt)


__all__ = ["Repl", "run_repl"]
