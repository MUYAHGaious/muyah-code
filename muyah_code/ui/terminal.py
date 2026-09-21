"""Interactive terminal rendering with Rich.

Improvements over a plain streaming CLI:
  * live Markdown: the answer renders as formatted markdown while it streams; only the last few lines
    stay "in motion" so long answers never flicker or duplicate
  * live stats while generating: elapsed time, tokens, tokens/s (vital on slow local models)
  * tool calls as bullets with results, Bash output tails, line-numbered colored diffs
  * themes (muyah, ocean, forest, mono, light)
"""

from __future__ import annotations

import _thread
import io
import re
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from rich.console import Console, Group
from rich.control import Control
from rich.live import Live
from rich.segment import Segment
from rich.style import Style
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from muyah_code.tools.base import ToolResult
from muyah_code.ui import blocks
from muyah_code.ui.base import UI, PermissionReply, PermissionRequest
from muyah_code.ui.theme import theme

MAX_DIFF_LINES = 40
LIVE_TAIL_LINES = 8
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
# While the model thinks, the label changes every few seconds (so a long wait visibly moves on)
THINKING_WORDS = ("Thinking", "Reasoning", "Working it out", "Weighing options", "Connecting the dots",
                  "Planning the next step", "Checking the details")
WORD_SECONDS = 4.0
PROSE_WIDTH = 110  # answers wrap at this width even on very wide terminals (readable line length)


def render_diff(diff: str, limit: int = MAX_DIFF_LINES) -> Text:
    """Line-numbered, colored diff (new-file previews without hunks are shown as additions)."""
    t = theme()
    out = Text()
    old_no = new_no = 0
    shown = 0
    lines = [ln for ln in diff.splitlines() if not ln.startswith(("---", "+++"))]
    for ln in lines:
        if shown >= limit:
            out.append(f"      ... {len(lines) - shown} more lines\n", style=t.dim)
            break
        m = HUNK_RE.match(ln)
        if m:
            old_no, new_no = int(m.group(1)), int(m.group(2))
            if shown:
                out.append("      ⋮\n", style=t.dim)
            continue
        if ln.startswith("+"):
            out.append(f"{new_no or '':>6} ", style=t.dim)
            out.append(f"+ {ln[1:]}\n", style=f"{t.ok} on {t.add_bg}")
            new_no += 1 if new_no else 0
        elif ln.startswith("-"):
            out.append(f"{old_no or '':>6} ", style=t.dim)
            out.append(f"- {ln[1:]}\n", style=f"{t.err} on {t.del_bg}")
            old_no += 1 if old_no else 0
        else:
            out.append(f"{new_no or '':>6}   {ln[1:] if ln.startswith(' ') else ln}\n", style=t.dim)
            old_no += 1 if old_no else 0
            new_no += 1 if new_no else 0
        shown += 1
    return out


CLEAR_SCREEN = "\x1b[2J\x1b[3J\x1b[H"  # clear screen + scrollback, cursor home


def _copy_to_clipboard(text: str) -> bool:
    """Best effort: put text on the system clipboard (Windows clip, macOS pbcopy, Linux wl-copy/xclip)."""
    import shutil
    import subprocess
    import sys

    if sys.platform == "win32":
        cmds = [["clip"]]
    elif sys.platform == "darwin":
        cmds = [["pbcopy"]]
    else:
        cmds = [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
    for cmd in cmds:
        if not shutil.which(cmd[0]):
            continue
        try:
            data = text.encode("utf-16-le" if cmd[0] == "clip" else "utf-8")
            subprocess.run(cmd, input=data, timeout=3, check=True, capture_output=True)
            return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


def _paused(console: Console):
    return console.paused() if isinstance(console, ReplayConsole) else nullcontext()


class ReplayConsole(Console):
    """A Rich console that remembers what it printed as renderables (not pre-wrapped text), so the whole
    conversation can be redrawn at a new width after the terminal is resized - the approach Claude Code
    uses instead of leaving old output wrapped at the old width."""

    MAX_ENTRIES = 4000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transcript: list[tuple[tuple, dict]] = []
        self._paused = 0

    def print(self, *objects, **kwargs):  # noqa: A003 - Rich API name
        super().print(*objects, **kwargs)
        if not self._paused and not (objects and all(isinstance(o, Control) for o in objects)):
            self.remember(*objects, **kwargs)

    def remember(self, *objects, **kwargs) -> None:
        """Remember something for replay without printing it (e.g. what prompt_toolkit drew)."""
        keep = {k: v for k, v in kwargs.items() if k in ("end", "style", "justify", "overflow", "no_wrap", "soft_wrap")}
        self.transcript.append((objects, keep))
        if len(self.transcript) > self.MAX_ENTRIES:
            del self.transcript[: len(self.transcript) - self.MAX_ENTRIES]

    @contextmanager
    def paused(self):
        self._paused += 1
        try:
            yield
        finally:
            self._paused -= 1

    def replay(self) -> None:
        """Clear the screen and scrollback, then redraw everything at the current width."""
        self.file.write(CLEAR_SCREEN)
        self.file.flush()
        with self.paused():
            for objects, kwargs in list(self.transcript):
                super().print(*objects, **kwargs)


class PrefixedMarkdown:
    """An assistant message: markdown with the ● bullet, laid out at whatever width it is printed at."""

    def __init__(self, text: str):
        self.text = text

    def __rich_console__(self, console, options):
        width = max(20, min(options.max_width - 2, PROSE_WIDTH))
        lines = console.render_lines(Markdown(self.text, code_theme=theme().code_theme),
                                     options.update(width=width), pad=False)
        while lines and not "".join(s.text for s in lines[-1]).strip():
            lines.pop()
        bullet = Segment("● ", Style.parse(f"bold {theme().accent}"))
        for i, line in enumerate(lines):
            yield bullet if i == 0 else Segment("  ")
            yield from line
            yield Segment.line()


class _MarkdownStream:
    """Render streaming markdown: print settled lines permanently, keep the tail live."""

    def __init__(self, console: Console, live: Live, stats):
        self.console = console
        self.live = live
        self.stats = stats
        self.text = ""
        self.printed = 0
        self._last_render = 0.0

    def _lines(self, text: str) -> list[str]:
        width = max(20, min(self.console.width - 2, PROSE_WIDTH))
        buf = io.StringIO()
        c = Console(file=buf, force_terminal=True, width=width, color_system=self.console.color_system or "standard",
                    highlight=False)
        c.print(Markdown(text, code_theme=theme().code_theme))
        lines = buf.getvalue().splitlines()
        while lines and not Text.from_ansi(lines[-1]).plain.strip():
            lines.pop()
        return lines

    def _prefix(self, i: int) -> Text:
        return Text("● ", style=f"bold {theme().accent}") if i == 0 else Text("  ")

    def update(self, chunk: str, final: bool = False) -> None:
        self.text += chunk
        if self.live is None:  # not animating (piped output, screenshots): render once at the end
            if final:
                self.console.print(PrefixedMarkdown(self.text))
            return
        now = time.monotonic()
        if not final and now - self._last_render < 0.08:
            return
        self._last_render = now
        lines = self._lines(self.text)
        settled = len(lines) if final else max(0, len(lines) - LIVE_TAIL_LINES)
        # The streamed lines are laid out for the current width; for redraws after a resize we remember
        # the whole message as markdown instead (see ReplayConsole).
        with _paused(self.console):
            for i in range(self.printed, settled):
                self.live.console.print(self._prefix(i) + Text.from_ansi(lines[i]))
        if final and isinstance(self.console, ReplayConsole):
            self.console.remember(PrefixedMarkdown(self.text))
        self.printed = max(self.printed, settled)
        tail = Text()
        for i in range(self.printed, len(lines)):
            tail.append_text(self._prefix(i) + Text.from_ansi(lines[i]))
            tail.append("\n")
        self.live.update(Group(tail, self.stats()) if not final else Text(""))


class TerminalUI(UI):
    headless = False

    def __init__(self, console: Console | None = None, show_reasoning: bool = False, animate: bool | None = None):
        self.console = console or ReplayConsole(highlight=False)
        self.show_reasoning = show_reasoning
        # spinners/live markdown only on a real terminal; plain output when piped or recorded
        self.animate = self.console.is_terminal if animate is None else animate
        self._live: Live | None = None
        self._stream: _MarkdownStream | None = None
        self._phase = ""
        self._t0 = 0.0
        self._first_token = 0.0
        self._chars = 0
        self._reasoning_chars = 0
        self.context_pct = None  # set by the REPL so the stats line can show context usage
        self.prompter = None     # arrow-key menus; set by the REPL (ui.select.Prompter)
        self.cwd: Path | None = None  # paths are shown relative to this
        self._last = None        # what was printed last: "inline" | "block" | None (spacing between blocks)
        self._tool: tuple[blocks.ToolTitle, float] | None = None   # the tool running now (live area)
        self._sub_current = ""   # a sub-agent's current step (shown under its Agent line while it runs)
        self._sub_steps = 0
        self._sub_errors = 0
        # typing while it works (see ui/typeahead.py)
        self._draft = ""
        self._queued: list[str] = []
        self._selected: int | None = None   # a queued message picked with ↑ (enter: edit it, delete: drop it)
        self.on_btw = None         # (question) -> None: answer a /btw side question now (set by the REPL)
        self.focused: bool | None = None   # the terminal window has focus (None: it never said)
        self.on_attention = None   # (message) -> None: it needs you (an approval, a question); set by the REPL
        self._keys = threading.Lock()
        self._reader = None
        self._turn_active = False
        self.events = None       # EventBus (set by the REPL): the live view shows the queue
        self.on_mode_cycle = None  # Shift+Tab while it works (set by the REPL)
        self.input_status = None   # () -> (label, description, color) for the status line under the box
        self.model_name = ""     # shown while waiting for the provider (set by the REPL)
        self._sent_at = 0.0      # when the current request left; 0 once its first byte arrived
        # One spinner for the whole live area. A Rich Spinner picks its frame from how long it has
        # existed: building a new one on every refresh froze the animation on its first frame.
        self._spinner = Spinner("dots")

    # ------------------------------------------------------------------ live status

    def _stats_line(self):
        t = theme()
        elapsed = time.monotonic() - self._t0
        if self._tool is not None:
            tt, _ = self._tool
            line = blocks.header(tt, True, self.cwd, self.console.width - 22)
            line.append(f"  {elapsed:.0f}s", style=t.dim)
            line.append("  (esc to interrupt · type to queue a message)", style=t.dim)
            spin = self._spin(line)
            if tt.name == "Agent" and self._sub_current:
                step = Text("  ⎿ ", style=t.dim)
                step.append(f"{self._sub_current}", style=t.dim)
                step.append(f"  · {self._sub_steps} step{'s' if self._sub_steps != 1 else ''}", style=t.dim)
                step.no_wrap, step.overflow = True, "ellipsis"
                return Group(spin, step)
            return spin
        parts = [f"{elapsed:.0f}s"]
        if self._phase == "thinking" and self._sent_at:
            # the request is out and nothing has come back yet: say so (slow providers queue requests)
            label = f"Waiting for {self.model_name or 'the model'}"
            parts = [f"sent {time.monotonic() - self._sent_at:.0f}s ago"]
        elif self._phase == "thinking":
            label = THINKING_WORDS[int(elapsed // WORD_SECONDS) % len(THINKING_WORDS)]
            if self._reasoning_chars:
                parts.append(f"{int(self._reasoning_chars / 3.5):,} reasoning tokens")
        elif self._phase == "writing":
            label = "Writing"
            toks = int(self._chars / 3.5)
            gen_time = max(0.001, time.monotonic() - self._first_token)
            parts.append(f"{toks:,} tokens")
            if gen_time > 1:
                parts.append(f"{toks / gen_time:.1f} tok/s")
        elif self._phase in ("", "Working"):
            label = THINKING_WORDS[int(elapsed // WORD_SECONDS) % len(THINKING_WORDS)]
        else:
            label = self._phase
        if self.context_pct is not None:
            parts.append(f"ctx {self.context_pct}%")
        return self._spin(Text.assemble((f"{label}… ", t.accent), (" · ".join(parts), t.dim),
                                        ("  (esc to interrupt · type to queue a message)", t.dim)))

    def _spin(self, text: Text) -> Spinner:
        self._spinner.update(text=text, style=theme().accent)
        return self._spinner

    def _with_typing(self, renderable):
        """While it works, the input box stays under the live status (like when idle): queued messages,
        the ❯ field with what you are typing, and the status line with the mode."""
        with self._keys:
            draft, queued = self._draft, list(self._queued)
        if self._reader is None and not queued:   # not listening (piped output, screenshots)
            return renderable
        t = theme()
        width = max(20, self.console.width - 1)
        rows = [renderable, Text("")]
        selected = self._selected
        for i, msg in enumerate(queued):
            if i == selected:
                line = Text("› ", style=f"bold {t.accent}")
                line.append(msg, style="reverse")
                line.append("  · enter: edit · delete: remove · ↑↓: move", style=f"italic {t.accent}")
            else:
                line = Text("› ", style=t.dim)
                line.append(msg, style=t.dim)
                line.append("  · queued: goes in at the next step (esc: send now · ↑: edit)", style=f"italic {t.dim}")
            line.no_wrap, line.overflow = True, "ellipsis"
            rows.append(line)
        if self._reader is None:
            return Group(*rows)
        label, what, color = self.input_status() if self.input_status else ("", "", t.accent)
        rows.append(Text("─" * width, style="bright_black"))
        field = Text("❯ ", style=f"bold {color}")
        if draft:
            field.append(draft)
            field.append("▌", style=color)
        else:
            field.append("type to queue a message · enter: queue · esc: send now · /btw: ask on the side",
                         style=t.dim)
        field.no_wrap, field.overflow = True, "ellipsis"
        rows.append(field)
        rows.append(Text("─" * width, style="bright_black"))
        status = Text("  ")
        if label:
            status.append(label, style=f"bold {color}")
            status.append(f" · {what} (shift+tab)", style=t.dim)
        status.no_wrap, status.overflow = True, "ellipsis"
        rows.append(status)
        return Group(*rows)

    # ------------------------------------------------------------------ typing while it works

    def begin_typing(self) -> None:
        """A turn is starting: listen for keys (if this is a real terminal)."""
        from muyah_code.ui.typeahead import KeyReader

        self._turn_active = True
        if self.animate and KeyReader.available():
            self._reader = KeyReader(self._on_key)
            self._reader.start()

    def end_typing(self) -> tuple[list[str], str]:
        """The turn ended: stop listening. Returns (queued messages not yet delivered, unsent draft)."""
        self._turn_active = False
        self._stop_live()
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        with self._keys:
            queued, draft = self._queued, self._draft
            self._queued, self._draft, self._selected = [], "", None
        return queued, draft

    def take_queued(self) -> list[str]:
        """Messages you queued while it worked: the agent takes them at its next step."""
        with self._keys:
            queued, self._queued, self._selected = self._queued, [], None
        if queued:
            self._emit_queue()
        for msg in queued:
            self._space("block")
            self._out(blocks.user_prompt(msg))
            self._last = "block"
        return queued

    def _emit_queue(self) -> None:
        if self.events is not None:
            with self._keys:
                items = list(self._queued)
            self.events.emit("queue", items=items)

    def _on_key(self, key: str) -> None:
        if key in ("focus-in", "focus-out"):
            self.focused = key == "focus-in"
            return
        interrupt = False
        btw = ""
        before = list(self._queued)
        with self._keys:
            sel = self._selected if self._selected is not None and self._selected < len(self._queued) else None
            if key == "up" and self._queued:
                self._selected = len(self._queued) - 1 if sel is None else max(0, sel - 1)
            elif key == "down" and sel is not None:
                self._selected = sel + 1 if sel + 1 < len(self._queued) else None
            elif key == "delete" and sel is not None:
                self._queued.pop(sel)
                self._selected = min(sel, len(self._queued) - 1) if self._queued else None
            elif key == "enter" and sel is not None and not self._draft:
                self._draft = self._queued.pop(sel)       # back into the box to edit, then enter queues it again
                self._selected = None
            elif key == "enter":
                text = self._draft.strip()
                if text.startswith("/btw ") and self.on_btw is not None:
                    btw = text[5:].strip()
                elif text:
                    self._queued.append(text)
                self._draft = ""
                self._selected = None
            elif key == "backspace":
                self._draft = self._draft[:-1]
            elif key in ("shift-tab", "mic"):
                pass                       # handled below, outside the lock
            elif key == "esc" and sel is not None:
                self._selected = None      # leave the queue; esc again sends now
            elif key in ("esc", "ctrl-c"):
                if self._draft.strip():
                    self._queued.append(self._draft.strip())
                self._draft = ""
                self._selected = None
                interrupt = True
            elif len(key) == 1:
                self._draft += key
                self._selected = None
        if btw:
            self.on_btw(btw)
        if key == "shift-tab" and self.on_mode_cycle is not None:
            self.on_mode_cycle()           # takes effect from the agent's next action
        if key == "mic":
            from muyah_code.ui.voice import start_dictation

            start_dictation()              # Windows voice typing types into this box
        if self._queued != before:
            self._emit_queue()
        if interrupt and self._turn_active:
            _thread.interrupt_main()   # same as Ctrl+C: the turn stops; the REPL sends what is queued

    def side_answer(self, question: str, answer: str) -> None:
        """A /btw answer: shown in its own panel, never added to the conversation."""
        t = theme()
        body = Group(Text(question, style=f"bold {t.dim}"), Text(""), Markdown(answer))
        self.console.print(Panel(body, title=f"[{t.accent}]btw[/] [dim]· not added to the conversation[/]",
                                 title_align="left", border_style=t.dim, padding=(0, 1)))

    def _keyboard_to_prompt(self):
        """Hand the keyboard to a menu/question for a moment."""
        reader = self._reader

        class _Pause:
            def __enter__(self_inner):
                if reader is not None:
                    reader.pause()

            def __exit__(self_inner, *exc):
                if reader is not None:
                    reader.resume()
        return _Pause()

    def _start_live(self, phase: str) -> None:
        self._stop_live()
        self._phase = phase
        self._t0 = time.monotonic()
        if not self.animate:
            return
        self._live = Live(self._StatsRenderable(self), console=self.console, refresh_per_second=8, transient=True)
        self._live.start()

    class _StatsRenderable:
        def __init__(self, ui):
            self.ui = ui

        def __rich__(self):
            return self.ui._with_typing(self.ui._stats_line())

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    # ------------------------------------------------------------------ assistant text

    def model_status(self, state: str) -> None:
        if state == "sent":
            self._sent_at = time.monotonic()
        elif state == "first_token":
            self._sent_at = 0.0

    def assistant_start(self) -> None:
        self._tool = None
        self._sent_at = 0.0
        self._chars = 0
        self._reasoning_chars = 0
        self._stream = None
        self._start_live("thinking")

    def reasoning(self, chunk: str) -> None:
        self._reasoning_chars += len(chunk)
        if self.show_reasoning and self._live is not None:
            self._live.console.print(Text(chunk, style=f"italic {theme().dim}"), end="")

    def text(self, chunk: str) -> None:
        if not chunk:
            return
        if self._stream is None:
            if not chunk.strip():
                return
            if self._live is None and self.animate:
                self._start_live("writing")
            self._phase = "writing"
            self._first_token = time.monotonic()
            self._space("block")
            self._stream = _MarkdownStream(self.console, self._live, self._stats_line)
            chunk = chunk.lstrip("\n")
        self._chars += len(chunk)
        self._stream.update(chunk)

    def busy(self, label: str) -> None:
        """Something is happening that prints nothing (a hook, the learning step): show it, animated."""
        if self._turn_active or self._live is None:
            self._start_live(label)

    def _keep_working(self) -> None:
        """Between steps of a turn the spinner never stops: the model is still at work."""
        if self._turn_active and self._live is None:
            self._start_live("Working")

    def assistant_end(self) -> None:
        if self._stream is not None:
            self._stream.update("", final=True)
            self._stream = None
            self._stop_live()
            self._keep_working()
        else:
            self._stop_live()

    # ------------------------------------------------------------------ spacing

    def _space(self, kind: str) -> None:
        """One blank line between blocks; quick one-line lookups stack without gaps."""
        if self._last is not None and not (kind == "inline" and self._last == "inline"):
            self._out(Text(""))
        self._last = kind

    def _out(self, renderable) -> None:
        if self._live is not None:
            self._live.console.print(renderable)
        else:
            self.console.print(renderable)

    def mark_prompt(self) -> None:
        """The REPL printed your prompt: the next block starts after one blank line."""
        self._last = "block"

    # ------------------------------------------------------------------ tools

    def tool_start(self, title: str) -> None:
        tt = blocks.parse_title(title)
        if tt.label and self._tool is not None and self._tool[0].name == "Agent":
            # a sub-agent's own step: shown live under its Agent line, not printed one by one
            self._sub_current = blocks.header(tt, True, self.cwd, 200).plain
            self._sub_steps += 1
            return
        self._stop_live()
        self._tool = (tt, time.monotonic())
        if tt.name == "Agent":
            self._sub_current, self._sub_steps, self._sub_errors = "", 0, 0
        self._start_live("tool")

    def tool_end(self, title: str, result: ToolResult) -> None:
        tt = blocks.parse_title(title)
        if tt.label and self._tool is not None and self._tool[0].name == "Agent":
            self._sub_errors += 1 if result.is_error else 0
            return
        started = self._tool[1] if self._tool is not None else time.monotonic()
        elapsed = time.monotonic() - started
        self._tool = None
        self._stop_live()
        if tt.name not in ("TodoWrite", "AskUser") or result.is_error:  # those print their own block
            self._print_tool(tt, result, elapsed)
        self._keep_working()

    def _print_tool(self, tt: blocks.ToolTitle, result: ToolResult, elapsed: float) -> None:
        t = theme()
        ok = not result.is_error
        head = blocks.header(tt, False, self.cwd, self.console.width - 3)
        glyph = blocks.glyph_for(tt, ok)
        if tt.label:
            head = Text(f"[{tt.label}] ", style=t.dim) + head
        children: list = []
        kind = tt.kind
        took = f" · {elapsed:.1f}s" if elapsed >= 1 else ""
        if not ok:
            lines = (result.content or result.summary or "failed").strip().splitlines()
            if lines and re.fullmatch(r"\[exit code -?\d+\]", lines[-1].strip()):
                lines.pop()  # shown below as "exit N"
            shown = lines[-6:] if tt.name == "Bash" else lines[:6]
            children.append(Text("\n".join(ln[:220] for ln in shown), style=t.err))
            if tt.name == "Bash":
                code = result.meta.get("exit_code")
                if code is not None:
                    children.append(Text(f"exit {code}{took}", style=t.err))
            self._space("block")
            self._out(blocks.gutter(glyph, head, children))
            return
        if kind == "explore":
            summary = result.summary or ""
            verb = blocks.KINDS.get(tt.name, ("", "", "", ""))[3]
            if verb and summary.lower().startswith(verb.lower() + " "):
                summary = summary[len(verb) + 1:]   # "Read · Read 3 lines" -> "Read · 3 lines"
            line = head.copy()
            if summary:
                line.append(f"  · {summary}", style=t.dim)
            line.truncate(max(20, self.console.width - 3), overflow="ellipsis")
            self._space("inline")
            self._out(blocks.gutter(glyph, line))
            return
        if kind == "shell":
            if took:
                head.append(took, style=t.dim)
            children.append(blocks.preview(result.content if result.content != "(no output)" else ""))
        elif kind == "edit":
            added, removed = blocks.diff_counts(result.display)
            if added or removed:
                head.append("  ")
                head.append(f"+{added}", style=t.ok)
                head.append(" ")
                head.append(f"-{removed}", style=t.err)
            if result.display:
                diff = render_diff(result.display)
                diff.rstrip()
                children.append(diff)
        elif kind == "agent":
            steps = f"{self._sub_steps} step{'s' if self._sub_steps != 1 else ''}"
            errs = f" · {self._sub_errors} failed" if self._sub_errors else ""
            children.append(Text(f"✓ Done · {steps}{errs}{took}", style=t.dim))
            self._sub_current, self._sub_steps, self._sub_errors = "", 0, 0
        elif kind == "mcp":
            children.append(blocks.preview(result.content, head=1, tail=3))
        else:
            if result.summary:
                children.append(Text(result.summary + took, style=t.dim))
        self._space("block")
        self._out(blocks.gutter(glyph, head, children))

    def on_todos(self, todos: list[dict]) -> None:
        t = theme()
        self._stop_live()
        done = sum(1 for td in todos if td.get("status") == "completed")
        head = Text("Updated plan", style="bold")
        head.append(f"  {done}/{len(todos)} done", style=t.dim)
        items = Text()
        for i, td in enumerate(todos):
            if i:
                items.append("\n")
            if td["status"] == "completed":
                items.append("✔ " + td["content"], style=f"{t.dim} strike")
            elif td["status"] == "in_progress":
                items.append("◼ " + td.get("activeForm", td["content"]), style=f"bold {t.accent}")
            else:
                items.append("□ " + td["content"])
        self._space("block")
        self._out(blocks.gutter(Text("▣", style=f"bold {t.accent}"), head, [items] if todos else []))

    # ------------------------------------------------------------------ messages

    def info(self, msg: str) -> None:
        self._space("inline")
        self._print_status(blocks.gutter(Text(" "), Text(msg, style=theme().dim)))

    def warn(self, msg: str) -> None:
        self._space("block")
        self._print_status(blocks.gutter(Text("⚠", style=f"bold {theme().warn}"), Text(msg, style=theme().warn)))

    def error(self, msg: str) -> None:
        self._space("block")
        self._print_status(blocks.gutter(Text("✗", style=f"bold {theme().err}"), Text(msg, style=theme().err)))

    def _print_status(self, renderable) -> None:
        if self._live is not None:
            self._live.console.print(renderable)
        else:
            self.console.print(renderable)

    def rerender(self) -> bool:
        """Redraw the whole conversation at the current terminal width (after a resize)."""
        if isinstance(self.console, ReplayConsole) and self.console.is_terminal:
            self._stop_live()
            self.console.replay()
            return True
        return False

    def turn_footer(self, status: str, seconds: float, tool_calls: int, files_changed: int, ctx_pct: int,
                    warnings: list[str] | None = None, cost: float = 0.0) -> None:
        t = theme()
        self._stop_live()
        ok = status == "ok"
        mark = Text("✓ " if ok else "• ", style=t.ok if ok else t.warn)
        self._space("block")
        self._last = None
        parts = [f"Worked {seconds:.0f}s" if ok else f"{seconds:.0f}s"]
        if tool_calls:
            parts.append(f"{tool_calls} tool call{'s' if tool_calls != 1 else ''}")
        if files_changed:
            parts.append(f"{files_changed} file{'s' if files_changed != 1 else ''} changed (/undo · /verify)")
        if cost > 0:
            from muyah_code.pricing import money

            parts.append(money(cost))
        parts.append(f"ctx {ctx_pct}%")
        if not ok:
            parts.append(status)
        self.console.print(mark + Text(" · ".join(parts), style=t.dim))
        if warnings:   # weakened tests are always pointed out, whatever the turn says about them
            self.console.print(Text("⚠ Tests weakened: " + "; ".join(warnings), style=t.warn))

    # ------------------------------------------------------------------ interaction

    def _attention(self, message: str) -> None:
        if self.on_attention is not None:
            self.on_attention(message)

    def ask_permission(self, req: PermissionRequest) -> PermissionReply:
        self._attention(f"Waiting for your approval: {req.title}")
        t = theme()
        self._stop_live()
        heading = {"Bash": "Run command", "Edit": "Edit file", "Write": "Write file", "WebFetch": "Fetch web page",
                   "WebSearch": "Search the web"}.get(req.tool_name, req.tool_name)
        body = Text()
        detail = (req.detail or "").rstrip()
        target = req.title[len(req.tool_name) + 1:-1] if req.title.startswith(req.tool_name + "(") else req.title
        if req.tool_name in ("Edit", "Write") and ("@@" in detail or detail.startswith("new file")):
            body.append(target + "\n\n", style="bold")
            body.append_text(render_diff(detail, limit=60))
        else:
            shown = detail or target  # show the command / URL once, not twice
            body.append("\n".join(shown.splitlines()[:30]) + "\n", style=f"bold {t.tool}")
        if req.reason:
            body.append(f"\n{req.reason[0].upper() + req.reason[1:]}", style=t.dim)
        self._space("block")
        self.console.print(Panel(body, title=f"[bold {t.warn}]{heading}[/]", title_align="left",
                                 border_style=t.warn, expand=False, padding=(0, 1)))
        if self.prompter is not None:
            with self._keyboard_to_prompt():
                return self._ask_permission_menu(req)
        return self._ask_permission_plain(req)

    def _ask_permission_menu(self, req: PermissionRequest) -> PermissionReply:
        picked = self.prompter.select("Do you want to proceed?", [
            ("yes", "Yes"),
            ("always", f"Yes, and don't ask again this session for {req.suggested_rule}"),
            ("project", "Yes, always in this project"),
            ("feedback", "No, and tell MUYAH-CODE what to do instead"),
            ("no", "No"),
        ], default="yes")
        if picked == "feedback":
            reply = PermissionReply("no", feedback=self.prompter.ask("What should it do instead? ").strip())
        else:
            reply = PermissionReply(picked or "no")
        self._decision(reply, req)
        return reply

    def _ask_permission_plain(self, req: PermissionRequest) -> PermissionReply:
        t = theme()
        self.console.print(
            f"  [bold]⏎/y[/] yes   [bold]a[/] always this session [{t.dim}]({escape(req.suggested_rule)})[/]   "
            f"[bold]p[/] always in project   [bold]n[/] no   [{t.dim}]or type what to do instead[/]")
        while True:
            try:
                ans = self.console.input(f"[{t.accent}]  ❯ [/]").strip()
            except EOFError:
                return PermissionReply("no")
            low = ans.lower()
            if low in ("", "y", "yes", "1"):
                return PermissionReply("yes")
            if low in ("a", "always", "2"):
                return PermissionReply("always")
            if low in ("p", "project", "3"):
                return PermissionReply("project")
            if low in ("n", "no", "4"):
                return PermissionReply("no")
            if len(ans) > 1:
                return PermissionReply("no", feedback=ans)

    def handoff(self, command: str, targets: list[str]) -> None:
        """A delete the agent wanted to run: shown to you to run yourself (MUYAH-CODE never deletes)."""
        t = theme()
        self._stop_live()
        body = Text()
        body.append(command.strip() + "\n", style=f"bold {t.tool}")
        if targets:
            body.append("\nWould remove:\n", style=t.dim)
            for path in targets[:12]:
                exists = Path(path).exists()
                body.append(f"  {path}", style="bold" if exists else t.dim)
                body.append("\n" if exists else "  (not found)\n", style=t.dim)
            if len(targets) > 12:
                body.append(f"  … and {len(targets) - 12} more\n", style=t.dim)
        copied = _copy_to_clipboard(command.strip())
        body.append("\nMUYAH-CODE never deletes files itself. If you want this done, run it in your own terminal"
                    + (" (it is on your clipboard)." if copied else "."), style=t.dim)
        self._space("block")
        self.console.print(Panel(body, title=f"[bold {t.warn}]Delete: run this yourself[/]", title_align="left",
                                 border_style=t.warn, expand=False, padding=(0, 1)))
        self._last = "block"
        self._keep_working()

    def _decision(self, reply: PermissionReply, req: PermissionRequest) -> None:
        t = theme()
        if reply.choice == "no":
            line = Text("✗ You declined", style=f"bold {t.err}")
            if reply.feedback:
                line.append(f" and said: {reply.feedback}", style=t.err)
        else:
            line = Text("✔ You approved", style=f"bold {t.ok}")
            line.append({"yes": " this once", "always": f" · allowed for this session: {req.suggested_rule}",
                         "project": f" · allowed in this project: {req.suggested_rule}"}.get(reply.choice, ""),
                        style=t.dim)
        self.console.print(line)
        self._last = "block"

    def ask_user(self, question: str, options: list[str]) -> str:
        self._attention(f"Question: {question[:120]}")
        with self._keyboard_to_prompt():
            return self._ask_user(question, options)

    def _ask_user(self, question: str, options: list[str]) -> str:
        t = theme()
        self._stop_live()
        self.console.print(Panel(Text(question), title=f"[bold {t.accent}]Question[/]", border_style=t.accent,
                                 expand=False))
        if self.prompter is not None:
            if not options:
                return self.prompter.ask("Your answer: ").strip()
            picked = self.prompter.select("Your answer", [(o, o) for o in options] + [("__type__", "Type my own answer")])
            if picked == "__type__":
                return self.prompter.ask("Your answer: ").strip()
            return picked or ""
        for i, opt in enumerate(options, 1):
            self.console.print(f"  [bold]{i}[/]. {escape(opt)}")
        try:
            ans = self.console.input(f"[{t.accent}]  ❯ [/]").strip()
        except EOFError:
            return ""
        if ans.isdigit() and 1 <= int(ans) <= len(options):
            return options[int(ans) - 1]
        return ans


BANNER = [
    "███╗   ███╗██╗   ██╗██╗   ██╗ █████╗ ██╗  ██╗",
    "████╗ ████║██║   ██║╚██╗ ██╔╝██╔══██╗██║  ██║",
    "██╔████╔██║██║   ██║ ╚████╔╝ ███████║███████║",
    "██║╚██╔╝██║██║   ██║  ╚██╔╝  ██╔══██║██╔══██║",
    "██║ ╚═╝ ██║╚██████╔╝   ██║   ██║  ██║██║  ██║",
    "╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝",
]


def _hex(c: str) -> tuple[int, int, int] | None:
    if not c.startswith("#") or len(c) != 7:
        return None
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def banner_text() -> Text:
    """The MUYAH wordmark with a left-to-right gradient in the theme colors."""
    t = theme()
    a, b = _hex(t.accent), _hex(t.accent2)
    out = Text()
    width = max(len(row) for row in BANNER)
    for row in BANNER:
        for i, ch in enumerate(row):
            if a and b:
                f = i / max(1, width - 1)
                rgb = tuple(int(a[k] + (b[k] - a[k]) * f) for k in range(3))
                out.append(ch, style=f"bold #{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
            else:
                out.append(ch, style=f"bold {t.accent}")
        out.append("\n")
    out.append("C O D E", style=f"bold {t.accent2}")
    return out
