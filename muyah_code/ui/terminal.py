"""Interactive terminal rendering with Rich.

Improvements over a plain streaming CLI:
  * live Markdown: the answer renders as formatted markdown while it streams; only the last few lines
    stay "in motion" so long answers never flicker or duplicate
  * live stats while generating: elapsed time, tokens, tokens/s (vital on slow local models)
  * tool calls as bullets with results, Bash output tails, line-numbered colored diffs
  * themes (muyah, ocean, forest, mono, light)
"""

from __future__ import annotations

import io
import re
import time

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from muyah_code.tools.base import ToolResult
from muyah_code.ui.base import UI, PermissionReply, PermissionRequest
from muyah_code.ui.theme import theme

MAX_DIFF_LINES = 40
LIVE_TAIL_LINES = 8
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
LONG_TOOLS = ("Bash", "Agent", "WebFetch", "WebSearch", "BashOutput")


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
        width = max(20, self.console.width - 2)
        buf = io.StringIO()
        c = Console(file=buf, force_terminal=True, width=width, color_system=self.console.color_system or "standard",
                    highlight=False)
        c.print(Markdown(text, code_theme=theme().code_theme))
        lines = buf.getvalue().splitlines()
        while lines and not Text.from_ansi(lines[-1]).plain.strip():
            lines.pop()
        return lines

    def _prefix(self, i: int) -> Text:
        return Text("● ", style=theme().accent) if i == 0 else Text("  ")

    def update(self, chunk: str, final: bool = False) -> None:
        self.text += chunk
        now = time.monotonic()
        if not final and now - self._last_render < 0.08:
            return
        self._last_render = now
        lines = self._lines(self.text)
        settled = len(lines) if final else max(0, len(lines) - LIVE_TAIL_LINES)
        for i in range(self.printed, settled):
            self.live.console.print(self._prefix(i) + Text.from_ansi(lines[i]))
        self.printed = max(self.printed, settled)
        tail = Text()
        for i in range(self.printed, len(lines)):
            tail.append_text(self._prefix(i) + Text.from_ansi(lines[i]))
            tail.append("\n")
        self.live.update(Group(tail, self.stats()) if not final else Text(""))


class TerminalUI(UI):
    headless = False

    def __init__(self, console: Console | None = None, show_reasoning: bool = False):
        self.console = console or Console(highlight=False)
        self.show_reasoning = show_reasoning
        self._live: Live | None = None
        self._stream: _MarkdownStream | None = None
        self._phase = ""
        self._t0 = 0.0
        self._first_token = 0.0
        self._chars = 0
        self._reasoning_chars = 0
        self.context_pct = None  # set by the REPL so the stats line can show context usage
        self.prompter = None     # arrow-key menus; set by the REPL (ui.select.Prompter)

    # ------------------------------------------------------------------ live status

    def _stats_line(self):
        t = theme()
        elapsed = time.monotonic() - self._t0
        parts = [f"{elapsed:.0f}s"]
        if self._phase == "thinking":
            label = "Thinking"
            if self._reasoning_chars:
                parts.append(f"{int(self._reasoning_chars / 3.5):,} reasoning tokens")
        elif self._phase == "writing":
            label = "Writing"
            toks = int(self._chars / 3.5)
            gen_time = max(0.001, time.monotonic() - self._first_token)
            parts.append(f"{toks:,} tokens")
            if gen_time > 1:
                parts.append(f"{toks / gen_time:.1f} tok/s")
        else:
            label = self._phase or "Working"
        if self.context_pct is not None:
            parts.append(f"ctx {self.context_pct}%")
        spinner = Spinner("dots", text=Text.assemble((f"{label}… ", t.accent), (" · ".join(parts), t.dim),
                                                     ("  (ctrl+c to interrupt)", t.dim)), style=t.accent)
        return spinner

    def _start_live(self, phase: str) -> None:
        self._stop_live()
        self._phase = phase
        self._t0 = time.monotonic()
        self._live = Live(self._StatsRenderable(self), console=self.console, refresh_per_second=8, transient=True)
        self._live.start()

    class _StatsRenderable:
        def __init__(self, ui):
            self.ui = ui

        def __rich__(self):
            return self.ui._stats_line()

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    # ------------------------------------------------------------------ assistant text

    def assistant_start(self) -> None:
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
            if self._live is None:
                self._start_live("writing")
            self._phase = "writing"
            self._first_token = time.monotonic()
            self._stream = _MarkdownStream(self.console, self._live, self._stats_line)
            chunk = chunk.lstrip("\n")
        self._chars += len(chunk)
        self._stream.update(chunk)

    def assistant_end(self) -> None:
        if self._stream is not None:
            self._stream.update("", final=True)
            self._stream = None
            self._stop_live()
            self.console.print()
        else:
            self._stop_live()

    # ------------------------------------------------------------------ tools

    def tool_start(self, title: str) -> None:
        t = theme()
        self._stop_live()
        name, _, rest = title.partition("(")
        line = Text("● ", style=t.tool)
        line.append(name, style="bold")
        if rest:
            line.append("(" + rest, style=t.dim)
        self.console.print(line)
        base = name.split("] ")[-1]
        if base in LONG_TOOLS or base.startswith("mcp__"):
            self._start_live("Running " + base)

    def tool_end(self, title: str, result: ToolResult) -> None:
        t = theme()
        elapsed = time.monotonic() - self._t0 if self._live is not None else 0
        self._stop_live()
        summary = result.summary or ("error" if result.is_error else "done")
        if elapsed >= 2:
            summary += f" · {elapsed:.0f}s"
        self.console.print(Text("  ⎿  ", style=t.dim) + Text(summary, style=t.err if result.is_error else t.dim))
        base = title.split("] ")[-1].split("(")[0]
        if result.is_error and result.content:
            lines = result.content.strip().splitlines()
            shown = lines[-6:] if base == "Bash" else lines[:6]
            self.console.print(Text("\n".join("     " + ln[:200] for ln in shown), style=t.err))
        elif result.display:
            self.console.print(render_diff(result.display), end="")
        elif base == "Bash" and result.content and result.content != "(no output)":
            lines = result.content.rstrip().splitlines()
            tail = lines[-3:]
            if len(lines) > 3:
                self.console.print(Text(f"     … {len(lines) - 3} more lines", style=t.dim))
            self.console.print(Text("\n".join("     " + ln[:160] for ln in tail), style=t.dim))

    def on_todos(self, todos: list[dict]) -> None:
        t = theme()
        self._stop_live()
        for td in todos:
            if td["status"] == "completed":
                self.console.print(Text("     ✔ " + td["content"], style=f"{t.dim} strike"))
            elif td["status"] == "in_progress":
                self.console.print(Text("     ▶ " + td.get("activeForm", td["content"]), style=f"bold {t.accent}"))
            else:
                self.console.print(Text("     ○ " + td["content"]))

    # ------------------------------------------------------------------ messages

    def info(self, msg: str) -> None:
        self._print_status(Text(msg, style=theme().dim))

    def warn(self, msg: str) -> None:
        self._print_status(Text("⚠ " + msg, style=theme().warn))

    def error(self, msg: str) -> None:
        self._print_status(Text("✗ " + msg, style=f"bold {theme().err}"))

    def _print_status(self, text: Text) -> None:
        if self._live is not None:
            self._live.console.print(text)
        else:
            self.console.print(text)

    def turn_footer(self, status: str, seconds: float, tool_calls: int, files_changed: int, ctx_pct: int) -> None:
        t = theme()
        ok = status == "ok"
        mark = Text("✓ " if ok else "• ", style=t.ok if ok else t.warn)
        parts = [f"{seconds:.0f}s"]
        if tool_calls:
            parts.append(f"{tool_calls} tool call{'s' if tool_calls != 1 else ''}")
        if files_changed:
            parts.append(f"{files_changed} file{'s' if files_changed != 1 else ''} changed (/undo)")
        parts.append(f"ctx {ctx_pct}%")
        if not ok:
            parts.append(status)
        self.console.print(mark + Text(" · ".join(parts), style=t.dim))

    # ------------------------------------------------------------------ interaction

    def ask_permission(self, req: PermissionRequest) -> PermissionReply:
        t = theme()
        self._stop_live()
        body = Text()
        body.append(req.title + "\n", style="bold")
        if req.reason:
            body.append(f"{req.reason}\n", style=t.dim)
        detail = req.detail or ""
        if detail:
            body.append("\n")
            if req.tool_name in ("Edit", "Write") and ("@@" in detail or detail.startswith("new file")):
                body.append_text(render_diff(detail, limit=60))
            else:
                body.append("\n".join(detail.splitlines()[:30]) + "\n", style=t.tool)
        self.console.print(Panel(body, title=f"[bold {t.accent}]Allow this {req.kind} action?[/]",
                                 border_style=t.accent, expand=False, padding=(0, 1)))
        if self.prompter is not None:
            picked = self.prompter.select("Allow?", [
                ("yes", "Yes"),
                ("always", f"Yes, and don't ask again this session for {req.suggested_rule}"),
                ("project", "Yes, always in this project"),
                ("feedback", "No, and tell MUYAH-CODE what to do instead"),
                ("no", "No"),
            ], default="yes")
            if picked == "feedback":
                return PermissionReply("no", feedback=self.prompter.ask("What should it do instead? ").strip())
            return PermissionReply(picked or "no")
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

    def ask_user(self, question: str, options: list[str]) -> str:
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
