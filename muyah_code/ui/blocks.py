"""How each kind of event looks in the transcript.

Rules (after studying Codex, Gemini CLI, OpenCode and Claude Code):
  * every block has a 2-column gutter: a status glyph, then the content. Wrapped lines hang under the
    content, never under the glyph, and child lines start with "⎿" in the same gutter.
  * each kind of action has its own glyph and verb: `$` Ran for commands, → Read, ✱ Searched,
    ± Edited, ◈ Fetched, ⇄ Called (MCP), ◇ Skill, ◆ Agent, ▣ Plan
    (glyphs chosen from blocks every common terminal font has). The glyph color is the outcome:
    green = worked, red = failed.
  * quick lookups (read, search, list, skill) are one line each and stack without blank lines;
    bigger blocks (commands, edits, agents, answers) get one blank line around them.
  * headers never wrap: long paths are shortened in the middle; output previews show the first and
    last lines with "… +N lines".

Everything here returns Rich renderables laid out at print time, so a redraw after a terminal resize
re-wraps them for the new width.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rich.console import Group
from rich.table import Table
from rich.text import Text

from muyah_code.ui.theme import theme

PREVIEW_HEAD = 2
PREVIEW_TAIL = 3

# name -> (kind, glyph, verb while running, verb when done)
KINDS: dict[str, tuple[str, str, str, str]] = {
    "Bash": ("shell", "$", "Running", "Ran"),
    "BashOutput": ("shell", "$", "Reading output", "Output"),
    "KillShell": ("shell", "$", "Stopping", "Stopped"),
    "Read": ("explore", "→", "Reading", "Read"),
    "LS": ("explore", "→", "Listing", "Listed"),
    "Glob": ("explore", "✱", "Finding", "Found"),
    "Grep": ("explore", "✱", "Searching", "Searched"),
    "Edit": ("edit", "±", "Editing", "Edited"),
    "MultiEdit": ("edit", "±", "Editing", "Edited"),
    "Write": ("edit", "±", "Writing", "Wrote"),
    "NotebookEdit": ("edit", "±", "Editing", "Edited"),
    "WebFetch": ("web", "◈", "Fetching", "Fetched"),
    "WebSearch": ("explore", "◈", "Searching the web for", "Searched the web for"),
    "Skill": ("explore", "◇", "Loading skill", "Using skill"),
    "TodoWrite": ("plan", "▣", "Updating plan", "Updated plan"),
    "Agent": ("agent", "◆", "Agent", "Agent"),
    "AskUser": ("ask", "?", "Asking", "Asked"),
}
INLINE_KINDS = ("explore",)   # one-liners that stack without blank lines


@dataclass
class ToolTitle:
    label: str | None   # sub-agent label for a sub-agent's own steps ("[explore] Read(x)")
    name: str           # tool name, e.g. "Bash", "mcp__tracker__lookup_issue", "Agent"
    variant: str        # "explore" in "Agent[explore](...)"
    subject: str        # what is inside the parentheses

    @property
    def kind(self) -> str:
        if self.name.startswith("mcp__"):
            return "mcp"
        return KINDS.get(self.name, ("tool", "●", "Running", "Ran"))[0]


_TITLE = re.compile(r"^(?:\[(?P<label>[^\]]+)\] )?(?P<name>[\w.-]+)(?:\[(?P<variant>[^\]]*)\])?(?:\((?P<subject>.*)\))?$",
                    re.DOTALL)


def parse_title(title: str) -> ToolTitle:
    m = _TITLE.match(title.strip())
    if not m:
        return ToolTitle(None, title, "", "")
    return ToolTitle(m["label"], m["name"], m["variant"] or "", m["subject"] or "")


def short_path(text: str, cwd: Path | None, width: int) -> str:
    """A path relative to the working folder, shortened in the middle if it still does not fit."""
    if cwd is not None and text:
        try:
            p = Path(text)
            if p.is_absolute():
                text = p.resolve().relative_to(cwd.resolve()).as_posix()
        except (ValueError, OSError):
            pass
    if len(text) <= width or width < 12:
        return text
    parts = re.split(r"([\\/])", text)
    if len(parts) >= 5:
        head, tail = parts[0] + parts[1], parts[-1]
        if len(head) + len(tail) + 2 <= width:
            return f"{head}…{parts[-2]}{tail}"
    keep = width - 1
    return text[: keep // 2] + "…" + text[-(keep - keep // 2):]


def gutter(glyph: Text, body, children: list | None = None):
    """glyph | body, then children under a ⎿ in the same gutter, all with hanging indents."""
    grid = Table.grid(padding=0)
    grid.add_column(width=2, no_wrap=True)
    grid.add_column(ratio=1, overflow="fold")
    grid.add_row(glyph, body)
    for i, child in enumerate(children or []):
        grid.add_row(Text("⎿ " if i == 0 else "  ", style=theme().dim), child)
    return grid


def preview(text: str, head: int = PREVIEW_HEAD, tail: int = PREVIEW_TAIL, width: int = 200) -> Text:
    """First and last lines of an output, with how many were left out."""
    t = theme()
    lines = [ln.rstrip() for ln in (text or "").replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return Text("(no output)", style=t.dim)
    if len(lines) > head + tail + 1:
        shown = lines[:head] + [f"… +{len(lines) - head - tail} lines"] + lines[-tail:]
    else:
        shown = lines
    out = Text(style=t.dim)
    for i, ln in enumerate(shown):
        out.append(ln[:width] + ("…" if len(ln) > width else ""))
        if i < len(shown) - 1:
            out.append("\n")
    return out


def _mcp_parts(name: str) -> tuple[str, str]:
    rest = name[5:]
    server, _, tool = rest.partition("__")
    return server, tool


def _args_text(subject: str) -> str:
    try:
        data = json.loads(subject)
    except (json.JSONDecodeError, TypeError):
        return subject
    if isinstance(data, dict):
        return ", ".join(f"{k}: {v}" for k, v in data.items())
    return subject


def header(tt: ToolTitle, running: bool, cwd: Path | None, width: int) -> Text:
    """The one-line header of a tool block: verb + target, never wrapped."""
    t = theme()
    room = max(20, width - 4)
    if tt.kind == "mcp":
        server, tool = _mcp_parts(tt.name)
        head = Text("Calling " if running else "Called ", style="bold")
        head.append(f"{server} · {tool}", style=t.accent)
        args = _args_text(tt.subject)
        if args:
            head.append(f"  {args}", style=t.dim)
    else:
        _, _, verb_running, verb_done = KINDS.get(tt.name, ("tool", "●", "Running", tt.name))
        verb = verb_running if running else verb_done
        head = Text(verb, style="bold")
        if tt.name == "Bash":
            head.append(" " + " ".join(tt.subject.split()), style=t.tool)
        elif tt.name == "Agent":
            head.append(f" {tt.variant or 'general'}", style=t.accent)
            if tt.subject:
                head.append(f" · {tt.subject}")
        elif tt.name in ("WebSearch", "Grep"):
            head.append(f' "{tt.subject}"', style=t.accent)
        elif tt.name == "WebFetch" or tt.name == "Skill":
            head.append(" " + tt.subject, style=t.accent)
        elif tt.name == "TodoWrite":
            pass
        elif tt.name in KINDS:
            head.append(" " + short_path(tt.subject, cwd, room - len(verb) - 1), style=t.accent)
        else:
            head = Text(tt.name, style="bold")
            if tt.subject:
                head.append(f"({tt.subject})", style=t.dim)
    head.no_wrap = True
    head.overflow = "ellipsis"
    head.truncate(room, overflow="ellipsis")
    return head


def glyph_for(tt: ToolTitle, ok: bool | None) -> Text:
    """ok=None: still running (dim), True: worked (green), False: failed (red)."""
    t = theme()
    g = "⇄" if tt.kind == "mcp" else KINDS.get(tt.name, ("tool", "●", "", ""))[1]
    style = t.dim if ok is None else (t.ok if ok else t.err)
    return Text(g, style=f"bold {style}")


def diff_counts(display: str | None) -> tuple[int, int]:
    lines = (display or "").splitlines()
    added = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
    return added, removed


def user_prompt(text: str):
    """Your message: highlighted as a block (no separator line)."""
    t = theme()
    body = Text(text.rstrip())
    grid = Table.grid(padding=0, expand=True)
    grid.add_column(width=2, no_wrap=True)
    grid.add_column(ratio=1, overflow="fold")
    grid.add_row(Text("› ", style=f"bold {t.accent}"), body, style=f"on {t.user_bg}")
    return grid


def blank() -> Group:
    return Group(Text(""))
