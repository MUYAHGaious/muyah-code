"""Resuming sessions: an arrow-key picker of recent sessions, and replaying a transcript on screen."""

from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console
from rich.rule import Rule
from rich.text import Text

from muyah_code.agent.context import is_real_user_message
from muyah_code.session import Session
from muyah_code.ui.theme import theme


def ago(ts: float, now: float | None = None) -> str:
    seconds = max(0, int((now or time.time()) - ts))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    if seconds < 30 * 86400:
        days = seconds // 86400
        return "yesterday" if days == 1 else f"{days} days ago"
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def pick_session(directory: Path, prompter, limit: int = 15) -> str | None:
    """Arrow-key list of recent sessions; returns the chosen session id (None if cancelled or none)."""
    infos = [s for s in Session.list_sessions(directory, limit=limit) if s.messages]
    if not infos:
        return None
    options = [(s.id, f"{s.title[:60]}  ·  {ago(s.modified)}  ·  {s.messages} messages") for s in infos]
    return prompter.select("Resume a conversation", options, default=infos[0].id)


def _tool_line(call: dict) -> Text:
    fn = call.get("function", {})
    name = fn.get("name", "?")
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}
    subject = ""
    for key in ("file_path", "path", "command", "pattern", "url", "query", "description"):
        if isinstance(args, dict) and args.get(key):
            subject = str(args[key]).replace("\n", " ")
            break
    if len(subject) > 80:
        subject = subject[:77] + "..."
    t = theme()
    line = Text("● ", style=t.tool)
    line.append(name, style="bold")
    line.append(f"({subject})", style=t.dim)
    return line


def print_history(console: Console, messages: list[dict]) -> None:
    """Show a resumed conversation the way it looked: your prompts, tool calls, and answers."""
    from muyah_code.ui.terminal import PrefixedMarkdown

    t = theme()
    shown = False
    for m in messages:
        role = m.get("role")
        content = m.get("content") if isinstance(m.get("content"), str) else ""
        if is_real_user_message(m):
            if content.startswith("[Summary of the earlier conversation"):
                console.print(Text("⋯ earlier messages were summarized to save context", style=t.dim))
                continue
            prompt = content.split("\n\n<lessons>")[0].split("\n\n<file path=")[0]
            console.print(Rule(style="bright_black"))
            console.print(Text("❯ ", style=t.accent) + Text(prompt.strip()))
            console.print()
            shown = True
        elif role == "assistant":
            for call in m.get("tool_calls") or []:
                console.print(_tool_line(call))
            if content.strip() and not content.startswith("Understood. I have the summary"):
                console.print(PrefixedMarkdown(content))
                console.print()
    if shown:
        console.print(Text("─── resumed ───", style=t.dim))
        console.print()
