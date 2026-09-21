"""Tips under the spinner while it works (like Claude Code's "⎿ Tip: ...").

Built fresh at the start of every turn from what is actually there, so anything new shows up without code
changes: every slash command (from its own help line), every skill you can run, every sub-agent type, every
tool, every connected MCP server, plus a few keyboard tips. Turn them off with "tips": false.
"""

from __future__ import annotations

import random

KEYS = [
    "Shift+Tab switches the mode: manual → edit → plan → auto",
    "Type while it works: your message goes in at the next step (Esc sends it now)",
    "↑ picks a queued message to edit or remove; with nothing queued it brings back earlier prompts",
    "Esc Esc on an empty prompt rewinds to before any earlier turn",
    "@path adds a file (or an image) to your message",
    "#note saves a note to MUYAH.md for future sessions",
    "Drag an image into the terminal to show it to the model",
    "F2 (or Ctrl+Space): talk instead of typing",
    "Ask mode (Shift+Tab): talk an idea through before planning or building",
    "/btw asks a side question without touching the conversation, even while it works",
]
SKIP = {"exit", "quit", "login", "cost", "help"}


def _first_sentence(text: str, limit: int = 110) -> str:
    text = " ".join((text or "").split())
    for end in (". ", "; "):
        if end in text:
            text = text.split(end)[0]
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_tips(app, commands: dict | None = None) -> list[str]:
    tips = list(KEYS)
    for name, c in (commands or {}).items():
        if name not in SKIP:
            tips.append(f"/{name}{(' ' + c.usage) if c.usage else ''}: {_first_sentence(c.help)}")
    for s in getattr(app, "skills", None).all() if getattr(app, "skills", None) else []:
        if getattr(s, "user_invocable", True):
            tips.append(f"Skill /{s.name}: {_first_sentence(s.description)}")
    for d in (getattr(app, "agent_defs", {}) or {}).values():
        tips.append(f"Sub-agent {d.name}: {_first_sentence(d.description)} (just ask for it)")
    registry = getattr(app, "registry", None)
    mcp_servers: dict[str, list[str]] = {}
    for t in registry.tools() if registry is not None else []:
        if t.name.startswith("mcp__"):
            _, server, tool = (t.name.split("__", 2) + ["", ""])[:3]
            mcp_servers.setdefault(server, []).append(tool)
        elif t.name not in ("AskUser",):
            tips.append(f"The agent's {t.name} tool: {_first_sentence(t.description)}")
    for server, tools in mcp_servers.items():
        shown = ", ".join(tools[:4]) + (f" and {len(tools) - 4} more" if len(tools) > 4 else "")
        tips.append(f"MCP server {server} gives the agent {len(tools)} tools: {shown} (/mcp to manage)")
    return tips


class TipRotation:
    """One tip at a time, changing every SECONDS while a turn runs; each turn starts somewhere new."""

    SECONDS = 12.0

    def __init__(self):
        self.tips: list[str] = []
        self.offset = 0

    def reset(self, tips: list[str]) -> None:
        self.tips = tips
        self.offset = random.randrange(len(tips)) if tips else 0

    def at(self, elapsed: float) -> str:
        if not self.tips:
            return ""
        return self.tips[(self.offset + int(elapsed // self.SECONDS)) % len(self.tips)]
