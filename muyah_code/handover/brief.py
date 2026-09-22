"""Turning other tools' conversations into something worth carrying: full copy, or a brief built locally.

Rules, in order:
  * The code is the truth. Everything imported is marked second-hand: it is what was SAID elsewhere, and the
    repository (and `git log`) is what actually happened.
  * Never blur what can be quoted. Facts (files touched, commands run, decisions) are kept word for word;
    only the prose around them is dropped.
  * Another tool's own compaction summary is already second-hand: it is kept as a quoted block, labelled, and
    not summarized again unless there is no room, and then it is dropped whole (with a note) rather than
    blurred further.
  * Newest first. When several tools worked in this folder, the most recent conversation keeps its own words;
    older ones become briefs, each with its tool and date, so a disagreement is visible instead of silent.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from muyah_code.handover.sources import Conversation, Turn

FULL_SHARE = 0.33          # a conversation this small (of the usable window) is copied as it is
VERBATIM_TURNS = 6         # how many recent exchanges a brief keeps word for word
MAX_FACTS = 12


@dataclass
class Imported:
    """What to put at the start of the new conversation, and what to tell the user it did."""
    messages: list[dict]
    report: list[str]
    tokens: int


def _fmt_when(ts: float) -> str:
    return time.strftime("%d %b %H:%M", time.localtime(ts)) if ts else "earlier"


def tools_used(turns: list[Turn]) -> tuple[list[str], list[str]]:
    """(files touched, commands run) in order, without repeats: the facts worth keeping."""
    files, commands = [], []
    for turn in turns:
        for call in turn.tools:
            name, _, target = call.partition("(")
            target = target.rstrip(")").strip()
            if not target:
                continue
            bucket = commands if name.lower() in ("bash", "shell", "run", "terminal", "execute") else files
            if target not in bucket:
                bucket.append(target)
    return files, commands


def _asked(turns: list[Turn]) -> list[str]:
    return [" ".join(t.text.split())[:120] for t in turns if t.role == "user" and t.kind == "message" and t.text.strip()]


def brief(conv: Conversation, turns: list[Turn], verbatim: int = VERBATIM_TURNS) -> str:
    """A brief built here, with no model call: what you asked, what was touched, and the last exchanges."""
    asked = _asked(turns)
    files, commands = tools_used(turns)
    summaries = [t for t in turns if t.kind == "summary"]
    lines = [f"## {conv.tool} · {_fmt_when(conv.updated)} · {conv.title}"]
    if asked:
        lines.append("\nWhat the user asked, in order:")
        lines += [f"- {a}" for a in asked[:MAX_FACTS]]
        if len(asked) > MAX_FACTS:
            lines.append(f"- … and {len(asked) - MAX_FACTS} more requests")
    if files:
        lines.append("\nFiles it worked on: " + ", ".join(files[:MAX_FACTS]) +
                     (f" (+{len(files) - MAX_FACTS} more)" if len(files) > MAX_FACTS else ""))
    if commands:
        lines.append("Commands it ran: " + " · ".join(commands[:8]) +
                     (f" (+{len(commands) - 8} more)" if len(commands) > 8 else ""))
    for summary in summaries[-2:]:
        lines.append(f"\n{conv.tool}'s own summary of its earlier work (second-hand, kept as written):\n"
                     f"> " + summary.text.replace("\n", "\n> "))
    tail = [t for t in turns if t.kind == "message"][-verbatim:]
    if tail:
        lines.append("\nThe last exchanges, word for word:")
        for turn in tail:
            who = "User" if turn.role == "user" else conv.tool
            body = " ".join(turn.text.split())[:600] if turn.text else ""
            calls = (" [" + ", ".join(turn.tools[:4]) + "]") if turn.tools else ""
            lines.append(f"- **{who}:** {body}{calls}")
    return "\n".join(lines)


def full(conv: Conversation, turns: list[Turn]) -> str:
    """The conversation as it was, tool calls named, tool output left out (it is stale; files are on disk)."""
    lines = [f"## {conv.tool} · {_fmt_when(conv.updated)} · {conv.title}"]
    for turn in turns:
        who = "User" if turn.role == "user" else conv.tool
        if turn.kind == "summary":
            lines.append(f"\n**{conv.tool}'s own summary (second-hand):**\n> " + turn.text.replace("\n", "\n> "))
            continue
        calls = ("\n  → " + "\n  → ".join(turn.tools)) if turn.tools else ""
        lines.append(f"\n**{who}:** {turn.text}{calls}")
    return "\n".join(lines)


HEADER = ("<imported_context>\nThis is work done in this folder with other AI tools, before this session. It is "
          "second-hand: it says what was SAID elsewhere. The repository and `git log` are the truth; check the "
          "files before relying on anything here, and raise any disagreement with the user instead of choosing "
          "quietly. Tool output was left out on purpose: read the files again if you need them.\n\n")
FOOTER = "\n</imported_context>"
REPLY = ("Understood. I have the earlier work from {tools} as background, and I will check the code itself "
         "before relying on it.")


def build(conversations: list[tuple[Conversation, list[Turn]]], budget_tokens: int,
          mode: str = "auto") -> Imported:
    """mode: auto | full | brief. The newest conversation keeps its own words when there is room for it."""
    report: list[str] = []
    blocks: list[str] = []
    spent = 0
    ordered = sorted(conversations, key=lambda pair: -pair[0].updated)
    for index, (conv, turns) in enumerate(ordered):
        size = sum(t.tokens() for t in turns)
        want_full = mode == "full" or (mode == "auto" and index == 0 and size <= budget_tokens * FULL_SHARE)
        text = full(conv, turns) if want_full else brief(conv, turns)
        cost = len(text) // 4
        if spent + cost > budget_tokens and blocks:
            report.append(f"{conv.tool} ({_fmt_when(conv.updated)}) was left out: no room left in the context")
            continue
        if spent + cost > budget_tokens:                      # even the first one is too big: shorten it
            text = brief(conv, turns, verbatim=2)
            cost = len(text) // 4
        blocks.append(text)
        spent += cost
        report.append(f"{conv.tool} ({_fmt_when(conv.updated)}, {conv.turns} turns): "
                      + ("copied as it is" if want_full else "brief built here, no model call")
                      + f" · ~{cost:,} tokens")
    if not blocks:
        return Imported([], ["nothing to import"], 0)
    names = ", ".join(sorted({conv.tool for conv, _ in ordered}))
    messages = [
        {"role": "user", "content": HEADER + "\n\n".join(blocks) + FOOTER},
        {"role": "assistant", "content": REPLY.format(tools=names)},
    ]
    return Imported(messages, report, spent)


def conflicts(conversations: list[tuple[Conversation, list[Turn]]]) -> list[str]:
    """Where two tools worked on the same file, worth naming so the model does not assume they agree."""
    by_file: dict[str, list[tuple[str, float]]] = {}
    for conv, turns in conversations:
        files, _ = tools_used(turns)
        for name in files:
            key = re.sub(r"^.*[\\/]", "", name)
            by_file.setdefault(key, []).append((conv.tool, conv.updated))
    out = []
    for name, who in by_file.items():
        tools = {tool for tool, _ in who}
        if len(tools) > 1:
            newest = max(who, key=lambda pair: pair[1])
            out.append(f"{name}: touched by {', '.join(sorted(tools))} — most recently {newest[0]} "
                       f"({_fmt_when(newest[1])})")
    return out[:8]
