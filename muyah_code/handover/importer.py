"""`muyah import` / `/import`: carry work from another AI tool into this session.

Nothing is fetched and nothing is changed in the other tool's files: its conversation is read, turned into
messages (see brief.py), and put at the start of this conversation. What was already imported is remembered,
so running it again brings only what is new.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from muyah_code.handover import brief as B
from muyah_code.handover.sources import Conversation, discover

MODES = ("auto", "full", "brief")


def listing(conversations: list[Conversation]) -> Table:
    table = Table(show_header=True, box=None, padding=(0, 2), header_style="bold")
    for column in ("tool", "when", "turns", "size", "about"):
        table.add_column(column, justify="right" if column in ("turns", "size") else "left")
    for conv in conversations:
        table.add_row(conv.tool, conv.when, str(conv.turns), f"~{conv.tokens // 1000}k" if conv.tokens >= 1000
                      else f"~{conv.tokens}", escape(conv.title[:60]))
    return table


def already_imported(app) -> set[str]:
    return set(getattr(app, "imported_ids", ()) or ())


def budget(app) -> int:
    """How much of the context an import may take: a third of what is usable."""
    usable = getattr(app.context, "usable", 8000)
    return max(1500, int(usable * B.FULL_SHARE))


def pick(conversations: list[Conversation], prompter, console: Console, take_all: bool) -> list[Conversation]:
    if not conversations:
        return []
    if take_all or len(conversations) == 1:
        return conversations if take_all else conversations[:1]
    options = [(f"{c.tool} · {c.when} · {c.title[:44]}",
                f"{c.turns} turns, about {max(1, c.tokens // 1000)}k tokens") for c in conversations[:8]]
    options.append((f"All {len(conversations)} of them", "merged newest first; older ones as short briefs"))
    ask = getattr(prompter, "question", None)
    if ask is None:
        picked = prompter.select("Import which conversation?", [(label, label) for label, _ in options])
    else:
        picked = ask("Which earlier work should be brought in?", options, header="Import")
    if not isinstance(picked, str) or picked == "__other__":
        return []
    if picked.startswith("All "):
        return conversations
    return [c for c, (label, _) in zip(conversations[:8], options, strict=False) if label == picked]


def run(app, console: Console, prompter=None, source: str = "", mode: str = "auto", take_all: bool = False,
        only_new: bool = True) -> int:
    """Find, choose, import. Returns how many conversations were brought in."""
    found = [c for c in discover(Path(app.cwd))
             if not source or c.tool.lower().startswith(source.lower())]
    if only_new:
        seen = already_imported(app)
        found = [c for c in found if f"{c.tool}:{c.id}" not in seen]
    if not found:
        console.print("[dim]No earlier work from other AI tools was found for this folder "
                      "(Claude Code, Codex, OpenCode, Gemini CLI and Aider are read).[/]")
        return 0
    console.print(listing(found))
    chosen = pick(found, prompter, console, take_all)
    if not chosen:
        console.print("[dim]Nothing imported.[/]")
        return 0
    loaded = [(c, c.load()) for c in chosen]
    imported = B.build(loaded, budget(app), mode=mode)
    if not imported.messages:
        console.print("[dim]Nothing to import: those conversations have no messages.[/]")
        return 0
    clashes = B.conflicts(loaded)
    if clashes:
        imported.messages[0]["content"] = imported.messages[0]["content"].replace(
            B.FOOTER, "\n\nSame files touched by more than one tool (check before trusting either):\n"
            + "\n".join(f"- {c}" for c in clashes) + B.FOOTER)
    app.add_imported(imported, [f"{c.tool}:{c.id}" for c in chosen])
    for line in imported.report:
        console.print(f"[dim]  {escape(line)}[/]")
    console.print(f"[green]✓[/] Brought in {len(chosen)} conversation{'s' if len(chosen) != 1 else ''} "
                  f"(~{imported.tokens:,} tokens). Ask your next question and it continues from there.")
    if clashes:
        console.print(f"[dim]  {len(clashes)} file(s) were worked on by more than one tool: it will check the code "
                      "before trusting either.[/]")
    return len(chosen)
