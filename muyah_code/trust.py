"""Workspace trust: ask once per folder before MUYAH-CODE reads, edits or executes anything there.

A folder's own configuration can run commands (hooks in .muyah/settings.json, MCP servers in .mcp.json),
so an untrusted folder is checked BEFORE any of it is loaded. Trusting a folder also trusts its subfolders.
Trusted folders are remembered in ~/.muyah/trusted.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.rule import Rule
from rich.text import Text

from muyah_code.ui.theme import theme


def _store(home: Path) -> Path:
    return home / "trusted.json"


def trusted_folders(home: Path) -> list[str]:
    try:
        data = json.loads(_store(home).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [str(p) for p in data] if isinstance(data, list) else []


def _norm(path: Path) -> str:
    return str(path.resolve()).rstrip("\\/").lower()


def is_trusted(home: Path, folder: Path) -> bool:
    target = _norm(folder)
    for t in trusted_folders(home):
        base = t.rstrip("\\/").lower()
        if target == base or target.startswith(base + "\\") or target.startswith(base + "/"):
            return True
    return False


def trust(home: Path, folder: Path) -> None:
    folders = trusted_folders(home)
    resolved = str(folder.resolve())
    if resolved not in folders:
        folders.append(resolved)
    home.mkdir(parents=True, exist_ok=True)
    _store(home).write_text(json.dumps(folders, indent=2) + "\n", encoding="utf-8")


def risky_config(folder: Path) -> list[str]:
    """Things in this folder that would run commands once trusted."""
    found = []
    for name in (".muyah/settings.json", ".muyah/settings.local.json", ".claude/settings.json"):
        p = folder / name
        try:
            if p.is_file() and '"hooks"' in p.read_text(encoding="utf-8", errors="replace"):
                found.append(f"{name} defines hooks (shell commands that run automatically)")
        except OSError:
            continue
    for name in (".mcp.json", ".muyah/mcp.json"):
        if (folder / name).is_file():
            found.append(f"{name} starts MCP servers (programs that run on your machine)")
    return found


def ensure_trusted(folder: Path, home: Path, console: Console, prompter) -> bool:
    """Returns True when it is OK to continue. Asks (once) if the folder is not trusted yet."""
    if is_trusted(home, folder):
        return True
    folder = folder.resolve()  # full path (no Windows 8.3 short names) in what the user reads
    t = theme()
    console.print(Rule(style=t.accent))
    console.print(Text("Accessing workspace:", style=f"bold {t.accent}"))
    console.print()
    console.print(Text(str(folder), style="bold"))
    console.print()
    console.print("Quick safety check: Is this a project you created or one you trust? (Like your own code, a "
                  "well-known open source project, or work from your team.) If not, take a moment to review "
                  "what's in this folder first.")
    console.print()
    console.print("MUYAH-CODE will be able to read, edit, and execute files here.")
    for item in risky_config(folder):
        console.print(f"[{t.warn}]⚠ This folder's {escape(item)}.[/]")
    console.print()
    picked = prompter.select("", [("no", "No, exit"), ("yes", "Yes, I trust this folder")], default="yes")
    if picked != "yes":
        return False
    trust(home, folder)
    return True
