"""Project instructions: MUYAH.md (plus AGENTS.md / CLAUDE.md for compatibility), with @imports.

Loaded, in order:
    ~/.muyah/MUYAH.md                                   (your global preferences)
    for each directory from the project root down to cwd:
        MUYAH.md, .muyah/MUYAH.md, MUYAH.local.md, AGENTS.md, CLAUDE.md*, .claude/CLAUDE.md*
                                                        (* when compat.claude_md is on)
A line containing `@relative/or/absolute/path.md` pulls that file in (max depth 3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

IMPORT_RE = re.compile(r"(?<![\w`])@((?:~|\.{1,2})?[\w./\\-]+\.\w+)")
MAX_TOTAL_CHARS = 16000
INIT_TEMPLATE = """# MUYAH.md

Project instructions for MUYAH-CODE. Keep this short, specific and true.

## Overview
<what this project is, in 1-3 sentences>

## Commands
- Install: <command>
- Test: <command>
- Lint/format: <command>
- Run: <command>

## Conventions
- <code style, naming, patterns to follow>

## Gotchas
- <things that break, surprising behavior>
"""


@dataclass
class InstructionFile:
    path: Path
    text: str


def _expand_imports(text: str, base: Path, depth: int, seen: set[Path]) -> str:
    if depth >= 3:
        return text
    out_lines = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        out_lines.append(line)
        if in_fence:
            continue
        for m in IMPORT_RE.finditer(line):
            target = Path(m.group(1)).expanduser()
            if not target.is_absolute():
                target = (base / target).resolve()
            if target in seen or not target.is_file():
                continue
            seen.add(target)
            try:
                inner = target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            out_lines.append(f"<!-- imported from {target} -->")
            out_lines.append(_expand_imports(inner, target.parent, depth + 1, seen))
    return "\n".join(out_lines)


def instruction_candidates(project_root: Path, cwd: Path, home: Path, claude_compat: bool) -> list[Path]:
    paths = [home / "MUYAH.md"]
    chain = [project_root]
    try:
        rel = cwd.resolve().relative_to(project_root.resolve())
        cur = project_root
        for part in rel.parts:
            cur = cur / part
            chain.append(cur)
    except ValueError:
        pass
    for d in chain:
        names = ["MUYAH.md", ".muyah/MUYAH.md", "MUYAH.local.md", "AGENTS.md"]
        if claude_compat:
            names += ["CLAUDE.md", ".claude/CLAUDE.md"]
        paths += [d / n for n in names]
    return paths


def load_instructions(project_root: Path, cwd: Path, home: Path, claude_compat: bool = True) -> list[InstructionFile]:
    files: list[InstructionFile] = []
    seen: set[Path] = set()
    total = 0
    for p in instruction_candidates(project_root, cwd, home, claude_compat):
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp in seen or not rp.is_file():
            continue
        seen.add(rp)
        try:
            text = rp.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        text = _expand_imports(text, rp.parent, 0, seen)
        remaining = MAX_TOTAL_CHARS - total
        if remaining <= 200:
            break
        if len(text) > remaining:
            text = text[:remaining] + f"\n... [truncated: {rp.name} is too long; keep instruction files concise]"
        total += len(text)
        files.append(InstructionFile(rp, text))
    return files


class NestedInstructions:
    """AGENTS.md / CLAUDE.md / MUYAH.md in subfolders, loaded the first time the agent works on a file there
    (like Codex's nested AGENTS.md): big repos keep per-package rules without paying for all of them upfront.
    The text goes into that tool result, so the system prompt stays the same (prompt caching)."""

    NAMES = ("MUYAH.md", "AGENTS.md", "CLAUDE.md")
    CAP = 8000

    def __init__(self, project_root: Path, already: list[InstructionFile], claude_compat: bool = True):
        self.root = project_root.resolve()
        self.names = self.NAMES if claude_compat else self.NAMES[:2]
        self.seen: set[Path] = set()
        for f in already:
            try:
                self.seen.add(Path(f.path).resolve())
            except (OSError, TypeError, ValueError):
                continue

    def for_path(self, path: Path) -> str:
        """Instructions not loaded yet from the folders between the project root and this file."""
        try:
            folder = path.resolve().parent
            rel = folder.relative_to(self.root)
        except (OSError, ValueError):
            return ""
        blocks = []
        cur = self.root
        for part in rel.parts:
            cur = cur / part
            for name in self.names:
                p = cur / name
                if p in self.seen or not p.is_file():
                    continue
                self.seen.add(p)
                try:
                    text = p.read_text(encoding="utf-8", errors="replace").strip()
                except OSError:
                    continue
                if text:
                    rel_p = p.relative_to(self.root).as_posix()
                    blocks.append(f'<instructions path="{rel_p}">\n{text[:self.CAP]}\n</instructions>')
        if not blocks:
            return ""
        return ("\n\nInstructions for this part of the project (follow them for files here):\n"
                + "\n".join(blocks))
