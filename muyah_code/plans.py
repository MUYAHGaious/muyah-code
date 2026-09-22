"""Plan files: in plan mode the plan is written to `.muyah/plans/<date>-<topic>.md` (like Claude Code's plan file).

The file is the only thing plan mode may change. You can open and edit it before approving; ExitPlanMode reads
it fresh. After approval it stays the reference for the build: every turn reminds the agent where it is, so
the plan survives a long session even when the conversation is summarized.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

PLANS_DIR = Path(".muyah") / "plans"
STOP = {"the", "a", "an", "to", "and", "or", "of", "for", "in", "on", "with", "this", "that", "it", "please", "can",
        "you", "me", "my", "i", "we", "our", "make", "let", "lets", "let's"}


def slug(prompt: str, words: int = 6) -> str:
    kept = [w for w in re.findall(r"[a-z0-9]+", prompt.lower()) if w not in STOP][:words]
    return "-".join(kept) or "plan"


def new_plan_path(project_root: Path, prompt: str, today: _dt.date | None = None) -> Path:
    """A fresh plan file path for this request (not created yet); never reuses an existing one."""
    base = project_root / PLANS_DIR / f"{(today or _dt.date.today()).isoformat()}-{slug(prompt)}"
    path, n = base.with_suffix(".md"), 2
    while path.exists():
        path, n = Path(f"{base}-{n}.md"), n + 1
    return path


def planning_note(path: Path, project_root: Path) -> str:
    rel = _rel(path, project_root)
    return (f"<plan_file>Plan mode: write the plan to {rel} (create it with Write, refine it with Edit). It is "
            "the only file you may change in plan mode. When it is complete, call ExitPlanMode.</plan_file>")


def building_note(path: Path, project_root: Path) -> str:
    rel = _rel(path, project_root)
    return (f"<plan_file>You are carrying out the approved plan in {rel}. Follow it; if you are unsure what is next, "
            "Read it again. If the plan turns out to be wrong, say so and propose the change instead of silently "
            "doing something else.</plan_file>")


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)
