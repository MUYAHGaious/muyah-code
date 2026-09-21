"""SKILL.md discovery and rendering (Claude Code compatible format).

Layout:  <dir>/<skill-name>/SKILL.md   with YAML frontmatter:
    ---
    name: debugging
    description: Use when ... (this is all the model sees until it loads the skill)
    allowed-tools: Read Grep Bash      # optional
    disable-model-invocation: false    # true = only the user can run it via /name
    user-invocable: true               # false = hidden from the / menu
    ---
    <markdown body loaded on demand>

Search order (later wins on name clash): bundled, ~/.claude/skills*, ~/.muyah/skills,
<project>/.claude/skills*, <project>/.muyah/skills      (* when compat.claude_skills is on)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

BUNDLED_DIR = Path(__file__).parent / "bundled"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    text = text.lstrip("﻿")
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines(keepends=True)
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            meta = yaml.safe_load("".join(lines[1:i])) or {}
            if not isinstance(meta, dict):
                raise ValueError("frontmatter must be a mapping")
            return meta, "".join(lines[i + 1:]).lstrip("\n")
    raise ValueError("frontmatter is not closed with ---")


def _as_list(val) -> list[str]:
    if not val:
        return []
    if isinstance(val, str):
        # "Read, Bash(git add:*) Bash(npm test)": commas or spaces separate items, but not inside (...)
        items, cur, depth = [], "", 0
        for ch in val:
            depth += (ch == "(") - (ch == ")")
            if ch in ", " and depth == 0:
                if cur.strip():
                    items.append(cur.strip())
                cur = ""
            else:
                cur += ch
        if cur.strip():
            items.append(cur.strip())
        return items
    return [str(v) for v in val]


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: Path
    source: str
    allowed_tools: list[str] = field(default_factory=list)
    disable_model_invocation: bool = False
    user_invocable: bool = True

    def render(self, args: str = "") -> str:
        body = self.body.replace("$ARGUMENTS", args or "")
        for var in ("${SKILL_DIR}", "${CLAUDE_SKILL_DIR}"):
            body = body.replace(var, str(self.path))
        header = f"# Skill: {self.name}\nBase directory for this skill: {self.path}\n"
        if args and "$ARGUMENTS" not in self.body:
            header += f"ARGUMENTS: {args}\n"
        return header + "\n" + body


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self.errors: list[str] = []

    @classmethod
    def load(cls, project_root: Path, home: Path, claude_compat: bool = True) -> SkillRegistry:
        reg = cls()
        dirs: list[tuple[Path, str]] = [(BUNDLED_DIR, "bundled")]
        if claude_compat:
            dirs.append((Path.home() / ".claude" / "skills", "user"))
        dirs.append((home / "skills", "user"))
        if claude_compat:
            dirs.append((project_root / ".claude" / "skills", "project"))
        dirs.append((project_root / ".muyah" / "skills", "project"))
        for d, source in dirs:
            reg.load_dir(d, source)
        return reg

    def load_dir(self, directory: Path, source: str) -> None:
        if not directory.is_dir():
            return
        for skill_md in sorted(directory.glob("*/SKILL.md")):
            try:
                skill = self._parse(skill_md, source)
            except (OSError, ValueError, yaml.YAMLError) as e:
                self.errors.append(f"{skill_md}: {e}")
                continue
            self._skills[skill.name] = skill

    @staticmethod
    def _parse(path: Path, source: str) -> Skill:
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        name = str(meta.get("name") or path.parent.name).strip()
        desc = str(meta.get("description") or "").strip()
        if not desc:
            first = next((ln.strip("# ").strip() for ln in body.splitlines() if ln.strip()), "")
            desc = first[:200]
        return Skill(
            name=name,
            description=" ".join(desc.split()),
            body=body,
            path=path.parent,
            source=source,
            allowed_tools=_as_list(meta.get("allowed-tools") or meta.get("allowed_tools")),
            disable_model_invocation=bool(meta.get("disable-model-invocation", False)),
            user_invocable=bool(meta.get("user-invocable", True)),
        )

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        name = name.strip().lstrip("/")
        if name in self._skills:
            return self._skills[name]
        # tolerate "plugin:skill" style and case differences
        short = name.split(":")[-1].lower()
        for k, v in self._skills.items():
            if k.lower() == short or k.split(":")[-1].lower() == short:
                return v
        return None

    def names(self) -> list[str]:
        return sorted(self._skills)

    def all(self) -> list[Skill]:
        return [self._skills[k] for k in self.names()]

    def index_text(self, max_chars: int = 3000) -> str:
        """The compact list the model sees in its system prompt."""
        lines = []
        used = 0
        for s in self.all():
            if s.disable_model_invocation:
                continue
            desc = s.description if len(s.description) <= 250 else s.description[:247] + "..."
            line = f"- {s.name}: {desc}"
            if used + len(line) > max_chars:
                lines.append(f"- ... and {len(self._skills) - len(lines)} more (see /skills)")
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)
