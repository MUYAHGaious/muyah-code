"""Permission modes and allow/ask/deny rules (Claude Code syntax).

Rules:
    "Read"                    every Read call
    "Bash(git status)"        exact command
    "Bash(npm run test:*)"    command prefix (':*' suffix)
    "Bash(git *)"             glob over the command
    "Edit(src/**)"            glob over the path, relative to the project root (or absolute)
    "WebFetch(domain:docs.python.org)"
    "mcp__github"             every tool of an MCP server

Evaluation order: deny > ask > allow > mode defaults.
Modes: default | acceptEdits | plan | bypassPermissions
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from muyah_code.tools.base import EXEC, META, NETWORK, READ, WRITE, Tool, ToolContext
from muyah_code.tools.search import matches_glob

MODES = ("default", "acceptEdits", "plan", "bypassPermissions")
MODE_ALIASES = {
    "default": "default", "ask": "default", "normal": "default", "manual": "default",
    "acceptedits": "acceptEdits", "accept-edits": "acceptEdits", "auto-edit": "acceptEdits", "edits": "acceptEdits",
    "plan": "plan", "readonly": "plan", "read-only": "plan",
    "bypasspermissions": "bypassPermissions", "bypass": "bypassPermissions", "yolo": "bypassPermissions",
}
PATH_TOOLS = {"Read", "Write", "Edit", "LS", "Glob", "Grep"}
# CLIs whose second word is a subcommand: "always allow" is scoped to e.g. "npm run" instead of all of npm.
SUBCOMMAND_CLIS = {"git", "npm", "pnpm", "yarn", "bun", "npx", "cargo", "go", "docker", "kubectl", "pip", "pip3",
                   "uv", "poetry", "dotnet", "gradle", "mvn", "deno", "make", "gh", "terraform", "helm", "conda"}
RULE_RE = re.compile(r"^\s*([\w.\-]+)\s*(?:\((.*)\))?\s*$", re.S)


def normalize_mode(mode: str) -> str:
    m = MODE_ALIASES.get((mode or "default").strip().lower().replace("_", ""))
    if m is None:
        m = MODE_ALIASES.get((mode or "").strip().lower())
    if m is None:
        raise ValueError(f"Unknown permission mode '{mode}'. Use one of: {', '.join(MODES)}")
    return m


@dataclass(frozen=True)
class Rule:
    tool: str
    pattern: str | None

    @classmethod
    def parse(cls, text: str) -> Rule:
        m = RULE_RE.match(text)
        if not m:
            raise ValueError(f"Invalid permission rule: {text!r}")
        pattern = m.group(2)
        return cls(m.group(1), pattern.strip() if pattern is not None and pattern.strip() not in ("", "*") else None)

    def __str__(self) -> str:
        return self.tool if self.pattern is None else f"{self.tool}({self.pattern})"

    def matches(self, tool_name: str, subject: str, project_root: Path) -> bool:
        if self.tool.startswith("mcp__") and "__" not in self.tool[5:]:
            if not tool_name.startswith(self.tool + "__"):
                return False
        elif self.tool != tool_name:
            return False
        if self.pattern is None:
            return True
        pat = self.pattern
        if tool_name == "Bash":
            cmd = " ".join(subject.split())
            if pat.endswith(":*"):
                prefix = pat[:-2].strip()
                return cmd == prefix or cmd.startswith(prefix + " ")
            return fnmatch.fnmatchcase(cmd, pat)
        if tool_name in ("WebFetch",) and pat.startswith("domain:"):
            host = (urlparse(subject).hostname or "").lower()
            dom = pat[len("domain:"):].lower()
            return host == dom or host.endswith("." + dom)
        if tool_name in PATH_TOOLS:
            path = Path(subject)
            candidates = [path.as_posix()]
            try:
                candidates.append(path.resolve().relative_to(project_root.resolve()).as_posix())
            except (ValueError, OSError):
                pass
            p = pat.replace("\\", "/")
            if p.startswith("./"):
                p = p[2:]
            return any(matches_glob(c, p) for c in candidates)
        return fnmatch.fnmatchcase(subject, pat)


@dataclass
class Decision:
    action: str  # allow | ask | deny
    reason: str = ""
    rule: str | None = None


def suggest_rule(tool: Tool, args: dict, ctx: ToolContext) -> str:
    """The rule offered for 'always allow' in the permission prompt."""
    if tool.name == "Bash":
        cmd = args.get("command", "").strip().split()
        if not cmd:
            return "Bash"
        prog = Path(cmd[0]).name.lower().removesuffix(".exe")
        two = prog in SUBCOMMAND_CLIS and len(cmd) > 1 and re.fullmatch(r"[a-z][\w:-]*", cmd[1])
        return f"Bash({' '.join(cmd[:2] if two else cmd[:1])}:*)"
    if tool.name in ("Write", "Edit"):
        path = ctx.resolve(args.get("file_path", ""))
        try:
            rel = path.relative_to(ctx.project_root).parent.as_posix()
        except ValueError:
            return f"{tool.name}({path.as_posix()})"
        return f"{tool.name}({rel}/**)" if rel != "." else f"{tool.name}(**)"
    if tool.name == "WebFetch":
        host = urlparse(args.get("url", "")).hostname
        return f"WebFetch(domain:{host})" if host else "WebFetch"
    return tool.name


class PermissionManager:
    def __init__(self, mode: str = "default", allow=(), ask=(), deny=(), project_root: Path | None = None):
        self.mode = normalize_mode(mode)
        self.project_root = (project_root or Path.cwd()).resolve()
        self.allow: list[Rule] = []
        self.ask: list[Rule] = []
        self.deny: list[Rule] = []
        self.errors: list[str] = []
        for kind, rules in (("allow", allow), ("ask", ask), ("deny", deny)):
            for r in rules:
                self.add(kind, r)

    @classmethod
    def from_config(cls, cfg, mode_override: str | None = None) -> PermissionManager:
        perms = cfg.get("permissions") or {}
        return cls(
            mode=mode_override or perms.get("mode", "default"),
            allow=perms.get("allow", []), ask=perms.get("ask", []), deny=perms.get("deny", []),
            project_root=cfg.project_root,
        )

    def add(self, kind: str, rule: str) -> None:
        try:
            parsed = Rule.parse(rule)
        except ValueError as e:
            self.errors.append(str(e))
            return
        bucket = getattr(self, kind)
        if parsed not in bucket:
            bucket.append(parsed)

    def set_mode(self, mode: str) -> str:
        self.mode = normalize_mode(mode)
        return self.mode

    def cycle_mode(self) -> str:
        order = ["default", "acceptEdits", "plan"]
        self.mode = order[(order.index(self.mode) + 1) % len(order)] if self.mode in order else "default"
        return self.mode

    def _find(self, rules: list[Rule], name: str, subject: str) -> Rule | None:
        return next((r for r in rules if r.matches(name, subject, self.project_root)), None)

    def check(self, tool: Tool, args: dict, ctx: ToolContext) -> Decision:
        name = tool.name
        subject = tool.permission_subject(args, ctx) or ""

        rule = self._find(self.deny, name, subject)
        if rule:
            return Decision("deny", f"denied by rule {rule}", str(rule))

        read_only = tool.is_read_only(args)
        if self.mode == "plan" and not read_only and tool.kind in (WRITE, EXEC):
            return Decision("deny", "plan mode is read-only: explore and present a plan; do not modify anything")

        rule = self._find(self.ask, name, subject)
        if rule:
            return Decision("ask", f"rule {rule} requires confirmation", str(rule))
        rule = self._find(self.allow, name, subject)
        if rule:
            return Decision("allow", f"allowed by rule {rule}", str(rule))

        if self.mode == "bypassPermissions":
            return Decision("allow", "bypassPermissions mode")
        if tool.kind in (READ, META) or read_only:
            return Decision("allow", "read-only")
        if tool.kind == NETWORK and self.mode == "plan":
            return Decision("allow", "research is allowed in plan mode")
        if tool.kind == WRITE and self.mode == "acceptEdits":
            target = Path(subject) if subject else None
            if target is not None and _is_within(target, self.project_root):
                return Decision("allow", "acceptEdits mode")
            return Decision("ask", "file is outside the project")
        return Decision("ask", f"{tool.kind} action")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except (ValueError, OSError):
        return False


__all__ = ["MODES", "Decision", "PermissionManager", "Rule", "normalize_mode", "suggest_rule"]
