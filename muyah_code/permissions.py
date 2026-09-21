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

from muyah_code.risk import is_delete_command, risky_command
from muyah_code.tools.base import EXEC, META, NETWORK, READ, WRITE, Tool, ToolContext
from muyah_code.tools.search import matches_glob

MODES = ("default", "acceptEdits", "plan", "auto", "bypassPermissions")
# Shift+Tab order. bypassPermissions (no checks at all) is deliberately not in it: only --mode sets it.
CYCLE = ("plan", "acceptEdits", "default", "auto")
MODE_NAMES = {"default": "manual", "acceptEdits": "edit", "plan": "plan", "auto": "auto",
              "bypassPermissions": "bypass"}
MODE_ALIASES = {
    "default": "default", "ask": "default", "normal": "default", "manual": "default",
    "acceptedits": "acceptEdits", "accept-edits": "acceptEdits", "auto-edit": "acceptEdits", "edits": "acceptEdits",
    "plan": "plan", "readonly": "plan", "read-only": "plan", "edit": "acceptEdits",
    "auto": "auto", "automatic": "auto",
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
            # match both the raw command and its normalized form ("C:\...\python.exe" -m x -> python -m x)
            for cmd in {" ".join(subject.split()), normalize_command(subject)}:
                if pat.endswith(":*"):
                    prefix = pat[:-2].strip()
                    if cmd == prefix or cmd.startswith(prefix + " "):
                        return True
                elif fnmatch.fnmatchcase(cmd, pat):
                    return True
            return False
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


def normalize_command(command: str) -> str:
    """'"C:\\Program Files\\Python\\python.exe" -m pytest' -> 'python -m pytest' (program name only)."""
    command = command.strip()
    if not command:
        return ""
    if command[0] in "\"'":
        end = command.find(command[0], 1)
        first, rest = (command[1:end], command[end + 1:]) if end > 0 else (command[1:], "")
    else:
        first, _, rest = command.partition(" ")
    prog = re.split(r"[\\/]", first)[-1].lower()
    prog = prog[:-4] if prog.endswith((".exe", ".cmd", ".bat")) else prog
    return " ".join([prog, *rest.split()])


KIND_REASON = {"write": "edits a file", "exec": "runs a command", "network": "uses the network"}


def suggest_rule(tool: Tool, args: dict, ctx: ToolContext) -> str:
    """The rule offered for 'always allow' in the permission prompt."""
    if tool.name == "Bash":
        cmd = normalize_command(args.get("command", "")).split()
        if not cmd:
            return "Bash"
        two = cmd[0] in SUBCOMMAND_CLIS and len(cmd) > 1 and re.fullmatch(r"[a-z][\w:-]*", cmd[1])
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
        """Shift+Tab: plan -> edit -> manual -> auto -> plan."""
        self.mode = CYCLE[(CYCLE.index(self.mode) + 1) % len(CYCLE)] if self.mode in CYCLE else CYCLE[0]
        return self.mode

    def _auto(self, tool: Tool, args: dict, subject: str, read_only: bool) -> Decision:
        return _auto_decision(tool, args, subject, read_only, self.project_root)

    def _find(self, rules: list[Rule], name: str, subject: str) -> Rule | None:
        return next((r for r in rules if r.matches(name, subject, self.project_root)), None)

    def check(self, tool: Tool, args: dict, ctx: ToolContext) -> Decision:
        name = tool.name
        subject = tool.permission_subject(args, ctx) or ""

        rule = self._find(self.deny, name, subject)
        if rule:
            return Decision("deny", f"denied by rule {rule}", str(rule))
        if name == "Bash" and is_delete_command(args.get("command", "")):
            return Decision("handoff", "deleting files is always left to you")

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
        if self.mode == "auto":
            return self._auto(tool, args, subject, read_only)
        if tool.kind in (READ, META) or read_only:
            return Decision("allow", "read-only")
        if tool.kind == NETWORK and self.mode == "plan":
            return Decision("allow", "research is allowed in plan mode")
        if tool.kind == WRITE and self.mode == "acceptEdits":
            target = Path(subject) if subject else None
            if target is not None and _is_within(target, self.project_root):
                return Decision("allow", "acceptEdits mode")
            return Decision("ask", "file is outside the project")
        return Decision("ask", KIND_REASON.get(tool.kind, f"{tool.kind} action"))


def _auto_decision(tool: Tool, args: dict, subject: str, read_only: bool, project_root: Path) -> Decision:
    """Auto mode: do the work without asking, except for actions that are hard to undo."""
    if tool.kind in (READ, META) or read_only:
        return Decision("allow", "read-only")
    if tool.name == "Bash":
        why = risky_command(args.get("command", ""))
        return Decision("ask", f"auto mode still asks: this {why}") if why else Decision("allow", "auto mode")
    if tool.kind == WRITE:
        target = Path(subject) if subject else None
        if target is not None and _is_within(target, project_root):
            return Decision("allow", "auto mode")
        return Decision("ask", "auto mode still asks: the file is outside the project")
    if tool.name.startswith("mcp__"):
        return Decision("ask", "auto mode still asks: this MCP tool can change things outside this project")
    if tool.name == "McpServers":
        return Decision("ask", "auto mode still asks: this starts a program on your machine (an MCP server)")
    return Decision("allow", "auto mode")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except (ValueError, OSError):
        return False


__all__ = ["CYCLE", "MODE_NAMES", "MODES", "Decision", "PermissionManager", "Rule", "normalize_mode", "suggest_rule"]
