"""Sub-agents: isolated child agents for searches and self-contained subtasks.

Definitions (Claude Code compatible) live in .muyah/agents/<name>.md or ~/.muyah/agents/<name>.md
(and .claude/agents when compat is on):

    ---
    name: reviewer
    description: Reviews a diff for bugs. Use after finishing a feature.
    tools: Read, Grep, Glob, Bash        # optional; default = all except Agent
    permissionMode: default              # optional
    max_steps: 30                        # optional
    ---
    System prompt for the sub-agent...

A child starts with a fresh context: its own system prompt + the task. Only its final report returns
to the parent, which keeps the parent's (small) context window clean.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from muyah_code.skills.loader import parse_frontmatter
from muyah_code.tools.base import META, Tool, ToolError, ToolResult
from muyah_code.ui.base import UI, PermissionReply, PermissionRequest

BUNDLED_AGENTS = Path(__file__).parent / "agents" / "bundled"
MAX_DEPTH = 2


@dataclass
class AgentDef:
    name: str
    description: str
    prompt: str
    tools: list[str] | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    permission_mode: str | None = None
    max_steps: int = 40
    source: str = "bundled"


def _list(val) -> list[str] | None:
    if val is None:
        return None
    if isinstance(val, str):
        return [v.strip() for v in val.replace(",", " ").split() if v.strip()]
    return [str(v) for v in val]


def load_agent_defs(project_root: Path, home: Path, claude_compat: bool = True) -> tuple[dict[str, AgentDef], list[str]]:
    defs: dict[str, AgentDef] = {}
    errors: list[str] = []
    dirs = [(BUNDLED_AGENTS, "bundled")]
    if claude_compat:
        dirs.append((Path.home() / ".claude" / "agents", "user"))
    dirs.append((home / "agents", "user"))
    if claude_compat:
        dirs.append((project_root / ".claude" / "agents", "project"))
    dirs.append((project_root / ".muyah" / "agents", "project"))
    for d, source in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.md")):
            try:
                meta, body = parse_frontmatter(f.read_text(encoding="utf-8"))
            except (OSError, ValueError, yaml.YAMLError) as e:
                errors.append(f"{f}: {e}")
                continue
            name = str(meta.get("name") or f.stem).strip()
            defs[name] = AgentDef(
                name=name,
                description=" ".join(str(meta.get("description") or "").split()),
                prompt=body.strip(),
                tools=_list(meta.get("tools")),
                disallowed_tools=_list(meta.get("disallowedTools") or meta.get("disallowed_tools")) or [],
                permission_mode=meta.get("permissionMode") or meta.get("permission_mode"),
                max_steps=int(meta.get("max_steps") or 40),
                source=source,
            )
    return defs, errors


class SubagentUI(UI):
    """Renders a child's activity compactly inside the parent's output."""

    def __init__(self, parent: UI, label: str):
        self.parent = parent
        self.label = label
        self.headless = parent.headless

    def tool_start(self, title: str) -> None:
        self.parent.tool_start(f"[{self.label}] {title}")

    def tool_end(self, title: str, result: ToolResult) -> None:
        brief = ToolResult(content="", is_error=result.is_error, summary=result.summary)
        self.parent.tool_end(f"[{self.label}] {title}", brief)

    def warn(self, msg: str) -> None:
        self.parent.warn(f"[{self.label}] {msg}")

    def error(self, msg: str) -> None:
        self.parent.error(f"[{self.label}] {msg}")

    def ask_permission(self, req: PermissionRequest) -> PermissionReply:
        req.title = f"[{self.label}] {req.title}"
        return self.parent.ask_permission(req)


class SubagentManager:
    def __init__(self, defs: dict[str, AgentDef], factory: Callable[[AgentDef, int], object], depth: int = 0):
        self.defs = defs
        self.factory = factory  # (definition, depth) -> Agent
        self.depth = depth

    def index_text(self) -> str:
        return "\n".join(f"- {d.name}: {d.description}" for d in self.defs.values())

    def run(self, agent_type: str, description: str, prompt: str) -> tuple[str, dict]:
        if self.depth >= MAX_DEPTH:
            raise ToolError("Sub-agents cannot spawn further sub-agents (depth limit).")
        d = self.defs.get(agent_type) or self.defs.get("general")
        if d is None:
            raise ToolError(f"Unknown agent type '{agent_type}'. Known: {', '.join(self.defs)}")
        child = self.factory(d, self.depth + 1)
        task = (f"{prompt}\n\nWhen you are done, reply with a concise, self-contained report of your findings "
                "or of what you changed (paths, line numbers, commands and results). The report is all the "
                "caller will see.")
        res = child.run(task)
        text = res.text.strip()
        if res.status not in ("ok",):
            text = (text + "\n\n" if text else "") + f"[sub-agent ended with status: {res.status}"
            text += f" - {res.error}]" if res.error else "]"
        return text or "(the sub-agent returned no report)", {"status": res.status, "steps": res.steps,
                                                              "tool_calls": res.tool_calls}


class AgentTool(Tool):
    name = "Agent"
    kind = META

    def __init__(self, manager: SubagentManager):
        self.manager = manager

    @property
    def description(self) -> str:  # type: ignore[override]
        return (
            "Launch a sub-agent with a fresh context to handle a broad search or a self-contained subtask; only its "
            "final report comes back, which keeps your context small. Give it a complete, standalone task "
            "description (it cannot see this conversation). Available agent types:\n" + self.manager.index_text()
        )

    parameters = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "3-5 word label"},
            "prompt": {"type": "string", "description": "The complete task for the sub-agent"},
            "subagent_type": {"type": "string", "description": "Agent type (default: general)"},
        },
        "required": ["prompt"],
    }

    def title(self, args):
        return f"Agent[{args.get('subagent_type') or 'general'}]({args.get('description') or args.get('prompt', '')[:60]})"

    def run(self, args, ctx):
        report, stats = self.manager.run(args.get("subagent_type") or "general", args.get("description", ""),
                                         args["prompt"])
        return ToolResult(report, summary=f"{stats['status']}, {stats['tool_calls']} tool calls")
