"""Tool registry: lookup (with aliases), schema export, validated dispatch."""

from __future__ import annotations

import traceback

from muyah_code.tools.base import Tool, ToolContext, ToolError, ToolResult, truncate_middle, validate_args

# Names other agents/models commonly use -> MUYAH-CODE canonical names.
COMMON_ALIASES = {
    "read_file": "Read", "view": "Read", "cat": "Read", "open_file": "Read",
    "write_file": "Write", "create_file": "Write",
    "edit_file": "Edit", "str_replace": "Edit", "replace_in_file": "Edit", "str_replace_editor": "Edit",
    "list_directory": "LS", "list_dir": "LS", "ls": "LS", "tree": "LS",
    "find_files": "Glob", "glob": "Glob",
    "search_code": "Grep", "grep": "Grep", "search": "Grep", "ripgrep": "Grep",
    "run_command": "Bash", "bash": "Bash", "shell": "Bash", "execute_command": "Bash", "run_terminal_cmd": "Bash",
    "web_search": "WebSearch", "fetch_webpage": "WebFetch", "web_fetch": "WebFetch", "fetch": "WebFetch",
    "todo_write": "TodoWrite", "todowrite": "TodoWrite", "update_todos": "TodoWrite",
    "spawn_agent": "Agent", "task": "Agent", "subagent": "Agent",
    "ask_user": "AskUser", "ask_followup_question": "AskUser",
}


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        self._lookup: dict[str, str] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._lookup[tool.name.lower()] = tool.name
        for a in tool.aliases:
            self._lookup[a.lower()] = tool.name

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._lookup = {k: v for k, v in self._lookup.items() if v != name}

    def resolve(self, name: str) -> str | None:
        if not name:
            return None
        if name in self._tools:
            return name
        key = name.strip().lower()
        if key in self._lookup:
            return self._lookup[key]
        alias = COMMON_ALIASES.get(key)
        if alias and alias in self._tools:
            return alias
        return None

    def get(self, name: str) -> Tool | None:
        canonical = self.resolve(name)
        return self._tools.get(canonical) if canonical else None

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def subset(self, names: list[str] | None, exclude: list[str] | None = None) -> ToolRegistry:
        chosen = self.tools() if not names else [t for n in names if (t := self.get(n))]
        excluded = {self.resolve(n) for n in (exclude or [])}
        return ToolRegistry([t for t in chosen if t.name not in excluded])

    def validate(self, name: str, args: dict | None) -> tuple[Tool | None, dict, list[str]]:
        tool = self.get(name)
        if tool is None:
            return None, {}, [f"unknown tool '{name}'. Available tools: {', '.join(self.names())}"]
        clean, errors = validate_args(tool.parameters, args)
        return tool, clean, errors

    def execute(self, tool: Tool, args: dict, ctx: ToolContext, max_chars: int) -> ToolResult:
        try:
            result = tool.run(args, ctx)
        except ToolError as e:
            result = ToolResult.error(f"Error: {e}")
        except KeyboardInterrupt:
            raise
        except Exception as e:  # a bug in a tool must never kill the session
            tb = traceback.format_exc(limit=3)
            result = ToolResult.error(f"Error: {tool.name} failed unexpectedly: {e.__class__.__name__}: {e}\n{tb}")
        if not isinstance(result, ToolResult):
            result = ToolResult(content=str(result))
        result.content = truncate_middle(result.content or "(no output)", max_chars, label=f"{tool.name} output")
        return result
