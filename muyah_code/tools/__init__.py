"""Built-in tool set."""

from __future__ import annotations

from muyah_code.tools.base import Tool, ToolContext, ToolError, ToolResult
from muyah_code.tools.fs import EditTool, LSTool, ReadTool, WriteTool
from muyah_code.tools.meta import AskUserTool, SkillTool, TodoWriteTool
from muyah_code.tools.registry import ToolRegistry
from muyah_code.tools.search import GlobTool, GrepTool
from muyah_code.tools.shell import BashOutputTool, BashTool, KillShellTool
from muyah_code.tools.web import WebFetchTool, WebSearchTool

READ_ONLY_TOOLS = ["Read", "LS", "Glob", "Grep", "WebFetch", "WebSearch", "Skill"]


def builtin_tools() -> list[Tool]:
    return [
        ReadTool(), WriteTool(), EditTool(), LSTool(), GlobTool(), GrepTool(),
        BashTool(), BashOutputTool(), KillShellTool(),
        WebSearchTool(), WebFetchTool(),
        TodoWriteTool(), AskUserTool(), SkillTool(),
    ]


def default_registry() -> ToolRegistry:
    return ToolRegistry(builtin_tools())


__all__ = ["READ_ONLY_TOOLS", "Tool", "ToolContext", "ToolError", "ToolResult", "ToolRegistry",
           "builtin_tools", "default_registry"]
