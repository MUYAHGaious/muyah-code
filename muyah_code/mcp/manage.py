"""McpServers: the agent sets up MCP servers itself (like `claude mcp add`), no restart needed.

    list                          what is configured and what is running
    add NAME (command+args | url) write it to .mcp.json (project) or ~/.muyah/mcp.json (user) and start it now
    connect NAME                  (re)start a configured server now
    disable NAME                  stop it and remember that for this project

Adding or starting a server runs a program on this machine, so it always goes through approval (auto mode
asks too), and the user sees the exact command first.
"""

from __future__ import annotations

import json

from muyah_code.tools.base import EXEC, Tool, ToolContext, ToolError, ToolResult

ACTIONS = ("list", "add", "connect", "disable")


class McpServersTool(Tool):
    name = "McpServers"
    kind = EXEC
    description = ("Set up and manage MCP servers (tools from other programs: browsers, databases, APIs, vision "
                   "models...). You CAN do this yourself: action=add writes the server into the MCP config and starts "
                   "it right away (no restart); its tools appear as mcp__<name>__<tool> from your next reply. "
                   "Stdio servers: command + args (e.g. npx -y some-mcp-package), env for keys the user gives you. "
                   "HTTP servers: url (+ headers). scope=project (.mcp.json, default) or user (~/.muyah/mcp.json). "
                   "Other actions: list, connect (restart one), disable. Never invent API keys: ask the user for them.")
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "name": {"type": "string", "description": "Server name (letters, digits, - and _)"},
            "command": {"type": "string", "description": "Program to start (stdio servers), e.g. npx"},
            "args": {"type": "array", "items": {"type": "string"}},
            "env": {"type": "object", "additionalProperties": {"type": "string"},
                    "description": "Environment variables, e.g. an API key the user gave you"},
            "url": {"type": "string", "description": "HTTP server URL (instead of command)"},
            "headers": {"type": "object", "additionalProperties": {"type": "string"}},
            "scope": {"type": "string", "enum": ["project", "user"]},
        },
        "required": ["action"],
    }

    def __init__(self, app):
        self.app = app

    def title(self, args):
        return f"McpServers({args.get('action', '')} {args.get('name', '')})".replace(" )", ")")

    def is_read_only(self, args):
        return args.get("action") == "list"

    def permission_subject(self, args, ctx):
        if args.get("action") == "add":
            what = args.get("url") or " ".join([str(args.get("command") or "")] + [str(a) for a in args.get("args") or []])
            return f"mcp add {args.get('name', '')}: {what}"
        return f"mcp {args.get('action', '')} {args.get('name', '')}"

    def preview(self, args, ctx):
        if args.get("action") != "add":
            return None
        shown = {k: v for k, v in args.items() if k in ("command", "args", "url", "scope")}
        if args.get("env"):
            shown["env"] = {k: "•••" for k in args["env"]}         # never echo secrets on screen
        return f"Add MCP server '{args.get('name', '')}' and start it now:\n" + json.dumps(shown, indent=2)

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        action = args.get("action")
        if action == "list":
            return ToolResult(self.app.mcp_list(), summary="MCP servers")
        name = str(args.get("name") or "").strip()
        if not name or not all(c.isalnum() or c in "-_" for c in name):
            raise ToolError("Give the server a name made of letters, digits, - and _.")
        if action == "add":
            if not args.get("command") and not args.get("url"):
                raise ToolError("An MCP server needs either a command (stdio) or a url (HTTP).")
            spec = {k: args[k] for k in ("command", "args", "env", "url", "headers") if args.get(k)}
            path, tools = self.app.mcp_add(name, spec, args.get("scope") or "project")
            return ToolResult(f"Saved '{name}' in {path} and started it: {len(tools)} tools - "
                              + ", ".join(tools[:30]) + ". Call them from your next reply.",
                              summary=f"{name}: {len(tools)} tools")
        if action in ("connect", "disable"):
            msg = self.app.mcp_set(name, "reconnect" if action == "connect" else "disable")
            return ToolResult(msg, summary=msg[:80])
        raise ToolError(f"Unknown action '{action}' (use one of: {', '.join(ACTIONS)}).")
