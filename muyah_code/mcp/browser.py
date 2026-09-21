"""The built-in browser: Microsoft's Playwright MCP server, started the first time the agent needs it.

Every session offers one small `Browser` tool. Calling it starts `npx @playwright/mcp` (headless, isolated
profile), connects it like any other MCP server, and replaces itself with the real browser_* tools
(navigate, click, type, fill forms, read the page as an accessibility snapshot, take screenshots...).
Nothing runs until then: no Node process, no startup delay, no memory.

Screenshots reach the model only when it can see images; snapshots (text) work with every model.
Needs Node.js 18+. Turn it off with "mcp": {"browser": false}; a server you configured yourself named
"browser" or "playwright" is used instead of this one.
"""

from __future__ import annotations

import re
import shutil
import subprocess

from muyah_code.mcp.client import ServerConfig
from muyah_code.tools.base import META, Tool, ToolContext, ToolError, ToolResult

PACKAGE = "@playwright/mcp@latest"
MIN_NODE = 18
FIRST_START_TIMEOUT = 180.0     # the first run downloads the package


def browser_config(headless: bool = True, output_dir=None) -> ServerConfig:
    """output_dir: where Playwright writes snapshots and screenshots (never the user's project)."""
    args = ["-y", PACKAGE, "--isolated"] + (["--headless"] if headless else [])
    if output_dir is not None:
        args += ["--output-dir", str(output_dir)]
    return ServerConfig(name="browser", command="npx", args=args, timeout=FIRST_START_TIMEOUT)


def node_version() -> int | None:
    node = shutil.which("node")
    if not node:
        return None
    try:
        out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.match(r"v?(\d+)", out.strip())
    return int(m.group(1)) if m else None


def wanted(cfg, configured: dict) -> bool:
    """Offer the built-in browser? (not when turned off or when you configured your own)"""
    if cfg.get("mcp.browser", True) is False:
        return False
    return not ({"browser", "playwright"} & {name.lower() for name in configured})


class BrowserTool(Tool):
    name = "Browser"
    kind = META
    description = ("Start the web browser tools (Playwright, headless): open a URL, click, type, fill forms, read "
                   "the page as an accessibility snapshot, take screenshots, see console errors. Use it to check a "
                   "web UI you built or to read a page that needs JavaScript. Call it once; the browser_* tools "
                   "then appear and you call those.")
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, activate):
        self.activate = activate   # () -> list of tool names now available (the app connects and registers)

    def title(self, args):
        return "Browser(start)"

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        version = node_version()
        if version is None:
            raise ToolError("The browser tools need Node.js 18 or newer (https://nodejs.org). Node is not installed "
                            "or not on PATH. Tell the user; you can still use WebFetch for static pages.")
        if version < MIN_NODE:
            raise ToolError(f"The browser tools need Node.js {MIN_NODE}+; this machine has Node {version}. Tell the "
                            "user to update Node; WebFetch still works for static pages.")
        busy = getattr(ctx.service("ui"), "busy", None)
        if callable(busy):
            busy("Starting the browser tools (the first time downloads @playwright/mcp)")
        names = self.activate()
        return ToolResult(
            "Browser tools are ready: " + ", ".join(names) + ".\nStart with browser_navigate, then browser_snapshot "
            "to read the page (refs in the snapshot are what browser_click / browser_type take). "
            "browser_take_screenshot gives a picture (only useful if you can see images). "
            "Close it with browser_close when you are done.", summary=f"{len(names)} browser tools")
