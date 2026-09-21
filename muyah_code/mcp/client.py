"""Minimal, dependency-free MCP client (stdio + streamable HTTP), enough to use MCP tools.

Config (Claude Code compatible) from ~/.muyah/mcp.json, <project>/.mcp.json and <project>/.muyah/mcp.json:
    {"mcpServers": {
        "github": {"type": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
                   "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}},
        "docs":   {"type": "http", "url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${TOKEN}"}}
    }}
Tools are exposed as mcp__<server>__<tool>.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from muyah_code.tools.base import EXEC, READ, Tool, ToolError, ToolResult

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "muyah-code", "version": "1.0"}
VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class MCPError(Exception):
    pass


def expand_vars(value, env: dict | None = None):
    env = env if env is not None else os.environ
    if isinstance(value, str):
        return VAR_RE.sub(lambda m: env.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [expand_vars(v, env) for v in value]
    if isinstance(value, dict):
        return {k: expand_vars(v, env) for k, v in value.items()}
    return value


@dataclass
class ServerConfig:
    name: str
    type: str = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    url: str = ""
    headers: dict = field(default_factory=dict)
    timeout: float = 60.0
    disabled: bool = False

    @classmethod
    def from_dict(cls, name: str, d: dict) -> ServerConfig:
        d = expand_vars(d)
        typ = d.get("type") or ("http" if d.get("url") else "stdio")
        timeout = d.get("timeout", 60000)
        return cls(
            name=name, type=typ, command=d.get("command", ""), args=list(d.get("args") or []),
            env=dict(d.get("env") or {}), url=d.get("url", ""), headers=dict(d.get("headers") or {}),
            timeout=float(timeout) / 1000 if float(timeout) > 1000 else float(timeout),
            disabled=bool(d.get("disabled", False)),
        )


class StdioTransport:
    def __init__(self, cfg: ServerConfig, log_path: Path):
        exe = shutil.which(cfg.command) or cfg.command
        if not exe:
            raise MCPError("stdio server has no command")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "ab")
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        self.proc = subprocess.Popen(
            [exe, *cfg.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            env={**os.environ, **{k: str(v) for k, v in cfg.env.items()}}, **kwargs,
        )
        self._pending: dict[int, dict] = {}
        self._cond = threading.Condition()
        self._closed = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        assert self.proc.stdout is not None
        for raw in iter(self.proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and "id" in msg and ("result" in msg or "error" in msg):
                with self._cond:
                    self._pending[msg["id"]] = msg
                    self._cond.notify_all()
            elif isinstance(msg, dict) and "id" in msg and "method" in msg:
                # server->client request (ping, roots/list...): answer minimally
                self._send({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _send(self, payload: dict) -> None:
        if self.proc.stdin is None or self.proc.poll() is not None:
            raise MCPError("server process is not running")
        self.proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def request(self, payload: dict, timeout: float) -> dict:
        self._send(payload)
        rid = payload["id"]
        with self._cond:
            ok = self._cond.wait_for(lambda: rid in self._pending or self._closed, timeout=timeout)
            if rid in self._pending:
                return self._pending.pop(rid)
        if not ok:
            raise MCPError(f"timed out after {timeout:.0f}s waiting for {payload.get('method')}")
        raise MCPError(f"server exited (code {self.proc.poll()})")

    def notify(self, payload: dict) -> None:
        self._send(payload)

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        self._log.close()


class HttpTransport:
    def __init__(self, cfg: ServerConfig):
        self.url = cfg.url
        self.headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json",
                        **cfg.headers}
        self.client = httpx.Client(timeout=httpx.Timeout(cfg.timeout, connect=15.0), follow_redirects=True)
        self.session_id: str | None = None

    def _post(self, payload: dict, timeout: float) -> httpx.Response:
        headers = dict(self.headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
        try:
            resp = self.client.post(self.url, json=payload, headers=headers, timeout=timeout)
        except httpx.HTTPError as e:
            raise MCPError(f"HTTP error: {e}") from e
        if resp.headers.get("mcp-session-id"):
            self.session_id = resp.headers["mcp-session-id"]
        if resp.status_code >= 400:
            raise MCPError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp

    def request(self, payload: dict, timeout: float) -> dict:
        resp = self._post(payload, timeout)
        ctype = resp.headers.get("content-type", "")
        if "text/event-stream" in ctype:
            for block in resp.text.split("\n\n"):
                data = "\n".join(ln[5:].strip() for ln in block.splitlines() if ln.startswith("data:"))
                if not data:
                    continue
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict) and msg.get("id") == payload["id"]:
                    return msg
            raise MCPError("no response found in event stream")
        try:
            return resp.json()
        except json.JSONDecodeError as e:
            raise MCPError(f"invalid JSON response: {resp.text[:200]}") from e

    def notify(self, payload: dict) -> None:
        self._post(payload, 15)

    def close(self) -> None:
        self.client.close()


class MCPClient:
    def __init__(self, cfg: ServerConfig, log_dir: Path):
        self.cfg = cfg
        self.log_dir = log_dir
        self.transport: StdioTransport | HttpTransport | None = None
        self._ids = itertools.count(1)
        self.server_info: dict = {}
        self.tools: list[dict] = []

    def connect(self) -> None:
        if self.cfg.type == "stdio":
            self.transport = StdioTransport(self.cfg, self.log_dir / f"mcp-{self.cfg.name}.log")
        elif self.cfg.type in ("http", "streamable-http", "sse"):
            self.transport = HttpTransport(self.cfg)
        else:
            raise MCPError(f"unsupported transport type '{self.cfg.type}'")
        init = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": CLIENT_INFO,
        })
        self.server_info = init.get("serverInfo", {})
        self.transport.notify({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.tools = self._list_tools()

    def request(self, method: str, params: dict | None = None, timeout: float | None = None) -> dict:
        if self.transport is None:
            raise MCPError("not connected")
        payload = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params or {}}
        msg = self.transport.request(payload, timeout or self.cfg.timeout)
        if "error" in msg:
            err = msg["error"] or {}
            raise MCPError(f"{method} failed: {err.get('message', err)}")
        return msg.get("result") or {}

    def _list_tools(self) -> list[dict]:
        tools, cursor = [], None
        for _ in range(20):
            res = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools.extend(res.get("tools") or [])
            cursor = res.get("nextCursor")
            if not cursor:
                break
        return tools

    def call_tool(self, name: str, arguments: dict) -> tuple[str, bool, list[tuple[str, str]]]:
        """(text, is_error, images as (media type, base64))."""
        res = self.request("tools/call", {"name": name, "arguments": arguments})
        parts, images = [], []
        for item in res.get("content") or []:
            t = item.get("type")
            if t == "text":
                parts.append(item.get("text", ""))
            elif t == "resource":
                r = item.get("resource") or {}
                parts.append(r.get("text") or f"[resource {r.get('uri', '')}]")
            elif t == "image" and item.get("data"):
                images.append((item.get("mimeType") or "image/png", item["data"]))
                parts.append(f"[image {len(images)}: {item.get('mimeType', 'image')}, attached]")
            else:
                parts.append(json.dumps(item)[:1000])
        if not parts and res.get("structuredContent") is not None:
            parts.append(json.dumps(res["structuredContent"], indent=2))
        return "\n".join(parts) or "(no content)", bool(res.get("isError")), images

    def close(self) -> None:
        if self.transport:
            try:
                self.transport.close()
            except Exception:
                pass


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


class MCPTool(Tool):
    def __init__(self, client: MCPClient, spec: dict):
        self.client = client
        self.remote_name = spec["name"]
        self.name = f"mcp__{_safe(client.cfg.name)}__{_safe(spec['name'])}"
        self.description = (spec.get("description") or f"MCP tool {spec['name']}")[:1500]
        schema = spec.get("inputSchema") or {"type": "object", "properties": {}}
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        self.parameters = schema
        annotations = spec.get("annotations") or {}
        self.kind = READ if annotations.get("readOnlyHint") else EXEC

    def permission_subject(self, args, ctx):
        return json.dumps(args, default=str)[:500]

    def run(self, args, ctx):
        try:
            text, is_error, images = self.client.call_tool(self.remote_name, args)
        except MCPError as e:
            raise ToolError(f"MCP server '{self.client.cfg.name}': {e}") from e
        summary = "error" if is_error else f"{len(text)} chars" + (f", {len(images)} image(s)" if images else "")
        return ToolResult(text, is_error=is_error, summary=summary, images=images)


CLAUDE_JSON = Path.home() / ".claude.json"


def _claude_code_servers(project_root: Path, scope: str = "project") -> dict:
    """MCP servers you set up for Claude Code in ~/.claude.json. scope "project" (default): the ones for this
    folder; "all": also the user-wide ones (they would start in every session); "off": none."""
    if scope == "off":
        return {}
    try:
        data = json.loads(CLAUDE_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    servers = dict(data.get("mcpServers") or {}) if scope == "all" else {}
    projects = data.get("projects") or {}
    for key in (str(project_root), project_root.as_posix(), str(project_root).replace("\\", "/")):
        servers.update((projects.get(key) or {}).get("mcpServers") or {})
    return servers


def load_server_configs(project_root: Path, home: Path, claude_scope: str = "project",
                        disabled: list[str] | None = None) -> dict[str, ServerConfig]:
    """User (~/.muyah/mcp.json, Claude Code's ~/.claude.json), then project (.mcp.json, .muyah/mcp.json);
    later ones win. Names in `disabled` (setting mcp.disabled) are kept but not started."""
    configs: dict[str, ServerConfig] = {}
    if claude_scope != "off":
        for name, d in _claude_code_servers(project_root, claude_scope).items():
            if isinstance(d, dict):
                configs[name] = ServerConfig.from_dict(name, d)
    for path in (home / "mcp.json", project_root / ".mcp.json", project_root / ".muyah" / "mcp.json"):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for name, d in (data.get("mcpServers") or {}).items():
            if isinstance(d, dict):
                configs[name] = ServerConfig.from_dict(name, d)
    for name in disabled or []:
        if name in configs:
            configs[name].disabled = True
    return configs


class MCPManager:
    def __init__(self, configs: dict[str, ServerConfig], log_dir: Path):
        self.configs = configs
        self.log_dir = log_dir
        self.clients: dict[str, MCPClient] = {}
        self.status: dict[str, str] = {}

    def connect_all(self, timeout: float = 30.0) -> list[Tool]:
        active = {n: c for n, c in self.configs.items() if not c.disabled}
        for n, c in self.configs.items():
            if c.disabled:
                self.status[n] = "disabled"
        if not active:
            return []

        def connect(cfg: ServerConfig):
            client = MCPClient(cfg, self.log_dir)
            client.connect()
            return client

        tools: list[Tool] = []
        with ThreadPoolExecutor(max_workers=min(8, len(active))) as pool:
            futures = {n: pool.submit(connect, c) for n, c in active.items()}
            for name, fut in futures.items():
                try:
                    client = fut.result(timeout=timeout)
                except Exception as e:
                    self.status[name] = f"failed: {e}"
                    continue
                self.clients[name] = client
                self.status[name] = f"connected ({len(client.tools)} tools)"
                tools.extend(MCPTool(client, spec) for spec in client.tools if spec.get("name"))
        return tools

    def disconnect(self, name: str) -> None:
        client = self.clients.pop(name, None)
        if client is not None:
            client.close()

    def reconnect(self, name: str) -> list[Tool]:
        """Stop the server (if running) and start it again. Raises on failure."""
        cfg = self.configs.get(name)
        if cfg is None:
            raise KeyError(name)
        self.disconnect(name)
        cfg.disabled = False
        return self.add(cfg)

    def add(self, cfg: ServerConfig) -> list[Tool]:
        """Connect one more server now (e.g. the built-in browser, on first use). Raises on failure."""
        self.configs[cfg.name] = cfg
        client = MCPClient(cfg, self.log_dir)
        try:
            client.connect()
        except Exception as e:
            self.status[cfg.name] = f"failed: {e}"
            raise
        self.clients[cfg.name] = client
        self.status[cfg.name] = f"connected ({len(client.tools)} tools)"
        return [MCPTool(client, spec) for spec in client.tools if spec.get("name")]

    def shutdown(self) -> None:
        for c in self.clients.values():
            c.close()
