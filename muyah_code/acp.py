"""`muyah acp`: MUYAH-CODE inside your editor, over the Agent Client Protocol (ACP, protocol version 1).

Zed, JetBrains IDEs and Neovim (CodeCompanion, avante.nvim) start an ACP agent as a subprocess and talk
JSON-RPC 2.0 over its stdin/stdout, one JSON message per line. This module is that agent:

  editor -> agent   initialize, session/new, session/load, session/prompt, session/set_mode, session/cancel
  agent -> editor   session/update (message and thought chunks, tool calls with real before/after diffs,
                    the plan, mode changes), session/request_permission (approvals, in the editor's own UI),
                    fs/read_text_file (so the agent sees unsaved editor buffers, when the editor offers it)

Everything else is the normal MUYAH-CODE: the same config, providers, tools, permissions, skills, hooks,
lessons, cost accounting and rewind snapshots. Nothing is written to stdout except protocol messages.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

from muyah_code import __version__
from muyah_code.ui.base import UI, PermissionReply, PermissionRequest

PROTOCOL_VERSION = 1
MODES = [("plan", "Plan", "Read-only: explore and propose a plan"),
         ("acceptEdits", "Edit", "Accept file edits; ask before commands"),
         ("default", "Manual", "Ask before edits, commands and network access"),
         ("auto", "Auto", "Run on its own; still asks for risky actions"),
         ("bypassPermissions", "Bypass", "Never ask (deletes are still never run)")]
KINDS = {"Read": "read", "LS": "read", "Grep": "search", "Glob": "search", "Edit": "edit", "MultiEdit": "edit",
         "Write": "edit", "Bash": "execute", "BashOutput": "execute", "KillShell": "execute", "WebFetch": "fetch",
         "WebSearch": "fetch", "TodoWrite": "think", "Agent": "think", "Skill": "think"}


class AcpError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class JsonRpc:
    """Newline-delimited JSON-RPC 2.0 on two byte streams, both directions (requests from either side)."""

    def __init__(self, rfile, wfile):
        self.rfile = rfile
        self.wfile = wfile
        self._write = threading.Lock()
        self._pending: dict[int, dict] = {}
        self._next = 0
        self._ids = threading.Lock()

    def send(self, message: dict) -> None:
        data = (json.dumps({"jsonrpc": "2.0", **message}, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write:
            self.wfile.write(data)
            self.wfile.flush()

    def notify(self, method: str, params: dict) -> None:
        self.send({"method": method, "params": params})

    def request(self, method: str, params: dict, cancel: threading.Event | None = None, timeout: float | None = None):
        with self._ids:
            self._next += 1
            rid = self._next
        slot = {"done": threading.Event()}
        self._pending[rid] = slot
        self.send({"id": rid, "method": method, "params": params})
        waited = 0.0
        while not slot["done"].wait(0.1):
            waited += 0.1
            if (cancel is not None and cancel.is_set()) or (timeout is not None and waited >= timeout):
                self._pending.pop(rid, None)
                raise AcpError(-32800, "cancelled")
        self._pending.pop(rid, None)
        if "error" in slot:
            err = slot["error"] or {}
            raise AcpError(int(err.get("code", -32603)), str(err.get("message", "error")))
        return slot.get("result")

    def serve(self, handle_request, handle_notification) -> None:
        """Read until the editor closes stdin. Requests run on their own threads (a prompt takes minutes,
        and meanwhile cancel notifications and permission answers must still get through)."""
        for raw in self.rfile:
            line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self.send({"id": None, "error": {"code": -32700, "message": "parse error"}})
                continue
            if "method" in msg and "id" in msg:
                threading.Thread(target=self._answer, args=(handle_request, msg), daemon=True).start()
            elif "method" in msg:
                try:
                    handle_notification(msg["method"], msg.get("params") or {})
                except Exception as e:   # a bad notification must not stop the server; the editor sees it
                    print(f"muyah acp: {msg['method']}: {e}", file=sys.stderr)
            elif "id" in msg:
                slot = self._pending.get(msg["id"])
                if slot is not None:
                    if "error" in msg:
                        slot["error"] = msg["error"]
                    else:
                        slot["result"] = msg.get("result")
                    slot["done"].set()

    def _answer(self, handle, msg) -> None:
        try:
            result = handle(msg["method"], msg.get("params") or {})
            self.send({"id": msg["id"], "result": result})
        except AcpError as e:
            self.send({"id": msg["id"], "error": {"code": e.code, "message": str(e)}})
        except Exception as e:     # report it to the editor instead of dying
            self.send({"id": msg["id"], "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}})


class AcpUI(UI):
    """What the terminal would show, sent to the editor as session updates."""

    headless = False

    def __init__(self, session: AcpSession):
        self.s = session

    def text(self, chunk: str) -> None:
        if chunk:
            self.s.update({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": chunk}})

    def reasoning(self, chunk: str) -> None:
        if chunk:
            self.s.update({"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": chunk}})

    def warn(self, msg: str) -> None:
        self.reasoning(f"\n[note] {msg}\n")

    def error(self, msg: str) -> None:
        self.text(f"\n**Error:** {msg}\n")

    def handoff(self, command: str, targets: list[str]) -> None:
        self.text(f"\nDeletes are left to you. To delete, run this yourself:\n\n```\n{command}\n```\n")

    def on_todos(self, todos: list[dict]) -> None:
        self.s.update({"sessionUpdate": "plan", "entries": [
            {"content": t.get("content", ""), "priority": "medium",
             "status": {"in_progress": "in_progress", "completed": "completed"}.get(t.get("status"), "pending")}
            for t in todos]})

    def ask_permission(self, req: PermissionRequest) -> PermissionReply:
        options = [{"optionId": "allow-once", "name": "Allow", "kind": "allow_once"},
                   {"optionId": "allow-always", "name": "Always allow", "kind": "allow_always"},
                   {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"}]
        tool_call = {"toolCallId": self.s.asking or "permission", "title": req.title,
                     "kind": KINDS.get(req.tool_name, "other"),
                     "rawInput": {"detail": (req.detail or "")[:4000], "reason": req.reason}}
        try:
            result = self.s.rpc.request("session/request_permission", {
                "sessionId": self.s.id, "toolCall": tool_call, "options": options}, cancel=self.s.cancel)
        except AcpError:
            return PermissionReply("no", "Cancelled in the editor.")
        outcome = (result or {}).get("outcome") or {}
        choice = outcome.get("optionId") if outcome.get("outcome") == "selected" else None
        return PermissionReply({"allow-once": "yes", "allow-always": "always"}.get(choice, "no"))

    def ask_user(self, question: str, options: list[str]) -> str:
        """No question dialog in ACP: the choices are offered as permission options (the editor shows them)."""
        if not options:
            self.text(f"\n**Question:** {question}\n(Answer in your next message.)\n")
            return ""
        opts = [{"optionId": str(i), "name": o[:80], "kind": "allow_once"} for i, o in enumerate(options)]
        try:
            result = self.s.rpc.request("session/request_permission", {
                "sessionId": self.s.id, "toolCall": {"toolCallId": "question", "title": question[:200], "kind": "think"},
                "options": opts}, cancel=self.s.cancel)
        except AcpError:
            return ""
        outcome = (result or {}).get("outcome") or {}
        idx = outcome.get("optionId")
        return options[int(idx)] if outcome.get("outcome") == "selected" and str(idx).isdigit() else ""


class AcpSession:
    def __init__(self, rpc: JsonRpc, client_caps: dict):
        self.rpc = rpc
        self.caps = client_caps or {}
        self.app = None
        self.id = ""
        self.cancel = threading.Event()
        self.asking = ""                      # the tool call waiting for approval (its ACP id)
        self._before: dict[str, str] = {}     # ACP tool id -> the file's text before an edit
        self._paths: dict[str, str] = {}

    def update(self, update: dict) -> None:
        if self.id:
            self.rpc.notify("session/update", {"sessionId": self.id, "update": update})

    def start(self, cwd: str, mcp_servers: list, resume: str | None = None) -> None:
        from muyah_code.app import App
        from muyah_code.config import load_config

        root = Path(cwd or ".").resolve()
        cfg = load_config(cwd=root)
        self.app = App(cfg, AcpUI(self), cwd=root, resume=resume, enable_mcp=True)
        self.id = self.app.session_id or f"muyah-{id(self)}"
        self.app.agent.cancel_event = self.cancel
        self.app.events.subscribe(self._on_event)
        self._add_mcp(mcp_servers or [])
        if self.caps.get("fs", {}).get("readTextFile"):
            self.app.ctx.services["editor_read"] = self._editor_read

    def _add_mcp(self, servers: list) -> None:
        """MCP servers the editor passes along (its own configured servers)."""
        from muyah_code.mcp.client import MCPManager, ServerConfig

        for s in servers:
            if not isinstance(s, dict) or not s.get("name"):
                continue
            env = {e.get("name"): e.get("value") for e in s.get("env") or [] if isinstance(e, dict)}
            headers = {h.get("name"): h.get("value") for h in s.get("headers") or [] if isinstance(h, dict)}
            cfg = ServerConfig(name=s["name"], type="http" if s.get("url") else "stdio", command=s.get("command", ""),
                               args=list(s.get("args") or []), env=env, url=s.get("url", ""), headers=headers)
            if self.app.mcp is None:
                self.app.mcp = MCPManager({}, self.app.home / "logs")
            try:
                tools = self.app.mcp.add(cfg)
            except Exception as e:     # one broken server must not stop the session: tell the user
                AcpUI(self).warn(f"MCP server {s['name']} did not start: {e}")
                continue
            for t in tools:
                self.app.registry.register(t)
                if self.app.agent.registry is not self.app.registry:
                    self.app.agent.registry.register(t)

    def _editor_read(self, path: Path) -> str | None:
        """The file as the editor has it (unsaved changes included); None to read from disk instead."""
        try:
            res = self.rpc.request("fs/read_text_file", {"sessionId": self.id, "path": str(path)}, timeout=10)
        except AcpError:
            return None
        return (res or {}).get("content")

    # ------------------------------------------------------------------ the agent's events -> tool calls

    def _tid(self, ev: dict) -> str:
        return f"{ev.get('agent') or 'main'}-{ev.get('id')}"

    def _on_event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "tool_request":
            tid, args = self._tid(ev), ev.get("args") or {}
            path = args.get("file_path") or args.get("path")
            self._paths[tid] = str(self.app.ctx.resolve(path)) if path and ev.get("name") in KINDS else ""
            update = {"sessionUpdate": "tool_call", "toolCallId": tid, "title": ev.get("title") or ev.get("name"),
                      "kind": KINDS.get(ev.get("name", ""), "other"), "status": "pending", "rawInput": args}
            if self._paths[tid]:
                update["locations"] = [{"path": self._paths[tid]}]
            self.update(update)
        elif kind == "tool_permission" and ev.get("state") == "asking":
            self.asking = self._tid(ev)
        elif kind == "tool_start":
            tid = self._tid(ev)
            if ev.get("name") in ("Edit", "MultiEdit", "Write") and self._paths.get(tid):
                self._before[tid] = _read(self._paths[tid])
            self.update({"sessionUpdate": "tool_call_update", "toolCallId": tid, "status": "in_progress"})
        elif kind == "tool_end":
            tid = self._tid(ev)
            update = {"sessionUpdate": "tool_call_update", "toolCallId": tid,
                      "status": "completed" if ev.get("ok") else "failed"}
            if tid in self._before and ev.get("ok"):
                path = self._paths.get(tid, "")
                update["content"] = [{"type": "diff", "path": path, "oldText": self._before.pop(tid) or None,
                                      "newText": _read(path)}]
            elif ev.get("output") or ev.get("summary"):
                update["content"] = [{"type": "content", "content": {
                    "type": "text", "text": str(ev.get("output") or ev.get("summary"))[-4000:]}}]
            self.update(update)
        elif kind == "tool_permission" and ev.get("state") in ("denied", "handoff"):
            self.update({"sessionUpdate": "tool_call_update", "toolCallId": self._tid(ev), "status": "failed"})

    # ------------------------------------------------------------------ prompts

    def prompt(self, blocks: list) -> str:
        texts, images = [], []
        for b in blocks or []:
            t = b.get("type")
            if t == "text":
                texts.append(b.get("text", ""))
            elif t == "image" and b.get("data"):
                images.append((b.get("mimeType") or "image/png", b["data"], "image from the editor"))
            elif t == "resource":
                r = b.get("resource") or {}
                if r.get("text") is not None:
                    texts.append(f'<file path="{_uri_path(r.get("uri", ""))}">\n{r["text"]}\n</file>')
            elif t == "resource_link":
                texts.append(f"@{_uri_path(b.get('uri', ''))}")
        prompt = "\n\n".join(x for x in texts if x)
        self.cancel.clear()
        app = self.app
        start = len(app.agent.messages)
        app.agent.llm = app.llm
        app.agent.escalation.reset()
        app.rewind.begin_turn(prompt, start)
        result = app.agent.run(app.expand_mentions(prompt), images=images or app.mention_images(prompt))
        app.emit_tree()
        if result.status == "interrupted" or self.cancel.is_set():
            return "cancelled"
        if result.status == "max_steps":
            return "max_turn_requests"
        if result.status not in ("ok",) and result.error:
            AcpUI(self).error(result.error)
        return "end_turn"

    def replay_history(self) -> None:
        """session/load: the earlier conversation, sent back as it happened."""
        from muyah_code.llm.content import text_of

        for m in self.app.agent.messages[1:]:
            role, text = m.get("role"), text_of(m.get("content") or "")
            if role == "user" and not m.get("_images") and text and not text.startswith("<tool_result"):
                self.update({"sessionUpdate": "user_message_chunk",
                             "content": {"type": "text", "text": text.split("\n\n<lessons>")[0]}})
            elif role == "assistant" and text.strip():
                self.update({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}})


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _uri_path(uri: str) -> str:
    if uri.startswith("file://"):
        from urllib.parse import unquote, urlparse

        p = unquote(urlparse(uri).path)
        return p[1:] if len(p) > 2 and p[0] == "/" and p[2] == ":" else p     # file:///C:/x -> C:/x
    return uri


class AcpAgent:
    def __init__(self, rpc: JsonRpc):
        self.rpc = rpc
        self.caps: dict = {}
        self.sessions: dict[str, AcpSession] = {}

    def modes(self, current: str) -> dict:
        return {"currentModeId": current, "availableModes": [{"id": i, "name": n, "description": d} for i, n, d in MODES]}

    def handle(self, method: str, p: dict):
        if method == "initialize":
            self.caps = p.get("clientCapabilities") or {}
            return {"protocolVersion": min(int(p.get("protocolVersion") or 1), PROTOCOL_VERSION),
                    "agentCapabilities": {"loadSession": True,
                                          "promptCapabilities": {"image": True, "audio": False, "embeddedContext": True},
                                          "mcpCapabilities": {"http": True, "sse": False}},
                    "agentInfo": {"name": "muyah-code", "title": "MUYAH-CODE", "version": __version__},
                    "authMethods": []}
        if method in ("session/new", "session/load"):
            s = AcpSession(self.rpc, self.caps)
            s.start(p.get("cwd", "."), p.get("mcpServers") or [],
                    resume=p.get("sessionId") if method == "session/load" else None)
            self.sessions[s.id] = s
            if method == "session/load":
                s.replay_history()
                return None
            return {"sessionId": s.id, "modes": self.modes(s.app.permissions.mode)}
        if method == "session/prompt":
            return {"stopReason": self._session(p).prompt(p.get("prompt") or [])}
        if method == "session/set_mode":
            s = self._session(p)
            s.app.set_mode(p.get("modeId", "default"))
            s.update({"sessionUpdate": "current_mode_update", "currentModeId": s.app.permissions.mode})
            return {}
        raise AcpError(-32601, f"method not found: {method}")

    def notification(self, method: str, p: dict) -> None:
        if method == "session/cancel":
            s = self.sessions.get(p.get("sessionId", ""))
            if s is not None:
                s.cancel.set()

    def _session(self, p: dict) -> AcpSession:
        s = self.sessions.get(p.get("sessionId", ""))
        if s is None:
            raise AcpError(-32602, f"unknown session {p.get('sessionId')!r}")
        return s

    def shutdown(self) -> None:
        for s in self.sessions.values():
            if s.app is not None:
                s.app.shutdown()


def serve(rfile=None, wfile=None) -> int:
    rpc = JsonRpc(rfile or sys.stdin.buffer, wfile or sys.stdout.buffer)
    agent = AcpAgent(rpc)
    try:
        rpc.serve(agent.handle, agent.notification)
    finally:
        agent.shutdown()
    return 0
