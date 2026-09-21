"""A scripted Anthropic Messages API server (streaming SSE) for testing the native Claude adapter."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def turn(text: str = "", tools: list[dict] | None = None, thinking: str = "", stop: str | None = None) -> dict:
    """tools: [{"id": "toolu_1", "name": "Read", "input": {...}}]"""
    return {"text": text, "tools": tools or [], "thinking": thinking, "stop": stop}


class FakeAnthropic:
    def __init__(self, script: list[dict], key: str = "sk-ant-test", models: list[dict] | None = None):
        self.script = list(script)
        self.key = key
        self.models = models or [{"type": "model", "id": "claude-opus-5", "display_name": "Claude Opus 5",
                                  "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1000000,
                                  "max_tokens": 128000}]
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.post_error: tuple[int, str] | None = None  # e.g. (400, "Your credit balance is too low ...")
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _auth_ok(self) -> bool:
                if self.headers.get("x-api-key") == server.key:
                    return True
                self._json(401, {"type": "error", "error": {"type": "authentication_error",
                                                            "message": "invalid x-api-key"}})
                return False

            def _json(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if not self._auth_ok():
                    return
                path = self.path.split("?")[0]
                if path == "/v1/models":
                    self._json(200, {"data": server.models, "has_more": False, "first_id": None, "last_id": None})
                elif path.startswith("/v1/models/"):
                    mid = path.rsplit("/", 1)[1]
                    found = [m for m in server.models if m["id"] == mid]
                    if found:
                        self._json(200, found[0])
                    else:
                        self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": mid}})
                else:
                    self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": "?"}})

            def do_POST(self):
                if not self._auth_ok():
                    return
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                server.requests.append(body)
                if server.post_error:
                    status, message = server.post_error
                    self._json(status, {"type": "error", "error": {"type": "invalid_request_error",
                                                                   "message": message}})
                    return
                server.headers.append({k.lower(): v for k, v in self.headers.items()})
                step = server.script.pop(0) if server.script else turn("(script exhausted)")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def ev(name, data):
                    self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
                    self.wfile.flush()

                ev("message_start", {"type": "message_start", "message": {
                    "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"], "content": [],
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 1200, "output_tokens": 1, "cache_read_input_tokens": 800,
                              "cache_creation_input_tokens": 0}}})
                idx = 0
                if step["thinking"]:
                    ev("content_block_start", {"type": "content_block_start", "index": idx,
                                               "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
                    ev("content_block_delta", {"type": "content_block_delta", "index": idx,
                                               "delta": {"type": "thinking_delta", "thinking": step["thinking"]}})
                    ev("content_block_delta", {"type": "content_block_delta", "index": idx,
                                               "delta": {"type": "signature_delta", "signature": "sig-abc"}})
                    ev("content_block_stop", {"type": "content_block_stop", "index": idx})
                    idx += 1
                if step["text"]:
                    ev("content_block_start", {"type": "content_block_start", "index": idx,
                                               "content_block": {"type": "text", "text": ""}})
                    for i in range(0, len(step["text"]), 6):
                        ev("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                   "delta": {"type": "text_delta", "text": step["text"][i:i + 6]}})
                    ev("content_block_stop", {"type": "content_block_stop", "index": idx})
                    idx += 1
                for t in step["tools"]:
                    ev("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {
                        "type": "tool_use", "id": t["id"], "name": t["name"], "input": {}}})
                    raw = json.dumps(t["input"])
                    for half in (raw[: len(raw) // 2], raw[len(raw) // 2:]):
                        ev("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                   "delta": {"type": "input_json_delta", "partial_json": half}})
                    ev("content_block_stop", {"type": "content_block_stop", "index": idx})
                    idx += 1
                stop = step["stop"] or ("tool_use" if step["tools"] else "end_turn")
                ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                     "usage": {"output_tokens": 42}})
                ev("message_stop", {"type": "message_stop"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> FakeAnthropic:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
