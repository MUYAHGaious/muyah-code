"""A scripted OpenAI-compatible server for end-to-end tests (streaming + non-streaming + tool calls)."""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def reply(content: str = "", tool_calls: list[dict] | None = None, finish: str | None = None,
          prompt_tokens: int | None = None, think: float = 0.0, cached_tokens: int = 0) -> dict:
    """tool_calls: [{"name": "Read", "arguments": {...}, "extra": {provider fields}}]. think: seconds to wait before streaming (demos)."""
    return {"content": content, "tool_calls": tool_calls or [], "finish": finish, "prompt_tokens": prompt_tokens,
            "think": think, "cached_tokens": cached_tokens}


def error(status: int, message: str) -> dict:
    return {"error": status, "message": message}


class FakeOpenAI:
    def __init__(self, script: list[dict] | None = None, models: list[dict] | None = None,
                 require_key: str | None = None, chunk_delay: float = 0.0, headers: dict | None = None):
        self.headers = headers or {}  # extra response headers (e.g. rate limits)
        self.chunk_delay = chunk_delay  # seconds between streamed chunks, to look like a real model (demos)
        self.script = list(script or [])
        self.models = models or [{"id": "fake-model", "object": "model", "max_model_len": 32768}]
        self.requests: list[dict] = []
        self.require_key = require_key
        self.default = reply("(script exhausted)")
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for k, v in server.headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _denied(self) -> bool:
                if server.require_key and self.headers.get("Authorization") != f"Bearer {server.require_key}":
                    self._json(401, {"error": {"message": "Invalid API key", "type": "invalid_request_error"}})
                    return True
                return False

            def do_GET(self):
                if self._denied():
                    return
                if self.path.rstrip("/").endswith("/models"):
                    self._json(200, {"object": "list", "data": server.models})
                elif self.path == "/health":
                    self._json(200, {"status": "ok"})
                else:
                    self._json(404, {"error": {"message": "not found"}})

            def do_POST(self):
                if self._denied():
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self._json(404, {"error": {"message": "not found"}})
                    return
                server.requests.append(body)
                step = server.script.pop(0) if server.script else server.default
                if "error" in step:
                    self._json(step["error"], {"error": {"message": step["message"], "type": "invalid_request_error"}})
                    return
                calls = [{"id": "call_" + uuid.uuid4().hex[:8], "type": "function",
                          "function": {"name": c["name"], "arguments": c["arguments"] if isinstance(c["arguments"], str)
                                       else json.dumps(c["arguments"])}, **c.get("extra", {})}
                         for c in step["tool_calls"]]
                finish = step["finish"] or ("tool_calls" if calls else "stop")
                # realistic default: ~4 chars per token over the whole request
                ptoks = step["prompt_tokens"] or max(1, len(json.dumps(body["messages"])) // 4)
                usage = {"prompt_tokens": ptoks, "completion_tokens": 10, "total_tokens": ptoks + 10}
                if step.get("cached_tokens"):   # OpenAI-style prompt caching report
                    usage["prompt_tokens_details"] = {"cached_tokens": step["cached_tokens"]}
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    for k, v in server.headers.items():
                        self.send_header(k, v)
                    self.end_headers()

                    time.sleep(step.get("think") or 0)

                    def send(obj):
                        if server.chunk_delay:
                            time.sleep(server.chunk_delay)
                        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                        self.wfile.flush()

                    base = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": body["model"]}
                    text = step["content"]
                    for i in range(0, len(text), 7):
                        send({**base, "choices": [{"index": 0, "delta": {"content": text[i:i + 7]},
                                                   "finish_reason": None}]})
                    for idx, c in enumerate(calls):
                        args = c["function"]["arguments"]
                        send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                            {"index": idx, "id": c["id"], "type": "function",
                             "function": {"name": c["function"]["name"], "arguments": args[: len(args) // 2]},
                             **{k: v for k, v in c.items() if k not in ("id", "type", "function")}}]},
                            "finish_reason": None}]})
                        send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                            {"index": idx, "function": {"arguments": args[len(args) // 2:]}}]},
                            "finish_reason": None}]})
                    send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                    send({**base, "choices": [], "usage": usage})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
                message = {"role": "assistant", "content": step["content"] or None}
                if calls:
                    message["tool_calls"] = calls
                self._json(200, {"id": "c1", "object": "chat.completion", "created": 0, "model": body["model"],
                                 "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                                 "usage": usage})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> FakeOpenAI:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
