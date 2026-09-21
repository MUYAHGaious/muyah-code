"""A tiny local web server for the visualization page.

    live    VizServer(bus=app.events)       streams events as the agent works (`/viz` in a session)
    replay  VizServer(events=load_events()) serves a recorded session; the page plays it back (`muyah viz`)

Security: it only listens on 127.0.0.1, every request needs the random token from the URL, and the Host
header must be the loopback address (so a web page cannot reach it through DNS rebinding). The page is
self-contained: no external scripts, fonts or requests.
"""

from __future__ import annotations

import hmac
import json
import queue
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from muyah_code.events import EventBus

PAGE = Path(__file__).with_name("page.html")
KEEPALIVE = 15.0  # seconds between SSE comments, so proxies and browsers keep the stream open


def find_events_file(directory: Path, session_id: str | None) -> Path | None:
    """The recorded events of a session: an id (prefix ok), or None / "last" for the most recent one."""
    if not directory.is_dir():
        return None
    files = sorted(directory.glob("*.events.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not session_id or session_id == "last":
        return files[0] if files else None
    matches = sorted(p for p in files if p.name.startswith(session_id))
    return matches[0] if matches else None


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    last_error: BaseException | None = None

    def handle_error(self, request, client_address):
        # The default prints a traceback to stderr, which would land on top of the terminal UI. A browser
        # tab closing mid-request (connection reset) is normal; anything else is kept for inspection.
        exc = sys.exc_info()[1]
        if not isinstance(exc, OSError):
            self.last_error = exc


class VizServer:
    def __init__(self, bus: EventBus | None = None, events: list[dict] | None = None, title: str = "",
                 host: str = "127.0.0.1", port: int = 0, following: str = ""):
        if (bus is None) == (events is None):
            raise ValueError("give either a live event bus or recorded events")
        self.bus = bus
        self.events = events
        self.title = title
        self.following = following  # the folder a follower watches ("" for a session's own /viz)
        self.token = secrets.token_urlsafe(18)
        self._stopping = threading.Event()
        self._httpd = _HTTPServer((host, port), self._handler())
        self.host, self.port = self._httpd.server_address[:2]
        self._thread: threading.Thread | None = None

    @property
    def mode(self) -> str:
        return "live" if self.bus is not None else "replay"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/?t={self.token}"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> str:
        if not self.running:
            self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.2},
                                            name="muyah-viz", daemon=True)
            self._thread.start()
        return self.url

    def serve_forever(self) -> None:
        self._httpd.serve_forever(poll_interval=0.2)

    def stop(self) -> None:
        self._stopping.set()
        if self.running:
            self._httpd.shutdown()
        self._httpd.server_close()

    # ------------------------------------------------------------------ HTTP

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):  # noqa: A002 - keep the terminal UI clean
                return

            def _allowed(self) -> bool:
                host = (self.headers.get("Host") or "").lower()
                if host not in (f"127.0.0.1:{server.port}", f"localhost:{server.port}"):
                    return False
                token = parse_qs(urlparse(self.path).query).get("t", [""])[0]
                return hmac.compare_digest(token.encode(), server.token.encode())

            def _send(self, code: int, body: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802 - http.server naming
                if not self._allowed():
                    self._send(403, b"Forbidden", "text/plain; charset=utf-8")
                    return
                route = urlparse(self.path).path
                if route == "/":
                    body = PAGE.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Security-Policy",
                                     "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                                     "connect-src 'self'; img-src data:")
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Referrer-Policy", "no-referrer")
                    self.end_headers()
                    self.wfile.write(body)
                elif route == "/events":
                    self._stream()
                else:
                    self._send(404, b"Not found", "text/plain; charset=utf-8")

            def _write_event(self, data: dict, name: str | None = None) -> None:
                payload = json.dumps(data, ensure_ascii=False, default=str)
                chunk = (f"event: {name}\n" if name else "") + f"data: {payload}\n\n"
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()

            def _stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    self._write_event({"mode": server.mode, "title": server.title, "following": server.following},
                                      "hello")
                    if server.bus is None:
                        for ev in server.events or []:
                            self._write_event(ev)
                        self._write_event({}, "end")
                        return
                    inbox: queue.Queue = queue.Queue()
                    unsubscribe = server.bus.subscribe(inbox.put)
                    try:
                        # history and subscription overlap for an instant: skip events already sent
                        backlog = list(server.bus.history)
                        seen = {id(ev) for ev in backlog}
                        for ev in backlog:
                            self._write_event(ev)
                        self._write_event({}, "caughtup")
                        idle = 0.0
                        while not server._stopping.is_set():
                            try:
                                ev = inbox.get(timeout=0.5)
                            except queue.Empty:
                                idle += 0.5
                                if idle >= KEEPALIVE:
                                    self.wfile.write(b": keepalive\n\n")
                                    self.wfile.flush()
                                    idle = 0.0
                                continue
                            idle = 0.0
                            if id(ev) in seen:
                                seen.discard(id(ev))
                                continue
                            self._write_event(ev)
                    finally:
                        unsubscribe()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return  # the browser tab was closed

        return Handler
