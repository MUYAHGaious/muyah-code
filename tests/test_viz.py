"""The /viz web server (live + replay) and the event stream it shows."""

import http.client
import json
import threading
import time

from fakeserver import FakeOpenAI, reply
from test_agent_e2e import make_app

from muyah_code.cli import main
from muyah_code.events import EventBus, load_events
from muyah_code.viz import VizServer, find_events_file


def request(server, path, host=None):
    conn = http.client.HTTPConnection(server.host, server.port, timeout=5)
    conn.putrequest("GET", path, skip_host=True)
    conn.putheader("Host", host or f"127.0.0.1:{server.port}")
    conn.endheaders()
    return conn, conn.getresponse()


def read_sse(resp, until, limit=200):
    """Parse server-sent events until one named `until` arrives. Returns [(name, data)]."""
    out, name, data = [], None, None
    for _ in range(limit * 3):
        line = resp.fp.readline().decode("utf-8")
        if not line:
            break
        line = line.rstrip("\n")
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            data = json.loads(line[6:])
        elif line == "" and data is not None:
            out.append((name or "message", data))
            if (name or "message") == until:
                break
            name, data = None, None
    return out


def test_requires_token_and_loopback_host(capfd):
    server = VizServer(events=[{"type": "session", "t": 0}])
    server.start()
    try:
        _, resp = request(server, "/")
        assert resp.status == 403
        _, resp = request(server, "/?t=wrong")
        assert resp.status == 403
        # DNS rebinding: right token, but the browser thinks it is talking to another site
        _, resp = request(server, f"/?t={server.token}", host=f"evil.example:{server.port}")
        assert resp.status == 403
        _, resp = request(server, f"/?t={server.token}", host=f"localhost:{server.port}")
        assert resp.status == 200
        body = resp.read().decode("utf-8")
        assert "MUYAH-CODE" in body and "EventSource" in body
        assert "default-src 'none'" in resp.getheader("Content-Security-Policy")
        assert "http://" not in body.split("<script>")[1]   # self-contained: no external requests
        _, resp = request(server, f"/nope?t={server.token}")
        assert resp.status == 404
    finally:
        server.stop()
    assert server.host == "127.0.0.1"
    time.sleep(0.2)
    assert "Traceback" not in capfd.readouterr().err   # dropped connections never print over the terminal UI


def test_replay_streams_recorded_events_then_ends():
    events = [{"type": "session", "t": 0, "model": "m"}, {"type": "turn_start", "t": 1, "prompt": "hi"},
              {"type": "turn_end", "t": 2, "status": "ok"}]
    server = VizServer(events=events, title="abc")
    server.start()
    try:
        _, resp = request(server, f"/events?t={server.token}")
        assert resp.getheader("Content-Type").startswith("text/event-stream")
        got = read_sse(resp, "end")
    finally:
        server.stop()
    assert got[0] == ("hello", {"mode": "replay", "title": "abc"})
    assert [d for n, d in got if n == "message"] == events
    assert got[-1][0] == "end"


def test_live_stream_sends_backlog_then_new_events():
    bus = EventBus()
    bus.emit("session", model="m")
    server = VizServer(bus=bus)
    server.start()
    try:
        _, resp = request(server, f"/events?t={server.token}")
        first = read_sse(resp, "caughtup")
        assert first[0][1]["mode"] == "live"
        assert [d["type"] for n, d in first if n == "message"] == ["session"]
        threading.Timer(0.2, lambda: bus.emit("tool_start", id=1, name="Read", title="Read(a.py)")).start()
        nxt = read_sse(resp, "message")
        assert nxt[-1][1]["type"] == "tool_start" and nxt[-1][1]["name"] == "Read"
    finally:
        server.stop()
    assert not server.running


def test_a_turn_records_the_events_the_page_draws(project):
    script = [reply("", [{"name": "Glob", "arguments": {"pattern": "*"}}]), reply("Nothing here.")]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project)
        app.run_prompt("what files are here?")
        app.shutdown()
    path = app.session.path.with_suffix(".events.jsonl")
    types = [e["type"] for e in load_events(path)]
    for kind in ("session", "context", "turn_start", "llm_start", "llm_end", "tool_start", "tool_end", "turn_end"):
        assert kind in types, kind
    assert types.index("turn_start") < types.index("tool_start") < types.index("tool_end") < types.index("turn_end")
    ends = [e for e in load_events(path) if e["type"] == "tool_end"]
    assert ends[0]["name"] == "Glob" and ends[0]["ok"] is True
    assert find_events_file(path.parent, "last") == path
    assert find_events_file(path.parent, app.session.id[:10]) == path
    assert find_events_file(path.parent, "nope") is None


def test_viz_command_serves_this_session_live(project, monkeypatch):
    from test_repl import run_session

    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    code, out, app = run_session(project, ["/viz", "/viz", "/viz stop", "/viz stop", "/exit"], [])
    assert code == 0
    assert "Live view: http://127.0.0.1:" in out and len(opened) == 2
    assert opened[0] == opened[1]           # a second /viz reuses the running server
    assert "Visualization stopped." in out and "not running" in out
    assert app.viz is None


def test_muyah_viz_without_recordings_explains(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["viz", "--no-open"]) == 1
    assert "No recorded session" in capsys.readouterr().out


def test_muyah_viz_replays_the_last_session(project, monkeypatch):
    script = [reply("Hello!")]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project)
        app.run_prompt("hi")
        app.shutdown()
    monkeypatch.chdir(project)
    started = []
    real_forever = VizServer.serve_forever

    def serve_briefly(self):
        started.append(self)
        threading.Timer(0.3, self._httpd.shutdown).start()
        real_forever(self)

    monkeypatch.setattr(VizServer, "serve_forever", serve_briefly)
    t0 = time.time()
    assert main(["viz", "--no-open", "--speed", "4"]) == 0
    assert started and started[0].mode == "replay" and time.time() - t0 < 5
    assert any(e["type"] == "turn_end" for e in started[0].events)
