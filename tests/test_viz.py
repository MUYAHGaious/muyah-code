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
        script = body.split("<script>")[1].replace("http://www.w3.org/2000/svg", "")  # an SVG namespace id, not a URL
        assert "http://" not in script and "https://" not in script   # self-contained: no external requests
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
    assert got[0] == ("hello", {"mode": "replay", "title": "abc", "following": "", "recording": ""})
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


def test_muyah_viz_replay_without_recordings_explains(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["viz", "--replay", "--no-open"]) == 1
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
    assert main(["viz", "--replay", "--no-open", "--speed", "0.25"]) == 0
    assert started and started[0].mode == "replay" and time.time() - t0 < 5
    assert any(e["type"] == "turn_end" for e in started[0].events)


def wait_for(predicate, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def write_events(path, *events):
    with open(path, "a", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_follower_tails_the_newest_session_and_switches_to_new_ones(tmp_path):
    from muyah_code.viz import SessionFollower

    first = tmp_path / "20260101-000000-aaaaaa.events.jsonl"
    write_events(first, {"type": "session", "t": 0, "model": "m"})
    follower = SessionFollower(tmp_path, poll=0.02)
    follower.start()
    seen = []
    follower.bus.subscribe(seen.append)
    try:
        assert [e["type"] for e in follower.bus.history] == ["reset", "session"]   # backlog before any viewer
        write_events(first, {"type": "turn_start", "t": 1, "prompt": "hi"})
        with open(first, "a", encoding="utf-8") as f:          # an event still being written...
            f.write('{"type": "llm_start", ')
        assert wait_for(lambda: any(e["type"] == "turn_start" for e in seen))
        time.sleep(0.1)
        assert not any(e["type"] == "llm_start" for e in seen)  # ...is not published half-way
        with open(first, "a", encoding="utf-8") as f:
            f.write('"t": 1.5}\n')
        assert wait_for(lambda: any(e["type"] == "llm_start" for e in seen))
        # a new session starts in the folder: the viewer is told to start over, then gets its events
        second = tmp_path / "20260102-000000-bbbbbb.events.jsonl"
        write_events(second, {"type": "session", "t": 0, "model": "m2"})
        assert wait_for(lambda: any(e["type"] == "reset" for e in seen))
        assert wait_for(lambda: any(e.get("model") == "m2" for e in seen))
        assert follower.path == second
        assert [e["type"] for e in follower.bus.history] == ["reset", "session"]
        reset = [e for e in seen if e["type"] == "reset"][0]
        assert reset["session_id"] == "20260102-000000-bbbbbb"
        # the old session keeps writing (another terminal): no flip-flopping back to it
        write_events(first, {"type": "turn_end", "t": 3})
        time.sleep(0.15)
        assert follower.path == second
    finally:
        follower.stop()


def test_muyah_viz_follows_a_session_running_in_another_process(project, monkeypatch):
    """The whole path: an agent writes its events file, `muyah viz` tails it, the page gets them live."""
    from muyah_code.viz import SessionFollower

    monkeypatch.chdir(project)
    started = []
    real_forever = VizServer.serve_forever

    def serve_in_background(self):
        started.append(self)
        threading.Thread(target=real_forever, args=(self,), daemon=True).start()
        assert wait_for(lambda: stop_now.is_set(), timeout=30)
        self._httpd.shutdown()

    stop_now = threading.Event()
    monkeypatch.setattr(VizServer, "serve_forever", serve_in_background)
    viewer = threading.Thread(target=main, args=(["viz", "--no-open"],), daemon=True)
    viewer.start()
    assert wait_for(lambda: started)
    server = started[0]
    assert server.mode == "live" and server.following == str(project) and isinstance(server.bus, EventBus)
    _, resp = request(server, f"/events?t={server.token}")
    hello = read_sse(resp, "caughtup")
    assert hello[0][1]["following"] == str(project)

    script = [reply("", [{"name": "Glob", "arguments": {"pattern": "*"}}], think=0.3), reply("All done here.")]
    with FakeOpenAI(script, chunk_delay=0.01) as srv:
        app = make_app(srv, project)          # "the other terminal": same folder, same sessions dir
        worker = threading.Thread(target=app.run_prompt, args=("what files are here?",))
        worker.start()
        got = read_sse(resp, "message", limit=1)
        types = [d["type"] for _, d in got]
        while "turn_end" not in types:
            nxt = read_sse(resp, "message", limit=1)
            assert nxt, types
            types.append(nxt[-1][1]["type"])
            got += nxt
        worker.join(10)
        app.shutdown()
    stop_now.set()
    viewer.join(10)
    assert types[0] == "reset"
    for kind in ("session", "turn_start", "llm_start", "tool_start", "tool_end", "llm_tokens", "turn_end"):
        assert kind in types, kind
    text = "".join(d.get("text", "") for _, d in got if d.get("type") == "llm_tokens")
    assert "All done here." in text                       # the answer streams to the page as it is written
    assert SessionFollower  # imported for the type check above


def test_resumed_session_keeps_its_clock_going(tmp_path):
    path = tmp_path / "s.events.jsonl"
    write_events(path, {"type": "session", "t": 0}, {"type": "turn_end", "t": 42.5})
    bus = EventBus(record_to=path)
    ev = bus.emit("session")
    assert ev["t"] > 42.5


def test_hooks_and_failed_model_calls_are_events(project, tmp_path):
    import sys

    from fakeserver import error

    from muyah_code.hooks import HookRunner

    script = tmp_path / "hook.py"
    script.write_text("import sys\nsys.exit(0)\n")
    runner = HookRunner({"PreToolUse": [{"matcher": "Bash", "hooks": [{"command": f'"{sys.executable}" "{script}"'}]}]},
                        tmp_path)
    runner.events = bus = EventBus()
    runner.run("PreToolUse", {"tool_name": "Bash", "tool_input": {}}, tool_name="Bash")
    hook = [e for e in bus.history if e["type"] == "hook"][0]
    assert hook["event"] == "PreToolUse" and hook["tool"] == "Bash" and hook["outcome"] == "ok"

    with FakeOpenAI([error(500, "boom")] * 5) as srv:
        app = make_app(srv, project)
        app.llm.max_retries = 0
        app.run_prompt("hi")
        app.shutdown()
    ends = [e for e in app.events.history if e["type"] == "llm_end"]
    assert ends and ends[-1].get("error")                 # the page shows the failure instead of "thinking" forever


def test_mcp_servers_and_calls_show_up(project):
    """End to end over a real stdio MCP server: connection status, a good call and a failing one."""
    import sys
    from pathlib import Path

    from muyah_code.app import App
    from muyah_code.config import load_config

    server = Path(__file__).with_name("fakemcp.py")
    (project / ".mcp.json").write_text(json.dumps({"mcpServers": {"tracker": {
        "command": sys.executable, "args": [str(server)]}}}), encoding="utf-8")
    script = [reply("", [{"name": "mcp__tracker__lookup_issue", "arguments": {"id": "42"}}]),
              reply("", [{"name": "mcp__tracker__fail", "arguments": {}}]),
              reply("Issue 42 is about divide().")]
    with FakeOpenAI(script) as srv:
        cfg = load_config(cwd=project, overrides={"base_url": srv.url, "model": "fake-model", "api_key": "k"})
        cfg.set("learning.reflect", False)
        from test_agent_e2e import RecUI

        app = App(cfg, RecUI(), cwd=project, mode="bypassPermissions", enable_mcp=True)
        try:
            res = app.run_prompt("what is issue 42?")
        finally:
            app.shutdown()
    assert res.status == "ok" and "divide" in res.text
    assert "divide() returns wrong results" in srv.requests[1]["messages"][-1]["content"]   # MCP result reached the model
    events = app.events.history
    session = [e for e in events if e["type"] == "session"][0]
    assert session["mcp"] == [{"name": "tracker", "status": "connected (3 tools)", "tools": ["lookup_issue", "fail", "screenshot"]}]
    assert any(s["name"] == "debugging" for s in session["skills"]) and any(a["name"] == "explore" for a in session["agents"])
    ends = {e["name"]: e["ok"] for e in events if e["type"] == "tool_end"}
    assert ends == {"mcp__tracker__lookup_issue": True, "mcp__tracker__fail": False}


def test_viewer_prefers_a_stable_port_and_falls_back_when_busy():
    from muyah_code.viz.server import PREFERRED_PORT

    first = VizServer(events=[{"type": "session", "t": 0}])
    second = VizServer(events=[{"type": "session", "t": 0}])   # the usual port is taken by the first
    try:
        assert second.port != first.port                  # never two viewers on one address
        assert first.port == PREFERRED_PORT or second.port != PREFERRED_PORT
    finally:
        first.stop()
        second.stop()


def test_recording_can_be_downloaded_and_limits_are_events(project):
    from fakeserver import FakeOpenAI as Fake

    limits = {"x-ratelimit-limit-tokens": "6000", "x-ratelimit-remaining-tokens": "5000",
              "x-ratelimit-reset-tokens": "1m30s"}
    with Fake([reply("done")], headers=limits) as srv:
        app = make_app(srv, project)
        app.run_prompt("hi")
    lim = [e for e in app.events.history if e["type"] == "limits"][0]
    assert lim["rows"] == [{"name": "tokens", "remaining": "5000", "limit": "6000", "reset_s": 90.0}]
    server = VizServer(bus=app.events, title="s", recording=app.events.record_to)
    server.start()
    try:
        _, resp = request(server, f"/events?t={server.token}")
        hello = read_sse(resp, "hello")[0][1]
        assert hello["recording"].endswith(".events.jsonl")
        _, resp = request(server, f"/recording?t={server.token}")
        assert resp.status == 200 and "attachment" in resp.getheader("Content-Disposition")
        lines = resp.read().decode("utf-8").splitlines()
        assert any(json.loads(ln)["type"] == "turn_end" for ln in lines)
        _, resp = request(server, "/recording")                   # the token is required here too
        assert resp.status == 403
    finally:
        server.stop()
        app.shutdown()
