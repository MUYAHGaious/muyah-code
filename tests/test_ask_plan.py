"""Ask mode (talk it through, change nothing), plan approval (ExitPlanMode), streamed compaction progress,
and voice input (record + Whisper) with a fake microphone and a fake transcription server."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from fakeserver import FakeOpenAI, reply
from test_agent_e2e import RecUI, make_app

from muyah_code.permissions import CYCLE, PermissionManager
from muyah_code.ui import voice


class PlanUI(RecUI):
    def __init__(self, choice):
        super().__init__()
        self.choice = choice
        self.plans = []
        self.questions = []

    def approve_plan(self, plan, options):
        self.plans.append(plan)
        return self.ask_user("Would you like to proceed?", options)

    def ask_user(self, question, options):
        self.questions.append((question, options))
        return next((o for o in options if o.startswith(self.choice)), self.choice)


def test_shift_tab_order_puts_ask_before_plan():
    assert CYCLE == ("default", "acceptEdits", "ask", "plan", "auto")
    pm = PermissionManager(mode="acceptEdits")
    assert [pm.cycle_mode() for _ in range(3)] == ["ask", "plan", "auto"]
    assert PermissionManager(mode="brainstorm").mode == "ask"


def test_ask_mode_talks_but_changes_nothing(project):
    write = {"name": "Write", "arguments": {"file_path": "idea.txt", "content": "x"}}
    read = {"name": "Read", "arguments": {"file_path": "README.md"}}
    (project / "README.md").write_text("hello\n", encoding="utf-8")
    with FakeOpenAI([reply("", [read]), reply("", [write]), reply("What should happen on failure?")]) as srv:
        ui = RecUI()
        app = make_app(srv, project, ui=ui, mode="ask")
        app.run_prompt("let's think about a retry feature")
        app.shutdown()
    system = srv.requests[0]["messages"][0]["content"]
    assert "ASK MODE IS ACTIVE" in system
    tools = [m["content"] for m in srv.requests[-1]["messages"] if m["role"] == "tool"]
    assert "hello" in tools[0]                                   # reading is fine
    assert "ask mode" in tools[1].lower()                        # writing is refused, without asking
    assert not (project / "idea.txt").exists()
    assert not [e for e in ui.events if e[0] == "ask"]


def plan_run(project, choice, after):
    plan = {"name": "ExitPlanMode", "arguments": {"plan": "## Plan\n1. add retry.py\n2. test it"}}
    write = {"name": "Write", "arguments": {"file_path": "retry.py", "content": "RETRIES = 3\n"}}
    script = [reply("", [plan])] + ([reply("", [write])] if after else []) + [reply("done")]
    with FakeOpenAI(script) as srv:
        ui = PlanUI(choice)
        app = make_app(srv, project, ui=ui, mode="plan")
        app.run_prompt("add retries")
        mode = app.permissions.mode
        app.shutdown()
    tools = [m["content"] for m in srv.requests[-1]["messages"] if m["role"] == "tool"]
    return ui, mode, tools, srv


def test_an_approved_plan_switches_mode_and_the_work_starts(project):
    ui, mode, tools, srv = plan_run(project, "Yes, auto", after=True)
    assert ui.plans == ["## Plan\n1. add retry.py\n2. test it"]
    question, options = ui.questions[0]
    assert options[-1] == "No, keep planning" and any("manual" in o for o in options)
    assert mode == "auto" and "Implement the plan now" in tools[0]
    assert (project / "retry.py").read_text(encoding="utf-8") == "RETRIES = 3\n"   # built in the same turn
    assert "PLAN MODE" not in srv.requests[-1]["messages"][0]["content"]         # the prompt followed the switch


def test_keep_planning_stays_read_only_and_passes_on_what_the_user_said(project):
    ui, mode, tools, _ = plan_run(project, "use exponential backoff", after=False)
    assert mode == "plan"
    assert "keep planning" in tools[0] and "exponential backoff" in tools[0]


def test_exit_plan_mode_outside_plan_mode_is_a_no_op(project):
    plan = {"name": "ExitPlanMode", "arguments": {"plan": "x"}}
    with FakeOpenAI([reply("", [plan]), reply("ok")]) as srv:
        ui = PlanUI("Y")
        app = make_app(srv, project, ui=ui, mode="acceptEdits")
        app.run_prompt("go")
        app.shutdown()
    assert not ui.questions and app.permissions.mode == "acceptEdits"


def test_compaction_progress_counts_the_summary_as_it_streams():
    from muyah_code.agent.context import ContextManager

    class StreamingLLM:
        def chat(self, messages, on_text=None, **kw):
            for _ in range(10):
                on_text("x" * 70)                               # ~20 tokens a chunk
            from muyah_code.llm.client import AssistantMessage
            return AssistantMessage(content="the summary")

    cm = ContextManager(window=32000, max_output=4000)
    seen = []
    head = [{"role": "user", "content": f"message {i} " * 30} for i in range(6)]
    assert cm._summarize(head, StreamingLLM(), "", on_progress=lambda w, lim: seen.append((w, lim))) == "the summary"
    written = [w for w, _ in seen]
    limit = seen[0][1]
    assert seen[0] == (0, limit) and written == sorted(written) and written[-1] == 200 and limit == 2000


# ---------------------------------------------------------------------------- voice


class FakeStream:
    def __init__(self, callback, **kw):
        self.callback = callback
        self.kw = kw

    def start(self):
        loud = (b"\x00\x40" * 1600)                            # speech
        self.callback(loud, 1600, None, None)

    def stop(self):
        pass

    def close(self):
        pass


class Cfg(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def test_recording_then_whisper_puts_the_words_in_the_prompt(tmp_path):
    got, notes = [], []
    done = threading.Event()
    v = voice.VoiceInput(Cfg({"voice.engine": "whisper", "voice.url": "http://x"}), tmp_path,
                         recorder_factory=lambda on_auto_stop: voice.Recorder(on_auto_stop, stream_factory=FakeStream),
                         transcriber=lambda wav: "add a retry" if wav[:4] == b"RIFF" else "")
    msg = v.toggle(lambda t: (got.append(t), done.set()), notes.append)
    assert "Listening" in msg and v.state == "recording"
    assert v.toggle(lambda t: None, notes.append) == "Transcribing…"        # the key again stops it
    assert done.wait(3) and got == ["add a retry"] and v.state == ""


def test_whisper_without_a_service_says_how_to_get_one(tmp_path, monkeypatch):
    monkeypatch.setattr("muyah_code.providers.resolve_key", lambda home, name: None)
    v = voice.VoiceInput(Cfg({"voice.engine": "whisper"}), tmp_path)
    assert "Groq" in v.toggle(lambda t: None, lambda n: None) and v.state == ""


def test_engine_auto_is_windows_voice_typing_on_windows_else_whisper(tmp_path, monkeypatch):
    monkeypatch.setattr(voice, "windows_dictation_available", lambda: True)
    assert voice.VoiceInput(Cfg(), tmp_path).engine() == "windows"
    monkeypatch.setattr(voice, "windows_dictation_available", lambda: False)
    assert voice.VoiceInput(Cfg(), tmp_path).engine() == "whisper"
    assert voice.VoiceInput(Cfg({"voice.engine": "local"}), tmp_path).engine() == "local"


def test_silence_after_speech_stops_recording_by_itself(monkeypatch):
    monkeypatch.setattr(voice, "SILENCE_STOP", 0.05)
    stopped = threading.Event()
    rec = voice.Recorder(on_auto_stop=stopped.set, stream_factory=FakeStream)
    rec.start()
    time.sleep(0.1)
    rec._callback(b"\x00\x00" * 1600, 1600, None, None)       # quiet starts
    time.sleep(0.1)
    rec._callback(b"\x00\x00" * 1600, 1600, None, None)       # still quiet, long enough
    assert stopped.wait(2)


def test_transcribe_api_posts_the_wav_as_multipart():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen["auth"] = self.headers["Authorization"]
            seen["body"] = body
            out = json.dumps({"text": " hello there "}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        text = voice.transcribe_api(b"RIFFdata", f"http://127.0.0.1:{srv.server_port}/v1/audio/transcriptions",
                                    "key1", "whisper-large-v3-turbo", "en")
    finally:
        srv.shutdown()
    assert text == "hello there" and seen["auth"] == "Bearer key1"
    assert b"whisper-large-v3-turbo" in seen["body"] and b"RIFFdata" in seen["body"] and b'name="language"' in seen["body"]


def test_changes_allowed_without_asking_say_why_under_the_tool(project):
    from muyah_code.agent.loop import allowed_note

    assert allowed_note("auto mode").startswith("Auto-approved · auto mode")
    assert allowed_note("acceptEdits mode") == "Auto-approved · edit mode"
    assert allowed_note("read-only") == ""

    class NoteUI(RecUI):
        def __init__(self):
            super().__init__()
            self.notes = []

        def tool_allowed(self, how):
            self.notes.append(how)

    write = {"name": "Write", "arguments": {"file_path": "n.txt", "content": "n"}}
    read = {"name": "Read", "arguments": {"file_path": "n.txt"}}
    with FakeOpenAI([reply("", [write]), reply("", [read]), reply("ok")]) as srv:
        ui = NoteUI()
        app = make_app(srv, project, ui=ui, mode="auto")
        app.run_prompt("go")
        app.shutdown()
    assert ui.notes == ["Auto-approved · auto mode (risky actions still ask)"]    # the write only, not the read
