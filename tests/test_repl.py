"""Drive the real interactive REPL (prompt, "/" menu, arrow-key selects, Ctrl+C) with piped keystrokes."""

import io

from fakeserver import FakeOpenAI, reply
from prompt_toolkit.application import create_app_session
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from muyah_code.app import App
from muyah_code.config import load_config
from muyah_code.ui.repl import Repl, _Completer
from muyah_code.ui.terminal import TerminalUI

ENTER, DOWN, CTRL_C = "\r", "\x1b[B", "\x03"


def run_session(project, keys, script, lines=True, **app_kwargs):
    """keys: lines of text (Enter appended) or, with lines=False, raw keystrokes."""
    out = io.StringIO()
    console = Console(file=out, width=100, force_terminal=False, color_system=None)
    ui = TerminalUI(console)
    with FakeOpenAI(script) as srv, create_pipe_input() as pipe, \
            create_app_session(input=pipe, output=DummyOutput()):
        cfg = load_config(cwd=project, overrides={"base_url": srv.url, "model": "fake-model"})
        cfg.set("learning.reflect", False)
        app = App(cfg, ui, cwd=project, mode="acceptEdits", enable_mcp=False, **app_kwargs)
        for k in keys:
            pipe.send_text(k + ENTER if lines else k)
        repl = Repl(app, ui)
        code = repl.run()
        app.shutdown()
    return code, out.getvalue(), app


def test_repl_session_end_to_end(project):
    script = [
        reply("", [{"name": "Write", "arguments": {"file_path": "notes.md", "content": "# Notes\n"}}]),
        reply("Created **notes.md**."),
    ]
    lines = ["/help", "/context", "/status", "/mode plan", "/mode acceptEdits", "create notes.md", "/todos",
             "/undo", "#always use type hints", "/lessons", "/theme ocean", "/cost", "/exit"]
    code, out, app = run_session(project, lines, script)
    assert code == 0
    assert "/compact" in out                      # help listed commands
    assert "Context:" in out                      # /context
    assert "endpoint" in out and "fake-model" in out   # /status has the details the header no longer shows
    assert "Mode: plan" in out and "Mode: acceptEdits" in out
    assert "Created notes.md" in out              # markdown rendered (bold markers stripped)
    assert "1 file changed" in out                # turn footer
    assert "Undid changes" in out and not (project / "notes.md").exists()
    assert "always use type hints" in (project / "MUYAH.md").read_text()
    assert "Theme set to ocean" in out
    import re

    assert re.search(r"This session\s+2\s", out)      # /cost (= /usage): 2 model requests this session
    assert re.search(r"Today\s+2\s", out)             # and they were recorded in the usage log


def test_header_is_compact(project):
    code, out, app = run_session(project, ["/exit"], [])
    header = out.split("Bye.")[0].split("› ")[0].strip().splitlines()
    assert len(header) == 3  # mascot beside: name + version, model + provider, folder - nothing else
    assert "MUYAH-CODE v" in header[0] and "fake-model" in header[1] and header[2].rstrip().endswith("proj")
    assert "███╗" not in out  # the old big banner is gone


def test_header_never_wraps_on_long_paths(tmp_path):
    deep = tmp_path.joinpath(*["a-very-long-folder-name"] * 8, "my-project")
    deep.mkdir(parents=True)
    (deep / ".muyah").mkdir()
    code, out, app = run_session(deep, ["/exit"], [])
    header = out.split("Bye.")[0].split("› ")[0].strip().splitlines()
    assert len(header) == 3 and all(len(line) <= 100 for line in header)
    assert header[2].rstrip().endswith("my-project")  # the part of the path that matters is kept


def test_input_field_has_rule_with_folder_and_status_line(project):
    """Render the real prompt into an 80-column fake terminal and read the screen."""
    import re

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output

    class FakeTerminal(Vt100_Output):
        # Real terminals report the rows below the cursor; prompt_toolkit draws the bottom toolbar
        # only once it knows that (Windows consoles answer directly, others via a CPR query).
        def get_rows_below_cursor_position(self) -> int:
            return 20

    screen = io.StringIO()
    term = FakeTerminal(screen, lambda: Size(rows=24, columns=80), term="xterm", enable_cpr=False)
    out = io.StringIO()
    ui = TerminalUI(Console(file=out, width=80, color_system=None))
    with FakeOpenAI([]) as srv, create_pipe_input() as pipe, create_app_session(input=pipe, output=term):
        cfg = load_config(cwd=project, overrides={"base_url": srv.url, "model": "fake-model"})
        app = App(cfg, ui, cwd=project, mode="acceptEdits", enable_mcp=False)
        pipe.send_text("/exit" + ENTER)
        Repl(app, ui).run()
        app.shutdown()
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07", "", screen.getvalue())
    lines = text.replace("\r", "").split("\n")
    at = next(i for i, ln in enumerate(lines) if ln.startswith("❯"))
    assert set(lines[at - 1].strip()) == {"─"} and len(lines[at - 1].strip()) >= 70   # rule above the input
    below = [ln for ln in lines[at + 1:] if ln.strip()]
    assert set(below[0].strip()) == {"─"}                                          # rule below the input
    assert "edit" in below[1] and "accepts file edits" in below[1] and "(shift+tab)" in below[1]  # status line


def test_double_ctrl_c_exits_without_typing_exit(project):
    code, out, app = run_session(project, [CTRL_C, CTRL_C], [], lines=False)
    assert code == 0 and "Bye." in out


def test_ctrl_c_clears_the_line_first(project):
    # typed text + Ctrl+C clears it (no exit); then two Ctrl+C on the empty line exit; nothing was sent
    code, out, app = run_session(project, ["half-typed prompt", CTRL_C, CTRL_C, CTRL_C], [], lines=False)
    assert code == 0 and len(app.agent.messages) == 1


def test_slash_menu_shows_descriptions(project):
    code, out, app = run_session(project, ["/exit"], [])
    repl = Repl.__new__(Repl)
    repl.app = app
    from muyah_code.ui.commands import CommandRouter

    repl.router = CommandRouter(repl)
    items = list(_Completer(repl).get_completions(Document("/pro"), None))
    names = {c.text: c.display_meta_text for c in items}
    assert "/provider" in names and "API key" in names["/provider"]
    assert "/profile" in names
    skills = list(_Completer(repl).get_completions(Document("/debu"), None))
    assert skills and skills[0].display_meta_text.startswith("skill")


def test_repl_provider_command_uses_arrow_key_menus(project, monkeypatch):
    import muyah_code.provider_setup as provider_setup
    from muyah_code.providers import Provider

    with FakeOpenAI([reply("PONG"), reply("hello from the new provider")],
                    models=[{"id": "cloud-model-1"}, {"id": "cloud-model-2"}], require_key="sk-t-999") as cloud:
        monkeypatch.setattr(provider_setup, "get_provider",
                            lambda _: Provider("cloudy", "Cloudy AI", cloud.url, "", key_hint="sk-t"))
        keys = ["/provider cloudy" + ENTER, "sk-t-999" + ENTER,  # command, pasted key (hidden)
                DOWN + ENTER,                                     # ↓ to the second model, Enter
                "hi" + ENTER, "/exit" + ENTER]
        code, out, app = run_session(project, keys, [], lines=False)
    assert "Key OK" in out and "Ready:" in out
    assert "sk-t-999" not in out  # the key is never echoed
    assert app.llm.base_url == cloud.url and app.llm.model == "cloud-model-1"  # ranked list: 2 is first
    assert "hello from the new provider" in out


def test_repl_unknown_command_and_skill_invocation(project):
    code, out, app = run_session(project, ["/nope", "/debugging the login test fails", "/exit"],
                                 [reply("I'll follow the debugging skill.")])
    assert "Unknown command /nope" in out
    assert "follow the debugging skill" in out
    sent = app.agent.messages[1]["content"]
    assert "Systematic debugging" in sent and "the login test fails" in sent


def test_trust_prompt_blocks_untrusted_folder_until_accepted(tmp_path):
    from muyah_code.trust import ensure_trusted, is_trusted, risky_config

    class Pick:
        def __init__(self, answer):
            self.answer, self.asked = answer, []

        def select(self, message, options, default=None):
            self.asked.append([label for _, label in options])
            return self.answer

    home, folder = tmp_path / "home", tmp_path / "work" / "repo"
    (folder / ".muyah").mkdir(parents=True)
    (folder / ".mcp.json").write_text("{}")
    (folder / ".muyah" / "settings.json").write_text('{"hooks": {}}')
    out = io.StringIO()
    console = Console(file=out, width=100, color_system=None)
    no = Pick("no")
    assert ensure_trusted(folder, home, console, no) is False and not is_trusted(home, folder)
    assert no.asked == [["No, exit", "Yes, I trust this folder"]]
    assert "Accessing workspace:" in out.getvalue() and "defines hooks" in out.getvalue()
    assert len(risky_config(folder)) == 2
    assert ensure_trusted(folder, home, console, Pick("yes")) is True
    asked_again = Pick("no")
    assert ensure_trusted(folder / "sub", home, console, asked_again) is True  # subfolders inherit trust
    assert asked_again.asked == []


def test_resize_redraws_everything_and_keeps_typed_text(project):
    """Shrink the terminal while typing: the conversation is cleared and redrawn at the new width
    (not left wrapped at the old one) and the half-typed prompt is restored."""
    import threading
    import time

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output

    from muyah_code.ui.terminal import CLEAR_SCREEN, ReplayConsole

    size = {"cols": 100}

    class FakeTerminal(Vt100_Output):
        def get_rows_below_cursor_position(self) -> int:
            return 10

    screen = io.StringIO()
    term = FakeTerminal(screen, lambda: Size(rows=30, columns=size["cols"]), term="xterm", enable_cpr=False)
    console = ReplayConsole(file=screen, width=100, force_terminal=True, color_system=None)
    ui = TerminalUI(console, animate=False)
    with FakeOpenAI([reply("hello")]) as srv, create_pipe_input() as pipe, create_app_session(input=pipe, output=term):
        cfg = load_config(cwd=project, overrides={"base_url": srv.url, "model": "fake-model"})
        app = App(cfg, ui, cwd=project, mode="acceptEdits", enable_mcp=False)

        def drive():
            pipe.send_text("say hi" + ENTER)
            time.sleep(1.0)
            pipe.send_text("half-typed")
            time.sleep(0.3)
            before = screen.tell()
            size["cols"] = 60        # user shrinks the window
            console.width = 60
            time.sleep(1.0)          # watcher notices, waits for the size to settle, redraws
            state["redrawn"] = CLEAR_SCREEN in screen.getvalue()[before:]
            pipe.send_text(ENTER)    # submits the restored "half-typed" text
            time.sleep(0.8)
            pipe.send_text("/exit" + ENTER)

        state = {}
        threading.Thread(target=drive, daemon=True).start()
        Repl(app, ui).run()
        app.shutdown()
    assert state.get("redrawn") is True
    after_clear = screen.getvalue().split(CLEAR_SCREEN)[-1]
    assert "say hi" in after_clear and "hello" in after_clear  # the conversation was redrawn
    assert any(m.get("content") == "half-typed" for m in app.agent.messages)  # typed text survived


def test_resume_shows_the_earlier_conversation(project):
    first = [reply("", [{"name": "Glob", "arguments": {"pattern": "*.py"}}]), reply("There are **no** Python files.")]
    _, _, app = run_session(project, ["list the python files", "/exit"], first)
    session_id = app.session.id

    code, out, app2 = run_session(project, ["/exit"], [], resume=session_id)
    assert code == 0 and app2.session.id == session_id
    assert "list the python files" in out         # your prompt
    assert "Glob(*.py)" in out                     # the tool call
    assert "There are no Python files." in out     # the answer, rendered
    assert "resumed" in out


def test_resume_command_without_id_opens_a_picker(project):
    _, _, first = run_session(project, ["hello there", "/exit"], [reply("Hi!")])
    # /resume, then Enter picks the most recent conversation in the arrow-key list
    code, out, app = run_session(project, ["/resume", "", "/exit"], [])
    assert app.session.id == first.session.id
    assert "hello there" in out and "Hi!" in out


def test_pick_session_lists_recent_sessions_newest_first(tmp_path):
    import os
    import time

    from test_providers import Answers

    from muyah_code.session import Session
    from muyah_code.ui.history import ago, pick_session

    assert pick_session(tmp_path / "none", Answers()) is None
    old = Session(tmp_path, "20260101-000000-aaaaaa")
    old.log_message({"role": "user", "content": "first task"})
    new = Session(tmp_path, "20260102-000000-bbbbbb")
    new.log_message({"role": "user", "content": "second task"})
    old.close()
    new.close()                                            # writes are queued; finish them first
    os.utime(old.path, (time.time() - 7200, time.time() - 7200))
    user = Answers()
    assert pick_session(tmp_path, user) == new.id          # Enter = newest
    kind, message, ids = user.asked[0]
    assert kind == "select" and ids == [new.id, old.id]
    assert ago(time.time() - 7200) == "2 hours ago" and ago(time.time()) == "just now"


def test_model_request_can_be_interrupted_while_the_server_is_silent():
    import threading
    import time

    import pytest

    from muyah_code.agent.loop import interruptible_call

    release = threading.Event()
    streamed = []

    def slow_chat(messages, on_text=None, on_reasoning=None, **kw):
        release.wait(5)            # the server is "thinking": a blocked network read
        on_text("late text")       # arrives after the user interrupted: must be dropped
        return "done"

    threading.Timer(0.2, lambda: __import__("_thread").interrupt_main()).start()
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        interruptible_call(slow_chat, [], on_text=streamed.append)
    assert time.monotonic() - t0 < 1.0            # not after the server's 5 s
    release.set()
    time.sleep(0.2)
    assert streamed == []
    assert interruptible_call(lambda m, **kw: "ok", []) == "ok"


def test_typing_while_working_queues_and_esc_sends_now(monkeypatch):
    ui = TerminalUI(Console(file=io.StringIO(), width=100, force_terminal=False, color_system=None), animate=False)
    interrupts = []
    monkeypatch.setattr("_thread.interrupt_main", lambda: interrupts.append(1))
    ui.begin_typing()

    class Reader:  # stands in for the real key reader (tests have no terminal)
        def stop(self): ...
        def pause(self): ...
        def resume(self): ...
    ui._reader = Reader()
    modes = []
    ui.on_mode_cycle = lambda: modes.append("cycled")
    ui.input_status = lambda: ("⏵⏵ auto", "runs on its own", "#facc15")
    for ch in "also run the tests":
        ui._on_key(ch)
    ui._on_key("backspace")
    ui._on_key("s")
    from rich.text import Text

    shown = io.StringIO()
    Console(file=shown, width=120, color_system=None).print(ui._with_typing(Text("status")))
    screen = shown.getvalue()
    assert "❯ also run the tests▌" in screen                  # the input box stays on screen while it works
    assert "⏵⏵ auto" in screen and "(shift+tab)" in screen    # with the mode status line under it
    ui._on_key("shift-tab")
    assert modes == ["cycled"]                                # Shift+Tab changes the mode mid-turn
    ui._on_key("enter")
    assert ui._queued == ["also run the tests"] and ui._draft == ""
    assert ui.take_queued() == ["also run the tests"] and ui._queued == []   # delivered at the next step
    ui._on_key("x")
    ui._on_key("esc")                                                      # send now
    assert interrupts == [1]
    queued, draft = ui.end_typing()
    assert queued == ["x"] and draft == ""


def test_startup_offers_the_live_view_and_remembers_always_or_never(project, monkeypatch):
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        _check_startup_offer(project, monkeypatch)


def _check_startup_offer(project, monkeypatch):
    from test_providers import Answers

    opened = []
    for answer, expected_setting, opens in (("never", False, False), ("always", True, True), ("no", None, False)):
        cfg = load_config(cwd=project, overrides={"base_url": "http://127.0.0.1:9/v1", "model": "m"})
        cfg.persist("viz.autostart", None)
        ui = TerminalUI(Console(file=io.StringIO(), width=100, color_system=None), animate=False)
        app = App(cfg, ui, cwd=project, mode="default", enable_mcp=False)
        repl = Repl(app, ui, prompter=Answers(answer))
        monkeypatch.setattr(repl.router, "viz", lambda arg, a=answer: opened.append(a))
        repl.offer_viz()
        assert load_config(cwd=project).get("viz.autostart") == expected_setting
        assert (answer in opened) == opens
        app.shutdown()
    cfg.persist("viz.autostart", False)          # "Never": no question at all
    asked = Answers()
    ui = TerminalUI(Console(file=io.StringIO(), width=100, color_system=None), animate=False)
    app = App(load_config(cwd=project, overrides={"base_url": "http://127.0.0.1:9/v1", "model": "m"}), ui,
              cwd=project, mode="default", enable_mcp=False)
    Repl(app, ui, prompter=asked).offer_viz()
    assert asked.asked == []
    app.shutdown()
