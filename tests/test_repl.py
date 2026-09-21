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


def run_session(project, keys, script, lines=True):
    """keys: lines of text (Enter appended) or, with lines=False, raw keystrokes."""
    out = io.StringIO()
    console = Console(file=out, width=100, force_terminal=False, color_system=None)
    ui = TerminalUI(console)
    with FakeOpenAI(script) as srv, create_pipe_input() as pipe, \
            create_app_session(input=pipe, output=DummyOutput()):
        cfg = load_config(cwd=project, overrides={"base_url": srv.url, "model": "fake-model"})
        cfg.set("learning.reflect", False)
        app = App(cfg, ui, cwd=project, mode="acceptEdits", enable_mcp=False)
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
    assert "Requests: 2" in out


def test_header_is_compact(project):
    code, out, app = run_session(project, ["/exit"], [])
    header = out.split("Bye.")[0].strip().splitlines()
    assert header[0].startswith("✻ MUYAH-CODE")
    assert len(header) <= 3  # name, model + folder, one hint - no banner, no stats box
    assert "███" not in out


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
