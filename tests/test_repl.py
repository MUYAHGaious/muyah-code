"""Drive the real interactive REPL with piped keystrokes."""

import io

from fakeserver import FakeOpenAI, reply
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from muyah_code.app import App
from muyah_code.config import load_config
from muyah_code.ui.repl import Repl
from muyah_code.ui.terminal import TerminalUI


def run_session(project, lines, script):
    out = io.StringIO()
    console = Console(file=out, width=100, force_terminal=False, color_system=None)
    ui = TerminalUI(console)
    with FakeOpenAI(script) as srv, create_pipe_input() as pipe:
        cfg = load_config(cwd=project, overrides={"base_url": srv.url, "model": "fake-model"})
        cfg.set("learning.reflect", False)
        app = App(cfg, ui, cwd=project, mode="acceptEdits", enable_mcp=False)
        for line in lines:
            pipe.send_text(line + "\r")
        repl = Repl(app, ui, input=pipe, output=DummyOutput())
        code = repl.run()
        app.shutdown()
    return code, out.getvalue(), app


def test_repl_session_end_to_end(project):
    script = [
        reply("", [{"name": "Write", "arguments": {"file_path": "notes.md", "content": "# Notes\n"}}]),
        reply("Created **notes.md**."),
    ]
    lines = ["/help", "/context", "/mode plan", "/mode acceptEdits", "create notes.md", "/todos", "/undo",
             "#always use type hints", "/lessons", "/theme ocean", "/cost", "/exit"]
    code, out, app = run_session(project, lines, script)
    assert code == 0
    assert "/compact" in out                      # help listed commands
    assert "Context:" in out                      # /context
    assert "Mode: plan" in out and "Mode: acceptEdits" in out
    assert "Created notes.md" in out              # markdown rendered (bold markers stripped)
    assert "1 file changed" in out                # turn footer
    assert "Undid changes" in out and not (project / "notes.md").exists()
    assert "always use type hints" in (project / "MUYAH.md").read_text()
    assert "Theme set to ocean" in out
    assert "Requests: 2" in out


def test_repl_provider_command_switches_the_session(project, monkeypatch):
    import muyah_code.provider_setup as provider_setup
    from muyah_code.providers import Provider

    with FakeOpenAI([reply("PONG"), reply("hello from the new provider")], models=[{"id": "cloud-model"}],
                    require_key="sk-t-999") as cloud:
        monkeypatch.setattr(provider_setup, "get_provider",
                            lambda _: Provider("cloudy", "Cloudy AI", cloud.url, "cloud-model", key_hint="sk-t"))
        lines = ["/provider cloudy", "sk-t-999", "", "hi", "/exit"]  # command, pasted key, default model, prompt
        code, out, app = run_session(project, lines, [])
    assert "Key OK" in out and "Ready:" in out
    assert "sk-t-999" not in out  # the key is never echoed
    assert app.llm.base_url == cloud.url and app.llm.model == "cloud-model"
    assert "hello from the new provider" in out


def test_repl_unknown_command_and_skill_invocation(project):
    code, out, app = run_session(project, ["/nope", "/debugging the login test fails", "/exit"],
                                 [reply("I'll follow the debugging skill.")])
    assert "Unknown command /nope" in out
    assert "follow the debugging skill" in out
    sent = app.agent.messages[1]["content"]
    assert "Systematic debugging" in sent and "the login test fails" in sent
