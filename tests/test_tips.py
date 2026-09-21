"""Tips under the spinner come from what is installed: new commands, skills, tools and MCP servers included."""

import io
import time

from fakeserver import FakeOpenAI
from rich.console import Console
from test_agent_e2e import make_app

from muyah_code.ui.commands import Command
from muyah_code.ui.terminal import TerminalUI
from muyah_code.ui.tips import TipRotation, build_tips


def test_tips_follow_what_is_installed(project):
    skill = project / ".muyah" / "skills" / "deploy-staging"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: deploy-staging\ndescription: Ship the branch to staging. Then smoke "
                                    "test it.\n---\nSteps...\n")
    with FakeOpenAI([]) as srv:
        app = make_app(srv, project)
        commands = {"frobnicate": Command("frobnicate", "Do the new thing", lambda a: None, "<x>")}
        tips = build_tips(app, commands)
        app.shutdown()
    assert "/frobnicate <x>: Do the new thing" in tips                        # a command added later
    assert "Skill /deploy-staging: Ship the branch to staging" in tips        # first sentence only
    assert any(t.startswith("The agent's Grep tool:") for t in tips)
    assert any(t.startswith("Sub-agent explore:") for t in tips)


def test_the_spinner_shows_turn_time_tokens_and_a_tip():
    ui = TerminalUI(Console(file=io.StringIO(), width=120))
    ui.tips.reset(["Shift+Tab switches the mode"])
    ui._turn_active, ui._turn_t0, ui._turn_chars = True, time.monotonic() - 95, 327000
    ui._phase, ui._t0 = "thinking", time.monotonic()
    out = io.StringIO()
    Console(file=out, width=120, color_system=None).print(ui._StatsRenderable(ui))
    text = out.getvalue()
    assert "1m 35s · ↓ 93.4k tokens" in text and "⎿ Tip: Shift+Tab switches the mode" in text


def test_tips_rotate():
    r = TipRotation()
    r.reset(["a", "b", "c"])
    seen = {r.at(i * TipRotation.SECONDS) for i in range(3)}
    assert seen == {"a", "b", "c"}
