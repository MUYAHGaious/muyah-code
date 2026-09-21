"""Nested instructions loaded on use, skills' allowed-tools, prompt hooks, /mcp enable/disable, ~/.claude.json."""

import json
import sys
from pathlib import Path

from fakeserver import FakeOpenAI, reply
from test_agent_e2e import RecUI, make_app

import muyah_code.mcp.client as mcp_client
from muyah_code.app import App
from muyah_code.config import load_config

FAKE = Path(__file__).with_name("fakemcp.py")


def tool_results(request):
    return [m["content"] for m in request["messages"] if m["role"] == "tool"]


def test_subfolder_instructions_arrive_with_the_first_file_there(project):
    (project / "pkg").mkdir()
    (project / "pkg" / "AGENTS.md").write_text("In pkg/: use tabs, never spaces.")
    (project / "pkg" / "a.py").write_text("x = 1\n")
    (project / "pkg" / "b.py").write_text("y = 2\n")
    script = [reply("", [{"name": "Read", "arguments": {"file_path": "pkg/a.py"}}]),
              reply("", [{"name": "Read", "arguments": {"file_path": "pkg/b.py"}}]), reply("ok")]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project)
        app.run_prompt("look at pkg")
        system_before = srv.requests[0]["messages"][0]["content"]
        app.shutdown()
    first, second = tool_results(srv.requests[2])
    assert '<instructions path="pkg/AGENTS.md">' in first and "use tabs" in first
    assert "use tabs" not in second                                   # only once
    assert "use tabs" not in system_before                            # the system prompt stays cache-stable
    assert srv.requests[2]["messages"][0]["content"] == system_before


def test_a_skills_allowed_tools_run_without_asking(project):
    skill = project / ".muyah" / "skills" / "release"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: release\ndescription: Cut a release.\nallowed-tools: Bash(echo *)\n"
                                    "---\nRun echo to announce the release.\n")
    script = [reply("", [{"name": "Skill", "arguments": {"skill": "release"}}]),
              reply("", [{"name": "Bash", "arguments": {"command": "echo shipping"}}]), reply("Released.")]
    with FakeOpenAI(script) as srv:
        ui = RecUI()
        app = make_app(srv, project, ui=ui, mode="default")
        res = app.run_prompt("release it")
        app.shutdown()
    assert not [e for e in ui.events if e[0] == "ask"]                 # manual mode, yet no approval asked
    assert "Allowed without asking for this session, from this skill: Bash(echo *)" in tool_results(srv.requests[1])[0]
    assert res.text == "Released."


def test_prompt_hooks_let_a_model_hold_the_agent_to_a_rule(project):
    (project / ".muyah" / "settings.json").write_text(json.dumps({"hooks": {"Stop": [{"hooks": [
        {"type": "prompt", "prompt": "The agent must run the tests before it stops."}]}]}}))
    script = [reply("Done."), reply('{"ok": false, "reason": "You did not run the tests."}'),
              reply("I ran the tests: all pass."), reply('{"ok": true}')]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project)
        res = app.run_prompt("fix it")
        app.shutdown()
    assert res.text == "I ran the tests: all pass."
    judge = srv.requests[1]["messages"]
    assert "run the tests before it stops" in judge[1]["content"] and '"hook_event_name": "Stop"' in judge[1]["content"]
    assert any("You did not run the tests." in str(m.get("content")) for m in srv.requests[2]["messages"])
    assert [c.purpose for c in app.ledger.calls] == ["main", "hook", "main", "hook"]


def app_with_mcp(srv, project):
    cfg = load_config(cwd=project, overrides={"base_url": srv.url, "model": "fake-model", "api_key": "k"})
    cfg.set("learning.reflect", False)
    return App(cfg, RecUI(), cwd=project, mode="bypassPermissions", enable_mcp=True)


def test_mcp_disable_enable_and_reconnect(project):
    (project / ".mcp.json").write_text(json.dumps({"mcpServers": {"tracker": {
        "command": sys.executable, "args": [str(FAKE)]}}}), encoding="utf-8")
    with FakeOpenAI([]) as srv:
        app = app_with_mcp(srv, project)
        try:
            assert "mcp__tracker__lookup_issue" in app.agent.registry.names()
            assert app.mcp_set("tracker", "disable").startswith("tracker: disabled")
            assert "mcp__tracker__lookup_issue" not in app.agent.registry.names()
            assert load_config(cwd=project).get("mcp.disabled") == ["tracker"]     # remembered
            assert app.mcp_set("tracker", "enable") == "tracker: connected (3 tools)"
            assert "mcp__tracker__lookup_issue" in app.agent.registry.names()
            assert load_config(cwd=project).get("mcp.disabled") == []
            assert app.mcp_set("tracker", "reconnect") == "tracker: connected (3 tools)"
        finally:
            app.shutdown()


def test_claude_code_servers_for_this_project_are_used(project, monkeypatch, tmp_path):
    claude_json = tmp_path / "claude.json"
    claude_json.write_text(json.dumps({
        "mcpServers": {"everywhere": {"command": sys.executable, "args": [str(FAKE)]}},
        "projects": {str(project): {"mcpServers": {"tracker": {"command": sys.executable, "args": [str(FAKE)]}}}}}))
    monkeypatch.setattr(mcp_client, "CLAUDE_JSON", claude_json)
    configs = mcp_client.load_server_configs(project, tmp_path / "home")
    assert list(configs) == ["tracker"]                                  # user-wide ones need "all"
    assert set(mcp_client.load_server_configs(project, tmp_path / "home", "all")) == {"everywhere", "tracker"}
    assert mcp_client.load_server_configs(project, tmp_path / "home", "off") == {}


def test_the_agent_sets_up_an_mcp_server_itself_and_uses_it_right_away(project):
    script = [reply("", [{"name": "McpServers", "arguments": {
                  "action": "add", "name": "tracker", "command": sys.executable, "args": [str(FAKE)],
                  "env": {"TRACKER_KEY": "secret-123"}}}]),
              reply("", [{"name": "mcp__tracker__lookup_issue", "arguments": {"id": "7"}}]),
              reply("Issue 7 is about divide().")]
    with FakeOpenAI(script) as srv:
        app = app_with_mcp(srv, project)
        try:
            res = app.run_prompt("add the issue tracker and look up issue 7")
        finally:
            app.shutdown()
    saved = json.loads((project / ".mcp.json").read_text(encoding="utf-8"))
    assert saved["mcpServers"]["tracker"]["args"] == [str(FAKE)]
    added = [m for m in srv.requests[1]["messages"] if m["role"] == "tool"][-1]["content"]
    assert "started it: 3 tools" in added
    assert "mcp__tracker__lookup_issue" in [t["function"]["name"] for t in srv.requests[1]["tools"]]
    assert res.text == "Issue 7 is about divide()."


def test_adding_a_server_asks_first_even_in_auto_mode(project):
    from muyah_code.mcp.manage import McpServersTool
    from muyah_code.permissions import PermissionManager
    from muyah_code.tools.base import ToolContext

    tool = McpServersTool(app=None)
    args = {"action": "add", "name": "x", "command": "npx", "args": ["-y", "pkg"], "env": {"KEY": "s3cret"}}
    ctx = ToolContext(cwd=project, project_root=project, config=load_config(cwd=project))
    decision = PermissionManager("auto", project_root=project).check(tool, args, ctx)
    assert decision.action == "ask" and "starts a program" in decision.reason
    assert "s3cret" not in tool.preview(args, ctx)          # keys are never shown on screen
    assert PermissionManager("auto", project_root=project).check(tool, {"action": "list"}, ctx).action == "allow"
