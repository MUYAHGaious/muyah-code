"""Phase C: several models working together: roles, escalation on hard failure, fallback providers."""

import json

import pytest
from fakeserver import FakeOpenAI, error, reply
from test_agent_e2e import RecUI, make_app

from muyah_code.config import ConfigError, load_config
from muyah_code.llm.pool import ModelPool, agent_model


def profiles(project, **servers):
    """Save one profile per fake server in the project settings (as /provider or muyah connect would)."""
    data = {"profiles": {name: {"base_url": srv.url, "model": f"{name}-model", "api_key": "k"}
                         for name, srv in servers.items()}}
    (project / ".muyah" / "settings.json").write_text(json.dumps(data))


def explore_call(**extra):
    return reply("", [{"name": "Agent", "arguments": {"prompt": "find the config", "subagent_type": "explore",
                                                       **extra}}])


def models_of(srv):
    return [r["model"] for r in srv.requests]


def test_the_explore_sub_agent_runs_on_its_own_model(project):
    main_script = [explore_call(), reply("It is in settings.py.")]
    small_script = [reply("", [{"name": "Glob", "arguments": {"pattern": "*.py"}}]), reply("settings.py:1")]
    with FakeOpenAI(main_script) as main, FakeOpenAI(small_script) as small:
        profiles(project, small=small)
        app = make_app(main, project, models={"explore": "small"})
        res = app.run_prompt("where is the config?")
        app.shutdown()
    assert res.text == "It is in settings.py."
    assert models_of(small) == ["small-model", "small-model"]      # the child's two requests
    assert models_of(main) == ["fake-model", "fake-model"]
    assert [c.purpose for c in app.ledger.calls] == ["main", "subagent:explore", "subagent:explore", "main"]
    assert {c.model for c in app.ledger.calls if c.purpose == "subagent:explore"} == {"small-model"}


def test_the_agent_tool_can_pick_a_role_and_rejects_unknown_ones(project):
    main_script = [explore_call(model="main"), reply("", [{"name": "Glob", "arguments": {"pattern": "*"}}]),
                   reply("child report"), explore_call(model="huge"), reply("done")]
    with FakeOpenAI(main_script) as main, FakeOpenAI([]) as small:
        profiles(project, small=small)
        ui = RecUI()
        app = make_app(main, project, ui=ui, models={"explore": "small"})
        app.run_prompt("go")
        app.shutdown()
    assert small.requests == []                                    # model=main overrode the explore role
    assert any(e[0] == "end" and "Unknown model role 'huge'" in e[3] for e in ui.events)


def test_a_small_model_that_cannot_drive_the_tools_escalates_to_main(project):
    bad = reply("", [{"name": "Grep", "arguments": "{not json"}])
    small_script = [bad, bad]
    main_script = [explore_call(), reply("", [{"name": "Glob", "arguments": {"pattern": "*"}}]),   # child, now on main
                   reply("found it"), reply("All done.")]
    with FakeOpenAI(main_script) as main, FakeOpenAI(small_script) as small:
        profiles(project, small=small)
        ui = RecUI()
        app = make_app(main, project, ui=ui, models={"explore": "small"})
        res = app.run_prompt("go")
        app.shutdown()
    assert len(small.requests) == 2 and res.text == "All done."
    assert ("warn", "[explore] Escalated explore (small-model) → main (fake-model): two malformed tool calls") in ui.events
    # the bigger model sees the failed attempts
    child_on_main = main.requests[1]["messages"]
    assert sum(1 for m in child_on_main if m["role"] == "tool" and "could not parse" in m["content"]) == 2


def test_main_escalates_to_strong_for_the_rest_of_the_turn_only(project, tmp_path):
    fail = reply("", [{"name": "Bash", "arguments": {"command": "exit 3"}}])
    main_script = [fail, fail, reply("second turn on main")]
    strong_script = [reply("Fixed it (strong).")]
    with FakeOpenAI(main_script) as main, FakeOpenAI(strong_script) as strong:
        profiles(project, strong=strong)
        ui = RecUI()
        app = make_app(main, project, ui=ui, models={"strong": "strong"})
        first = app.run_prompt("make it pass")
        second = app.run_prompt("thanks")
        app.shutdown()
    assert first.text == "Fixed it (strong)."
    assert any(e[0] == "warn" and e[1].startswith("Escalated main (fake-model) → strong (strong-model): "
                                                  "the same command failed twice") for e in ui.events)
    assert second.text == "second turn on main"


def test_a_rate_limited_provider_falls_back_for_that_request(project):
    main_script = [error(429, "Rate limit reached for requests"), reply("main again")]
    backup_script = [reply("answered by the backup")]
    with FakeOpenAI(main_script) as main, FakeOpenAI(backup_script) as backup:
        profiles(project, backup=backup)
        ui = RecUI()
        app = make_app(main, project, ui=ui, fallback=["backup"])
        app.llm.max_retries = 0                 # the provider's own retries are exhausted at once
        first = app.run_prompt("hi")
        second = app.run_prompt("again")
        app.shutdown()
    assert first.status == "ok" and first.text == "answered by the backup"
    assert ("warn", "Using the fallback model backup-model for this request (rate limit from fake-model).") in ui.events
    assert second.text == "main again"              # only that request moved
    assert [c.model for c in app.ledger.calls] == ["backup-model", "fake-model"]


def test_no_fallback_for_errors_another_provider_cannot_fix(project):
    with FakeOpenAI([error(400, "Your request is malformed")]) as main, FakeOpenAI([]) as backup:
        profiles(project, backup=backup)
        app = make_app(main, project, fallback=["backup"])
        res = app.run_prompt("hi")
        app.shutdown()
    assert res.status == "error" and backup.requests == []


def test_provider_specs_use_that_providers_key_never_yours(project, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = load_config(cwd=project, overrides={"base_url": "https://api.openai.com/v1", "api_key": "sk-main-secret"})
    pool = ModelPool(cfg, main=None)
    with pytest.raises(ConfigError, match="No API key for Groq"):
        pool.config_for("groq:llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-groq")
    groq = pool.config_for("groq:llama-3.3-70b-versatile")
    assert groq.get("api_key") == "gsk-groq" and "groq.com" in groq.get("base_url")
    assert groq.get("model") == "llama-3.3-70b-versatile"
    same = pool.config_for("gpt-4.1-mini")                  # a bare model id: same endpoint and key
    assert same.get("base_url") == "https://api.openai.com/v1" and same.get("model") == "gpt-4.1-mini"
    assert cfg.get("model") != "gpt-4.1-mini"               # the session's own settings are untouched


def test_claude_code_agent_models_map_sensibly():
    assert agent_model("inherit", "anthropic") is None
    assert agent_model("haiku", "anthropic") == "claude-haiku-4-5"
    assert agent_model("haiku", "openai") is None           # a Claude alias means nothing elsewhere
    assert agent_model("explore", "openai") == "explore"


def test_the_editor_agent_only_exists_with_an_edit_model(project):
    with FakeOpenAI([]) as srv:
        plain = make_app(srv, project)
        with_edit = make_app(srv, project, models={"edit": "gpt-4.1-mini"})
        plain.shutdown()
        with_edit.shutdown()
    assert "editor" not in plain.agent_defs and "editor" in with_edit.agent_defs
