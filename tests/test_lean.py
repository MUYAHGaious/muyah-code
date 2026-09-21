"""Lean mode for small models: a short prompt, 6 core tools, everything else one find_tools call away."""

import json

from fakeserver import FakeOpenAI, reply
from test_agent_e2e import make_app

from muyah_code.agent.lean import choose_profile, lean_instructions, param_billions
from muyah_code.memory import InstructionFile

SMALL_WINDOW = [{"id": "fake-model", "object": "model", "max_model_len": 16384}]


def tokens(app):
    """What every request carries before the conversation: system prompt + tool definitions."""
    tools = app.agent.registry.schemas()
    app.agent.refresh_system_prompt()
    return app.context.count(app.agent.messages[:1], tools)


def test_profile_choice():
    assert param_billions("qwen2.5-coder:7b") == 7 and param_billions("Qwen3-Coder-30B-A3B-Instruct") == 30
    assert param_billions("llama-3.1-8b-instant") == 8 and param_billions("gpt-4.1") is None
    assert choose_profile("auto", 16384, "gpt-4.1", False) == "lean"            # small window
    assert choose_profile("auto", 32768, "Qwen2.5-Coder-32B-Instruct-AWQ", True) == "full"
    assert choose_profile("auto", 131072, "qwen2.5-coder:14b", True) == "lean"   # small model on your machine
    assert choose_profile("auto", 131072, "llama-3.1-8b-instant", False) == "full"  # a hosted API: its window
    assert choose_profile("full", 8192, "x", True) == "full" and choose_profile("lean", 10**6, "x", False) == "lean"


def test_lean_mode_cuts_the_fixed_cost_of_every_request(project):
    with FakeOpenAI([], models=SMALL_WINDOW) as srv:
        lean = make_app(srv, project)
        full = make_app(srv, project, prompt_profile="full")
        assert lean.prompt_profile == "lean" and full.prompt_profile == "full"
        lean_tokens, full_tokens = tokens(lean), tokens(full)
        lean.shutdown()
        full.shutdown()
    assert lean.agent.registry.names() == ["Read", "Edit", "Write", "Bash", "Grep", "Glob", "find_tools"]
    assert lean_tokens < full_tokens * 0.4, (lean_tokens, full_tokens)


def test_find_tools_switches_tools_on_for_the_next_request(project):
    script = [
        reply("", [{"name": "find_tools", "arguments": {"query": "todo list"}}]),
        reply("", [{"name": "TodoWrite", "arguments": {"todos": [
            {"content": "Fix the bug", "status": "in_progress", "activeForm": "Fixing the bug"}]}}]),
        reply("Planned."),
    ]
    with FakeOpenAI(script, models=SMALL_WINDOW) as srv:
        app = make_app(srv, project)
        res = app.run_prompt("plan it")
        app.shutdown()
    first_tools = [t["function"]["name"] for t in srv.requests[0]["tools"]]
    second_tools = [t["function"]["name"] for t in srv.requests[1]["tools"]]
    assert "TodoWrite" not in first_tools and "TodoWrite" in second_tools
    found = [m for m in srv.requests[1]["messages"] if m["role"] == "tool"][-1]["content"]
    assert found.startswith("Switched on") and "## TodoWrite" in found
    assert res.text == "Planned." and app.agent.ctx.todos[0]["content"] == "Fix the bug"


def test_lean_prompt_keeps_the_essentials_and_caps_instructions(project, tmp_path):
    (project / "MUYAH.md").write_text("Use tabs.\n" + "x" * 9000)
    with FakeOpenAI([], models=SMALL_WINDOW) as srv:
        app = make_app(srv, project)
        app.agent.refresh_system_prompt()
        prompt = app.agent.messages[0]["content"]
        app.shutdown()
    assert "find_tools" in prompt and "Never delete files yourself" in prompt and "Be honest" in prompt
    assert "Use tabs." in prompt and "[... cut to fit lean mode ...]" in prompt and len(prompt) < 7000
    nearest = lean_instructions([InstructionFile(tmp_path / "a.md", "global " * 900),
                                 InstructionFile(tmp_path / "b.md", "project rule")], cap=1000)
    assert "project rule" in nearest                           # the nearest file always makes it in


def test_eval_records_the_prompt_profile(tmp_path):
    from muyah_code.learning.eval import _previous

    log = tmp_path / "evals.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in [
        {"model": "m", "prompt_profile": "lean", "pass_rate": 0.5},
        {"model": "m", "prompt_profile": "full", "pass_rate": 0.9}]))
    assert _previous(log, "m", "lean") == 0.5 and _previous(log, "m", "full") == 0.9
