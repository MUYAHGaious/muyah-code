"""End-to-end: the real App + LLMClient + tools, against a scripted OpenAI-compatible server."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fakeserver import FakeOpenAI, error, reply

from muyah_code.app import App
from muyah_code.config import load_config
from muyah_code.ui.base import UI, PermissionReply


class RecUI(UI):
    def __init__(self, answers=None):
        self.events = []
        self.answers = list(answers or [])
        self.text_out = []

    def text(self, chunk):
        self.text_out.append(chunk)

    def tool_start(self, title):
        self.events.append(("start", title))

    def tool_end(self, title, result):
        self.events.append(("end", title, result.is_error, result.content))

    def warn(self, msg):
        self.events.append(("warn", msg))

    def error(self, msg):
        self.events.append(("error", msg))

    def info(self, msg):
        self.events.append(("info", msg))

    def ask_permission(self, req):
        self.events.append(("ask", req.title, req.detail))
        return self.answers.pop(0) if self.answers else PermissionReply("yes")


def make_app(server, project, ui=None, mode="bypassPermissions", headless=False, **cfg_over):
    cfg = load_config(cwd=project, overrides={"base_url": server.url, "model": "fake-model", "api_key": "k",
                                              **cfg_over})
    cfg.set("learning.reflect", False)
    return App(cfg, ui or RecUI(), cwd=project, mode=mode, headless=headless, enable_mcp=False)


def test_native_tool_loop_writes_file_and_answers(project):
    script = [
        reply("I'll create it.", [{"name": "Write", "arguments": {"file_path": "hello.py",
                                                                 "content": "print('hi')\n"}}]),
        reply("", [{"name": "Bash", "arguments": {"command": f'"{sys.executable}" hello.py'}}]),
        reply("Created hello.py and ran it: it prints hi."),
    ]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project)
        res = app.run_prompt("create hello.py that prints hi and run it")
    assert res.status == "ok" and "prints hi" in res.text
    assert (project / "hello.py").read_text() == "print('hi')\n"
    bash_result = [e for e in app.ui.events if e[0] == "end" and e[1].startswith("Bash")][0]
    assert "hi" in bash_result[3]
    # the transcript is valid OpenAI format: tool results answer the calls
    roles = [m["role"] for m in app.agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    # streaming used, and the context window came from the server's /v1/models
    assert srv.requests[0]["stream"] is True
    assert app.window == 32768 and app.window_source == "server"


def test_falls_back_to_text_protocol_when_server_rejects_tools(project):
    (project / "notes.txt").write_text("the secret is 7\n")
    script = [
        error(400, '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'),
        reply('Reading.\n<tool_call>\n{"name": "Read", "arguments": {"file_path": "notes.txt"}}\n</tool_call>'),
        reply("The secret is 7."),
    ]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project)
        res = app.run_prompt("what is the secret in notes.txt?")
        assert res.status == "ok" and res.text == "The secret is 7."
        assert app.agent.text_mode
        assert "tools" not in srv.requests[1]
        assert "<tool_call>" in srv.requests[1]["messages"][0]["content"]  # protocol instructions in prompt
        last_user = srv.requests[2]["messages"][-1]
        assert last_user["role"] == "user" and last_user["content"].startswith('<tool_result name="Read"')
        assert "the secret is 7" in last_user["content"]
    # the raw tool_call markup never reached the user's screen
    assert "<tool_call>" not in "".join(app.ui.text_out)


def test_invalid_arguments_are_reported_back_and_model_recovers(project):
    script = [
        reply("", [{"name": "Read", "arguments": {"path": "x.py"}}]),  # wrong param name
        reply("", [{"name": "Read", "arguments": '{"file_path": "x.py"'}]),  # broken JSON
        reply("Sorry, x.py does not exist."),
    ]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project)
        res = app.run_prompt("read x.py")
        tool_msgs = [m for m in srv.requests[1]["messages"] if m["role"] == "tool"]
        assert "missing required parameter 'file_path'" in tool_msgs[-1]["content"]
        tool_msgs = [m for m in srv.requests[2]["messages"] if m["role"] == "tool"]
        assert "could not parse" in tool_msgs[-1]["content"] or "not valid JSON" in tool_msgs[-1]["content"]
    assert res.status == "ok"


def test_repeated_identical_calls_are_stopped(project):
    call = [{"name": "LS", "arguments": {"path": "."}}]
    with FakeOpenAI([reply("", call) for _ in range(8)]) as srv:
        app = make_app(srv, project)
        res = app.run_prompt("list forever")
        # request[3] carries the result of the 3rd identical call
        warned = [m for m in srv.requests[3]["messages"] if m["role"] == "tool"][-1]["content"]
    assert "exact LS call 3 times" in warned
    assert res.status == "loop"
    assert any(s["type"] == "loop" for s in res.signals)


def test_headless_denies_actions_needing_approval(project):
    script = [reply("", [{"name": "Write", "arguments": {"file_path": "a.txt", "content": "x"}}]),
              reply("I could not write the file because permission was denied.")]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project, mode="default", headless=True)
        app.ctx.headless = True
        res = app.run_prompt("write a.txt")
        denied = [m for m in srv.requests[1]["messages"] if m["role"] == "tool"][-1]["content"]
    assert "needs approval" in denied and not (project / "a.txt").exists()
    assert res.status == "ok"


def test_interactive_permission_prompt_always_rule(project):
    script = [reply("", [{"name": "Bash", "arguments": {"command": "mkdir d1"}}]),
              reply("", [{"name": "Bash", "arguments": {"command": "mkdir d2"}}]),
              reply("done")]
    ui = RecUI(answers=[PermissionReply("always")])
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project, ui=ui, mode="default")
        app.run_prompt("make dirs")
    asks = [e for e in ui.events if e[0] == "ask"]
    assert len(asks) == 1  # 'always' allowed the second mkdir via the new Bash(mkdir:*) rule
    assert "mkdir d1" in asks[0][2]
    assert (project / "d1").is_dir() and (project / "d2").is_dir()


def test_user_rejection_stops_the_turn(project):
    script = [reply("", [{"name": "Write", "arguments": {"file_path": "b.txt", "content": "x"}}]),
              reply("should not be reached")]
    ui = RecUI(answers=[PermissionReply("no")])
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project, ui=ui, mode="default")
        res = app.run_prompt("write b.txt")
        assert len(srv.requests) == 1
    assert res.status == "denied" and not (project / "b.txt").exists()
    assert app.agent.messages[-1]["role"] == "tool"  # transcript stays valid for the next turn


def test_plan_mode_blocks_writes(project):
    script = [reply("", [{"name": "Write", "arguments": {"file_path": "c.txt", "content": "x"}}]),
              reply("Plan: 1. create c.txt")]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project, mode="plan")
        res = app.run_prompt("make c.txt")
        assert "PLAN MODE IS ACTIVE" in srv.requests[0]["messages"][0]["content"]
        denied = [m for m in srv.requests[1]["messages"] if m["role"] == "tool"][-1]["content"]
    assert "plan mode is read-only" in denied and not (project / "c.txt").exists()
    assert res.text.startswith("Plan:")


def test_context_overflow_triggers_emergency_compaction(project):
    big = "\n".join("y" * 60 for _ in range(400))  # ~24k chars across many lines
    (project / "big.txt").write_text(big)
    script = [reply("", [{"name": "Read", "arguments": {"file_path": "big.txt"}}]),
              reply("", [{"name": "Read", "arguments": {"file_path": "big.txt", "offset": 1}}]),
              error(400, "This model's maximum context length is 16384 tokens. However, your messages resulted "
                         "in 20000 tokens."),
              reply("The file is all y's.")]
    with FakeOpenAI(script, models=[{"id": "fake-model"}]) as srv:
        app = make_app(srv, project, context_window=16384, max_tokens=1024)
        res = app.run_prompt("what is in big.txt?")
        retried = srv.requests[-1]["messages"]
        before = srv.requests[-2]["messages"]
    assert res.status == "ok" and res.text == "The file is all y's."
    assert any(e[0] == "warn" and "context window is full" in e[1] for e in app.ui.events)
    # the retry really was smaller: both big tool outputs were pruned
    assert len(json.dumps(retried)) < len(json.dumps(before)) / 3
    assert all("pruned" in m["content"] for m in retried if m["role"] == "tool")


def test_system_prompt_is_stable_across_turns_for_prefix_caching(project):
    with FakeOpenAI([reply("one"), reply("two")]) as srv:
        app = make_app(srv, project)
        app.run_prompt("first")
        app.run_prompt("second")
        assert srv.requests[0]["messages"][0] == srv.requests[1]["messages"][0]
        assert srv.requests[1]["messages"][:3] == srv.requests[0]["messages"][:2] + [
            {"role": "assistant", "content": "one"}]


def test_lessons_are_injected_into_the_user_message(project):
    from muyah_code.learning.lessons import Lesson

    with FakeOpenAI([reply("ok")]) as srv:
        app = make_app(srv, project)
        app.lessons.add(Lesson("running pytest in this project", "use python -m pytest -q from the root"))
        app.run_prompt("please run pytest and fix failures")
        user = srv.requests[0]["messages"][1]["content"]
    assert "<lessons>" in user and "python -m pytest -q" in user


def test_correction_updates_lesson_scores(project):
    from muyah_code.learning.lessons import Lesson

    with FakeOpenAI([reply("done"), reply("fixed")]) as srv:
        app = make_app(srv, project)
        lesson, _ = app.lessons.add(Lesson("deploying the docs site", "run mkdocs gh-deploy --force"))
        app.run_prompt("deploy the docs site")
        app.run_prompt("no, that's wrong - never force deploy")
    assert app.lessons.get(lesson.id).losses == 1


def test_think_tags_are_hidden_from_output(project):
    with FakeOpenAI([reply("<think>secret reasoning</think>Final answer.")]) as srv:
        app = make_app(srv, project)
        res = app.run_prompt("q")
    assert res.text == "Final answer." and "secret" not in "".join(app.ui.text_out)


def test_subagent_returns_only_its_report(project):
    (project / "mod.py").write_text("def target():\n    pass\n")
    script = [
        reply("", [{"name": "Agent", "arguments": {"prompt": "find where target is defined",
                                                   "subagent_type": "explore"}}]),
        reply("", [{"name": "Grep", "arguments": {"pattern": "def target", "output_mode": "content"}}]),  # child
        reply("target is defined in mod.py:1"),  # child's report
        reply("It is in mod.py line 1."),  # parent
    ]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project)
        res = app.run_prompt("where is target defined?")
        child_first = srv.requests[1]["messages"]
        parent_last = srv.requests[3]["messages"]
    assert "Sub-agent role: explore" in child_first[0]["content"]
    assert [m for m in parent_last if m["role"] == "tool"][-1]["content"] == "target is defined in mod.py:1"
    assert res.text == "It is in mod.py line 1."


def test_undo_reverts_last_turn(project):
    (project / "u.txt").write_text("original\n")
    script = [reply("", [{"name": "Read", "arguments": {"file_path": "u.txt"}}]),
              reply("", [{"name": "Edit", "arguments": {"file_path": "u.txt", "old_string": "original",
                                                        "new_string": "changed"}}]),
              reply("edited")]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project)
        app.run_prompt("change u.txt")
    assert (project / "u.txt").read_text() == "changed\n"
    assert "restored" in app.undo()
    assert (project / "u.txt").read_text() == "original\n"


def test_at_file_mentions_are_inlined(project):
    (project / "spec.md").write_text("Build a calculator.")
    with FakeOpenAI([reply("ok")]) as srv:
        app = make_app(srv, project)
        app.run_prompt("implement @spec.md")
        assert "Build a calculator." in srv.requests[0]["messages"][1]["content"]


def test_session_resume(project):
    with FakeOpenAI([reply("first answer"), reply("second answer")]) as srv:
        app = make_app(srv, project)
        app.run_prompt("remember the number 42")
        sid = app.session_id
        app2 = App(app.cfg, RecUI(), cwd=project, mode="bypassPermissions", resume=sid, enable_mcp=False)
        app2.run_prompt("what number?")
        msgs = srv.requests[1]["messages"]
    assert any(m.get("content") == "remember the number 42" for m in msgs)


@pytest.mark.parametrize("fmt", ["json", "text"])
def test_cli_headless(project, fmt, isolated_home):
    with FakeOpenAI([reply("", [{"name": "LS", "arguments": {}}]), reply("There is a .muyah folder.")]) as srv:
        env = {**os.environ, "MUYAH_HOME": str(isolated_home), "PYTHONIOENCODING": "utf-8"}
        root = Path(__file__).resolve().parent.parent
        env["PYTHONPATH"] = str(root)
        proc = subprocess.run(
            [sys.executable, "-m", "muyah_code", "-p", "what is here?", "--base-url", srv.url, "-m", "fake-model",
             "--output-format", fmt, "--no-mcp", "--cwd", str(project)],
            capture_output=True, text=True, env=env, timeout=60, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    if fmt == "json":
        data = json.loads(proc.stdout)
        assert data["result"] == "There is a .muyah folder." and data["status"] == "ok" and data["tool_calls"] == 1
    else:
        assert proc.stdout.strip() == "There is a .muyah folder."


GEMINI_SIG = {"extra_content": {"google": {"thought_signature": "EpgFCpUF-signature"}}}
GEMINI_ERROR = ("Function call is missing a thought_signature in functionCall parts. This is required for tools to "
                "work correctly. Additional data, function call `default_api:LS` , position 6.")


def test_provider_tool_call_fields_are_sent_back_gemini_thought_signature(project):
    # Gemini attaches a thought signature to each tool call and rejects the next request without it
    script = [reply("", [{"name": "Glob", "arguments": {"pattern": "*"}, "extra": GEMINI_SIG}]), reply("Nothing.")]
    for stream in (True, False):
        with FakeOpenAI(list(script)) as srv:
            app = make_app(srv, project)
            app.llm.stream = stream
            res = app.run_prompt("list files")
        assert res.status == "ok", stream
        sent = [m for m in srv.requests[1]["messages"] if m.get("tool_calls")][0]["tool_calls"][0]
        assert sent["extra_content"] == GEMINI_SIG["extra_content"], stream
        assert not any(k.startswith("_") for m in srv.requests[1]["messages"] for k in m)


def test_provider_tool_call_fields_are_dropped_for_another_endpoint(project):
    from muyah_code.llm.client import ENDPOINT_KEY

    with FakeOpenAI([]) as srv:
        app = make_app(srv, project)
        msg = {"role": "assistant", "content": "", ENDPOINT_KEY: "https://generativelanguage.googleapis.com/v1beta/openai",
               "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "LS", "arguments": "{}"},
                               **GEMINI_SIG}]}
        out = app.llm._outgoing(msg)
    assert out["tool_calls"] == [{"id": "c1", "type": "function", "function": {"name": "LS", "arguments": "{}"}}]
    assert ENDPOINT_KEY not in out


def test_malformed_tool_request_is_not_mistaken_for_no_tool_support(project):
    from muyah_code.llm.models import is_tools_unsupported

    assert not is_tools_unsupported(GEMINI_ERROR)
    assert is_tools_unsupported('"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser')
    assert is_tools_unsupported("registry.ollama.ai/library/gemma:2b does not support tools")
    script = [reply("", [{"name": "Glob", "arguments": {"pattern": "*"}}]), error(400, GEMINI_ERROR)]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project)
        res = app.run_prompt("list files")
    assert res.status == "error"
    assert not app.agent.text_mode       # it stays on native tool calling
    assert len(srv.requests) == 2        # and does not retry the same broken request in text mode


def test_the_agent_never_runs_a_delete_the_user_gets_the_command(project):
    victim = project / "important.txt"
    victim.write_text("keep me")

    class Seen(RecUI):
        def handoff(self, command, targets):
            self.events.append(("handoff", command, targets))

    script = [reply("", [{"name": "Bash", "arguments": {"command": "rm -f important.txt"}}]), reply("ok")]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project, ui=Seen())                 # bypassPermissions: still not run
        app.run_prompt("clean up")
        tool_msg = [m for m in srv.requests[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert victim.exists()                                      # nothing was deleted
    handoff = [e for e in app.ui.events if e[0] == "handoff"][0]
    assert handoff[1] == "rm -f important.txt" and handoff[2][0].endswith("important.txt")
    assert "Not run" in tool_msg and "run in their own terminal" in tool_msg
    states = [e.get("state") for e in app.events.history if e["type"] == "tool_permission"]
    assert "handoff" in states
