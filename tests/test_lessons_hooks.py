import json
import sys

from muyah_code.hooks import HookRunner
from muyah_code.learning.lessons import Lesson, LessonStore, Reflector, is_correction, parse_lessons
from muyah_code.llm.client import AssistantMessage


def store(tmp_path):
    return LessonStore(tmp_path / "g.jsonl", tmp_path / "p.jsonl")


def test_add_dedupe_persist(tmp_path):
    s = store(tmp_path)
    a, merged = s.add(Lesson("running pytest on windows", "use python -m pytest instead of pytest", ["pytest"]))
    assert not merged
    b, merged = s.add(Lesson("running pytest on windows machines", "use python -m pytest instead of bare pytest",
                             ["windows"]))
    assert merged and b.id == a.id and b.reinforced == 1 and "windows" in b.tags
    s.add(Lesson("editing yaml config files", "validate with yamllint after editing", scope="global"))
    reloaded = store(tmp_path)
    assert len(reloaded.lessons) == 2
    assert {x.scope for x in reloaded.lessons} == {"project", "global"}


def test_search_relevance_and_score_weighting(tmp_path):
    s = store(tmp_path)
    good, _ = s.add(Lesson("running pytest tests", "run python -m pytest -q from the repo root"))
    s.add(Lesson("writing react components", "prefer function components with hooks"))
    hits = s.search("the pytest tests are failing, fix them")
    assert hits and hits[0].id == good.id
    assert s.search("completely unrelated astronomy question") == []


def test_outcomes_and_pruning(tmp_path):
    s = store(tmp_path)
    bad, _ = s.add(Lesson("deploying the service", "always force push to main"))
    for _ in range(3):
        s.record_outcome([bad.id], success=False)
    assert s.get(bad.id) is None  # consistently harmful lessons are pruned
    ok, _ = s.add(Lesson("formatting python", "run ruff format before committing"))
    s.record_outcome([ok.id], success=True)
    assert s.get(ok.id).wins == 1 and s.get(ok.id).score > 0.5


def test_correction_detection():
    assert is_correction("no, that's wrong, use the other API")
    assert is_correction("It still fails with the same error")
    assert is_correction("you forgot to update the tests")
    assert not is_correction("now add a --verbose flag")
    assert not is_correction("nothing else, thanks")


def test_parse_lessons_is_defensive():
    text = 'Sure!\n```json\n[{"trigger": "running npm scripts here", "lesson": "use pnpm, npm is not installed", ' \
           '"scope": "project", "tags": ["pnpm"]}, {"trigger": "x", "lesson": "too short"}]\n```'
    out = parse_lessons(text)
    assert len(out) == 1 and out[0]["lesson"].startswith("use pnpm")
    assert parse_lessons("no json here") == []


def test_reflector_stores_lessons(tmp_path):
    class LLM:
        def chat(self, messages, **kw):
            return AssistantMessage(content=json.dumps([{"trigger": "tests fail with ModuleNotFoundError",
                                                         "lesson": "install the package in editable mode first",
                                                         "scope": "project", "tags": ["pip"]}]))

    s = store(tmp_path)
    r = Reflector(LLM(), s)
    signals = [{"type": "recovered", "tool": "Bash", "previous_error": "ModuleNotFoundError"}]
    assert Reflector.worth_reflecting(signals, "ok")
    assert not Reflector.worth_reflecting([{"type": "tool_error"}], "ok")
    stored = r.reflect("run the tests", signals, [{"role": "user", "content": "run the tests"}], "ok")
    assert len(stored) == 1 and s.lessons[0].source == "recovered"


def _hook(tmp_path, code: str) -> str:
    script = tmp_path / "hook.py"
    script.write_text(code)
    return f'"{sys.executable}" "{script}"'


def test_pre_tool_hook_blocks_with_exit_2(tmp_path):
    cmd = _hook(tmp_path, "import sys, json\nd = json.load(sys.stdin)\n"
                          "if 'rm' in d['tool_input']['command']:\n    print('no rm allowed', file=sys.stderr)\n"
                          "    sys.exit(2)\n")
    runner = HookRunner({"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": cmd}]}]}, tmp_path)
    out = runner.run("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "rm -rf x"}}, tool_name="Bash")
    assert out.blocked and "no rm allowed" in out.reason
    ok = runner.run("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "ls"}}, tool_name="Bash")
    assert not ok.blocked
    other = runner.run("PreToolUse", {"tool_name": "Edit", "tool_input": {}}, tool_name="Edit")
    assert not other.blocked  # matcher does not match


def test_hook_json_output(tmp_path):
    cmd = _hook(tmp_path, "import json\nprint(json.dumps({'hookSpecificOutput': {'permissionDecision': 'allow', "
                          "'updatedInput': {'command': 'ls -la'}, 'additionalContext': 'ctx!'}}))\n")
    runner = HookRunner({"PreToolUse": [{"hooks": [{"command": cmd}]}]}, tmp_path)
    out = runner.run("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "ls"}}, tool_name="Bash")
    assert out.permission == "allow" and out.updated_input == {"command": "ls -la"} and "ctx!" in out.additional_context


def test_user_prompt_hook_plain_stdout_is_context(tmp_path):
    cmd = _hook(tmp_path, "print('Today is release day')\n")
    runner = HookRunner({"UserPromptSubmit": [{"hooks": [{"command": cmd}]}]}, tmp_path)
    out = runner.run("UserPromptSubmit", {"prompt": "hi"})
    assert "release day" in out.additional_context


def test_unknown_hook_event_reported(tmp_path):
    assert HookRunner({"OnMagic": []}, tmp_path).errors
