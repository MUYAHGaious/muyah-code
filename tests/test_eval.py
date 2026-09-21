import json
from pathlib import Path

from fakeserver import FakeOpenAI, reply
from rich.console import Console

from muyah_code.learning.eval import discover_tasks, run_eval, run_task

TASKS = Path(__file__).resolve().parent.parent / "muyah_code" / "evals" / "tasks"


def test_bundled_tasks_are_discoverable():
    names = [t.name for t in discover_tasks(TASKS)]
    assert names == ["add-cli-flag", "fix-off-by-one", "implement-function", "keyerror-traceback", "rename-refactor"]


def test_eval_passes_when_the_model_fixes_the_bug(monkeypatch):
    script = [
        reply("", [{"name": "Read", "arguments": {"file_path": "stats.py"}}]),
        reply("", [{"name": "Edit", "arguments": {"file_path": "stats.py", "old_string": "range(len(values) - window)",
                                                  "new_string": "range(len(values) - window + 1)"}}]),
        reply("Fixed the off-by-one in moving_average."),
    ]
    with FakeOpenAI(script) as srv:
        monkeypatch.setenv("MUYAH_BASE_URL", srv.url)
        res = run_task(TASKS / "fix-off-by-one", {"model": "fake-model"}, learn=False, verbose=False)
    assert res.passed and res.status == "ok" and res.tool_calls == 2


def test_eval_fails_when_the_model_does_nothing_and_records_history(monkeypatch, isolated_home):
    with FakeOpenAI([reply("I think it is fine.")]) as srv:
        monkeypatch.setenv("MUYAH_BASE_URL", srv.url)
        code = run_eval(TASKS, ["fix-off-by-one"], {"model": "fake-model"}, False, False, Console(quiet=True))
    assert code == 1
    history = [json.loads(line) for line in (isolated_home / "evals.jsonl").read_text().splitlines()]
    assert history[-1]["pass_rate"] == 0.0 and history[-1]["model"] == "fake-model"
