"""/verify (only when you ask), the commands it detects, and the weakened-tests warning in the footer."""

import json
import shutil
import sys

import pytest
from fakeserver import reply
from test_repl import run_session

from muyah_code.config import load_config
from muyah_code.verify import (Check, configured, detect, e2e_prompt, is_test_file, quick_checks,
                               summary_for_model, weakened_tests)

needs_git = pytest.mark.skipif(not shutil.which("git"), reason="turn changes come from rewind snapshots (git)")

CALC = "def add(a, b):\n    return a + b\n"
TEST_CALC = "from calc import add\n\n\ndef test_add():\n    assert add(2, 2) == 4\n    assert add(0, 0) == 0\n"


def py_project(project):
    (project / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath = ["."]\n')
    (project / "calc.py").write_text(CALC)
    (project / "tests").mkdir()
    (project / "tests" / "test_calc.py").write_text(TEST_CALC)
    return project


def commands(checks):
    return [c.command for c in checks]


def test_detects_the_projects_test_and_lint_commands(tmp_path):
    py = tmp_path / "py"
    py.mkdir()
    (py / "pyproject.toml").write_text("[tool.pytest.ini_options]\n[tool.ruff]\nline-length = 100\n")
    assert commands(detect(py)) == ["python -m pytest -q", "python -m ruff check ."]

    node = tmp_path / "node"
    node.mkdir()
    (node / "package.json").write_text(json.dumps({"scripts": {"test": "vitest run", "lint": "eslint ."}}))
    (node / "pnpm-lock.yaml").write_text("")
    assert commands(detect(node)) == ["pnpm test", "pnpm run lint"]

    placeholder = tmp_path / "fresh"
    placeholder.mkdir()
    (placeholder / "package.json").write_text(json.dumps(
        {"scripts": {"test": 'echo "Error: no test specified" && exit 1'}}))
    assert detect(placeholder) == []                          # npm init's placeholder is not a test suite

    go = tmp_path / "go"
    go.mkdir()
    (go / "go.mod").write_text("module x\n")
    assert commands(detect(go)) == ["go test ./...", "go vet ./..."]


def test_your_own_commands_win(project):
    (project / ".muyah" / "settings.json").write_text(json.dumps(
        {"verify": {"test": "make test", "lint": ["make lint", "make types"]}}))
    cfg = load_config(cwd=project)
    checks, from_settings = configured(cfg, project)
    assert from_settings and commands(checks) == ["make test", "make lint", "make types"]


def test_quick_runs_only_the_tests_of_the_changed_files(project):
    py_project(project)
    (project / "pyproject.toml").write_text('[tool.pytest.ini_options]\n[tool.ruff]\n')
    (project / "other.py").write_text("X = 1\n")
    cfg = load_config(cwd=project)
    checks, notes = quick_checks(cfg, project, ["calc.py"])
    assert commands(checks) == ["python -m ruff check calc.py", "python -m pytest -q tests/test_calc.py"]
    checks, notes = quick_checks(cfg, project, ["other.py"])
    assert commands(checks) == ["python -m ruff check other.py"]
    assert "no tests found for the changed Python files" in notes[0]
    checks, _ = quick_checks(cfg, project, ["tests/test_calc.py"])  # a changed test runs itself
    assert "python -m pytest -q tests/test_calc.py" in commands(checks)


def test_recognises_test_files():
    for rel in ("tests/test_calc.py", "pkg/calc_test.py", "src/a.test.ts", "web/b.spec.jsx", "x/y_test.go",
                "__tests__/c.js", "conftest.py"):
        assert is_test_file(rel), rel
    for rel in ("calc.py", "src/app.ts", "testing_utils.go", "docs/tests.md"):
        assert not is_test_file(rel), rel


def test_weakened_tests_are_spotted_without_a_model(project):
    (project / "tests").mkdir()
    before = {"tests/test_calc.py": TEST_CALC, "tests/test_ok.py": "def test_x():\n    assert 1\n"}
    (project / "tests" / "test_calc.py").write_text(
        "import pytest\nfrom calc import add\n\n\n@pytest.mark.skip\ndef test_add():\n    add(2, 2)\n")
    (project / "tests" / "test_ok.py").write_text("def test_x():\n    assert 1\n    assert 2\n")
    (project / "tests" / "test_new.py").write_text("import pytest\n\n\n@pytest.mark.xfail\ndef test_y():\n    pass\n")
    changes = [("M", "tests/test_calc.py"), ("M", "tests/test_ok.py"), ("A", "tests/test_new.py"),
               ("D", "tests/test_gone.py"), ("M", "calc.py")]
    warnings = weakened_tests(changes, before.get, project)
    assert warnings == ["tests/test_calc.py: 2 assertions removed, 1 skip/xfail added", "tests/test_gone.py deleted"]


def test_the_model_hears_about_failures_and_e2e_is_a_prompt():
    from muyah_code.verify import CheckResult

    note = summary_for_model("quick", [CheckResult(Check("test", "pytest"), 1, 0.5, "E   assert 5 == 4\n1 failed"),
                                       CheckResult(Check("lint", "ruff check ."), 0, 0.1, "")])
    assert "pytest: exit 1 (FAILED)" in note and "assert 5 == 4" in note and "ruff check .: exit 0 (passed)" in note
    prompt = e2e_prompt(["app.py"])
    assert "Do not change any code" in prompt and "- app.py" in prompt


# --------------------------------------------------------------------------- the real REPL

def read(path):
    return reply("", [{"name": "Read", "arguments": {"file_path": path}}])


def break_calc():
    return [read("calc.py"), reply("", [{"name": "Write", "arguments": {
        "file_path": "calc.py", "content": "def add(a, b):\n    return a + b + 1\n"}}])]


@needs_git
def test_verify_quick_runs_the_related_tests_and_tells_the_model(project):
    py_project(project)
    script = [*break_calc(), reply("Changed add."), reply("I see the failing test.")]
    code, out, app = run_session(project, ["change add", "/verify quick", "why did it fail?", "/exit"], script)
    assert "1 file changed (/undo · /verify)" in out
    assert "✗ python -m pytest -q tests/test_calc.py" in out and "exit 1" in out
    assert "1 of 1 failed" in out
    last_user = [m for m in app.agent.messages if m["role"] == "user"][-1]["content"]
    assert "<verify" in last_user
    assert "python -m pytest -q tests/test_calc.py: exit 1 (FAILED)" in last_user
    assert "why did it fail?" in last_user


@needs_git
def test_verify_full_and_nothing_changed(project):
    py_project(project)
    code, out, app = run_session(project, ["/verify quick", "/verify full", "/verify fast", "/exit"], [])
    assert "The last turn changed no files" in out
    assert "✓ python -m pytest -q" in out and "All 1 passed" in out
    assert "Unknown depth 'fast'" in out


@needs_git
def test_footer_warns_about_weakened_tests_and_counts_changes_made_by_commands(project):
    py_project(project)
    (project / ".muyah" / "settings.json").write_text(json.dumps({"permissions": {"allow": ["Bash"]}}))
    weaker = TEST_CALC.replace("    assert add(0, 0) == 0\n", "")
    make_file = f'"{sys.executable}" -c "open(\'made.txt\', \'w\')"'
    script = [read("tests/test_calc.py"),
              reply("", [{"name": "Write", "arguments": {"file_path": "tests/test_calc.py", "content": weaker}},
                         {"name": "Bash", "arguments": {"command": make_file}}]),
              reply("Done.")]
    code, out, app = run_session(project, ["tidy the tests", "/exit"], script)
    assert "2 files changed" in out                            # the command's file counts too
    assert "Tests weakened: tests/test_calc.py: 1 assertion removed" in out


@needs_git
def test_auto_verify_runs_after_a_turn_that_changed_files(project):
    py_project(project)
    (project / ".muyah" / "settings.json").write_text(json.dumps({"verify": {"auto": "quick"}}))
    code, out, app = run_session(project, ["change add", "/exit"], [*break_calc(), reply("Changed add.")])
    assert "Verify quick" in out and "✗ python -m pytest -q tests/test_calc.py" in out


def test_verify_e2e_asks_the_agent_to_run_it(project):
    code, out, app = run_session(project, ["/verify e2e", "/exit"], [reply("It works: ran python app.py, exit 0.")])
    first_user = [m for m in app.agent.messages if m["role"] == "user"][0]["content"]
    assert "Verify, end to end" in first_user and "It works" in out
