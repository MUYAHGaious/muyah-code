import sys

import pytest

from muyah_code.tools.search import GlobTool, GrepTool, matches_glob
from muyah_code.tools.shell import BashTool, detect_shell, is_read_only_command


@pytest.mark.parametrize("path,pattern,ok", [
    ("src/a.py", "**/*.py", True),
    ("a.py", "**/*.py", True),
    ("src/deep/a.py", "src/**/*.py", True),
    ("src/a.ts", "*.{ts,tsx}", True),
    ("src/a.js", "*.{ts,tsx}", False),
    ("tests/test_x.py", "test_*.py", True),
    ("src/a.py", "src/*.py", True),
    ("src/x/a.py", "src/*.py", False),
])
def test_glob_matching(path, pattern, ok):
    assert matches_glob(path, pattern) is ok


def _tree(project):
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("def main():\n    return helper()\n\ndef helper():\n    return 42\n")
    (project / "src" / "util.js").write_text("function helper() {}\n")
    (project / "README.md").write_text("helper docs\n")


def test_glob_tool(ctx, project):
    _tree(project)
    out = GlobTool().run({"pattern": "**/*.py"}, ctx).content
    assert "src/app.py" in out and "util.js" not in out


def test_grep_modes(ctx, project):
    _tree(project)
    files = GrepTool().run({"pattern": "helper"}, ctx).content
    assert "src/app.py" in files and "README.md" in files
    content = GrepTool().run({"pattern": r"def \w+", "output_mode": "content", "type": "py"}, ctx).content
    assert "src/app.py:1:def main():" in content and "util.js" not in content
    count = GrepTool().run({"pattern": "helper", "output_mode": "count", "glob": "*.py"}, ctx).content
    assert count.strip() == "src/app.py:2"


def test_grep_context_lines(ctx, project):
    _tree(project)
    out = GrepTool().run({"pattern": "return 42", "output_mode": "content", "-B": 1}, ctx).content
    assert "src/app.py-4-def helper():" in out and "src/app.py:5:" in out


@pytest.mark.parametrize("cmd,ok", [
    ("git status", True), ("git diff --stat", True), ("git log --oneline -5", True),
    ("ls -la", True), ("cat README.md", True), ("python --version", True),
    ("git push", False), ("git branch -D main", False), ("git branch new-branch", False),
    ("rm -rf /", False), ("ls; rm x", False), ("cat a > b", False), ("echo $(whoami)", False),
    ("find . -name '*.py'", True), ("find . -delete", False), ("pip install x", False),
    ("git config user.name bob", False), ("git config --get user.name", True),
])
def test_read_only_detection(cmd, ok):
    assert is_read_only_command(cmd) is ok


def test_bash_runs_and_reports_exit_code(ctx):
    res = BashTool().run({"command": f'"{sys.executable}" -c "print(6*7)"'}, ctx)
    assert "42" in res.content and not res.is_error
    bad = BashTool().run({"command": f'"{sys.executable}" -c "import sys; sys.exit(3)"'}, ctx)
    assert bad.is_error and "[exit code 3]" in bad.content


def test_bash_timeout(ctx):
    res = BashTool().run({"command": f'"{sys.executable}" -c "import time; time.sleep(10)"', "timeout": 1}, ctx)
    assert res.is_error and "timed out" in res.content


def test_shell_detection_never_picks_wsl_launcher():
    spec = detect_shell("auto")
    assert "system32" not in spec.executable.lower()
    assert "windowsapps" not in spec.executable.lower()
