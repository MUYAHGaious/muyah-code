import pytest

from muyah_code.permissions import PermissionManager, Rule, normalize_mode, suggest_rule
from muyah_code.tools.fs import EditTool, ReadTool, WriteTool
from muyah_code.tools.shell import BashTool
from muyah_code.tools.web import WebFetchTool


def pm(project, mode="default", **rules):
    return PermissionManager(mode, project_root=project, **rules)


def test_defaults(ctx, project):
    p = pm(project)
    assert p.check(ReadTool(), {"file_path": "a.py"}, ctx).action == "allow"
    assert p.check(EditTool(), {"file_path": "a.py"}, ctx).action == "ask"
    assert p.check(BashTool(), {"command": "npm test"}, ctx).action == "ask"
    assert p.check(BashTool(), {"command": "git status"}, ctx).action == "allow"  # read-only


def test_accept_edits_inside_project_only(ctx, project):
    p = pm(project, "acceptEdits")
    assert p.check(WriteTool(), {"file_path": "src/x.py"}, ctx).action == "allow"
    outside = str(project.parent / "elsewhere.py")
    assert p.check(WriteTool(), {"file_path": outside}, ctx).action == "ask"
    assert p.check(BashTool(), {"command": "npm test"}, ctx).action == "ask"


def test_plan_mode_is_read_only(ctx, project):
    p = pm(project, "plan")
    assert p.check(EditTool(), {"file_path": "a.py"}, ctx).action == "deny"
    assert p.check(BashTool(), {"command": "pytest"}, ctx).action == "deny"
    assert p.check(BashTool(), {"command": "git log"}, ctx).action == "allow"
    assert p.check(ReadTool(), {"file_path": "a.py"}, ctx).action == "allow"


def test_rules_precedence(ctx, project):
    p = pm(project, "bypassPermissions", allow=["Bash(npm run test:*)"], deny=["Bash(rm *)"],
           ask=["Edit(secrets/**)"])
    assert p.check(BashTool(), {"command": "rm -rf build"}, ctx).action == "deny"
    assert p.check(EditTool(), {"file_path": "secrets/key.txt"}, ctx).action == "ask"
    assert p.check(BashTool(), {"command": "ls"}, ctx).action == "allow"


def test_prefix_rule(ctx, project):
    p = pm(project, allow=["Bash(npm run test:*)"])
    assert p.check(BashTool(), {"command": "npm run test -- --watch"}, ctx).action == "allow"
    assert p.check(BashTool(), {"command": "npm run test"}, ctx).action == "allow"
    assert p.check(BashTool(), {"command": "npm run testing"}, ctx).action == "ask"


def test_path_rule(ctx, project):
    p = pm(project, allow=["Edit(src/**)"])
    assert p.check(EditTool(), {"file_path": "src/a/b.py"}, ctx).action == "allow"
    assert p.check(EditTool(), {"file_path": "tests/b.py"}, ctx).action == "ask"


def test_domain_rule(ctx, project):
    p = pm(project, allow=["WebFetch(domain:python.org)"])
    assert p.check(WebFetchTool(), {"url": "https://docs.python.org/3/"}, ctx).action == "allow"
    assert p.check(WebFetchTool(), {"url": "https://evil.com/python.org"}, ctx).action == "ask"


def test_rule_parse_and_modes():
    assert str(Rule.parse("Bash(git status)")) == "Bash(git status)"
    assert Rule.parse("Read").pattern is None
    assert normalize_mode("yolo") == "bypassPermissions"
    assert normalize_mode("accept-edits") == "acceptEdits"
    with pytest.raises(ValueError):
        normalize_mode("chaos")


def test_suggest_rule(ctx, project):
    assert suggest_rule(BashTool(), {"command": "npm run build --prod"}, ctx) == "Bash(npm run:*)"
    assert suggest_rule(EditTool(), {"file_path": "src/app/x.py"}, ctx) == "Edit(src/app/**)"


def test_cycle_mode(project):
    p = pm(project)
    assert [p.cycle_mode(), p.cycle_mode(), p.cycle_mode()] == ["acceptEdits", "plan", "default"]


def test_quoted_program_paths_are_normalized(ctx, project):
    from muyah_code.permissions import normalize_command

    cmd = '"C:/Program Files/Python314/python.exe" -m pytest -q'
    assert normalize_command(cmd) == "python -m pytest -q"
    assert suggest_rule(BashTool(), {"command": cmd}, ctx) == "Bash(python:*)"
    p = pm(project, allow=["Bash(python:*)"])
    assert p.check(BashTool(), {"command": cmd}, ctx).action == "allow"
    assert p.check(BashTool(), {"command": "pythonx evil"}, ctx).action == "ask"
