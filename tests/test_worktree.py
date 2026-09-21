"""Worktrees: a separate folder and branch per session (or sub-agent), removed only when nothing would be lost."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fakeserver import FakeOpenAI, reply

from muyah_code import worktree as wt

pytestmark = pytest.mark.skipif(not shutil.which("git"), reason="worktrees need git")
ROOT = Path(__file__).resolve().parent.parent


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")
    (r / "app.py").write_text("print('hi')\n")
    (r / ".gitignore").write_text(".env\n")
    (r / ".env").write_text("SECRET=1\n")
    (r / ".worktreeinclude").write_text("# untracked files a fresh checkout needs\n.env\n")
    git(r, "add", "app.py", ".gitignore", ".worktreeinclude")
    git(r, "commit", "-qm", "init")
    return r


def test_create_copies_includes_hides_the_folder_and_reuses(repo):
    path, created = wt.create(repo, "feat")
    assert created and path == repo / ".muyah" / "worktrees" / "feat"
    assert (path / "app.py").exists() and (path / ".env").read_text() == "SECRET=1\n"
    assert git(path, "branch", "--show-current").strip() == "muyah/feat"
    assert ".muyah" not in git(repo, "status", "--porcelain")          # the main checkout stays clean
    again, created_again = wt.create(repo, "feat")
    assert again == path and not created_again
    with pytest.raises(wt.WorktreeError, match="not a usable worktree name"):
        wt.create(repo, "../escape")


def test_finish_removes_a_clean_worktree_and_keeps_one_with_work(repo):
    clean, _ = wt.create(repo, "clean")
    assert wt.finish(clean, "clean") == "Worktree clean had no changes and was removed."
    assert not clean.exists() and "muyah/clean" not in git(repo, "branch")

    busy, _ = wt.create(repo, "busy")
    (busy / "new.py").write_text("x = 1\n")
    message = wt.finish(busy, "busy")
    assert message.startswith("Worktree busy kept (1 uncommitted file)") and busy.exists()
    git(busy, "add", "new.py")
    git(busy, "commit", "-qm", "work")
    assert "1 new commit" in wt.finish(busy, "busy") and busy.exists()
    assert "git merge muyah/busy" in message and "git worktree remove --force" in message


def test_headless_session_in_a_worktree(repo, isolated_home):
    script = [reply("", [{"name": "Write", "arguments": {"file_path": "feature.py", "content": "ok = True\n"}}]),
              reply("Added feature.py.")]
    with FakeOpenAI(script) as srv:
        env = {**os.environ, "MUYAH_HOME": str(isolated_home), "MUYAH_BASE_URL": srv.url, "MUYAH_MODEL": "fake-model",
               "MUYAH_API_KEY": "k", "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
        res = subprocess.run([sys.executable, "-m", "muyah_code", "-w", "feat", "-p", "add a feature",
                              "--mode", "bypassPermissions", "--output-format", "json"],
                             cwd=repo, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout)["result"] == "Added feature.py."          # stdout stays pure JSON
    assert "Worktree feat kept (1 uncommitted file)" in res.stderr
    assert (repo / ".muyah" / "worktrees" / "feat" / "feature.py").exists() and not (repo / "feature.py").exists()


def test_a_sub_agent_can_work_in_its_own_worktree(repo):
    (repo / ".muyah").mkdir(exist_ok=True)
    agents = repo / ".muyah" / "agents"
    agents.mkdir()
    (agents / "builder.md").write_text("---\nname: builder\ndescription: Builds things in isolation.\n"
                                       "isolation: worktree\n---\nBuild what you are asked.\n")
    script = [reply("", [{"name": "Agent", "arguments": {"prompt": "create lib.py", "subagent_type": "builder"}}]),
              reply("", [{"name": "Write", "arguments": {"file_path": "lib.py", "content": "x = 1\n"}}]),
              reply("Created lib.py."),
              reply("The builder made lib.py in its worktree.")]
    from test_agent_e2e import make_app

    with FakeOpenAI(script) as srv:
        app = make_app(srv, repo)
        res = app.run_prompt("build it")
        app.shutdown()
    report = [m for m in srv.requests[3]["messages"] if m["role"] == "tool"][-1]["content"]
    assert "Created lib.py." in report and "kept (1 uncommitted file)" in report
    made = list((repo / ".muyah" / "worktrees").glob("agent-builder-*/lib.py"))
    assert len(made) == 1 and not (repo / "lib.py").exists()
    assert res.text == "The builder made lib.py in its worktree."
