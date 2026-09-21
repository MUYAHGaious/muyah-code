"""Rewind: code and/or conversation back to before any turn, including files changed by commands, after
a restart, and without ever touching the project's own git repository."""

import shutil
import subprocess
import sys

import pytest
from fakeserver import FakeOpenAI, reply
from test_agent_e2e import make_app

from muyah_code.rewind import MAX_FILE_BYTES

pytestmark = pytest.mark.skipif(not shutil.which("git"), reason="rewind snapshots need git")


def write(name, content):
    return {"name": "Write", "arguments": {"file_path": name, "content": content}}


def bash(cmd):
    return {"name": "Bash", "arguments": {"command": cmd}}


def two_turns(project):
    """Turn 1 writes new.txt and edits a.txt; turn 2 runs a command that creates b.txt and changes a.txt."""
    (project / "a.txt").write_text("original\n")
    (project / "mk.py").write_text("open('b.txt', 'w').write('from a command\\n')\n"
                                   "open('a.txt', 'a').write('appended by a command\\n')\n")
    return [
        reply("", [{"name": "Read", "arguments": {"file_path": "a.txt"}}]),
        reply("", [write("new.txt", "made in turn 1\n"),
                   {"name": "Edit", "arguments": {"file_path": "a.txt", "old_string": "original",
                                                  "new_string": "edited in turn 1"}}]),
        reply("turn 1 done"),
        reply("", [bash(f'"{sys.executable}" mk.py')]),
        reply("turn 2 done"),
    ]


def test_rewind_code_and_conversation_including_command_changes(project):
    with FakeOpenAI(two_turns(project)) as srv:
        app = make_app(srv, project)
        app.run_prompt("turn one")
        app.run_prompt("turn two")
        assert (project / "b.txt").exists() and "appended by a command" in (project / "a.txt").read_text()
        rw = app.rewind
        assert [p.turn for p in rw.turns()] == [1, 2] and rw.covers_commands
        before_two = len(rw.turns()[1:2]) and rw.turns()[1]
        msgs_before_two = before_two.msg_index

        result = rw.restore(before_two, code=True, conversation=True, agent=app.agent)
        assert not (project / "b.txt").exists()                        # made by a command: gone
        assert (project / "a.txt").read_text() == "edited in turn 1\n"  # the command's change undone
        assert (project / "new.txt").exists()                            # turn 1's work is kept
        assert len(app.agent.messages) == msgs_before_two and result["conversation"]
        assert ("D", "b.txt") in result["code"]

        # undo the rewind: everything comes back, conversation included
        back = rw.undo_rewind(app.agent)
        assert back["conversation"] and (project / "b.txt").exists()
        assert "appended by a command" in (project / "a.txt").read_text()
        assert app.agent.messages[-1]["content"] == "turn 2 done"

        # code only, back to before turn 1: files as at the start; the conversation stays
        n = len(app.agent.messages)
        rw.restore(rw.turns()[0], code=True, conversation=False, agent=app.agent)
        assert (project / "a.txt").read_text() == "original\n" and not (project / "new.txt").exists()
        assert len(app.agent.messages) == n
        app.shutdown()


def test_conversation_only_rewind_keeps_files(project):
    with FakeOpenAI(two_turns(project)) as srv:
        app = make_app(srv, project)
        app.run_prompt("turn one")
        app.run_prompt("turn two")
        point = app.rewind.turns()[1]
        app.rewind.restore(point, code=False, conversation=True, agent=app.agent)
        assert (project / "b.txt").exists()                  # files untouched
        assert len(app.agent.messages) == point.msg_index    # the conversation went back
        app.shutdown()


def test_rewind_works_after_a_restart(project):
    with FakeOpenAI(two_turns(project)) as srv:
        app = make_app(srv, project)
        app.run_prompt("turn one")
        app.run_prompt("turn two")
        session_id = app.session.id
        app.shutdown()
    from test_agent_e2e import RecUI

    from muyah_code.app import App
    from muyah_code.config import load_config

    with FakeOpenAI([]) as srv:  # a new process: MUYAH-CODE restarted with --resume
        cfg = load_config(cwd=project, overrides={"base_url": srv.url, "model": "fake-model", "api_key": "k"})
        again = App(cfg, RecUI(), cwd=project, mode="bypassPermissions", enable_mcp=False, resume=session_id)
        turns = again.rewind.turns()
        assert [p.turn for p in turns] == [1, 2] and all(p.sha for p in turns)
        result = again.rewind.restore(turns[1], code=True, conversation=True, agent=again.agent)
        assert not (project / "b.txt").exists() and result["conversation"]
        assert again.agent.messages[-1]["content"] == "turn 1 done"
        again.shutdown()


def test_the_projects_own_git_repo_is_never_touched(project):
    def git(*args):
        return subprocess.run(["git", *args], cwd=project, capture_output=True, text=True).stdout.strip()

    script = two_turns(project)          # creates a.txt and mk.py; both go into the user's first commit
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"], cwd=project, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], cwd=project,
                   check=True)
    head, branches, status = git("rev-parse", "HEAD"), git("branch", "-a"), git("status", "--porcelain")
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project)
        app.run_prompt("turn one")
        app.run_prompt("turn two")
        app.rewind.restore(app.rewind.turns()[0], code=True, conversation=True, agent=app.agent)
        app.shutdown()
    assert git("rev-parse", "HEAD") == head and git("branch", "-a") == branches   # no commits, no branches
    assert git("status", "--porcelain") == status                                   # the index is as it was
    assert not str(app.rewind.shadow.git_dir).startswith(str(project))             # the shadow repo is elsewhere


def test_a_sub_agent_does_not_split_the_users_turn(project):
    script = [
        reply("", [{"name": "Agent", "arguments": {"description": "write a file", "subagent_type": "general",
                                                   "prompt": "create sub.txt"}}]),
        reply("", [write("sub.txt", "by the sub-agent\n")]),   # the sub-agent
        reply("sub-agent done"),
        reply("", [write("main.txt", "by the main agent\n")]),
        reply("all done"),
    ]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project)
        app.run_prompt("do it")
        assert len(app.rewind.turns()) == 1                       # one turn, not split by the sub-agent
        assert (project / "sub.txt").exists() and (project / "main.txt").exists()
        app.undo()
        assert not (project / "sub.txt").exists() and not (project / "main.txt").exists()
        app.shutdown()


def test_after_compaction_earlier_turns_rewind_code_only(project):
    with FakeOpenAI(two_turns(project)) as srv:
        app = make_app(srv, project)
        app.run_prompt("turn one")
        app.agent.clear()                                         # like /clear (compaction does the same)
        point = app.rewind.turns()[0]
        assert not app.rewind.can_rewind_conversation(point, app.agent)
        result = app.rewind.restore(point, code=True, conversation=True, agent=app.agent)
        assert not result["conversation"] and not (project / "new.txt").exists()
        app.shutdown()


def test_big_files_are_left_out_of_snapshots(project):
    big = project / "data.bin"
    with open(big, "wb") as f:
        f.truncate(MAX_FILE_BYTES + 1)
    with FakeOpenAI([reply("ok")]) as srv:
        app = make_app(srv, project)
        app.run_prompt("hi")
        app.rewind.wait_ready()
        assert "data.bin" in app.rewind.shadow.skipped
        app.shutdown()
