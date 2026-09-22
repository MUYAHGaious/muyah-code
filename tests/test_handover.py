"""Continuing work started in another AI tool: reading its conversation, and what enters the context."""

import json
import sqlite3

from muyah_code.handover import brief as B
from muyah_code.handover.sources import CLAUDE, CODEX, OPENCODE, discover


def claude_store(home, project, messages):
    import re

    folder = home / ".claude" / "projects" / re.sub(r"[\\/:]+", "-", str(project.resolve()))
    folder.mkdir(parents=True)
    path = folder / "abc123.jsonl"
    rows = [{"type": "ai-title", "aiTitle": "discount codes", "sessionId": "abc123"}]
    rows += messages
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_claude_code_conversation_is_read_with_its_tool_calls(tmp_path):
    home, project = tmp_path / "home", tmp_path / "shop"
    project.mkdir()
    claude_store(home, project, [
        {"type": "user", "cwd": str(project), "message": {"role": "user", "content": "add discount codes"}},
        {"type": "assistant", "cwd": str(project), "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Looking at the cart."},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "cart.py"}}]}},
        {"type": "user", "cwd": str(project), "isSidechain": True,
         "message": {"role": "user", "content": "a sub-agent's own conversation"}},
        {"type": "summary", "summary": "Earlier: chose percentage discounts."},
    ])
    found = discover(project, home)
    assert [c.tool for c in found] == [CLAUDE]
    conv = found[0]
    assert conv.title == "discount codes" and conv.turns == 2
    turns = conv.load()
    assert [t.role for t in turns] == ["user", "assistant", "user"]
    assert turns[1].tools == ["Read(cart.py)"] and "Looking at the cart." in turns[1].text
    assert turns[2].kind == "summary"                       # its own compaction summary, kept apart
    assert all("sub-agent's own" not in t.text for t in turns)


def test_codex_and_opencode_are_read_too(tmp_path):
    home, project = tmp_path / "home", tmp_path / "shop"
    project.mkdir()
    sessions = home / ".codex" / "sessions" / "2026" / "09" / "22"
    sessions.mkdir(parents=True)
    rows = [
        {"timestamp": "2026-09-22T10:00:00Z", "type": "session_meta", "payload": {"id": "x", "cwd": str(project)}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
                                              "content": [{"type": "input_text", "text": "seed the database"}]}},
        {"type": "response_item", "payload": {"type": "function_call", "name": "shell",
                                              "arguments": json.dumps({"command": "python seed.py"})}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                              "content": [{"type": "output_text", "text": "Seeded 4 items."}]}},
    ]
    (sessions / "rollout-2026-09-22T10-00-00-x.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    db = home / ".local" / "share" / "opencode" / "opencode.db"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE session (id TEXT, directory TEXT, title TEXT, time_updated INTEGER)")
    con.execute("CREATE TABLE message (id TEXT, session_id TEXT, time_created INTEGER, data TEXT)")
    con.execute("CREATE TABLE part (id TEXT, message_id TEXT, session_id TEXT, time_created INTEGER, data TEXT)")
    con.execute("INSERT INTO session VALUES ('s1', ?, 'checkout flow', 1790000000000)", (str(project),))
    con.execute("INSERT INTO message VALUES ('m1', 's1', 1, ?)", (json.dumps({"role": "user"}),))
    con.execute("INSERT INTO part VALUES ('p1', 'm1', 's1', 1, ?)",
                (json.dumps({"type": "text", "text": "build checkout"}),))
    con.commit()
    con.close()

    found = discover(project, home)
    assert {c.tool for c in found} == {CODEX, OPENCODE}
    codex = next(c for c in found if c.tool == CODEX)
    turns = codex.load()
    assert turns[0].text == "seed the database"
    assert turns[-1].tools == ["shell(python seed.py)"] and "Seeded 4 items." in turns[-1].text
    opencode = next(c for c in found if c.tool == OPENCODE)
    assert opencode.title == "checkout flow" and opencode.load()[0].text == "build checkout"


def test_what_enters_the_context_is_capped_and_marked_second_hand(tmp_path):
    home, project = tmp_path / "home", tmp_path / "shop"
    project.mkdir()
    long_talk = [{"type": "user", "cwd": str(project),
                  "message": {"role": "user", "content": f"request number {i} " + "word " * 200}} for i in range(40)]
    claude_store(home, project, long_talk)
    conv = discover(project, home)[0]
    turns = conv.load()

    big = B.build([(conv, turns)], budget_tokens=2000)
    text = big.messages[0]["content"]
    assert big.tokens <= 2000 and "brief built here" in big.report[0]
    assert "second-hand" in text and "repository and `git log` are the truth" in text
    assert "request number 39" in text                      # the most recent exchanges are kept word for word
    assert big.messages[1]["role"] == "assistant"

    small = B.build([(conv, turns)], budget_tokens=200_000, mode="full")
    assert "copied as it is" in small.report[0] and small.tokens > big.tokens


def test_several_tools_in_one_folder_keep_their_names_and_dates(tmp_path):
    home, project = tmp_path / "home", tmp_path / "shop"
    project.mkdir()
    claude_store(home, project, [
        {"type": "user", "cwd": str(project), "message": {"role": "user", "content": "use phone sign-in"}},
        {"type": "assistant", "cwd": str(project), "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Edit", "input": {"file_path": "auth.py"}}]}}])
    sessions = home / ".codex" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "rollout-old.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"type": "session_meta", "payload": {"cwd": str(project)}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "email sign-in first"}]}},
        {"type": "response_item", "payload": {"type": "function_call", "name": "apply_patch",
                                              "arguments": json.dumps({"path": "auth.py"})}},
    ]), encoding="utf-8")
    pairs = [(c, c.load()) for c in discover(project, home)]
    assert len(pairs) == 2
    out = B.build(pairs, budget_tokens=4000)
    text = out.messages[0]["content"]
    assert CLAUDE in text and CODEX in text                  # every block says which tool and when
    clashes = B.conflicts(pairs)
    assert clashes and "auth.py" in clashes[0] and CLAUDE in clashes[0] and CODEX in clashes[0]
