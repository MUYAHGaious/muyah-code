import json

import pytest

import muyah_code.config as config
from muyah_code.config import ConfigError, load_config
from muyah_code.memory import load_instructions
from muyah_code.session import Checkpoints, Session
from muyah_code.skills.loader import SkillRegistry, parse_frontmatter


def test_layering_and_rule_concatenation(project, isolated_home):
    isolated_home.mkdir(parents=True, exist_ok=True)
    (isolated_home / "settings.json").write_text(json.dumps({
        "model": "user-model", "permissions": {"allow": ["Read"]}}))
    (project / ".muyah" / "settings.json").write_text(json.dumps({
        "model": "project-model", "permissions": {"allow": ["Bash(git *)"]}}))
    cfg = load_config(cwd=project)
    assert cfg["model"] == "project-model"
    assert cfg.get("permissions.allow") == ["Read", "Bash(git *)"]
    assert cfg.project_root == project.resolve()


def test_env_and_overrides(project, monkeypatch):
    monkeypatch.setenv("MUYAH_MODEL", "env-model")
    assert load_config(cwd=project)["model"] == "env-model"
    assert load_config(cwd=project, overrides={"model": "cli-model"})["model"] == "cli-model"


def test_profiles(project, isolated_home):
    isolated_home.mkdir(parents=True, exist_ok=True)
    (isolated_home / "settings.json").write_text(json.dumps({
        "profiles": {"colab": {"base_url": "https://x.pinggy.link/v1", "model": "qwen",
                               "learning": {"reflect": False}}},
        "profile": "colab"}))
    cfg = load_config(cwd=project)
    assert cfg["base_url"] == "https://x.pinggy.link/v1"
    assert cfg.get("learning.reflect") is False and cfg.get("learning.enabled") is True
    with pytest.raises(ConfigError):
        load_config(cwd=project, overrides={"profile": "missing"})


def test_legacy_migration(tmp_path, project, monkeypatch, isolated_home):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"base_url": "https://old.trycloudflare.com/v1", "model": "old", "api_key": "k",
                                  "command_timeout": 99}))
    monkeypatch.setattr(config, "LEGACY_CONFIG", legacy)
    cfg = load_config(cwd=project)
    assert cfg["base_url"] == "https://old.trycloudflare.com/v1"
    assert cfg["bash_timeout"] == 99
    assert "colab" in cfg.profiles()


def test_invalid_settings_file_is_reported(project):
    (project / ".muyah" / "settings.json").write_text("{nope")
    with pytest.raises(ConfigError):
        load_config(cwd=project)


def test_persist(project):
    cfg = load_config(cwd=project)
    cfg.persist("model", "saved", scope="user")
    assert load_config(cwd=project)["model"] == "saved"
    cfg.append_rule("allow", "Bash(npm test)")
    assert "Bash(npm test)" in load_config(cwd=project).get("permissions.allow")


def test_frontmatter():
    meta, body = parse_frontmatter("---\nname: x\ndescription: Use when y\n---\n# Body\n")
    assert meta == {"name": "x", "description": "Use when y"} and body.startswith("# Body")
    assert parse_frontmatter("no frontmatter") == ({}, "no frontmatter")


def test_bundled_and_project_skills(project, isolated_home):
    d = project / ".muyah" / "skills" / "deploy"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: deploy\ndescription: Deploy the app\ndisable-model-invocation: true\n"
                                "---\nRun ./deploy.sh $ARGUMENTS\n")
    reg = SkillRegistry.load(project, isolated_home, claude_compat=False)
    for name in ("debugging", "planning", "verification", "tdd", "code-review", "brainstorming", "commit"):
        assert reg.get(name) is not None, name
    s = reg.get("deploy")
    assert s.disable_model_invocation and "Run ./deploy.sh prod" in s.render("prod")
    assert "deploy" not in reg.index_text()  # user-only skills are hidden from the model
    assert not reg.errors


def test_instruction_files_and_imports(project, isolated_home):
    (project / "docs").mkdir()
    (project / "docs" / "style.md").write_text("Use tabs.")
    (project / "MUYAH.md").write_text("# Rules\nSee @docs/style.md\n")
    (project / "AGENTS.md").write_text("Agents rule")
    files = load_instructions(project, project, isolated_home)
    text = "\n".join(f.text for f in files)
    assert "Use tabs." in text and "Agents rule" in text


def test_session_roundtrip_and_repair(tmp_path):
    s = Session(tmp_path / "s")
    s.start({"cwd": "x"})
    s.log_message({"role": "user", "content": "hello world"})
    s.log_message({"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]})
    # crash before the tool result: the dangling call must be dropped on load
    msgs, meta = Session.load(s.path)
    assert msgs == [{"role": "user", "content": "hello world"}]
    assert meta["title"] == "hello world"
    s.log_replace([{"role": "user", "content": "summary"}], "compact")
    msgs, _ = Session.load(s.path)
    assert msgs == [{"role": "user", "content": "summary"}]
    s2, msgs2, _ = Session.resume(tmp_path / "s", s.id[:10])
    assert s2.id == s.id and msgs2 == msgs
    assert Session.list_sessions(tmp_path / "s")[0].title == "hello world"


def test_checkpoints_restore_first_state(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("v0")
    cp = Checkpoints()
    cp.begin_turn("t1")
    cp.record(f, b"v0")
    f.write_text("v1")
    cp.record(f, b"v1")  # second change in the same turn keeps the original snapshot
    f.write_text("v2")
    label, restored = cp.undo()
    assert label == "t1" and f.read_text() == "v0"
    assert cp.undo() is None


def test_global_muyah_home_is_not_a_project_marker(tmp_path, monkeypatch):
    from muyah_code.config import find_project_root

    user = tmp_path / "user"
    (user / ".muyah").mkdir(parents=True)                 # the global ~/.muyah
    monkeypatch.setenv("MUYAH_HOME", str(user / ".muyah"))
    sandbox = user / "Desktop" / "sandbox"
    sandbox.mkdir(parents=True)
    assert find_project_root(sandbox) == sandbox.resolve()   # not the whole home directory
    proj = user / "code" / "app"
    (proj / ".muyah").mkdir(parents=True)                  # a real project marker still counts
    (proj / "src").mkdir()
    assert find_project_root(proj / "src") == proj.resolve()


def test_append_writer_keeps_order_and_flushes(tmp_path):
    import threading

    from muyah_code.appendlog import append_line, flush

    path = tmp_path / "log.jsonl"
    for i in range(500):
        append_line(path, str(i))
    flush(path)
    assert path.read_text(encoding="utf-8").split() == [str(i) for i in range(500)]
    # writes from several threads all land, and close=True releases the file so it can be deleted
    threads = [threading.Thread(target=lambda k=k: [append_line(path, f"t{k}") for _ in range(50)]) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    flush(path, close=True)
    assert len(path.read_text(encoding="utf-8").split()) == 700
    path.unlink()                                   # would fail on Windows if the handle were still open


def test_requests_send_the_conversation_as_plain_json(project):
    """Messages go straight into the request body (the SDK's slow per-request transform is skipped)."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from fakeserver import FakeOpenAI, reply
    from test_agent_e2e import make_app

    with FakeOpenAI([reply("", [{"name": "Glob", "arguments": {"pattern": "*"}}]), reply("ok")]) as srv:
        app = make_app(srv, project)
        app.run_prompt("hi")
    first = srv.requests[0]
    assert first["messages"][0]["role"] == "system" and first["messages"][-1]["content"].endswith("hi")
    assert any(t["function"]["name"] == "Glob" for t in first["tools"])
    states = [e["state"] for e in app.events.history if e["type"] == "llm_status"]
    assert states[:2] == ["sent", "first_token"]          # the live view learns when it was really sent
