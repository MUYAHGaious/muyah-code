"""Images end to end: Read, MCP screenshots and @mentions reach models that can see; others get a note."""

import base64
import json
import sys
from pathlib import Path

from fakeserver import FakeOpenAI, reply
from test_agent_e2e import make_app

from muyah_code.agent.context import ContextManager
from muyah_code.llm.anthropic_client import to_anthropic
from muyah_code.llm.content import IMAGE_TOKENS, image_part, image_size, model_sees_images, text_of

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def images_in(request) -> list[str]:
    return [p["image_url"]["url"] for m in request["messages"] if isinstance(m.get("content"), list)
            for p in m["content"] if p.get("type") == "image_url"]


def test_image_sizes_and_vision_detection():
    assert image_size(PNG) == (1, 1)
    gif = b"GIF89a" + (640).to_bytes(2, "little") + (480).to_bytes(2, "little") + b"\x00" * 10
    assert image_size(gif) == (640, 480)
    jpeg = (b"\xff\xd8\xff\xe0\x00\x10" + b"\x00" * 14 + b"\xff\xc0\x00\x11\x08" + (600).to_bytes(2, "big")
            + (800).to_bytes(2, "big") + b"\x00" * 10)
    assert image_size(jpeg) == (800, 600)
    assert model_sees_images("qwen2.5-vl-7b", None, "auto") and not model_sees_images("qwen2.5-coder:7b", None, "auto")
    assert model_sees_images("anything", False, True) and not model_sees_images("gpt-4o", True, "false")


def test_read_attaches_an_image_for_a_vision_model(project):
    (project / "shot.png").write_bytes(PNG)
    script = [reply("", [{"name": "Read", "arguments": {"file_path": "shot.png"}}]), reply("A single pixel.")]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project, vision=True)
        res = app.run_prompt("what is in shot.png?")
        app.shutdown()
    second = srv.requests[1]
    assert images_in(second) == ["data:image/png;base64," + base64.b64encode(PNG).decode()]
    tool = [m for m in second["messages"] if m["role"] == "tool"][-1]["content"]
    assert "shot.png (1×1, 0 KB) is an image; it is attached" in tool
    assert not any("_images" in m for m in second["messages"])            # private keys never leave
    assert res.text == "A single pixel."


def test_models_that_cannot_see_get_a_note_instead(project):
    (project / "shot.png").write_bytes(PNG)
    script = [reply("", [{"name": "Read", "arguments": {"file_path": "shot.png"}}]), reply("ok")]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project, vision=False)
        app.run_prompt("look")
        app.shutdown()
    assert images_in(srv.requests[1]) == []
    tool = [m for m in srv.requests[1]["messages"] if m["role"] == "tool"][-1]["content"]
    assert "not sent, because fake-model cannot see images" in tool


def test_image_mentions_attach_to_the_prompt(project):
    (project / "ui.png").write_bytes(PNG)
    with FakeOpenAI([reply("I see it.")]) as srv:
        app = make_app(srv, project, vision=True)
        app.run_prompt("does @ui.png look right?")
        app.shutdown()
    user = [m for m in srv.requests[0]["messages"] if m["role"] == "user"][-1]["content"]
    assert user[0]["type"] == "text" and "does @ui.png look right?" in user[0]["text"]
    assert "ui.png (1×1" in user[1]["text"] and user[2]["type"] == "image_url"
    assert "\x89PNG" not in json.dumps(srv.requests[0])                   # never inlined as text


def test_mcp_screenshots_reach_the_model(project):
    server = Path(__file__).with_name("fakemcp.py")
    (project / ".mcp.json").write_text(json.dumps({"mcpServers": {"browser": {
        "command": sys.executable, "args": [str(server)]}}}), encoding="utf-8")
    script = [reply("", [{"name": "mcp__browser__screenshot", "arguments": {}}]), reply("The page is blank.")]
    from muyah_code.app import App
    from muyah_code.config import load_config
    from test_agent_e2e import RecUI

    with FakeOpenAI(script) as srv:
        cfg = load_config(cwd=project, overrides={"base_url": srv.url, "model": "fake-model", "api_key": "k",
                                                  "vision": True})
        cfg.set("learning.reflect", False)
        app = App(cfg, RecUI(), cwd=project, mode="bypassPermissions", enable_mcp=True)
        try:
            app.run_prompt("check the page")
        finally:
            app.shutdown()
    assert len(images_in(srv.requests[1])) == 1
    tool = [m for m in srv.requests[1]["messages"] if m["role"] == "tool"][-1]["content"]
    assert "[image 1: image/png, attached]" in tool


def test_claude_gets_image_blocks_after_the_tool_results():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "look"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "t1", "content": "attached"},
                {"role": "user", "_images": True, "content": [{"type": "text", "text": "Images:"},
                                                              image_part("image/png", "QUJD")]}]
    _, out = to_anthropic(messages)
    last = out[-1]
    assert last["role"] == "user" and [b["type"] for b in last["content"]] == ["tool_result", "text", "image"]
    assert last["content"][2]["source"] == {"type": "base64", "media_type": "image/png", "data": "QUJD"}


def test_images_are_counted_and_pruned_first():
    ctx = ContextManager(32768, 4096)
    msg = {"role": "user", "_images": True, "content": [{"type": "text", "text": "x"}, image_part("image/png", "A" * 50000)]}
    assert ctx.raw_count([msg]) < IMAGE_TOKENS + 50            # an image costs a fixed estimate, not its base64
    history = [{"role": "system", "content": "s"}, msg] + [{"role": "tool", "content": "r" * 10} for _ in range(8)]
    assert ctx.prune(history, keep_recent=2) >= 1
    assert text_of(history[1]["content"]) == "x\n[image removed to save context]"


def test_dragging_an_image_into_the_prompt_makes_a_mention(project):
    from test_repl import run_session

    (project / "shots").mkdir()
    (project / "shots" / "a.png").write_bytes(PNG)
    keys = "\x1b[200~" + str(project / "shots" / "a.png") + "\x1b[201~what is this\r/exit\r"
    code, out, app = run_session(project, [keys], [reply("A pixel.")], lines=False)
    first_user = [m for m in app.agent.messages if m["role"] == "user"][0]["content"]
    text = first_user if isinstance(first_user, str) else first_user[0]["text"]
    assert text.startswith("@shots/a.png what is this")
