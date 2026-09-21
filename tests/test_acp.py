"""`muyah acp`: a fake editor drives the real agent over JSON-RPC pipes, the way Zed or a JetBrains IDE does."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from fakeserver import FakeOpenAI, reply

from muyah_code import acp

ROOT = Path(__file__).resolve().parent.parent


class FakeEditor:
    """Speaks ACP to the agent: sends requests, answers the agent's own requests, collects its updates."""

    def __init__(self, answers=None):
        a_in_r, a_in_w = os.pipe()
        a_out_r, a_out_w = os.pipe()
        self.to_agent = os.fdopen(a_in_w, "wb", buffering=0)
        self.from_agent = os.fdopen(a_out_r, "rb")
        self.agent = threading.Thread(target=acp.serve, args=(os.fdopen(a_in_r, "rb"), os.fdopen(a_out_w, "wb")),
                                      daemon=True)
        self.agent.start()
        self.answers = answers or {}        # method -> callable(params) -> result
        self.updates, self.asked, self.results = [], [], {}
        self._id = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _send(self, msg):
        self.to_agent.write((json.dumps({"jsonrpc": "2.0", **msg}) + "\n").encode())

    def _read(self):
        for line in self.from_agent:
            msg = json.loads(line)
            if msg.get("method") == "session/update":
                self.updates.append(msg["params"]["update"])
            elif "method" in msg:                       # the agent asks the editor something
                self.asked.append(msg)
                result = self.answers[msg["method"]](msg["params"])
                self._send({"id": msg["id"], "result": result})
            else:
                self.results[msg["id"]] = msg

    def call(self, method, params, timeout=60):
        self._id += 1
        rid = self._id
        self._send({"id": rid, "method": method, "params": params})
        end = time.time() + timeout
        while rid not in self.results and time.time() < end:
            time.sleep(0.02)
        msg = self.results[rid]
        assert "error" not in msg, msg
        return msg["result"]

    def notify(self, method, params):
        self._send({"method": method, "params": params})

    def close(self):
        self.to_agent.close()
        self.agent.join(10)


def settings(project):
    (project / ".muyah" / "settings.json").write_text(json.dumps({"learning": {"reflect": False}}))


def test_a_prompt_from_the_editor_with_approval_diff_and_plan(project, monkeypatch):
    settings(project)
    (project / "app.py").write_text("x = 2\n")
    script = [reply("", [{"name": "Read", "arguments": {"file_path": "app.py"}}]),
              reply("", [{"name": "Edit", "arguments": {"file_path": "app.py", "old_string": "x = 2",
                                                        "new_string": "x = 3"}}]),
              reply("Changed x to 3.")]
    with FakeOpenAI(script) as srv:
        monkeypatch.setenv("MUYAH_BASE_URL", srv.url)
        monkeypatch.setenv("MUYAH_MODEL", "fake-model")
        editor = FakeEditor({
            "session/request_permission": lambda p: {"outcome": {"outcome": "selected", "optionId": "allow-once"}},
            "fs/read_text_file": lambda p: {"content": "x = 2\n# unsaved note\n"},   # the editor's buffer
        })
        init = editor.call("initialize", {"protocolVersion": 1, "clientCapabilities": {"fs": {"readTextFile": True}}})
        assert init["protocolVersion"] == 1 and init["agentCapabilities"]["loadSession"] is True
        new = editor.call("session/new", {"cwd": str(project), "mcpServers": []})
        assert new["sessionId"] and new["modes"]["currentModeId"] == "default"
        done = editor.call("session/prompt", {"sessionId": new["sessionId"],
                                              "prompt": [{"type": "text", "text": "make x 3"}]})
        editor.close()
    assert done == {"stopReason": "end_turn"}
    kinds = [u["sessionUpdate"] for u in editor.updates]
    assert "tool_call" in kinds and "tool_call_update" in kinds and "agent_message_chunk" in kinds
    text = "".join(u["content"]["text"] for u in editor.updates if u["sessionUpdate"] == "agent_message_chunk")
    assert text == "Changed x to 3."
    # the model saw the editor's unsaved buffer, and the edit was approved in the editor
    assert "# unsaved note" in [m for m in srv.requests[1]["messages"] if m["role"] == "tool"][-1]["content"]
    asked = [a for a in editor.asked if a["method"] == "session/request_permission"]
    assert len(asked) == 1 and asked[0]["params"]["toolCall"]["kind"] == "edit"
    diff = [u for u in editor.updates if u["sessionUpdate"] == "tool_call_update" and u.get("content")
            and u["content"][0]["type"] == "diff"]
    assert diff and diff[0]["content"][0]["newText"] == "x = 3\n"
    call = [u for u in editor.updates if u["sessionUpdate"] == "tool_call" and u["kind"] == "edit"][0]
    assert call["locations"][0]["path"].endswith("app.py")


def test_the_editor_can_cancel_a_turn(project, monkeypatch):
    settings(project)
    with FakeOpenAI([reply("slow answer", think=10)]) as srv:
        monkeypatch.setenv("MUYAH_BASE_URL", srv.url)
        monkeypatch.setenv("MUYAH_MODEL", "fake-model")
        editor = FakeEditor()
        editor.call("initialize", {"protocolVersion": 1})
        sid = editor.call("session/new", {"cwd": str(project), "mcpServers": []})["sessionId"]
        result = {}
        t = threading.Thread(target=lambda: result.update(editor.call(
            "session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "think long"}]})))
        t.start()
        time.sleep(1.5)
        started = time.time()
        editor.notify("session/cancel", {"sessionId": sid})
        t.join(15)
        editor.close()
    assert result == {"stopReason": "cancelled"} and time.time() - started < 5


def test_modes_and_unknown_methods(project, monkeypatch):
    settings(project)
    with FakeOpenAI([]) as srv:
        monkeypatch.setenv("MUYAH_BASE_URL", srv.url)
        monkeypatch.setenv("MUYAH_MODEL", "fake-model")
        editor = FakeEditor()
        editor.call("initialize", {"protocolVersion": 1})
        sid = editor.call("session/new", {"cwd": str(project), "mcpServers": []})["sessionId"]
        editor.call("session/set_mode", {"sessionId": sid, "modeId": "plan"})
        editor._id += 1
        editor._send({"id": editor._id, "method": "no/such", "params": {}})
        time.sleep(0.5)
        editor.close()
    assert {"sessionUpdate": "current_mode_update", "currentModeId": "plan"} in editor.updates
    assert editor.results[editor._id]["error"]["code"] == -32601


def test_muyah_acp_speaks_only_protocol_on_stdout(isolated_home):
    env = {**os.environ, "MUYAH_HOME": str(isolated_home), "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run([sys.executable, "-m", "muyah_code", "acp"], input=json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1}}) + "\n",
        capture_output=True, text=True, env=env, timeout=60)
    lines = [json.loads(x) for x in p.stdout.splitlines()]
    assert lines == [{"jsonrpc": "2.0", "id": 1, "result": lines[0]["result"]}]
    assert lines[0]["result"]["agentInfo"]["name"] == "muyah-code"
