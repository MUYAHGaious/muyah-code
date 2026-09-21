"""Record a demo session for the visualization, and optionally serve its replay.

It runs the REAL agent (planning, a sub-agent, tools, a failing then passing test, a recalled lesson) against
the scripted local model server from tests/fakeserver.py, paced like a real model. The recording is the
same `.events.jsonl` file every session writes, so the replay is exactly what `muyah viz` shows.

    python scripts/make_viz_demo.py            # record, print the events file
    python scripts/make_viz_demo.py --serve    # record, then serve the replay and print its URL
    python scripts/make_viz_demo.py --live 10  # watch it live: prints a URL, starts the turn 10 s later
    python scripts/make_viz_demo.py --scenario long --live 10   # the long tour: most features, two turns
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
os.environ.setdefault("MUYAH_HOME", tempfile.mkdtemp(prefix="muyah-viz-demo-"))
for var in ("MUYAH_BASE_URL", "MUYAH_API_KEY", "MUYAH_MODEL", "MUYAH_PROFILE"):
    os.environ.pop(var, None)

from fakeserver import FakeOpenAI, reply  # noqa: E402

from muyah_code.app import App  # noqa: E402
from muyah_code.config import load_config  # noqa: E402
from muyah_code.learning.lessons import Lesson  # noqa: E402
from muyah_code.ui.base import UI  # noqa: E402

CALC = "def add(a, b):\n    return a + b\n\n\ndef divide(a, b):\n    return a * b\n"
TEST = ("from calc import add, divide\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_divide():\n    assert divide(6, 3) == 2\n")
PYTEST = "python -m pytest -q"


def todos(*states):
    names = ["Find the failing test", "Fix the bug", "Run the tests"]
    return {"todos": [{"content": n, "status": s} for n, s in zip(names, states, strict=True)]}


SCRIPT = [
    reply("I'll plan this first.", [{"name": "TodoWrite", "arguments": todos("in_progress", "pending", "pending")}],
          think=1.2),
    reply("Checking the tracker first.", [{"name": "mcp__tracker__lookup_issue", "arguments": {"id": "42"}}],
          think=0.8),
    reply("", [{"name": "Agent", "arguments": {
        "description": "Find divide code", "subagent_type": "explore",
        "prompt": "Find where divide() is defined and tested. Report the file paths and anything wrong."}}], think=0.9),
    # --- the explore sub-agent
    reply("", [{"name": "Grep", "arguments": {"pattern": "def divide", "output_mode": "content"}},
               {"name": "Glob", "arguments": {"pattern": "**/*.py"}}], think=0.8),
    reply("", [{"name": "Read", "arguments": {"file_path": "calc.py"}}], think=0.6),
    reply("divide() is in calc.py (line 5) and returns a * b instead of a / b. It is tested by test_divide in "
          "test_calc.py.", think=0.9),
    # --- back in the main agent
    reply("Let me confirm with the tests.", [{"name": "Bash", "arguments": {"command": PYTEST}}], think=0.8),
    reply("", [{"name": "Read", "arguments": {"file_path": "calc.py"}}], think=0.6),
    reply("divide() multiplies. Fixing it.", [{"name": "Edit", "arguments": {
        "file_path": "calc.py", "old_string": "return a * b", "new_string": "return a / b"}}], think=1.0),
    reply("", [{"name": "TodoWrite", "arguments": todos("completed", "completed", "in_progress")}], think=0.5),
    reply("", [{"name": "Bash", "arguments": {"command": PYTEST}}], think=0.7),
    reply("", [{"name": "TodoWrite", "arguments": todos("completed", "completed", "completed")}], think=0.5),
    reply("Fixed. `divide()` in **calc.py** returned `a * b`; it now returns `a / b`.\n\n"
          "- Found with an explore sub-agent (Grep + Read)\n- Confirmed by the failing `test_divide`\n"
          "- Both tests pass now (`python -m pytest -q`)", think=1.0),
]


class QuietUI(UI):
    def ask_permission(self, req):
        raise AssertionError("the demo runs in bypassPermissions mode")


def record(project: Path, live_wait: float | None = None) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    (project / "calc.py").write_text(CALC, encoding="utf-8")
    (project / "test_calc.py").write_text(TEST, encoding="utf-8")
    # an MCP server (a fake issue tracker) and a hook that runs after every Bash command
    (project / ".mcp.json").write_text(json.dumps({"mcpServers": {"tracker": {
        "command": sys.executable, "args": [str(ROOT / "tests" / "fakemcp.py"), "--delay", "0.8"]}}}), encoding="utf-8")
    (project / ".muyah").mkdir(exist_ok=True)
    (project / ".muyah" / "after_bash.py").write_text("import sys, json\njson.load(sys.stdin)\n", encoding="utf-8")
    hook = f'"{sys.executable}" "{project / ".muyah" / "after_bash.py"}"'
    (project / ".muyah" / "settings.json").write_text(json.dumps({"hooks": {"PostToolUse": [
        {"matcher": "Bash", "hooks": [{"type": "command", "command": hook}]}]}}), encoding="utf-8")
    with FakeOpenAI(SCRIPT, chunk_delay=0.035) as srv:
        cfg = load_config(cwd=project, overrides={"base_url": srv.url, "model": "qwen3-coder-30b", "api_key": "k"})
        cfg.set("learning.reflect", False)
        app = App(cfg, QuietUI(), cwd=project, mode="bypassPermissions", headless=True, enable_mcp=True)
        app.lessons.add(Lesson(trigger="fixing a failing test",
                               lesson="Run the failing test first to confirm the bug before editing.",
                               tags=["pytest", "test", "failing"]))
        server = None
        if live_wait is not None:
            from muyah_code.viz import VizServer

            server = VizServer(bus=app.events, title="calc-demo (live)")
            print(f"live: {server.start()}", flush=True)
            time.sleep(live_wait)  # time to open the page before the turn starts
        result = app.run_prompt("The divide test is failing. Find the bug and fix it.")
        app.shutdown()
        if server is not None:
            server.stop()
    if result.status != "ok":
        raise SystemExit(f"demo turn ended with status {result.status}")
    return app.session.path.with_suffix(".events.jsonl")


def record_long(project: Path, live_wait: float | None = None) -> Path:
    """The long scenario (scripts/demo_long.py): two turns, skills, MCP, two sub-agents, approvals, a hook,
    tests written first, a message typed while it works."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import demo_long

    project.mkdir(parents=True, exist_ok=True)
    for rel, text in demo_long.FILES.items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(text, encoding="utf-8")
    (project / ".mcp.json").write_text(json.dumps({"mcpServers": {"tracker": {
        "command": sys.executable, "args": [str(ROOT / "tests" / "fakemcp.py"), "--delay", "0.8"]}}}), encoding="utf-8")
    (project / ".muyah").mkdir(exist_ok=True)
    (project / ".muyah" / "after_bash.py").write_text("import sys, json\njson.load(sys.stdin)\n", encoding="utf-8")
    hook = f'"{sys.executable}" "{project / ".muyah" / "after_bash.py"}"'
    (project / ".muyah" / "settings.json").write_text(json.dumps({"hooks": {"PostToolUse": [
        {"matcher": "Bash", "hooks": [{"type": "command", "command": hook}]}]}}), encoding="utf-8")
    with FakeOpenAI(demo_long.script(), chunk_delay=0.03) as srv:
        cfg = load_config(cwd=project, overrides={"base_url": srv.url, "model": "qwen3-coder-30b", "api_key": "k"})
        cfg.set("learning.reflect", False)
        app = App(cfg, demo_long.DemoUser(), cwd=project, mode="default", enable_mcp=True)
        app.lessons.add(Lesson(trigger="adding a feature with tests",
                               lesson="Write the failing test first, then the code, then run the whole suite.",
                               tags=["test", "tests", "feature", "priority"]))
        server = None
        if live_wait is not None:
            from muyah_code.viz import VizServer

            server = VizServer(bus=app.events, title="tasks-demo (live)", recording=app.events.record_to)
            print(f"live: {server.start()}", flush=True)
            time.sleep(live_wait)
        results = demo_long.run(app)
        app.shutdown()
        if server is not None:
            server.stop()
    bad = [r.status for r in results if r.status != "ok"]
    if bad:
        raise SystemExit(f"demo turns ended with {bad}")
    return app.session.path.with_suffix(".events.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=["short", "long"], default="short",
                    help="short: fix one failing test (17 s); long: two turns using most features (~1 min)")
    ap.add_argument("--serve", action="store_true", help="Serve the replay after recording")
    ap.add_argument("--live", type=float, metavar="SECONDS",
                    help="Serve the live view while recording; wait SECONDS before starting the turn")
    ap.add_argument("--out", help="Copy the recording to this path")
    ap.add_argument("--project", help="Folder for the demo project (default: a new temp folder). Run "
                    "`muyah viz` there, with the same MUYAH_HOME, to watch from another terminal.")
    a = ap.parse_args()
    name = "tasks-demo" if a.scenario == "long" else "calc-demo"
    project = Path(a.project) if a.project else Path(tempfile.mkdtemp(prefix="muyah-viz-proj-")) / name
    path = (record_long if a.scenario == "long" else record)(project, a.live)
    if a.out:
        shutil.copyfile(path, a.out)
        path = Path(a.out)
    print(f"events: {path}", flush=True)
    if a.serve:
        from muyah_code.events import load_events
        from muyah_code.viz import VizServer

        server = VizServer(events=load_events(path), title="calc-demo")
        print(f"replay: {server.url}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
