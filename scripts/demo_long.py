"""The long demo scenario for the live view: two turns that use most of what MUYAH-CODE can do.

Used by `scripts/make_viz_demo.py --scenario long`. The model is scripted (tests/fakeserver.py), but
everything else is the real agent: real files, real pytest runs, a real stdio MCP server, a real hook,
real permission checks (answered by the scripted user after a short pause), a message typed while it
works, skills and sub-agents.
"""

from __future__ import annotations

import time

from fakeserver import reply

from muyah_code.ui.base import UI, PermissionReply

TASKS_PY = '''from dataclasses import dataclass, field
from itertools import count

_ids = count(1)


@dataclass
class Task:
    title: str
    done: bool = False
    id: int = field(default_factory=lambda: next(_ids))
'''

API_PY = '''from tasks import Task

TASKS: list[Task] = []


def add_task(title: str) -> Task:
    task = Task(title)
    TASKS.append(task)
    return task


def list_tasks() -> list[Task]:
    return list(TASKS)
'''

TEST_TASKS_PY = '''from api import TASKS, add_task, list_tasks


def setup_function():
    TASKS.clear()


def test_add_and_list():
    add_task("write docs")
    assert [t.title for t in list_tasks()] == ["write docs"]
'''

README = "# tasks\n\nA tiny task list.\n"

TEST_PRIORITY_PY = '''from api import TASKS, add_task, list_tasks


def setup_function():
    TASKS.clear()


def test_default_priority_is_normal():
    assert add_task("a").priority == "normal"


def test_list_is_sorted_by_priority():
    add_task("low one", priority="low")
    add_task("urgent one", priority="high")
    add_task("normal one")
    assert [t.title for t in list_tasks()] == ["urgent one", "normal one", "low one"]
'''

FILES = {"tasks.py": TASKS_PY, "api.py": API_PY, "tests/test_tasks.py": TEST_TASKS_PY, "README.md": README,
         "conftest.py": "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parent))\n"}
PYTEST = "python -m pytest -q"

PROMPT_1 = "Add a priority (low/normal/high) to tasks, list them by priority, with tests. Issue #7 has details."
QUEUED = "also mention priorities in the README"
PROMPT_2 = "Review the change before I merge it."


def todos(*states):
    names = ["Read issue #7 and the code", "Write failing tests", "Implement priority", "Update the README",
             "Run the full test suite"]
    return {"todos": [{"content": n, "status": s} for n, s in zip(names, states, strict=True)]}


def script() -> list[dict]:
    t = 0.9  # model "thinking" before each reply
    return [
        # ---- turn 1
        reply("I'll start with the planning skill.", [{"name": "Skill", "arguments": {"skill": "planning"}}], think=t),
        reply("", [{"name": "TodoWrite", "arguments": todos("in_progress", "pending", "pending", "pending",
                                                            "pending")}], think=t),
        reply("", [{"name": "mcp__tracker__lookup_issue", "arguments": {"id": "7"}}], think=t),
        reply("", [{"name": "Agent", "arguments": {
            "description": "Map the task code", "subagent_type": "explore",
            "prompt": "Find where tasks are defined, created and listed, and how the tests are set up."}}], think=t),
        #   the explore sub-agent
        reply("", [{"name": "Glob", "arguments": {"pattern": "**/*.py"}},
                   {"name": "Grep", "arguments": {"pattern": "def ", "output_mode": "content"}}], think=0.7),
        reply("", [{"name": "Read", "arguments": {"file_path": "tasks.py"}},
                   {"name": "Read", "arguments": {"file_path": "api.py"}}], think=0.7),
        reply("Task is a dataclass in tasks.py (title, done, id). api.py has add_task() and list_tasks() over a "
              "module-level TASKS list. Tests in tests/ clear TASKS in setup_function.", think=0.8),
        #   back in the main agent: tests first
        reply("", [{"name": "TodoWrite", "arguments": todos("completed", "in_progress", "pending", "pending",
                                                            "pending")}], think=0.5),
        reply("Writing the tests first.", [{"name": "Write", "arguments": {
            "file_path": "tests/test_priority.py", "content": TEST_PRIORITY_PY}}], think=t),
        reply("", [{"name": "Bash", "arguments": {"command": PYTEST}}], think=0.6),
        reply("", [{"name": "TodoWrite", "arguments": todos("completed", "completed", "in_progress", "pending",
                                                            "pending")}], think=0.5),
        reply("They fail as expected. Reading the files I'll change.", [
            {"name": "Read", "arguments": {"file_path": "tasks.py"}},
            {"name": "Read", "arguments": {"file_path": "api.py"}},
            {"name": "Read", "arguments": {"file_path": "README.md"}}], think=0.6),
        reply("Adding the field.", [{"name": "Edit", "arguments": {
            "file_path": "tasks.py", "old_string": "    done: bool = False\n",
            "new_string": "    done: bool = False\n    priority: str = \"normal\"\n"}}], think=t),
        reply("", [{"name": "Edit", "arguments": {
            "file_path": "api.py",
            "old_string": "def add_task(title: str) -> Task:\n    task = Task(title)",
            "new_string": "PRIORITY = {\"high\": 0, \"normal\": 1, \"low\": 2}\n\n\n"
                          "def add_task(title: str, priority: str = \"normal\") -> Task:\n"
                          "    if priority not in PRIORITY:\n"
                          "        raise ValueError(f\"priority must be one of {list(PRIORITY)}\")\n"
                          "    task = Task(title, priority=priority)"}}], think=t),
        reply("", [{"name": "Edit", "arguments": {
            "file_path": "api.py", "old_string": "    return list(TASKS)",
            "new_string": "    return sorted(TASKS, key=lambda t: PRIORITY[t.priority])"}}], think=0.7),
        #   (the queued message "also mention priorities in the README" arrives here)
        reply("You asked for the README too.", [{"name": "Edit", "arguments": {
            "file_path": "README.md", "old_string": "A tiny task list.\n",
            "new_string": "A tiny task list. Tasks have a priority (low, normal, high); "
                          "`list_tasks()` returns the most urgent first.\n"}}], think=t),
        reply("", [{"name": "TodoWrite", "arguments": todos("completed", "completed", "completed", "completed",
                                                            "in_progress")}], think=0.5),
        reply("", [{"name": "Bash", "arguments": {"command": PYTEST}}], think=0.6),
        reply("", [{"name": "TodoWrite", "arguments": todos("completed", "completed", "completed", "completed",
                                                            "completed")}], think=0.5),
        reply("Done. Tasks now have a **priority** (`low`, `normal`, `high`, default `normal`):\n\n"
              "- `tasks.py`: new `priority` field\n- `api.py`: `add_task(title, priority)` validates it; "
              "`list_tasks()` returns the most urgent first\n- `tests/test_priority.py`: 2 new tests (failed "
              "first, pass now); all 3 tests pass\n- `README.md`: documents priorities, as you asked", think=1.0),
        # ---- turn 2
        reply("Loading the code-review skill.", [{"name": "Skill", "arguments": {"skill": "code-review"}}], think=t),
        reply("", [{"name": "Agent", "arguments": {
            "description": "Check edge cases", "subagent_type": "general",
            "prompt": "Check list_tasks() and add_task() for edge cases: invalid priority, stable order."}}],
              think=t),
        reply("", [{"name": "Read", "arguments": {"file_path": "api.py"}}], think=0.7),
        reply("", [{"name": "Bash", "arguments": {"command": "python -c \"import api; api.add_task('x', 'urgent')\""}}],
              think=0.7),
        reply("Invalid priorities raise ValueError with the allowed values; sorted() is stable, so equal "
              "priorities keep their insertion order. No issues found.", think=0.8),
        reply("Review: **ready to merge.** Invalid priorities are rejected with a clear error, the order is "
              "stable for equal priorities, and the tests cover the default and the sorting. One suggestion for "
              "later: accept priorities case-insensitively.", think=1.0),
    ]


class DemoUser(UI):
    """The scripted user: approves after a short look, and types one message while the agent works."""

    def __init__(self, think: float = 1.2):
        self.think = think
        self.steps = 0
        self.queue_at = 12  # deliver the typed message at this agent step (just before the README edit)

    def ask_permission(self, req):
        time.sleep(self.think)  # "reading" the request: the live view shows it waiting for approval
        return PermissionReply("always" if req.tool_name == "Bash" else "yes")

    def take_queued(self):
        self.steps += 1
        return [QUEUED] if self.steps == self.queue_at else []


def run(app) -> list:
    """Run both turns; returns the turn results."""
    results = [app.run_prompt(PROMPT_1)]
    time.sleep(1.0)
    results.append(app.run_prompt(PROMPT_2))
    return results
