"""The interface between the agent loop and whatever displays it (terminal, headless, tests)."""

from __future__ import annotations

from dataclasses import dataclass

from muyah_code.tools.base import ToolResult


@dataclass
class PermissionRequest:
    tool_name: str
    title: str
    detail: str
    reason: str
    suggested_rule: str
    kind: str


@dataclass
class PermissionReply:
    choice: str  # yes | always | project | no
    feedback: str = ""


class UI:
    """No-op base. Subclasses override what they render."""

    headless = False

    def assistant_start(self) -> None: ...
    def text(self, chunk: str) -> None: ...
    def reasoning(self, chunk: str) -> None: ...
    def assistant_end(self) -> None: ...
    def model_status(self, state: str) -> None: ...   # "sent" (waiting for the provider) / "first_token"
    def busy(self, label: str) -> None: ...           # work with no output of its own (keep the spinner going)
    def handoff(self, command: str, targets: list[str]) -> None: ...  # a delete the user should run themselves
    def compact_started(self, messages: int, yours: int, tokens: int) -> None: ...   # compaction begins
    def compact_progress(self, written: int, limit: int) -> None: ...   # summary tokens written so far

    def tool_allowed(self, how: str) -> None:   # the next tool ran without asking, and why
        pass

    def show_plan(self, plan: str) -> None:   # plan mode: the plan, before the user decides
        self.info(plan)

    def compact_finished(self, summary: str) -> None:   # what it did, one line
        self.info(summary)

    def tool_start(self, title: str) -> None: ...
    def tool_end(self, title: str, result: ToolResult) -> None: ...
    def on_todos(self, todos: list[dict]) -> None: ...
    def info(self, msg: str) -> None: ...
    def warn(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...

    def ask_permission(self, req: PermissionRequest) -> PermissionReply:
        return PermissionReply("no", "No interactive user is available to approve this action.")

    def ask_user(self, question: str, options: list[str]) -> str:
        return ""


class NullUI(UI):
    headless = True
