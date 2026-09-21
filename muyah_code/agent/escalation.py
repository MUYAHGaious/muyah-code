"""When to hand a task to a bigger model: hard signals only, never a guess about "difficulty".

Research on routing (FrugalGPT's cascades, Aider's architect/editor split, and agents that retry on a stronger
model) points the same way: let the cheap model try, watch for concrete failure, then move up once, carrying
the failed attempt along so the bigger model does not repeat it. The signals here are all observable:

  * two malformed tool calls (the model cannot drive the tools),
  * the same file's Edit failing twice (it keeps guessing old_string),
  * the same failing command twice (no new idea),
  * NO_PROGRESS_STEPS steps in a row where every tool call failed.
"""

from __future__ import annotations

NO_PROGRESS_STEPS = 4


class Escalation:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.malformed = 0
        self.edit_failures: dict[str, int] = {}
        self.command_failures: dict[str, int] = {}
        self.failed_steps = 0
        self._reason = ""

    def malformed_call(self) -> None:
        self.malformed += 1
        if self.malformed >= 2:
            self._reason = self._reason or "two malformed tool calls"

    def tool_result(self, tool_name: str, args: dict, is_error: bool) -> None:
        if not is_error:
            return
        if tool_name in ("Edit", "Write"):
            path = str(args.get("file_path") or "")
            self.edit_failures[path] = self.edit_failures.get(path, 0) + 1
            if self.edit_failures[path] >= 2:
                self._reason = self._reason or f"the edit to {path} failed twice"
        elif tool_name == "Bash":
            command = " ".join(str(args.get("command") or "").split())
            self.command_failures[command] = self.command_failures.get(command, 0) + 1
            if self.command_failures[command] >= 2:
                self._reason = self._reason or f"the same command failed twice ({command[:60]})"

    def step(self, calls: int, errors: int) -> None:
        if calls and errors == calls:
            self.failed_steps += 1
            if self.failed_steps >= NO_PROGRESS_STEPS:
                self._reason = self._reason or f"{NO_PROGRESS_STEPS} steps in a row with only failing tool calls"
        elif calls:
            self.failed_steps = 0

    def take_reason(self) -> str:
        """The reason to escalate now ("" if none); counting starts over afterwards."""
        reason = self._reason
        if reason:
            self.reset()
        return reason
