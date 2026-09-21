"""Lifecycle hooks (Claude Code compatible).

settings.json:
    {"hooks": {"PreToolUse": [{"matcher": "Bash|Edit", "hooks": [{"type": "command", "command": "...", "timeout": 60}]}]}}

The hook receives JSON on stdin:
    {session_id, transcript_path, cwd, permission_mode, hook_event_name, tool_name?, tool_input?, tool_response?, prompt?}

Exit code 0   -> continue. If stdout is JSON it may contain:
                   {"decision": "block", "reason": "..."}                      (any event)
                   {"hookSpecificOutput": {"permissionDecision": "allow|deny|ask",
                                           "permissionDecisionReason": "...",
                                           "updatedInput": {...}, "additionalContext": "..."}}
                 Plain (non-JSON) stdout from UserPromptSubmit/SessionStart is added as context.
Exit code 2   -> block. stderr is fed back to the model (PreToolUse: tool is not run;
                 UserPromptSubmit: prompt is rejected; Stop: the agent keeps working).
Other codes   -> non-blocking error, shown to the user.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from muyah_code.tools.shell import detect_shell, kill_tree, shell_env

EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure",
          "Stop", "SubagentStop", "PreCompact", "SessionEnd")
TOOL_EVENTS = {"PreToolUse", "PostToolUse", "PostToolUseFailure"}


@dataclass
class HookOutcome:
    blocked: bool = False
    reason: str = ""
    permission: str | None = None  # allow | deny | ask
    updated_input: dict | None = None
    context: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def additional_context(self) -> str:
        return "\n".join(c for c in self.context if c.strip())


class HookRunner:
    def __init__(self, config: dict | None, cwd: Path, shell_pref: str = "auto"):
        self.config = config or {}
        self.cwd = cwd
        self.shell_pref = shell_pref
        self.errors: list[str] = []
        self.events = None  # EventBus: each hook run is shown in the live view
        for event in self.config:
            if event not in EVENTS:
                self.errors.append(f"Unknown hook event '{event}' (known: {', '.join(EVENTS)})")

    def has(self, event: str) -> bool:
        return bool(self.config.get(event))

    def _matching(self, event: str, tool_name: str | None) -> list[dict]:
        out = []
        for group in self.config.get(event) or []:
            matcher = (group.get("matcher") or "").strip()
            if event in TOOL_EVENTS and matcher not in ("", "*"):
                try:
                    if not re.fullmatch(matcher, tool_name or ""):
                        continue
                except re.error:
                    if matcher != tool_name:
                        continue
            for h in group.get("hooks") or []:
                if h.get("type", "command") == "command" and h.get("command"):
                    out.append(h)
        return out

    def run(self, event: str, payload: dict, tool_name: str | None = None) -> HookOutcome:
        outcome = HookOutcome()
        hooks = self._matching(event, tool_name)
        if not hooks:
            return outcome
        data = json.dumps({"hook_event_name": event, "cwd": str(self.cwd), **payload}, default=str)
        shell = detect_shell(self.shell_pref)
        for h in hooks:
            started, warnings_before = time.time(), len(outcome.warnings)
            self._run_one(event, h, data, shell, outcome)
            if self.events is not None:
                result = ("blocked" if outcome.blocked else "warning" if len(outcome.warnings) > warnings_before
                          else "ok")
                self.events.emit("hook", event=event, tool=tool_name, command=h["command"][:160], outcome=result,
                                 duration=round(time.time() - started, 3))
            if outcome.blocked:
                return outcome
        return outcome

    def _run_one(self, event: str, h: dict, data: str, shell, outcome: HookOutcome) -> None:
        timeout = float(h.get("timeout", 60))
        try:
            proc = subprocess.Popen(
                shell.argv(h["command"]), cwd=str(self.cwd), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**shell_env(), "MUYAH_PROJECT_DIR": str(self.cwd)},
            )
        except OSError as e:
            outcome.warnings.append(f"{event} hook failed to start: {e}")
            return
        try:
            out, err = proc.communicate(data.encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            outcome.warnings.append(f"{event} hook timed out after {timeout:.0f}s: {h['command'][:80]}")
            return
        stdout = out.decode("utf-8", errors="replace").strip()
        stderr = err.decode("utf-8", errors="replace").strip()
        if proc.returncode == 2:
            outcome.blocked = True
            outcome.reason = stderr or stdout or f"blocked by {event} hook"
            return
        if proc.returncode != 0:
            outcome.warnings.append(f"{event} hook exited {proc.returncode}: {(stderr or stdout)[:300]}")
            return
        self._apply_stdout(event, stdout, outcome)

    @staticmethod
    def _apply_stdout(event: str, stdout: str, outcome: HookOutcome) -> None:
        if not stdout:
            return
        parsed = None
        if stdout.startswith("{"):
            try:
                parsed = json.loads(stdout)
            except json.JSONDecodeError:
                parsed = None
        if not isinstance(parsed, dict):
            if event in ("UserPromptSubmit", "SessionStart"):
                outcome.context.append(stdout)
            return
        if parsed.get("decision") == "block" or parsed.get("continue") is False:
            outcome.blocked = True
            outcome.reason = parsed.get("reason") or parsed.get("stopReason") or f"blocked by {event} hook"
        specific = parsed.get("hookSpecificOutput") or {}
        decision = specific.get("permissionDecision") or parsed.get("permissionDecision")
        if decision in ("allow", "deny", "ask"):
            outcome.permission = decision
            outcome.reason = specific.get("permissionDecisionReason") or outcome.reason
            if decision == "deny":
                outcome.blocked = True
        if isinstance(specific.get("updatedInput"), dict):
            outcome.updated_input = specific["updatedInput"]
        ctx = specific.get("additionalContext") or parsed.get("additionalContext")
        if ctx:
            outcome.context.append(str(ctx))
        if parsed.get("systemMessage"):
            outcome.warnings.append(str(parsed["systemMessage"]))
