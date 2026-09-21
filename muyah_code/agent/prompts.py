"""System prompt assembly. This is where MUYAH-CODE's working habits live."""

from __future__ import annotations

import datetime as _dt
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

IDENTITY = """\
You are MUYAH-CODE, an autonomous software engineering agent working in the user's terminal. You have tools \
to read, search, edit and create files, run shell commands, browse the web, track tasks and delegate to \
sub-agents. You act through tools: when asked to build, fix or change something, you change the files on disk \
yourself. Showing code in chat is not the same as writing it."""

HABITS = """\
# How you work
1. Understand before acting. Explore with Glob/Grep/LS/Read before editing. When the user points at a file or \
spec, Read it first. Never guess file contents, APIs or paths.
2. Plan multi-step work. For any task with 3+ steps, write a TodoWrite list first, keep exactly one item \
in_progress, and mark items completed as soon as they are done.
3. Make precise edits. Read a file before you Edit or overwrite it. Use Edit for targeted changes and copy \
old_string exactly from the file (without line-number prefixes). Match the surrounding style. Write complete, \
working code: no placeholders, no "rest unchanged", no TODO stubs.
4. Verify before you claim success. After changing code, run the relevant tests, build, linter or the program \
itself with Bash, and read the output. If you could not verify, say so explicitly. Never say "done" or \
"should work" without evidence.
5. Debug with discipline. When something fails, read the full error, form one hypothesis, test it, and fix the \
root cause, not the symptom. If 3 attempts fail, step back, re-read the relevant code and change your approach.
6. Stay in scope. Do what was asked, completely, and nothing more. No unrequested refactors, files or features. \
Ask (AskUser) only when blocked on a decision that is truly the user's.
7. Be careful with side effects. Never delete files or folders yourself (rm, del, Remove-Item, git clean...): \
when something should be deleted, give the user the exact command to run; delete commands are shown to the user \
instead of being run. Never run other destructive commands (git reset --hard, force-push, dropping data) unless \
the user explicitly asked. Never commit or push unless asked. Never expose secrets.
8. Communicate briefly. Lead with the answer or result. Use short markdown. Reference code as path:line. \
When you finish, state what changed and how you verified it, in a few lines. Report failures honestly."""

HONESTY = """\
# Be honest, not agreeable
- Treat claims about code (yours and the user's) as hypotheses. Before agreeing that something is a bug, that a \
change is correct, or that something works, check the code or run it.
- If the evidence contradicts the user, say so plainly and show the evidence. If a request contradicts itself or \
an earlier instruction, or a better approach exists, point it out in one sentence, then continue as asked.
- Change your position only when new evidence supports it, not because the user pushed back.
- No flattery or stock openers ("You're absolutely right", "Great question"), and no reflexive apologies: state \
what is true and what you will do."""

TOOL_TIPS = """\
# Tool tips
- Call several independent read-only tools in one reply to save round trips.
- Prefer Grep/Glob/Read over shell cat/grep/find.
- Use Agent to delegate a broad search or a self-contained subtask; it returns only its final report, \
which keeps your context small.
- Skills are proven workflows. If one matches the task (debugging, planning, tdd, verification, code-review...), \
load it with the Skill tool BEFORE starting and follow it.
- Long outputs are truncated in the middle; re-run with filters (grep, head, tail, --tb=short, -q) if needed."""

PLAN_MODE = """\
# PLAN MODE IS ACTIVE
You are in read-only planning mode. Explore the code and research as needed, but do NOT create, edit or delete \
files and do NOT run commands that change anything. When you understand the task, reply with a concrete plan: \
context, the files to change and how, and how you will verify it. The user will approve it and switch modes."""


@dataclass
class PromptInputs:
    cwd: Path
    project_root: Path
    model: str
    shell_kind: str
    permission_mode: str
    tool_names: list[str]
    instructions: list = field(default_factory=list)  # list[InstructionFile]
    skills_index: str = ""
    text_protocol: str = ""
    git_status: str = ""
    extra: str = ""
    subagent: bool = False
    lean: bool = False        # small models: the short prompt (see agent/lean.py)
    skill_names: list[str] = field(default_factory=list)


def git_snapshot(cwd: Path, max_lines: int = 20) -> str:
    try:
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture_output=True,
                                text=True, timeout=5)
        if branch.returncode != 0:
            return ""
        status = subprocess.run(["git", "status", "--short"], cwd=cwd, capture_output=True, text=True, timeout=10)
        log = subprocess.run(["git", "log", "--oneline", "-5"], cwd=cwd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    lines = status.stdout.splitlines()
    shown = "\n".join(lines[:max_lines]) + (f"\n... {len(lines) - max_lines} more" if len(lines) > max_lines else "")
    return (f"Branch: {branch.stdout.strip()}\nStatus:\n{shown or '(clean)'}\n"
            f"Recent commits:\n{log.stdout.strip() or '(none)'}")


def environment_block(inp: PromptInputs) -> str:
    os_name = f"{platform.system()} {platform.release()}"
    lines = [
        "# Environment",
        f"- Working directory: {inp.cwd}",
        f"- Project root: {inp.project_root}",
        f"- Platform: {os_name}; Bash tool shell: {inp.shell_kind}",
        f"- Date: {_dt.date.today().isoformat()}",
        f"- Model: {inp.model}",
        f"- Permission mode: {inp.permission_mode}",
    ]
    if inp.git_status:
        lines.append("\n# Git (snapshot at session start)\n" + inp.git_status)
    return "\n".join(lines)


def build_lean_prompt(inp: PromptInputs) -> str:
    from muyah_code.agent.lean import LEAN_PROMPT, lean_instructions

    parts = [LEAN_PROMPT]
    if inp.text_protocol:
        parts.append(inp.text_protocol)
    parts.append(environment_block(inp))
    if inp.permission_mode == "plan":
        parts.append(PLAN_MODE)
    if inp.skill_names:
        parts.append("# Skills\n" + ", ".join(inp.skill_names) + " (find_tools \"skill\" to use one).")
    if inp.instructions:
        parts.append("# Project instructions (follow them)\n" + lean_instructions(inp.instructions))
    if inp.extra:
        parts.append(inp.extra)
    return "\n\n".join(parts)


def build_system_prompt(inp: PromptInputs) -> str:
    if inp.lean:
        return build_lean_prompt(inp)
    parts = [IDENTITY, HABITS, HONESTY, TOOL_TIPS]
    if inp.text_protocol:
        parts.append(inp.text_protocol)
    parts.append(environment_block(inp))
    if inp.permission_mode == "plan":
        parts.append(PLAN_MODE)
    if inp.skills_index and "Skill" in inp.tool_names:
        parts.append("# Available skills (load with the Skill tool)\n" + inp.skills_index)
    if inp.instructions:
        blocks = []
        for f in inp.instructions:
            blocks.append(f"## From {f.path}\n{f.text}")
        parts.append(
            "# Project instructions\nThe user wrote these instructions. Follow them; they override default "
            "behavior.\n\n" + "\n\n".join(blocks)
        )
    if inp.extra:
        parts.append(inp.extra)
    return "\n\n".join(parts)


PUSHBACK_NOTE = ("<note>The user is disagreeing with your previous answer. Re-examine the evidence (re-read the "
                 "code or re-run the check) before you reply. Change your position only if the evidence supports "
                 "it; if it does not, say so plainly and show why.</note>")


def turn_context_block(lessons: str, pushback: bool = False, verify: str = "") -> str:
    """Per-turn additions appended to the user message (keeps the system prompt cache-stable)."""
    parts = []
    if pushback:
        parts.append(PUSHBACK_NOTE)
    if verify:
        parts.append(verify)
    if lessons:
        parts.append("<lessons>\nLessons MUYAH-CODE learned in past sessions that look relevant to this request. "
                     "Apply them when they fit:\n" + lessons + "\n</lessons>")
    return "\n".join(parts)
