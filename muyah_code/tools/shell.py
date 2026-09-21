"""Bash tool (Git Bash / bash / PowerShell), background jobs, and read-only command detection."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from muyah_code.tools.base import EXEC, READ, Tool, ToolError, ToolResult

IS_WINDOWS = sys.platform == "win32"
MAX_TIMEOUT = 1800


@dataclass(frozen=True)
class ShellSpec:
    kind: str  # bash | powershell | cmd
    executable: str

    def argv(self, command: str) -> list[str]:
        if self.kind == "bash":
            return [self.executable, "-c", command]
        if self.kind == "powershell":
            prelude = "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; $ProgressPreference='SilentlyContinue'; "
            return [self.executable, "-NoProfile", "-NonInteractive", "-Command", prelude + command]
        return [self.executable, "/d", "/s", "/c", command]


def _windows_git_bash() -> str | None:
    candidates = []
    git = shutil.which("git")
    if git:
        gp = Path(git).resolve()
        for up in (gp.parent.parent, gp.parent.parent.parent):
            candidates += [up / "bin" / "bash.exe", up / "usr" / "bin" / "bash.exe"]
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LOCALAPPDATA")):
        if base:
            candidates += [Path(base) / "Git" / "bin" / "bash.exe", Path(base) / "Programs" / "Git" / "bin" / "bash.exe"]
    for c in candidates:
        if c.is_file():
            return str(c)
    found = shutil.which("bash")
    # System32\bash.exe and WindowsApps\bash.exe are WSL launchers: a different filesystem view. Avoid.
    if found and "system32" not in found.lower() and "windowsapps" not in found.lower():
        return found
    return None


@lru_cache(maxsize=4)
def detect_shell(preference: str = "auto") -> ShellSpec:
    if preference == "powershell" or (IS_WINDOWS and preference == "auto" and not _windows_git_bash()):
        exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        return ShellSpec("powershell", exe)
    if preference == "cmd":
        return ShellSpec("cmd", os.environ.get("COMSPEC", "cmd.exe"))
    if IS_WINDOWS:
        bash = _windows_git_bash()
        if bash:
            return ShellSpec("bash", bash)
        return ShellSpec("powershell", shutil.which("powershell") or "powershell")
    return ShellSpec("bash", shutil.which("bash") or "/bin/sh")


def shell_env() -> dict:
    env = dict(os.environ)
    env.update({
        "MUYAH_CODE": "1",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "NO_COLOR": "1",
        "TERM": env.get("TERM", "dumb"),
    })
    return env


def _popen(argv: list[str], cwd: Path, stdout) -> subprocess.Popen:
    kwargs: dict = {}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        argv, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=stdout, stderr=subprocess.STDOUT,
        env=shell_env(), **kwargs,
    )


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=15)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def run_command(command: str, cwd: Path, timeout: float, shell: ShellSpec) -> tuple[int | None, str, bool]:
    """Run to completion. Returns (exit_code, output, timed_out). Ctrl+C kills the process tree."""
    proc = _popen(shell.argv(command), cwd, subprocess.PIPE)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out.decode("utf-8", errors="replace"), False
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        out, _ = proc.communicate()
        return None, (out or b"").decode("utf-8", errors="replace"), True
    except KeyboardInterrupt:
        kill_tree(proc)
        raise


# --------------------------------------------------------------------------- read-only detection

_SAFE_SIMPLE = {
    "ls", "dir", "pwd", "cat", "head", "tail", "wc", "which", "where", "whoami", "tree", "file", "stat",
    "du", "df", "env", "printenv", "date", "uname", "hostname", "type", "rg", "grep", "egrep", "fgrep",
    "diff", "cmp", "sort", "uniq", "basename", "dirname", "realpath", "readlink", "echo",
    "get-childitem", "get-content", "get-location", "get-command", "test-path", "select-string",
}
_SAFE_SUBCOMMANDS = {
    "git": {"status", "diff", "log", "show", "branch", "rev-parse", "ls-files", "blame", "remote", "describe",
            "shortlog", "tag", "config", "grep", "reflog"},
    "pip": {"list", "show", "freeze", "--version"}, "pip3": {"list", "show", "freeze", "--version"},
    "npm": {"ls", "list", "view", "--version", "-v"}, "node": {"--version", "-v"},
    "python": {"--version", "-V"}, "python3": {"--version", "-V"}, "py": {"--version", "-V"},
    "cargo": {"--version", "tree"}, "go": {"version", "env", "list"}, "uv": {"--version", "pip"},
}
_UNSAFE_GIT_FLAGS = {"--output", "-o", "--delete", "-d", "-D", "-m", "-M", "--set-upstream-to", "--unset",
                     "--add", "--replace-all", "--global", "--system"}
_META = re.compile(r"[;&|<>`$()]|\n")


def is_read_only_command(command: str) -> bool:
    cmd = command.strip()
    if not cmd or _META.search(cmd):
        return False
    try:
        parts = shlex.split(cmd, posix=True)
    except ValueError:
        return False
    if not parts:
        return False
    prog = Path(parts[0]).name.lower()
    if prog.endswith(".exe"):
        prog = prog[:-4]
    if prog in _SAFE_SUBCOMMANDS:
        if len(parts) < 2 or parts[1] not in _SAFE_SUBCOMMANDS[prog]:
            return False
        sub, rest = parts[1], parts[2:]
        if prog == "git":
            if any(p in _UNSAFE_GIT_FLAGS for p in rest):
                return False
            if sub == "config":
                return any(p in ("--get", "--list", "-l", "--get-all") for p in rest)
            if sub in ("branch", "tag"):
                return all(p.startswith("-") for p in rest)  # listing only
            if sub == "remote":
                return rest in ([], ["-v"], ["show"])
        if prog == "uv" and sub == "pip":
            return bool(rest) and rest[0] in ("list", "show", "freeze")
        return True
    if prog in _SAFE_SIMPLE:
        return True
    if prog == "find":
        return not any(p in ("-exec", "-execdir", "-delete", "-ok", "-fprint", "-fls") for p in parts)
    return False


# --------------------------------------------------------------------------- background jobs

@dataclass
class Job:
    id: str
    command: str
    proc: subprocess.Popen
    started: float = field(default_factory=time.time)
    chunks: list[str] = field(default_factory=list)
    read_pos: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def status(self) -> str:
        rc = self.proc.poll()
        return "running" if rc is None else f"exited with code {rc}"


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}

    def start(self, command: str, cwd: Path, shell: ShellSpec) -> Job:
        proc = _popen(shell.argv(command), cwd, subprocess.PIPE)
        job = Job(id="bg_" + uuid.uuid4().hex[:6], command=command, proc=proc)

        def pump():
            assert proc.stdout is not None
            for raw in iter(proc.stdout.readline, b""):
                with job.lock:
                    job.chunks.append(raw.decode("utf-8", errors="replace"))
            proc.stdout.close()

        threading.Thread(target=pump, daemon=True).start()
        self.jobs[job.id] = job
        return job

    def read(self, job_id: str, filter_re: str | None = None) -> tuple[Job, str]:
        job = self.jobs.get(job_id)
        if job is None:
            raise ToolError(f"No background job '{job_id}'. Known: {', '.join(self.jobs) or 'none'}")
        with job.lock:
            new = job.chunks[job.read_pos:]
            job.read_pos = len(job.chunks)
        text = "".join(new)
        if filter_re:
            rx = re.compile(filter_re)
            text = "".join(ln for ln in text.splitlines(keepends=True) if rx.search(ln))
        return job, text

    def kill(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise ToolError(f"No background job '{job_id}'.")
        kill_tree(job.proc)
        return job

    def shutdown(self) -> None:
        for job in self.jobs.values():
            kill_tree(job.proc)


# --------------------------------------------------------------------------- tools

class BashTool(Tool):
    name = "Bash"
    kind = EXEC

    def __init__(self, shell: ShellSpec | None = None):
        self._shell = shell

    def shell(self, ctx) -> ShellSpec:
        return self._shell or detect_shell(ctx.config.get("shell", "auto"))

    @property
    def description(self) -> str:  # type: ignore[override]
        spec = self._shell or detect_shell("auto")
        flavor = {
            "bash": "bash (on Windows this is Git Bash: use forward slashes and POSIX commands)",
            "powershell": "Windows PowerShell (use PowerShell syntax; `&&` may not work, use `;`)",
            "cmd": "cmd.exe",
        }[spec.kind]
        return (
            f"Run a shell command in {flavor}. Each call starts in the project directory; `cd` does not persist "
            "between calls, so chain with `cd dir && cmd` or use absolute paths. Output (stdout+stderr) is "
            "returned with the exit code. Default timeout 120s (max 1800). Set run_in_background=true for "
            "servers/watchers and read them with BashOutput. Never run interactive commands (editors, "
            "prompts, REPLs) - pass non-interactive flags (-y, --yes, CI=1). Prefer Read/Grep/Glob over "
            "cat/grep/find for reading files."
        )

    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run"},
            "description": {"type": "string", "description": "5-10 word summary of what it does"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 120, max 1800)"},
            "run_in_background": {"type": "boolean", "description": "Start and return immediately"},
        },
        "required": ["command"],
    }

    def permission_subject(self, args, ctx):
        return args.get("command", "")

    def title(self, args):
        cmd = args.get("command", "").replace("\n", " ")
        return f"Bash({cmd[:100]}{'...' if len(cmd) > 100 else ''})"

    def is_read_only(self, args):
        return is_read_only_command(args.get("command", "")) and not args.get("run_in_background")

    def run(self, args, ctx):
        command = args["command"]
        spec = self.shell(ctx)
        if args.get("run_in_background"):
            jobs: JobManager = ctx.service("jobs")
            if jobs is None:
                raise ToolError("Background jobs are not available in this context.")
            job = jobs.start(command, ctx.cwd, spec)
            time.sleep(1.0)
            _, first = jobs.read(job.id)
            return ToolResult(
                f"Started background job {job.id} ({job.status()}). Read output with BashOutput(job_id=\"{job.id}\")."
                + (f"\nInitial output:\n{first}" if first.strip() else ""),
                summary=f"Background job {job.id}",
            )
        timeout = min(int(args.get("timeout") or ctx.config.get("bash_timeout", 120)), MAX_TIMEOUT)
        code, out, timed_out = run_command(command, ctx.cwd, timeout, spec)
        out = out.rstrip()
        if timed_out:
            return ToolResult.error(
                f"Command timed out after {timeout}s and was killed.\n{out}\n"
                "If it is a long-running server or watcher, use run_in_background=true."
            )
        is_error = code != 0
        body = out if out else "(no output)"
        if is_error:
            body += f"\n[exit code {code}]"
        lines = out.count("\n") + 1 if out else 0
        summary = f"exit {code}" + (f", {lines} lines" if lines else "")
        return ToolResult(body, is_error=is_error, summary=summary, meta={"exit_code": code})


class BashOutputTool(Tool):
    name = "BashOutput"
    kind = READ
    description = "Read new output from a background job started with Bash(run_in_background=true)."
    parameters = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "description": "The job id, e.g. bg_1a2b3c"},
            "filter": {"type": "string", "description": "Optional regex; only return matching lines"},
        },
        "required": ["job_id"],
    }

    def run(self, args, ctx):
        jobs: JobManager = ctx.service("jobs")
        if jobs is None:
            raise ToolError("Background jobs are not available in this context.")
        job, text = jobs.read(args["job_id"], args.get("filter"))
        return ToolResult(f"[{job.id}: {job.status()}]\n{text or '(no new output)'}", summary=job.status())


class KillShellTool(Tool):
    name = "KillShell"
    kind = EXEC
    description = "Stop a background job started with Bash(run_in_background=true)."
    parameters = {
        "type": "object",
        "properties": {"job_id": {"type": "string", "description": "The job id"}},
        "required": ["job_id"],
    }

    def permission_subject(self, args, ctx):
        return args.get("job_id", "")

    def is_read_only(self, args):
        return True  # stopping our own job is always allowed

    def run(self, args, ctx):
        jobs: JobManager = ctx.service("jobs")
        if jobs is None:
            raise ToolError("Background jobs are not available in this context.")
        job = jobs.kill(args["job_id"])
        return ToolResult(f"Killed {job.id} ({job.command[:80]}).", summary="killed")


__all__ = ["BashTool", "BashOutputTool", "KillShellTool", "JobManager", "detect_shell", "is_read_only_command",
           "run_command"]
