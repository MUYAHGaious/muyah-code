"""What "auto" mode still asks about: shell commands that destroy data, publish, escalate, or pipe code
from the internet into a shell.

Auto mode runs everything else on its own (edits inside the project, builds, tests, installs, reads,
web research). It is a speed setting, not a sandbox: these patterns catch the common irreversible
mistakes, and your own `ask`/`deny` rules still apply first.
"""

from __future__ import annotations

import re

_RULES: list[tuple[re.Pattern, str]] = [(re.compile(p, re.IGNORECASE), why) for p, why in [
    # deleting in bulk / by force
    (r"(^|[\s;&|(])rm\s+(-[a-z]*[rf][a-z]*\b|--recursive|--force)", "deletes files recursively or by force"),
    (r"(^|[\s;&|(])(rd|rmdir)\s+(/s|-r)", "deletes a folder tree"),
    (r"(^|[\s;&|(])del\s+.*(/s|/q)", "deletes files in bulk"),
    (r"remove-item\b.*-(recurse|force)", "deletes files recursively or by force"),
    (r"(^|[\s;&|(])find\b.*\s-delete\b", "deletes every file it finds"),
    (r"(^|[\s;&|(])shred\b", "destroys file contents"),
    # git history and remotes
    (r"git\s+push\b", "publishes commits to a remote"),
    (r"git\s+reset\s+.*--hard|git\s+reset\s+--hard", "throws away uncommitted work"),
    (r"git\s+clean\s+.*-[a-z]*f", "deletes untracked files"),
    (r"git\s+(checkout|restore)\s+(.*\s)?(--\s+)?\.(\s|$)", "discards changes to every file"),
    (r"git\s+branch\s+.*-D\b", "force-deletes a branch"),
    (r"git\s+stash\s+(drop|clear)\b", "deletes stashed work"),
    (r"git\s+filter-(branch|repo)\b", "rewrites history"),
    # publishing / releasing
    (r"(npm|pnpm|yarn)\s+publish\b", "publishes a package"),
    (r"(twine\s+upload|uv\s+publish|poetry\s+publish|cargo\s+publish)\b", "publishes a package"),
    (r"docker\s+push\b|gh\s+release\s+create\b", "publishes an artifact"),
    # privilege and system changes
    (r"(^|[\s;&|(])(sudo|su|runas|doas)\s", "runs with elevated privileges"),
    (r"(^|[\s;&|(])(chmod|chown)\s+(-[a-z]*R|--recursive)", "changes permissions recursively"),
    (r"(^|[\s;&|(])(mkfs\S*|format|diskpart|fdisk)\b", "formats or partitions a disk"),
    (r"(^|[\s;&|(])dd\s+.*\bif=", "writes raw disk data"),
    (r"(^|[\s;&|(])(shutdown|reboot|halt|poweroff|stop-computer|restart-computer)\b", "shuts the machine down"),
    (r"(^|[\s;&|(])(killall|pkill|taskkill|stop-process)\b|kill\s+-9", "kills processes"),
    (r"(reg\s+(add|delete)|set-itemproperty\s+.*hk(lm|cu))", "changes the Windows registry"),
    (r"(pip|pip3|npm|choco|winget|apt(-get)?|brew)\s+(uninstall|remove|purge)\b", "uninstalls software"),
    # code from the internet straight into an interpreter
    (r"(curl|wget|iwr|irm|invoke-webrequest|invoke-restmethod)\b.*\|\s*(sh|bash|zsh|python\d?|iex|invoke-expression|pwsh|powershell)\b",
     "runs code downloaded from the internet"),
    # databases
    (r"\b(drop\s+(table|database|schema)|truncate\s+table)\b", "deletes database data"),
    # fork bomb, raw device writes
    (r":\(\)\s*\{", "fork bomb"),
    (r">\s*/dev/(sd|nvme|disk)", "writes to a raw disk"),
]]


# --------------------------------------------------------------------------- deletes (never run by the agent)

DELETE_PROGRAMS = {"rm", "rmdir", "rd", "del", "erase", "unlink", "shred", "srm", "wipe", "trash", "trash-put",
                   "remove-item", "ri", "rimraf", "deltree"}
_PREFIXES = {"sudo", "doas", "time", "nohup", "command", "builtin", "exec", "nice", "env", "&", "."}
_SEGMENT = re.compile(r"\s*(?:&&|\|\||;|\||\n|\$\(|`|\))\s*")
_INLINE_DELETE = re.compile(
    r"(os\.(remove|unlink|rmdir|removedirs)|shutil\.rmtree|\.unlink\s*\(|\.rmdir\s*\(|send2trash|"
    r"fs(?:\.promises)?\.(rm|rmSync|unlink|unlinkSync|rmdir|rmdirSync)|\brimraf\b|Remove-Item|"
    r"\[System\.IO\.(File|Directory)\]::Delete|File\.Delete|Directory\.Delete|FileUtils\.rm)", re.IGNORECASE)


def _program(tokens: list[str]) -> tuple[str, list[str]]:
    """The program a command segment runs (after sudo/env/VAR=x prefixes), and its arguments."""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        low = tok.lower()
        if low in _PREFIXES or ("=" in tok and not tok.startswith(("-", "/")) and "\\" not in tok.split("=")[0]):
            i += 1
            continue
        break
    if i >= len(tokens):
        return "", []
    prog = re.split(r"[\\/]", tokens[i].strip("'\""))[-1].lower()
    for ext in (".exe", ".cmd", ".bat", ".ps1"):
        if prog.endswith(ext):
            prog = prog[: -len(ext)]
    return prog, tokens[i + 1:]


def is_delete_command(command: str) -> bool:
    """True if any part of the command deletes files or folders. The agent never runs these: the user
    gets the command to run themselves (the user asked for this after agents wiped whole drives)."""
    if not command or not command.strip():
        return False
    if _INLINE_DELETE.search(command):
        return True
    for segment in _SEGMENT.split(command):
        tokens = segment.split()
        prog, args = _program(tokens)
        if prog == "xargs":  # xargs [flags] rm ...
            rest = [a for a in args if not a.startswith("-")]
            prog, args = _program(rest)
        if prog in DELETE_PROGRAMS:
            return True
        if prog == "git" and args and args[0].lower() in ("clean", "rm"):
            return True
        if prog in ("find", "fd"):
            low = [a.lower().strip("'\"") for a in args]
            if "-delete" in low or "--delete" in low:
                return True
            if any(a in ("-exec", "-execdir", "-x", "--exec") for a in low) and \
                    any(a in DELETE_PROGRAMS for a in low):
                return True
        if prog in ("cmd", "powershell", "pwsh", "bash", "sh", "zsh", "wsl") and args:
            # cmd /c del x · bash -c "rm -rf x" · pwsh -Command "..." : look at the command inside
            rest = list(args)
            while rest and rest[0].startswith(("-", "/")):
                rest.pop(0)
            inner = " ".join(rest).strip("'\"")
            if inner and is_delete_command(inner):
                return True
    return False


def delete_targets(command: str, cwd) -> list[str]:
    """Best-effort list of what a delete command would remove, with ~ and variables expanded."""
    import os
    from pathlib import Path

    targets: list[str] = []
    for segment in _SEGMENT.split(command):
        prog, args = _program(segment.split())
        if prog == "xargs":
            prog, args = _program([a for a in args if not a.startswith("-")])
        if prog not in DELETE_PROGRAMS and not (prog == "git" and args and args[0] in ("clean", "rm")):
            continue
        for arg in args:
            if arg.startswith("-") or arg.lower() in ("/s", "/q", "/f", "rm", "clean") or arg.startswith("2>"):
                continue
            arg = arg.strip("'\"")
            expanded = os.path.expandvars(os.path.expanduser(arg))
            path = Path(expanded)
            if not path.is_absolute():
                path = Path(cwd) / path
            targets.append(str(path))
    return targets


def risky_command(command: str) -> str | None:
    """Why this shell command needs a human even in auto mode, or None if it can run."""
    for pattern, why in _RULES:
        if pattern.search(command or ""):
            return why
    return None
