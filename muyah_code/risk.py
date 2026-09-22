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
# delete calls written directly in a shell line (PowerShell / .NET), looked for OUTSIDE quoted text only
_SHELL_INLINE = re.compile(r"\[System\.IO\.(File|Directory)\]::Delete|\bFileUtils\.rm", re.IGNORECASE)
# delete calls in JavaScript / Ruby / Perl one-liners, looked for in their code with string literals blanked
_CODE_DELETE = re.compile(
    r"\.(rm|rmSync|unlink|unlinkSync|rmdir|rmdirSync)\s*\(|\brimraf\s*\(|"   # fs.rmSync, require('fs').rmSync
    r"\bFileUtils\.(rm|rm_rf|remove_dir)|\bFile\.(delete|unlink)\s*\(|\bDir\.(delete|rmdir)\s*\(|\bunlink\s*\(", re.I)
PYTHON = {"python", "python3", "py", "pypy", "pypy3"}
SCRIPTING = {"node": ("-e", "--eval", "-p", "--print"), "deno": ("eval",), "bun": ("-e", "--eval"),
             "ruby": ("-e",), "perl": ("-e", "-E")}
SHELLS = {"cmd", "powershell", "pwsh", "bash", "sh", "zsh", "wsl", "dash"}


def mask_quotes(text: str) -> str:
    """The same text with the inside of every quoted string blanked out (same length). What is left is what the
    shell itself acts on: `echo "rm -rf /"` prints text, it does not delete anything."""
    out, quote, i = [], "", 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < len(text):
                out.append("  ")
                i += 2
                continue
            if ch == quote:
                quote = ""
                out.append(ch)
            else:
                out.append(" ")         # a line break inside quotes is text too (a multi-line python -c)
        else:
            if ch in "'\"":
                quote = ch
            out.append(ch)
        i += 1
    return "".join(out)


_SPLIT = re.compile(r"&&|\|\||;|\||\n|\$\(|`|\)")


def segments(command: str) -> list[str]:
    """The command's parts, split where the SHELL splits it (never inside quotes: a python -c script with
    many lines is one part)."""
    masked = mask_quotes(command)
    parts, last = [], 0
    for m in _SPLIT.finditer(masked):
        parts.append(command[last:m.start()])
        last = m.end()
    parts.append(command[last:])
    return [p.strip() for p in parts if p.strip()]


def _tokens(segment: str) -> list[str]:
    import shlex

    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


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


def python_deletes(code: str) -> bool:
    """True if Python code deletes files: os.remove/unlink/rmdir/removedirs, shutil.rmtree, Path.unlink/rmdir,
    send2trash, or a subprocess/os.system call that runs a delete. Text inside strings is just text (a route
    called '/api/admin/remove-item', or SQL like 'DELETE FROM items', deletes no file)."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return bool(_CODE_DELETE.search(mask_quotes(code)) or
                    re.search(r"\b(os\.(remove|unlink|rmdir|removedirs)|shutil\.rmtree)\s*\(|\.(unlink|rmdir)\s*\(",
                              mask_quotes(code)))

    def dotted(node) -> str:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func)
        last = name.rsplit(".", 1)[-1]
        if name in ("os.remove", "os.unlink", "os.rmdir", "os.removedirs", "shutil.rmtree", "send2trash",
                    "send2trash.send2trash") or last in ("rmtree", "unlink", "rmdir", "removedirs"):
            return True
        if last in ("system", "run", "call", "Popen", "check_call", "check_output", "getoutput", "getstatusoutput"):
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str) and is_delete_command(first.value):
                return True
            if isinstance(first, (ast.List, ast.Tuple)) and first.elts and isinstance(first.elts[0], ast.Constant) \
                    and str(first.elts[0].value).lower() in DELETE_PROGRAMS:
                return True
    return False


def _inline_code(prog: str, args: list[str]) -> tuple[str, str] | None:
    """("python" | "script" | "shell", code) for `python -c CODE`, `node -e CODE`, `bash -c CODE`..."""
    if prog in PYTHON and "-c" in args:
        i = args.index("-c")
        return ("python", args[i + 1]) if i + 1 < len(args) else None
    flags = SCRIPTING.get(prog)
    if flags:
        for i, a in enumerate(args):
            if a in flags and i + 1 < len(args):
                return "script", args[i + 1]
    if prog in SHELLS and args:
        rest = list(args)
        while rest and rest[0].startswith(("-", "/")):
            rest.pop(0)
        if rest:
            return "shell", " ".join(rest)
    return None


def is_delete_command(command: str) -> bool:
    """True if running the command would delete files or folders. The agent never runs these: the user gets
    the command to run themselves (the user asked for this after agents wiped whole drives).

    Judged by what the command DOES, like Claude Code's auto mode: the program each part of the command runs
    (`rm`, `del`, `Remove-Item`, `git clean`, `find -delete`...), and the calls inside inline code (python -c,
    node -e, bash -c...). Words inside quoted text are not commands: `curl .../remove-item`, a SQL string with
    DELETE, or a Python `del x` are not file deletes."""
    if not command or not command.strip():
        return False
    if _SHELL_INLINE.search(mask_quotes(command)):
        return True
    for segment in segments(command):
        prog, args = _program(_tokens(segment))
        if prog == "xargs":  # xargs [flags] rm ...
            prog, args = _program([a for a in args if not a.startswith("-")])
        if prog in DELETE_PROGRAMS:
            return True
        if prog == "git" and args and args[0].lower() in ("clean", "rm"):
            return True
        if prog in ("find", "fd"):
            low = [a.lower() for a in args]
            if "-delete" in low or "--delete" in low:
                return True
            if any(a in ("-exec", "-execdir", "-x", "--exec") for a in low) and any(a in DELETE_PROGRAMS for a in low):
                return True
        inline = _inline_code(prog, args)
        if inline is not None:
            kind, code = inline
            if kind == "python" and python_deletes(code):
                return True
            if kind == "script" and _CODE_DELETE.search(mask_quotes(code)):
                return True
            if kind == "shell" and is_delete_command(code):
                return True
    return False


def delete_targets(command: str, cwd) -> list[str]:
    """Best-effort list of what a delete command would remove, with ~ and variables expanded."""
    import os
    from pathlib import Path

    targets: list[str] = []
    for segment in segments(command):
        prog, args = _program(_tokens(segment))
        if prog == "xargs":
            prog, args = _program([a for a in args if not a.startswith("-")])
        if prog not in DELETE_PROGRAMS and not (prog == "git" and args and args[0] in ("clean", "rm")):
            continue
        for arg in args:
            if arg.startswith("-") or arg.lower() in ("/s", "/q", "/f", "rm", "clean") or arg.startswith("2>"):
                continue
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
