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


def risky_command(command: str) -> str | None:
    """Why this shell command needs a human even in auto mode, or None if it can run."""
    for pattern, why in _RULES:
        if pattern.search(command or ""):
            return why
    return None
