"""Glob and Grep. Respect .gitignore inside git repos; prune heavy directories everywhere."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from muyah_code.tools.base import READ, Tool, ToolError, ToolResult
from muyah_code.tools.fs import IGNORED_DIRS, is_binary

MAX_GLOB_RESULTS = 250
MAX_GREP_FILE_BYTES = 5 * 1024 * 1024
TYPE_EXT = {
    "py": [".py", ".pyi"], "js": [".js", ".mjs", ".cjs", ".jsx"], "ts": [".ts", ".tsx", ".mts", ".cts"],
    "go": [".go"], "rust": [".rs"], "java": [".java"], "kotlin": [".kt", ".kts"], "c": [".c", ".h"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h"], "cs": [".cs"], "rb": [".rb"], "php": [".php"],
    "swift": [".swift"], "dart": [".dart"], "md": [".md", ".markdown"], "json": [".json"],
    "yaml": [".yml", ".yaml"], "toml": [".toml"], "html": [".html", ".htm"], "css": [".css", ".scss", ".sass"],
    "sh": [".sh", ".bash"], "ps": [".ps1", ".psm1"], "sql": [".sql"], "vue": [".vue"], "svelte": [".svelte"],
}


def glob_to_regex(pattern: str) -> re.Pattern:
    """Translate a glob with **, *, ?, [..] and {a,b} into a regex over posix relative paths."""
    i, out = 0, []
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
                i += 1
            else:
                body = pattern[i + 1:j].replace("\\", "\\\\")
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                i = j + 1
        elif c == "{":
            j = pattern.find("}", i + 1)
            if j == -1:
                out.append(re.escape(c))
                i += 1
            else:
                alts = [glob_to_regex(a).pattern[:-2] for a in pattern[i + 1:j].split(",")]  # strip trailing \Z
                out.append("(?:" + "|".join(alts) + ")")
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z")


def matches_glob(rel_posix: str, pattern: str) -> bool:
    pattern = pattern.replace("\\", "/")
    while pattern.startswith("./"):
        pattern = pattern[2:]
    if "/" not in pattern:
        return bool(glob_to_regex(pattern).match(rel_posix.rsplit("/", 1)[-1]))
    return bool(glob_to_regex(pattern).match(rel_posix))


def _git_files(root: Path) -> list[Path] | None:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=root, capture_output=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    files = []
    for rel in proc.stdout.decode("utf-8", errors="replace").split("\0"):
        if rel:
            p = root / rel
            if p.is_file():
                files.append(p)
    return files


def list_files(root: Path) -> list[Path]:
    """All candidate files under root, honoring .gitignore where possible."""
    if (root / ".git").exists() or _inside_git(root):
        files = _git_files(root)
        if files is not None:
            return files
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.endswith(".egg-info")]
        for f in filenames:
            out.append(Path(dirpath) / f)
    return out


def _inside_git(path: Path) -> bool:
    return any((p / ".git").exists() for p in path.parents)


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


class GlobTool(Tool):
    name = "Glob"
    kind = READ
    description = (
        "Find files by glob pattern, e.g. '**/*.py', 'src/**/*.{ts,tsx}', 'test_*.py'. A pattern without '/' "
        "matches file names at any depth. Returns paths sorted by modification time (newest first). "
        "Respects .gitignore."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern"},
            "path": {"type": "string", "description": "Directory to search (default: cwd)"},
        },
        "required": ["pattern"],
    }

    def permission_subject(self, args, ctx):
        return str(ctx.resolve(args.get("path") or "."))

    def run(self, args, ctx):
        root = ctx.resolve(args.get("path") or ".")
        if not root.is_dir():
            raise ToolError(f"Not a directory: {root}")
        pattern = args["pattern"].strip()
        hits = []
        for p in list_files(root):
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                continue
            if matches_glob(rel, pattern):
                hits.append(p)
        hits.sort(key=_mtime, reverse=True)
        total = len(hits)
        if not hits:
            return ToolResult(f"No files match '{pattern}' under {ctx.rel(root)}.", summary="0 files")
        shown = hits[:MAX_GLOB_RESULTS]
        body = "\n".join(ctx.rel(p) for p in shown)
        if total > len(shown):
            body += f"\n... {total - len(shown)} more; narrow the pattern."
        return ToolResult(body, summary=f"Found {total} file{'s' if total != 1 else ''}")


class GrepTool(Tool):
    name = "Grep"
    kind = READ
    description = (
        "Search file contents with a regular expression (Python re syntax). output_mode: 'files_with_matches' "
        "(default, paths only), 'content' (matching lines with line numbers), or 'count'. Filter with glob "
        "(e.g. '*.py') or type (e.g. 'py', 'js'). Use -i for case-insensitive, -C/-A/-B for context lines."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression"},
            "path": {"type": "string", "description": "File or directory (default: cwd)"},
            "glob": {"type": "string", "description": "Only search files matching this glob"},
            "type": {"type": "string", "description": "File type, e.g. py, js, ts, go, rust, java"},
            "output_mode": {"type": "string", "enum": ["files_with_matches", "content", "count"]},
            "-i": {"type": "boolean", "description": "Case-insensitive"},
            "-C": {"type": "integer", "description": "Context lines before and after (content mode)"},
            "-A": {"type": "integer", "description": "Lines after each match (content mode)"},
            "-B": {"type": "integer", "description": "Lines before each match (content mode)"},
            "multiline": {"type": "boolean", "description": "Let . match newlines; patterns may span lines"},
            "head_limit": {"type": "integer", "description": "Max output lines/entries (default 200)"},
        },
        "required": ["pattern"],
    }

    def permission_subject(self, args, ctx):
        return str(ctx.resolve(args.get("path") or "."))

    def title(self, args):
        return f'Grep("{args.get("pattern", "")[:60]}"' + (f", {args['path']}" if args.get("path") else "") + ")"

    def run(self, args, ctx):
        target = ctx.resolve(args.get("path") or ".")
        if not target.exists():
            raise ToolError(f"Path does not exist: {target}")
        mode = args.get("output_mode") or "files_with_matches"
        limit = int(args.get("head_limit") or 200)
        flags = re.IGNORECASE if args.get("-i") else 0
        if args.get("multiline"):
            flags |= re.DOTALL | re.MULTILINE
        try:
            rx = re.compile(args["pattern"], flags)
        except re.error as e:
            raise ToolError(f"Invalid regex: {e}") from e

        files = [target] if target.is_file() else list_files(target)
        base = target if target.is_dir() else target.parent
        if args.get("glob"):
            files = [f for f in files if matches_glob(_rel(f, base), args["glob"])]
        if args.get("type"):
            exts = TYPE_EXT.get(args["type"].lower().lstrip("."), ["." + args["type"].lower().lstrip(".")])
            files = [f for f in files if f.suffix.lower() in exts]

        before = int(args.get("-B") or args.get("-C") or 0)
        after = int(args.get("-A") or args.get("-C") or 0)
        results: list[tuple[Path, list[tuple[int, str, bool]], int]] = []
        for f in files:
            try:
                if f.stat().st_size > MAX_GREP_FILE_BYTES:
                    continue
                raw = f.read_bytes()
            except OSError:
                continue
            if is_binary(raw):
                continue
            text = raw.decode("utf-8", errors="replace")
            if args.get("multiline"):
                ms = list(rx.finditer(text))
                if not ms:
                    continue
                lines_out = [(text.count("\n", 0, m.start()) + 1, m.group(0).split("\n")[0], True) for m in ms]
                results.append((f, lines_out, len(ms)))
                continue
            lines = text.splitlines()
            hit_idx = [i for i, ln in enumerate(lines) if rx.search(ln)]
            if not hit_idx:
                continue
            if mode != "content":
                results.append((f, [], len(hit_idx)))
                continue
            keep: dict[int, bool] = {}
            for i in hit_idx:
                for j in range(max(0, i - before), min(len(lines), i + after + 1)):
                    keep[j] = keep.get(j, False) or j == i
            results.append((f, [(j + 1, lines[j], keep[j]) for j in sorted(keep)], len(hit_idx)))

        if not results:
            return ToolResult(f"No matches for /{args['pattern']}/.", summary="0 matches")
        results.sort(key=lambda r: _mtime(r[0]), reverse=True)
        total_matches = sum(r[2] for r in results)
        out: list[str] = []
        if mode == "files_with_matches":
            out = [ctx.rel(r[0]) for r in results]
        elif mode == "count":
            out = [f"{ctx.rel(r[0])}:{r[2]}" for r in results]
        else:
            for f, lines_out, _ in results:
                prev = None
                for num, line, is_hit in lines_out:
                    if prev is not None and num > prev + 1:
                        out.append("--")
                    sep = ":" if is_hit else "-"
                    out.append(f"{ctx.rel(f)}{sep}{num}{sep}{line[:500]}")
                    prev = num
        truncated = len(out) > limit
        body = "\n".join(out[:limit])
        if truncated:
            body += f"\n... [{len(out) - limit} more lines; refine the pattern or raise head_limit]"
        return ToolResult(body, summary=f"{total_matches} matches in {len(results)} files")


def _rel(p: Path, base: Path) -> str:
    try:
        return p.relative_to(base).as_posix()
    except ValueError:
        return p.name
