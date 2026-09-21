"""File tools: Read, Write, Edit, LS.

Safety rules (inherited from Claude Code):
  * Write/Edit on an existing file require that the agent Read it first in this session,
    and that it has not changed on disk since (prevents clobbering the user's edits).
  * Edit needs an exact, unique match unless replace_all=true. As a small-model aid, a
    whole-line match that differs only in trailing whitespace or uniform indentation is
    accepted when it is unique.
  * Line endings are preserved (CRLF files stay CRLF).
  * Every change is checkpointed so /undo can revert it.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path

from muyah_code.tools.base import READ, WRITE, Tool, ToolContext, ToolError, ToolResult

IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", "dist", "build", ".next", ".nuxt", "target", ".gradle", ".idea", ".egg-info",
    ".dart_tool", ".cache", "coverage", ".turbo",
}
MAX_LINE = 2000
DEFAULT_LIMIT = 2000
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico"}


def is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def read_text(path: Path) -> str:
    """Read preserving line endings exactly."""
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        return f.read()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.muyah-tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


def unified_diff(old: str, new: str, name: str, context: int = 3) -> str:
    diff = difflib.unified_diff(
        old.replace("\r\n", "\n").splitlines(), new.replace("\r\n", "\n").splitlines(),
        fromfile=f"a/{name}", tofile=f"b/{name}", lineterm="", n=context,
    )
    return "\n".join(diff)


def diff_stats(diff: str) -> tuple[int, int]:
    added = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    return added, removed


def check_fresh(path: Path, ctx: ToolContext, action: str) -> None:
    key = str(path)
    if key not in ctx.file_state:
        raise ToolError(f"You must Read {ctx.rel(path)} before you {action} it (it already exists).")
    if mtime_ns(path) != ctx.file_state[key]:
        raise ToolError(
            f"{ctx.rel(path)} was modified on disk since you last read it (by the user or a command). "
            f"Read it again before you {action} it."
        )


def record_checkpoint(ctx: ToolContext, path: Path, previous: bytes | None) -> None:
    cp = ctx.service("checkpoints")
    if cp is not None:
        cp.record(path, previous)


def numbered(lines: list[str], start: int) -> str:
    out = []
    for i, line in enumerate(lines, start=start):
        line = line.rstrip("\r\n")
        if len(line) > MAX_LINE:
            line = line[:MAX_LINE] + " ... [line truncated]"
        out.append(f"{i:>6}\t{line}")
    return "\n".join(out)


class ReadTool(Tool):
    name = "Read"
    kind = READ
    description = (
        "Read a text file. Returns lines prefixed with line numbers (cat -n format: number, TAB, content). "
        "The line-number prefix is NOT part of the file - never include it in Edit old_string. "
        f"Reads up to {DEFAULT_LIMIT} lines by default; use offset (1-based line) and limit for large files. "
        "You must Read a file before editing or overwriting it. Accepts absolute or relative paths."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file (absolute or relative to cwd)"},
            "offset": {"type": "integer", "description": "1-based line number to start from"},
            "limit": {"type": "integer", "description": "Maximum number of lines to read"},
        },
        "required": ["file_path"],
    }

    def permission_subject(self, args, ctx):
        return str(ctx.resolve(args.get("file_path", "")))

    def run(self, args, ctx):
        path = ctx.resolve(args["file_path"])
        if not path.exists():
            hint = _similar_paths(path, ctx)
            raise ToolError(f"File does not exist: {args['file_path']} (resolved to {path}).{hint}")
        if path.is_dir():
            raise ToolError(f"{args['file_path']} is a directory. Use LS or Glob to list it.")
        from muyah_code.llm.content import describe, is_image_path, load_image

        if is_image_path(path):
            try:
                media, b64, data = load_image(path)
            except ValueError as e:
                return ToolResult(f"{ctx.rel(path)} is an image, but it cannot be attached: {e}.", summary="image")
            return ToolResult(f"{describe(ctx.rel(path), data)} is an image; it is attached for you to look at.",
                              summary="image", images=[(media, b64)])
        raw = path.read_bytes()
        editor_read = ctx.service("editor_read")
        if editor_read is not None and not is_binary(raw):
            unsaved = editor_read(path)        # the editor's buffer, unsaved changes included (ACP)
            if unsaved is not None:
                raw = unsaved.encode("utf-8")
        if is_binary(raw):
            kind = "image" if path.suffix.lower() in IMAGE_EXT else "binary"
            return ToolResult(f"{ctx.rel(path)} is a {kind} file ({len(raw)} bytes); it cannot be shown as text.",
                              summary=f"{kind} file")
        text = raw.decode("utf-8", errors="replace")
        ctx.file_state[str(path)] = mtime_ns(path)
        if not text:
            return ToolResult(f"{ctx.rel(path)} exists but is empty.", summary="empty file")
        lines = text.splitlines(keepends=True)
        offset = max(1, int(args.get("offset") or 1))
        limit = max(1, int(args.get("limit") or DEFAULT_LIMIT))
        if offset > len(lines):
            raise ToolError(f"offset {offset} is past the end of the file ({len(lines)} lines).")
        chunk = lines[offset - 1: offset - 1 + limit]
        body = numbered(chunk, offset)
        end = offset + len(chunk) - 1
        note = ""
        if end < len(lines) or offset > 1:
            note = f"\n\n[Showing lines {offset}-{end} of {len(lines)}. Use offset/limit to read more.]"
        return ToolResult(body + note, summary=f"Read {len(chunk)} lines" + (f" (of {len(lines)})" if note else ""))


class WriteTool(Tool):
    name = "Write"
    kind = WRITE
    description = (
        "Create a new file or completely overwrite an existing one with the given content. "
        "Parent directories are created. For existing files you must Read them first; prefer Edit for "
        "targeted changes. Write the COMPLETE file content - never placeholders like '... rest unchanged'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file"},
            "content": {"type": "string", "description": "The full file content"},
        },
        "required": ["file_path", "content"],
    }

    def permission_subject(self, args, ctx):
        return str(ctx.resolve(args.get("file_path", "")))

    def preview(self, args, ctx):
        path = ctx.resolve(args.get("file_path", ""))
        content = args.get("content", "")
        if path.is_file():
            try:
                return unified_diff(read_text(path), content, ctx.rel(path))
            except OSError:
                return None
        lines = content.replace("\r\n", "\n").splitlines()
        shown = "\n".join("+" + ln for ln in lines[:60])
        return f"new file {ctx.rel(path)} ({len(lines)} lines)\n{shown}" + ("\n+..." if len(lines) > 60 else "")

    def run(self, args, ctx):
        path = ctx.resolve(args["file_path"])
        content: str = args["content"]
        if path.is_dir():
            raise ToolError(f"{args['file_path']} is a directory.")
        previous: bytes | None = None
        old_text = ""
        if path.exists():
            check_fresh(path, ctx, "overwrite")
            previous = path.read_bytes()
            old_text = previous.decode("utf-8", errors="replace")
            if "\r\n" in old_text and "\r\n" not in content:
                content = content.replace("\n", "\r\n")
        record_checkpoint(ctx, path, previous)
        write_text(path, content)
        ctx.file_state[str(path)] = mtime_ns(path)
        n_lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        name = ctx.rel(path)
        if previous is None:
            preview = "\n".join("+" + ln for ln in content.replace("\r\n", "\n").splitlines()[:40])
            return ToolResult(f"Created {name} ({n_lines} lines).", display=preview,
                              summary=f"Created {n_lines} lines", meta={"path": str(path), "created": True})
        diff = unified_diff(old_text, content, name)
        added, removed = diff_stats(diff)
        return ToolResult(f"Overwrote {name} ({n_lines} lines).", display=diff,
                          summary=f"Updated: +{added} -{removed}", meta={"path": str(path)})


class EditTool(Tool):
    name = "Edit"
    kind = WRITE
    description = (
        "Replace text in a file. old_string must match the file content EXACTLY (including indentation) and "
        "be unique; include 2-3 surrounding lines to make it unique, or set replace_all=true to change every "
        "occurrence. Do not include Read's line-number prefixes. You must Read the file first."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text (must differ from old_string)"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"},
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def permission_subject(self, args, ctx):
        return str(ctx.resolve(args.get("file_path", "")))

    def preview(self, args, ctx):
        try:
            path, _, text, updated, _, _, _ = self._compute(args, ctx)
        except (ToolError, OSError):
            return None
        return unified_diff(text, updated, ctx.rel(path))

    def _compute(self, args, ctx):
        path = ctx.resolve(args["file_path"])
        old_s: str = args["old_string"]
        new_s: str = args["new_string"]
        replace_all = bool(args.get("replace_all", False))
        if not path.exists():
            raise ToolError(f"File does not exist: {args['file_path']}. Use Write to create it.")
        if old_s == "":
            raise ToolError("old_string is empty. Use Write to create or overwrite a whole file.")
        if old_s == new_s:
            raise ToolError("old_string and new_string are identical; nothing to change.")
        check_fresh(path, ctx, "edit")
        previous = path.read_bytes()
        original = previous.decode("utf-8", errors="replace")
        crlf = "\r\n" in original
        text = original.replace("\r\n", "\n")
        old_n = old_s.replace("\r\n", "\n")
        new_n = new_s.replace("\r\n", "\n")

        count = text.count(old_n)
        fuzzy_note = ""
        if count == 0:
            updated = _fuzzy_line_replace(text, old_n, new_n)
            if updated is None:
                raise ToolError(_not_found_message(text, old_n, ctx.rel(path)))
            fuzzy_note = " (matched ignoring whitespace differences)"
            count = 1
        elif count > 1 and not replace_all:
            raise ToolError(
                f"old_string matches {count} places in {ctx.rel(path)}. Add surrounding lines to make it unique, "
                "or set replace_all=true to replace every occurrence."
            )
        else:
            updated = text.replace(old_n, new_n) if replace_all else text.replace(old_n, new_n, 1)
        return path, previous, text, updated, count, fuzzy_note, crlf

    def run(self, args, ctx):
        path, previous, text, updated, count, fuzzy_note, crlf = self._compute(args, ctx)
        new_n = args["new_string"].replace("\r\n", "\n")
        replace_all = bool(args.get("replace_all", False))
        final = updated.replace("\n", "\r\n") if crlf else updated
        record_checkpoint(ctx, path, previous)
        write_text(path, final)
        ctx.file_state[str(path)] = mtime_ns(path)
        name = ctx.rel(path)
        diff = unified_diff(text, updated, name)
        added, removed = diff_stats(diff)
        snippet = _snippet_around(updated, new_n)
        n = count if replace_all else 1
        msg = f"Edited {name}: replaced {n} occurrence{'s' if n != 1 else ''}{fuzzy_note}."
        if snippet:
            msg += f"\nResult around the change:\n{snippet}"
        return ToolResult(msg, display=diff, summary=f"Updated: +{added} -{removed}", meta={"path": str(path)})


class LSTool(Tool):
    name = "LS"
    kind = READ
    description = (
        "List a directory as a tree (default depth 2). Skips .git, node_modules, virtualenvs, caches and build "
        "output. Use Glob to find files by pattern and Grep to search contents."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list (default: cwd)"},
            "depth": {"type": "integer", "description": "How many levels deep to go (default 2, max 6)"},
        },
    }

    def permission_subject(self, args, ctx):
        return str(ctx.resolve(args.get("path") or "."))

    def run(self, args, ctx):
        root = ctx.resolve(args.get("path") or ".")
        if not root.exists():
            raise ToolError(f"Directory does not exist: {root}")
        if not root.is_dir():
            raise ToolError(f"{root} is a file, not a directory. Use Read.")
        depth = min(max(int(args.get("depth") or 2), 1), 6)
        lines = [f"{root}/"]
        count = [0]
        limit = 400

        def walk(d: Path, prefix: str, level: int) -> None:
            try:
                entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError as e:
                lines.append(f"{prefix}[unreadable: {e.strerror}]")
                return
            for e in entries:
                if count[0] >= limit:
                    return
                if e.is_dir() and (e.name in IGNORED_DIRS or e.name.endswith(".egg-info")):
                    continue
                count[0] += 1
                if e.is_dir():
                    lines.append(f"{prefix}{e.name}/")
                    if level < depth:
                        walk(e, prefix + "  ", level + 1)
                else:
                    try:
                        size = e.stat().st_size
                    except OSError:
                        size = 0
                    lines.append(f"{prefix}{e.name}  ({_human(size)})")

        walk(root, "  ", 1)
        if count[0] >= limit:
            lines.append(f"  ... truncated at {limit} entries; list a subdirectory or use Glob.")
        return ToolResult("\n".join(lines), summary=f"Listed {count[0]} entries")


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _similar_paths(path: Path, ctx: ToolContext) -> str:
    parent = path.parent
    if not parent.is_dir():
        return ""
    try:
        names = [p.name for p in parent.iterdir()]
    except OSError:
        return ""
    close = difflib.get_close_matches(path.name, names, n=3, cutoff=0.5)
    return f" Did you mean: {', '.join(close)}?" if close else ""


def _indent(s: str) -> str:
    return s[: len(s) - len(s.lstrip(" \t"))]


def _fuzzy_line_replace(text: str, old: str, new: str) -> str | None:
    """Whole-line match ignoring trailing whitespace, then ignoring uniform indentation. Must be unique."""
    lines = text.split("\n")
    old_lines = old.strip("\n").split("\n")
    if not old_lines or not any(ln.strip() for ln in old_lines):
        return None
    n = len(old_lines)
    for mode in ("rstrip", "strip"):
        norm = (lambda s: s.rstrip()) if mode == "rstrip" else (lambda s: s.strip())
        target = [norm(x) for x in old_lines]
        hits = [i for i in range(len(lines) - n + 1) if [norm(x) for x in lines[i:i + n]] == target]
        if len(hits) != 1:
            if len(hits) > 1:
                return None
            continue
        i = hits[0]
        new_lines = new.strip("\n").split("\n") if new.strip("\n") else []
        if mode == "strip":
            file_first = next((ln for ln in lines[i:i + n] if ln.strip()), "")
            old_first = next((ln for ln in old_lines if ln.strip()), "")
            fi, oi = _indent(file_first), _indent(old_first)
            if len(fi) > len(oi):
                new_lines = [(fi[len(oi):] + ln) if ln.strip() else ln for ln in new_lines]
            elif len(oi) > len(fi):
                cut = len(oi) - len(fi)
                new_lines = [ln[cut:] if ln[:cut].strip() == "" else ln.lstrip() for ln in new_lines]
        return "\n".join(lines[:i] + new_lines + lines[i + n:])
    return None


def _not_found_message(text: str, old: str, name: str) -> str:
    first = next((ln.strip() for ln in old.split("\n") if ln.strip()), "")
    lines = text.split("\n")
    candidates = difflib.get_close_matches(first, [ln.strip() for ln in lines], n=3, cutoff=0.6)
    hint = ""
    if candidates:
        locs = []
        for c in candidates:
            for idx, ln in enumerate(lines, start=1):
                if ln.strip() == c:
                    locs.append(f"  line {idx}: {ln.rstrip()[:150]}")
                    break
        hint = "\nClosest lines in the file:\n" + "\n".join(locs)
    return (
        f"old_string was not found in {name}. It must match exactly (whitespace, indentation, quotes). "
        f"Read the file again and copy the exact text (without line-number prefixes).{hint}"
    )


def _snippet_around(text: str, new: str, context: int = 3, max_lines: int = 16) -> str:
    if not new.strip():
        return ""
    idx = text.find(new)
    if idx == -1:
        return ""
    start_line = text.count("\n", 0, idx)
    end_line = start_line + new.count("\n")
    lines = text.split("\n")
    lo = max(0, start_line - context)
    hi = min(len(lines), end_line + context + 1)
    if hi - lo > max_lines:
        hi = lo + max_lines
    return numbered(lines[lo:hi], lo + 1)
