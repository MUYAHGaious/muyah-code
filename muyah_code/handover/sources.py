"""Reading other AI coding tools' conversations for this folder (Claude Code, Codex, OpenCode, Gemini, Aider).

Everything here is local and read-only: each tool writes its conversation to a file (or a small database) on
this machine as you work, so moving to MUYAH-CODE is a matter of reading those messages. Nothing is sent
anywhere, and no file of another tool is ever changed.

Each adapter returns the same two things:
  * `Conversation`: which tool, which folder, when, a title, how big;
  * `Turn`s: role (you / the assistant), text, and what tools were used (name + target), with tool OUTPUT
    dropped (it is stale: the files are on disk and can be read fresh) and the other tool's own compaction
    summaries kept apart, so they are never summarized again without saying so.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

CLAUDE = "Claude Code"
CODEX = "Codex CLI"
OPENCODE = "OpenCode"
GEMINI = "Gemini CLI"
AIDER = "Aider"
MAX_TEXT = 4000          # one message longer than this is trimmed (its middle is the least useful part)


@dataclass
class Turn:
    """One exchange, as it will be shown to the model."""
    role: str                     # "user" | "assistant"
    text: str = ""
    tools: list[str] = field(default_factory=list)   # "Read(cart.py)", "Bash(pytest -q)"
    kind: str = "message"         # "message" | "summary" (the other tool's own compaction summary)
    ts: float = 0.0

    def tokens(self) -> int:
        return (len(self.text) + sum(len(t) + 2 for t in self.tools)) // 4 + 4


@dataclass
class Conversation:
    tool: str
    id: str
    source: Path                  # the file or database it came from
    cwd: str = ""
    title: str = ""
    updated: float = 0.0
    turns: int = 0
    tokens: int = 0
    load: object = None           # () -> list[Turn]

    @property
    def when(self) -> str:
        return time.strftime("%d %b %H:%M", time.localtime(self.updated)) if self.updated else "?"


def _same_folder(a: str, b: Path) -> bool:
    try:
        return a and Path(a).resolve() == b.resolve()
    except (OSError, ValueError):
        return False


def _trim(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= MAX_TEXT:
        return text
    half = MAX_TEXT // 2
    return text[:half] + f"\n… [{len(text) - MAX_TEXT} characters left out] …\n" + text[-half:]


def _short(value, limit: int = 60) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------------------- Claude Code

def claude_conversations(project: Path, home: Path | None = None) -> list[Conversation]:
    """~/.claude/projects/<folder with separators replaced by ->/<session id>.jsonl"""
    base = (home or Path.home()) / ".claude" / "projects"
    if not base.is_dir():
        return []
    slug = re.sub(r"[\\/:]+", "-", str(project.resolve()))
    out = []
    for folder in base.iterdir():
        if folder.name.lower() != slug.lower() or not folder.is_dir():
            continue
        for path in folder.glob("*.jsonl"):
            conv = _claude_head(path)
            if conv is not None:
                out.append(conv)
    return out


def _claude_head(path: Path) -> Conversation | None:
    title, turns, tokens, cwd = "", 0, 0, ""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                ev = _json(line)
                if ev is None:
                    continue
                kind = ev.get("type")
                if kind == "ai-title" and not title:
                    title = str(ev.get("aiTitle") or "")
                elif kind in ("user", "assistant") and not ev.get("isSidechain"):
                    turns += 1
                    cwd = cwd or str(ev.get("cwd") or "")
                    tokens += len(json.dumps(ev.get("message") or {})) // 4
                elif kind == "summary" and not title:
                    title = str(ev.get("summary") or "")
    except OSError:
        return None
    if not turns:
        return None
    return Conversation(CLAUDE, path.stem, path, cwd, title or "(untitled)", path.stat().st_mtime, turns, tokens,
                        load=lambda p=path: list(_claude_turns(p)))


def _claude_turns(path: Path) -> Iterator[Turn]:
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            ev = _json(line)
            if ev is None or ev.get("isSidechain"):      # a sub-agent's own conversation: not yours
                continue
            kind = ev.get("type")
            ts = _timestamp(ev.get("timestamp"))
            if kind == "summary":
                yield Turn("user", _trim(str(ev.get("summary") or "")), kind="summary", ts=ts)
                continue
            if kind not in ("user", "assistant"):
                continue
            message = ev.get("message") or {}
            text, tools = _blocks(message.get("content"))
            if text or tools:
                yield Turn("assistant" if kind == "assistant" else "user", _trim(text), tools, ts=ts)


def _blocks(content) -> tuple[str, list[str]]:
    """Anthropic-style content: a string, or blocks of text / tool_use / tool_result (output dropped)."""
    if isinstance(content, str):
        return content, []
    text, tools = [], []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text.append(str(block.get("text") or ""))
        elif kind == "tool_use":
            tools.append(f"{block.get('name')}({_target(block.get('input'))})")
        elif kind == "thinking":
            continue                                    # another tool's private reasoning: not carried over
    return "\n".join(t for t in text if t.strip()), tools


def _target(args) -> str:
    if not isinstance(args, dict):
        return ""
    for key in ("file_path", "path", "command", "pattern", "url", "query", "notebook_path"):
        if args.get(key):
            return _short(args[key])
    return ""


# ---------------------------------------------------------------------------- Codex CLI

def codex_conversations(project: Path, home: Path | None = None) -> list[Conversation]:
    """~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl; the first line says which folder it ran in."""
    base = (home or Path.home()) / ".codex" / "sessions"
    if not base.is_dir():
        return []
    out = []
    for path in base.rglob("rollout-*.jsonl"):
        head = _json(_first_line(path))
        payload = (head or {}).get("payload") or {}
        if not _same_folder(str(payload.get("cwd") or ""), project):
            continue
        turns = tokens = 0
        title = ""
        for ev in _lines(path):
            item = ev.get("payload") or {}
            if item.get("type") in ("message", "user_message", "agent_message"):
                turns += 1
                tokens += len(json.dumps(item)) // 4
                if not title and item.get("role") in (None, "user"):
                    title = _short(_codex_text(item), 70)
        if turns:
            out.append(Conversation(CODEX, path.stem.replace("rollout-", ""), path, str(payload.get("cwd") or ""),
                                    title or "(untitled)", path.stat().st_mtime, turns, tokens,
                                    load=lambda p=path: list(_codex_turns(p))))
    return out


def _codex_text(item: dict) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    parts = [str(b.get("text") or "") for b in (content or []) if isinstance(b, dict)]
    return "\n".join(p for p in parts if p.strip()) or str(item.get("message") or "")


def _codex_turns(path: Path) -> Iterator[Turn]:
    pending: list[str] = []
    for ev in _lines(path):
        item = ev.get("payload") or {}
        kind = item.get("type")
        ts = _timestamp(ev.get("timestamp"))
        if kind in ("function_call", "custom_tool_call"):
            name = item.get("name") or item.get("tool_name") or "tool"
            pending.append(f"{name}({_target(_json(item.get('arguments')) or {})})")
        elif kind in ("message", "user_message", "agent_message"):
            role = item.get("role") or ("user" if kind == "user_message" else "assistant")
            text = _codex_text(item)
            if text.strip() or pending:
                yield Turn("assistant" if role == "assistant" else "user", _trim(text), pending, ts=ts)
                pending = []
    if pending:      # tool calls after the last message (the conversation ended mid-work)
        yield Turn("assistant", "", pending)


# ---------------------------------------------------------------------------- OpenCode

def opencode_conversations(project: Path, home: Path | None = None) -> list[Conversation]:
    """~/.local/share/opencode/opencode.db: session / message / part tables (read-only)."""
    db = (home or Path.home()) / ".local" / "share" / "opencode" / "opencode.db"
    if not db.exists():
        return []
    out = []
    try:
        con = _read_only(db)
        for sid, directory, title, updated in con.execute(
                "SELECT id, directory, title, time_updated FROM session"):
            if not _same_folder(str(directory or ""), project):
                continue
            n = con.execute("SELECT COUNT(*) FROM message WHERE session_id = ?", (sid,)).fetchone()[0]
            size = con.execute("SELECT COALESCE(SUM(LENGTH(data)), 0) FROM part WHERE session_id = ?",
                               (sid,)).fetchone()[0]
            if n:
                out.append(Conversation(OPENCODE, str(sid), db, str(directory or ""), _short(title, 70) or "(untitled)",
                                        _seconds(updated), n, int(size) // 4,
                                        load=lambda s=str(sid), d=db: list(_opencode_turns(d, s))))
        con.close()
    except sqlite3.DatabaseError:
        return []
    return out


def _opencode_turns(db: Path, session_id: str) -> Iterator[Turn]:
    con = _read_only(db)
    try:
        rows = con.execute("SELECT id, data, time_created FROM message WHERE session_id = ? ORDER BY time_created",
                           (session_id,)).fetchall()
        for mid, data, created in rows:
            info = _json(data) or {}
            role = "assistant" if info.get("role") == "assistant" else "user"
            text, tools, summary = [], [], False
            for (part,) in con.execute("SELECT data FROM part WHERE message_id = ? ORDER BY time_created", (mid,)):
                item = _json(part) or {}
                kind = item.get("type")
                if kind == "text":
                    text.append(str(item.get("text") or ""))
                elif kind == "tool":
                    state = item.get("state") or {}
                    args = state.get("input") if isinstance(state, dict) else None
                    tools.append(f"{item.get('tool')}({_target(args)})")
                elif kind == "compaction":
                    summary = True
            joined = "\n".join(t for t in text if t.strip())
            if joined or tools:
                yield Turn(role, _trim(joined), tools, kind="summary" if summary else "message", ts=_seconds(created))
    finally:
        con.close()


def _read_only(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)


# ---------------------------------------------------------------------------- Gemini CLI

def gemini_conversations(project: Path, home: Path | None = None) -> list[Conversation]:
    """~/.gemini/tmp/<hash>/logs.json (and chats/*.json); the folder is named in .project_root."""
    base = (home or Path.home()) / ".gemini" / "tmp"
    if not base.is_dir():
        return []
    out = []
    for folder in base.iterdir():
        root = folder / ".project_root"
        try:
            where = root.read_text(encoding="utf-8").strip() if root.exists() else ""
        except OSError:
            where = ""
        if not _same_folder(where, project):
            continue
        for path in [folder / "logs.json", *sorted((folder / "chats").glob("*.json"))]:
            if not path.exists():
                continue
            entries = _json(_read(path)) or []
            entries = entries if isinstance(entries, list) else entries.get("messages") or []
            if not entries:
                continue
            out.append(Conversation(GEMINI, f"{folder.name}/{path.stem}", path, where,
                                    _short(_gemini_text(entries[0]), 70) or "(untitled)", path.stat().st_mtime,
                                    len(entries), len(_read(path)) // 4,
                                    load=lambda p=path: list(_gemini_turns(p))))
    return out


def _gemini_text(entry: dict) -> str:
    if not isinstance(entry, dict):
        return ""
    for key in ("message", "content", "text"):
        value = entry.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(str(p.get("text") or "") for p in value if isinstance(p, dict))
    return ""


def _gemini_turns(path: Path) -> Iterator[Turn]:
    entries = _json(_read(path)) or []
    entries = entries if isinstance(entries, list) else entries.get("messages") or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("type") or entry.get("role") or "user").lower()
        text = _gemini_text(entry)
        if text.strip():
            yield Turn("assistant" if kind in ("gemini", "model", "assistant") else "user", _trim(text),
                       ts=_timestamp(entry.get("timestamp")))


# ---------------------------------------------------------------------------- Aider

def aider_conversations(project: Path, home: Path | None = None) -> list[Conversation]:
    """`.aider.chat.history.md` in the project: markdown, "#### " marks what you typed."""
    path = project / ".aider.chat.history.md"
    if not path.exists():
        return []
    text = _read(path)
    turns = text.count("\n#### ")
    if not turns:
        return []
    return [Conversation(AIDER, "chat-history", path, str(project), "aider chat history", path.stat().st_mtime,
                         turns, len(text) // 4, load=lambda p=path: list(_aider_turns(p)))]


def _aider_turns(path: Path) -> Iterator[Turn]:
    role, buffer, ts = "user", [], path.stat().st_mtime

    def flush(current_role, lines):
        text = "\n".join(lines).strip()
        return Turn(current_role, _trim(text), ts=ts) if text else None

    for line in _read(path).splitlines():
        if line.startswith("#### "):
            done = flush(role, buffer)
            if done:
                yield done
            role, buffer = "user", [line[5:]]
        elif line.startswith("# aider chat started"):
            continue
        else:
            if role == "user" and line.strip() and not line.startswith("####") and buffer and \
                    not buffer[-1].startswith(">"):
                pass
            buffer.append(line)
    done = flush(role, buffer)
    if done:
        yield done


# ---------------------------------------------------------------------------- all of them

ADAPTERS = (claude_conversations, codex_conversations, opencode_conversations, gemini_conversations,
            aider_conversations)


def discover(project: Path, home: Path | None = None) -> list[Conversation]:
    """Every conversation any supported tool has for this folder, newest first."""
    found: list[Conversation] = []
    for adapter in ADAPTERS:
        try:
            found += adapter(project, home)
        except Exception:      # one unreadable store must never stop the others
            continue
    return sorted(found, key=lambda c: -c.updated)


# ---------------------------------------------------------------------------- small helpers

def _json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _first_line(path: Path) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.readline()
    except OSError:
        return ""


def _lines(path: Path) -> Iterator[dict]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                ev = _json(line)
                if isinstance(ev, dict):
                    yield ev
    except OSError:
        return


def _timestamp(value) -> float:
    if isinstance(value, (int, float)):
        return _seconds(value)
    if isinstance(value, str) and value:
        try:
            from datetime import datetime

            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _seconds(value) -> float:
    """Tools store milliseconds or seconds; both end up as seconds."""
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return number / 1000 if number > 1e11 else number


def tool_home(name: str) -> Path | None:
    """Where a tool keeps its data (for messages that tell the user where to look)."""
    return {CLAUDE: Path.home() / ".claude", CODEX: Path.home() / ".codex",
            OPENCODE: Path.home() / ".local" / "share" / "opencode",
            GEMINI: Path.home() / ".gemini"}.get(name) or (Path(os.getcwd()) if name == AIDER else None)
