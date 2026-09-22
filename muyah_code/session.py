"""Session transcripts (JSONL, resumable) and file checkpoints (/undo)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from muyah_code.appendlog import append_line, flush


def project_slug(root: Path) -> str:
    resolved = str(root.resolve())
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(resolved).name).strip("-") or "root"
    return f"{name}-{hashlib.sha1(resolved.lower().encode()).hexdigest()[:8]}"


def sessions_dir(home: Path, project_root: Path) -> Path:
    return home / "sessions" / project_slug(project_root)


@dataclass
class SessionInfo:
    id: str
    path: Path
    modified: float
    title: str
    messages: int


class Session:
    def __init__(self, directory: Path, session_id: str | None = None):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.id = session_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.path = self.dir / f"{self.id}.jsonl"
        self.title: str | None = None

    def _append(self, event: dict) -> None:
        event["ts"] = time.time()
        append_line(self.path, json.dumps(event, ensure_ascii=False, default=str))

    def close(self) -> None:
        flush(self.path, close=True)

    def start(self, meta: dict) -> None:
        if not self.path.exists():
            self._append({"t": "meta", **meta})

    def log_message(self, message: dict) -> None:
        if self.title is None and message.get("role") == "user":
            from muyah_code.llm.content import text_of

            content = text_of(message.get("content") or "")
            if not message.get("_images") and content and not content.startswith("<"):
                self.title = " ".join(content.split())[:80]
                self._append({"t": "title", "title": self.title})
        self._append({"t": "msg", "m": message})

    def log_replace(self, messages: list[dict], reason: str) -> None:
        """History was rewritten (compaction / clear / rewind)."""
        self._append({"t": "replace", "reason": reason, "messages": messages})

    def log_event(self, kind: str, **data) -> None:
        self._append({"t": kind, **data})

    @staticmethod
    def load(path: Path) -> tuple[list[dict], dict]:
        flush(path)
        messages: list[dict] = []
        display: list[dict] = []   # what you saw: compaction does not remove messages from it (the model's copy
        meta: dict = {}            # is compacted; the screen shows the whole conversation, like Claude Code)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn last line after a crash
                t = ev.get("t")
                if t == "meta":
                    meta.update({k: v for k, v in ev.items() if k not in ("t", "ts")})
                elif t == "msg":
                    messages.append(ev["m"])
                    display.append(ev["m"])
                elif t == "replace":
                    messages = list(ev.get("messages") or [])
                    if ev.get("reason") != "compact":      # /clear and rewinds really remove messages
                        display = list(messages)
                elif t == "title":
                    meta["title"] = ev.get("title")
                elif t == "mode":                   # the last mode wins: resuming puts you back in it
                    meta["mode"] = {k: ev.get(k) for k in ("mode", "plan_file", "active_plan")}
                elif t in ("checkpoint", "epoch"):
                    meta.setdefault("rewind", []).append(ev)
        meta["display"] = display
        return _repair(messages), meta

    @classmethod
    def resume(cls, directory: Path, session_id: str) -> tuple[Session, list[dict], dict]:
        matches = sorted(p for p in directory.glob(f"{session_id}*.jsonl") if not p.name.endswith(".events.jsonl"))
        if not matches:
            raise FileNotFoundError(f"No session '{session_id}' in {directory}")
        path = matches[0]
        messages, meta = cls.load(path)
        s = cls(directory, path.stem)
        s.title = meta.get("title")
        return s, messages, meta

    @staticmethod
    def list_sessions(directory: Path, limit: int = 20) -> list[SessionInfo]:
        flush()
        if not directory.is_dir():
            return []
        out = []
        transcripts = [p for p in directory.glob("*.jsonl") if not p.name.endswith(".events.jsonl")]
        for p in sorted(transcripts, key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
            title, count = "", 0
            try:
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        if '"t": "title"' in line and not title:
                            title = json.loads(line).get("title", "")
                        elif '"t": "msg"' in line:
                            count += 1
            except (OSError, json.JSONDecodeError):
                continue
            out.append(SessionInfo(p.stem, p, p.stat().st_mtime, title or "(untitled)", count))
        return out


def _repair(messages: list[dict]) -> list[dict]:
    """Drop a dangling assistant tool-call message without results (crash mid-turn)."""
    out = list(messages)
    while out:
        last = out[-1]
        if last.get("role") == "assistant" and last.get("tool_calls"):
            out.pop()
            continue
        break
    # tool results must follow an assistant message that issued them
    ids: set[str] = set()
    fixed = []
    for m in out:
        if m.get("role") == "assistant":
            ids = {tc.get("id") for tc in m.get("tool_calls") or []}
        elif m.get("role") == "tool" and m.get("tool_call_id") not in ids:
            continue
        fixed.append(m)
    return fixed


class Checkpoints:
    """Per-turn snapshots of files the agent changed, so /undo can restore them."""

    def __init__(self, max_turns: int = 50):
        self.turns: list[dict] = []
        self.max_turns = max_turns

    def begin_turn(self, label: str) -> None:
        self.turns.append({"label": label[:80], "changes": {}})
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def record(self, path: Path, previous: bytes | None) -> None:
        if not self.turns:
            self.begin_turn("(no turn)")
        changes = self.turns[-1]["changes"]
        key = str(path)
        if key not in changes:  # keep the state from BEFORE the turn's first change
            changes[key] = previous

    def undo(self) -> tuple[str, list[str]] | None:
        while self.turns and not self.turns[-1]["changes"]:
            self.turns.pop()
        if not self.turns:
            return None
        turn = self.turns.pop()
        restored = []
        for key, previous in turn["changes"].items():
            p = Path(key)
            try:
                if previous is None:
                    if p.exists():
                        p.unlink()
                        restored.append(f"deleted {p}")
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    tmp = p.with_name(f".{p.name}.muyah-undo")
                    tmp.write_bytes(previous)
                    os.replace(tmp, p)
                    restored.append(f"restored {p}")
            except OSError as e:
                restored.append(f"FAILED {p}: {e}")
        return turn["label"], restored
