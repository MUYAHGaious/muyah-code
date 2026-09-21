"""Live event stream of what the agent is doing - powers `/viz` (live) and `muyah viz --replay` (recorded).

Events are small JSON-able dicts with a "type" and a "t" timestamp (seconds since the session started):

    session       model, provider, window, cwd, mode
    turn_start    prompt                         turn_end     status, duration, tool_calls
    llm_start     step                           llm_tokens   chars (batched ~10/s while streaming)
    llm_end       prompt_tokens, completion_tokens, duration, calls
    tool_start    id, name, title                tool_end     id, name, ok, summary, duration
    context       used, usable, window, parts{system, conversation, tools}
    compact       description                    lessons      items
    subagent_start name, task                    subagent_end name, status
    todos         items

Publishing is cheap when nobody listens, so the agent always emits. Every event is also appended to the
session's `.events.jsonl` file so any session can be replayed later.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

Listener = Callable[[dict], None]


class EventBus:
    def __init__(self, record_to: Path | None = None):
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()
        self.started = time.time()
        self.history: list[dict] = []   # this process's events, so a late viewer can catch up
        self.max_history = 5000
        self.record_to = record_to

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)
        return unsubscribe

    def emit(self, type_: str, **data) -> dict:
        event = {"type": type_, "t": round(time.time() - self.started, 3), **data}
        with self._lock:
            self.history.append(event)
            if len(self.history) > self.max_history:
                del self.history[: len(self.history) - self.max_history]
            listeners = list(self._listeners)
        if self.record_to is not None:
            try:
                with open(self.record_to, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            except OSError:
                self.record_to = None  # never let recording break the session
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                pass
        return event


class TokenMeter:
    """Batches streamed text into llm_tokens events (~10 per second) instead of one per chunk."""

    def __init__(self, bus: EventBus, interval: float = 0.1):
        self.bus = bus
        self.interval = interval
        self.pending = 0
        self.last = 0.0

    def feed(self, chunk: str) -> None:
        self.pending += len(chunk)
        now = time.monotonic()
        if now - self.last >= self.interval:
            self.flush()
            self.last = now

    def flush(self) -> None:
        if self.pending:
            self.bus.emit("llm_tokens", chars=self.pending)
            self.pending = 0


def load_events(path: Path) -> list[dict]:
    events = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return events
