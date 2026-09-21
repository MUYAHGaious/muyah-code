"""Live event stream of what the agent is doing - powers `/viz`, `muyah viz` (live) and `muyah viz --replay`.

Events are small JSON-able dicts with a "type" and a "t" timestamp (seconds since the session started;
a resumed session continues its clock). Agent events also carry "agent" ("main" or the sub-agent type):

    session       model, provider, window, cwd, mode, tools, mcp[{name, status}]
    turn_start    prompt                         turn_end     status, duration, tool_calls
    llm_start     model                          llm_tokens   chars, text, thinking (batched ~10/s)
    llm_end       prompt_tokens, completion_tokens, duration, calls
    tool_start    id, name, title                tool_end     id, name, ok, summary, duration, chars
    context       used, usable, window, parts{system, conversation, tools}
    compact       description, emergency         lessons      items
    subagent_start prompt                        subagent_end status, duration, tool_calls
    todos         items                          hook         event, command, outcome, duration
    reset         (a viewer following a folder switched to a newer session)

Publishing is cheap when nobody listens, so the agent always emits. Every event is also appended to the
session's `.events.jsonl` file (flushed per event), so another process can follow a session live and any
session can be replayed later.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

Listener = Callable[[dict], None]


def _last_time(path: Path) -> float:
    """The "t" of the last event already recorded in a file (0 if none), read from the file's tail."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            lines = f.read().splitlines()
    except OSError:
        return 0.0
    for line in reversed(lines):
        try:
            return float(json.loads(line).get("t") or 0)
        except (ValueError, AttributeError):
            continue
    return 0.0


class EventBus:
    def __init__(self, record_to: Path | None = None):
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()
        self.started = time.time()
        if record_to is not None and record_to.exists():
            # a resumed session: keep its timeline increasing instead of restarting at 0
            self.started -= _last_time(record_to) + 1.0
        self.history: list[dict] = []   # events so far, so a late viewer can catch up
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
        if self.record_to is not None:
            try:
                with open(self.record_to, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            except OSError:
                self.record_to = None  # never let recording break the session
        return self.publish(event)

    def publish(self, event: dict) -> dict:
        """Deliver an already-built event (keeps its "t"): used by followers that replay another process."""
        with self._lock:
            self.history.append(event)
            if len(self.history) > self.max_history:
                del self.history[: len(self.history) - self.max_history]
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:  # a broken viewer must never break the agent
                continue
        return event

    def reset(self, **data) -> None:
        """Forget the history and tell viewers to start over (a follower switched sessions)."""
        with self._lock:
            self.history.clear()
        self.publish({"type": "reset", "t": 0, **data})


class TokenMeter:
    """Batches streamed text into llm_tokens events (~10 per second) instead of one per chunk."""

    def __init__(self, bus: EventBus, agent: str = "main", interval: float = 0.1):
        self.bus = bus
        self.agent = agent
        self.interval = interval
        self.text: list[str] = []
        self.thinking: list[str] = []
        self.last = 0.0

    def feed(self, chunk: str, thinking: bool = False) -> None:
        (self.thinking if thinking else self.text).append(chunk)
        now = time.monotonic()
        if now - self.last >= self.interval:
            self.flush()
            self.last = now

    def flush(self) -> None:
        if self.text or self.thinking:
            text, thinking = "".join(self.text), "".join(self.thinking)
            self.bus.emit("llm_tokens", agent=self.agent, chars=len(text) + len(thinking), text=text,
                          thinking=thinking)
            self.text, self.thinking = [], []


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
