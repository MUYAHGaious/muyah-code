"""Follow a folder's sessions live from another process (`muyah viz` in a second terminal).

Every session appends its events to `<sessions dir>/<id>.events.jsonl` as they happen. The follower tails
the newest of those files and republishes each new event on its own EventBus, which the web server
streams to the page. When a newer session starts in the folder (you start `muyah` again, or resume
another conversation), it switches to it and tells the page to start over.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

from muyah_code.events import EventBus


IDLE_SWITCH = 5.0  # seconds the current session must be quiet before switching to another active one


class SessionFollower:
    def __init__(self, directory: Path | Callable[[], Path], poll: float = 0.1):
        """directory: the sessions folder, or a function giving it (re-checked every second, because a
        folder's project root can change while you work, e.g. when `git init` or `.muyah/` appears)."""
        self._resolve = directory if callable(directory) else (lambda: directory)
        self.directory = self._resolve()
        self._resolved_at = time.monotonic()
        self.poll = poll
        self.bus = EventBus()
        self.path: Path | None = None
        self._offset = 0
        self._partial = b""
        self._known: set[Path] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.step()  # the current session's events so far are there before the first viewer connects
        self._thread = threading.Thread(target=self._loop, name="muyah-viz-follow", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.wait(self.poll):
            self.step()

    def _pick(self) -> Path | None:
        """The session to show: the current one, unless a brand-new session appeared or the current one
        went quiet while another is being written (so two sessions running at once do not flip-flop)."""
        if time.monotonic() - self._resolved_at >= 1.0:
            self.directory, self._resolved_at = self._resolve(), time.monotonic()
        try:
            files = {p: p.stat().st_mtime for p in self.directory.glob("*.events.jsonl") if p.is_file()}
        except OSError:  # a file vanished between glob and stat
            return self.path
        if not files:
            return self.path
        newest = max(files, key=files.get)
        fresh = [p for p in files if p not in self._known]
        self._known.update(files)
        if self.path is None or self.path not in files:
            return newest
        if fresh and self.path is not None and newest in fresh:
            return newest
        if newest != self.path and files[newest] - files[self.path] > IDLE_SWITCH:
            return newest
        return self.path

    def step(self) -> None:
        """Pick up the newest session and publish any events appended since the last step."""
        newest = self._pick()
        if newest is not None and newest != self.path:
            self.path, self._offset, self._partial = newest, 0, b""
            self.bus.reset(session_id=newest.name.removesuffix(".events.jsonl"))
        if self.path is None:
            return
        try:
            with open(self.path, "rb") as f:
                f.seek(self._offset)
                chunk = f.read()
        except OSError:
            return
        if not chunk:
            return
        self._offset += len(chunk)
        data = self._partial + chunk
        lines = data.split(b"\n")
        self._partial = lines.pop()  # an event still being written stays until its newline arrives
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(event, dict) and "type" in event:
                self.bus.publish(event)
