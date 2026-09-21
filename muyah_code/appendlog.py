"""Append-only JSONL writing off the main thread.

Session transcripts and event recordings get a line per message/event. Opening a file costs little on
Linux, but on Windows (with antivirus scanning each open) it measured ~60 ms, and it happened on the main
thread in the middle of a turn: streaming stalled with it. Here each file gets a writer thread that keeps
the file open while lines keep coming, writes them in order, and closes it after a quiet second.

`flush(path)` waits until everything queued for a file is on disk; readers call it first.
"""

from __future__ import annotations

import atexit
import queue
import threading
from pathlib import Path

IDLE_CLOSE = 1.0    # close the file after this many quiet seconds
IDLE_EXIT = 10.0    # and end the writer thread after this many


class _Writer:
    def __init__(self, path: Path):
        self.path = path
        self.queue: queue.Queue = queue.Queue()
        self.alive = True
        self.error: OSError | None = None
        self.thread = threading.Thread(target=self._run, name=f"muyah-write-{path.name}", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        handle = None
        idle = 0.0
        while True:
            try:
                line = self.queue.get(timeout=IDLE_CLOSE)
            except queue.Empty:
                if handle is not None:
                    handle.close()
                    handle = None
                idle += IDLE_CLOSE
                if idle >= IDLE_EXIT and _retire(self):
                    return
                continue
            idle = 0.0
            if line is _CLOSE:
                if handle is not None:
                    handle.close()
                    handle = None
                self.queue.task_done()
                continue
            lines = [line]
            close_after = False
            while True:  # take everything already queued: one write for a burst of events
                try:
                    more = self.queue.get_nowait()
                except queue.Empty:
                    break
                if more is _CLOSE:
                    close_after = True
                    self.queue.task_done()
                    break
                lines.append(more)
            try:
                if handle is None:
                    handle = open(self.path, "a", encoding="utf-8")
                handle.write("".join(lines))
                handle.flush()
            except OSError as e:  # disk full, folder deleted...: never break the session over a log
                self.error = e
                handle = None
            finally:
                if close_after and handle is not None:
                    handle.close()
                    handle = None
                for _ in lines:
                    self.queue.task_done()


_CLOSE = object()  # queue marker: close the file now (the session is ending)
_writers: dict[str, _Writer] = {}
_lock = threading.Lock()


def _key(path: Path) -> str:
    return str(path)


def _retire(writer: _Writer) -> bool:
    """End an idle writer, unless a line arrived meanwhile."""
    with _lock:
        if not writer.queue.empty():
            return False
        writer.alive = False
        if _writers.get(_key(writer.path)) is writer:
            del _writers[_key(writer.path)]
        return True


def append_line(path: Path, line: str) -> None:
    """Queue one line (a newline is added) for `path`. Returns immediately."""
    with _lock:
        writer = _writers.get(_key(path))
        if writer is None or not writer.alive:
            writer = _writers[_key(path)] = _Writer(path)
        writer.queue.put(line + "\n")


def flush(path: Path | None = None, close: bool = False) -> None:
    """Wait until the queued lines of `path` (or of every file) are written; `close` also releases the
    file handle (so the file can be moved or deleted right away)."""
    with _lock:
        writers = [w for k, w in _writers.items() if path is None or k == _key(path)]
        if close:
            for w in writers:
                w.queue.put(_CLOSE)
    for w in writers:
        w.queue.join()


atexit.register(flush, None, True)
