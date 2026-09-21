"""Typing while MUYAH-CODE works.

While a turn runs, a small reader thread takes your keystrokes (the normal input box is not active then):
  * the text you type shows as a draft under the spinner,
  * Enter queues it; the agent gets queued messages at its next step (between tool calls), not only
    after the whole turn, and the transcript shows them as sent at that moment,
  * Esc stops the current step and sends what you queued (or typed) right away.

The reader is paused whenever MUYAH-CODE itself needs the keyboard (permission menus, questions), so the
two never compete for the same keys. Windows uses msvcrt; macOS/Linux put the terminal in cbreak mode
(Ctrl+C still interrupts) and restore it afterwards.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable

POLL = 0.02


class KeyReader:
    """Reads single keys from the terminal in a background thread and hands them to `on_key`.

    Keys are delivered as text; special keys as names: "enter", "backspace", "esc"."""

    def __init__(self, on_key: Callable[[str], None]):
        self.on_key = on_key
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._idle = threading.Event()      # set while the thread is not reading (safe to hand over stdin)
        self._idle.set()
        self._thread: threading.Thread | None = None
        self._saved_tty = None

    @staticmethod
    def available() -> bool:
        try:
            return sys.stdin is not None and sys.stdin.isatty() and sys.stdout.isatty()
        except (ValueError, OSError):
            return False

    def start(self) -> None:
        if self._thread is not None or not self.available():
            return
        self._stop.clear()
        self._paused.clear()
        if os.name != "nt":
            self._enter_cbreak()
        self._thread = threading.Thread(target=self._run, name="muyah-typeahead", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1)
        self._thread = None
        if os.name != "nt":
            self._leave_cbreak()

    def pause(self) -> None:
        """Stop reading until resume() (someone else is about to read the keyboard)."""
        self._paused.set()
        self._idle.wait(timeout=1)
        if os.name != "nt":
            self._leave_cbreak()

    def resume(self) -> None:
        if self._thread is None:
            return
        if os.name != "nt":
            self._enter_cbreak()
        self._paused.clear()

    # ------------------------------------------------------------------ platform

    def _run(self) -> None:
        reader = self._read_windows if os.name == "nt" else self._read_posix
        while not self._stop.is_set():
            if self._paused.is_set():
                self._idle.set()
                time.sleep(POLL)
                continue
            self._idle.clear()
            try:
                key = reader()
            except (OSError, ValueError):
                key = None
            self._idle.set()
            if key:
                try:
                    self.on_key(key)
                except Exception:  # a display problem must not kill the reader
                    continue
            else:
                time.sleep(POLL)

    def _read_windows(self) -> str | None:
        import msvcrt

        if not msvcrt.kbhit():
            return None
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):   # arrows, function keys: ignore the second half
            msvcrt.getwch()
            return None
        return _name(ch)

    def _read_posix(self) -> str | None:
        import select

        fd = sys.stdin.fileno()
        ready, _, _ = select.select([fd], [], [], POLL)
        if not ready:
            return None
        data = os.read(fd, 64)
        if data.startswith(b"\x1b") and len(data) > 1:
            return None               # an escape sequence (arrow keys...), not a bare Esc
        text = data.decode("utf-8", errors="ignore")
        if len(text) == 1:
            return _name(text)
        for ch in text[:-1]:          # pasted text arrives in one read
            self.on_key(_name(ch) or "")
        return _name(text[-1])

    def _enter_cbreak(self) -> None:
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            if self._saved_tty is None:
                self._saved_tty = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except (ImportError, OSError, ValueError):
            self._saved_tty = None

    def _leave_cbreak(self) -> None:
        if self._saved_tty is None:
            return
        try:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved_tty)
        except (ImportError, OSError, ValueError):
            pass
        self._saved_tty = None


def _name(ch: str) -> str | None:
    if ch in ("\r", "\n"):
        return "enter"
    if ch in ("\x08", "\x7f"):
        return "backspace"
    if ch == "\x1b":
        return "esc"
    if ch == "\x03":
        return "ctrl-c"
    if ch.isprintable():
        return ch
    return None
