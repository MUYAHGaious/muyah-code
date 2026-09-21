"""Typing while MUYAH-CODE works.

While a turn runs, a small reader thread takes your keystrokes (the normal input box is not active then):
  * the input box stays on screen under the spinner, and what you type shows in it,
  * Shift+Tab changes the mode (it applies from the agent's next action),
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

    Keys are delivered as text; special keys as names: "enter", "backspace", "esc", "shift-tab"."""

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
        # msvcrt.getwch() only returns characters, so Shift+Tab arrives as a plain Tab. Console input
        # records carry the key code and the Shift state.
        return _WIN.read_key()

    def _read_posix(self) -> str | None:
        import select

        fd = sys.stdin.fileno()
        ready, _, _ = select.select([fd], [], [], POLL)
        if not ready:
            return None
        data = os.read(fd, 64)
        if data == b"\x1b[Z":
            return "shift-tab"
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


class _WindowsConsole:
    """Key presses from the Windows console input buffer (ReadConsoleInputW), with modifier state."""

    KEY_EVENT = 0x0001
    SHIFT = 0x0010
    CTRL = 0x0004 | 0x0008   # left / right ctrl
    VK = {0x09: "tab", 0x0D: "enter", 0x08: "backspace", 0x1B: "esc"}

    def __init__(self):
        self._api = None

    def _setup(self):
        import ctypes
        from ctypes import wintypes

        class KeyEvent(ctypes.Structure):
            _fields_ = [("bKeyDown", wintypes.BOOL), ("wRepeatCount", wintypes.WORD),
                        ("wVirtualKeyCode", wintypes.WORD), ("wVirtualScanCode", wintypes.WORD),
                        ("uChar", wintypes.WCHAR), ("dwControlKeyState", wintypes.DWORD)]

        class EventUnion(ctypes.Union):
            _fields_ = [("KeyEvent", KeyEvent), ("_pad", ctypes.c_byte * 16)]

        class InputRecord(ctypes.Structure):
            _fields_ = [("EventType", wintypes.WORD), ("Event", EventUnion)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = k32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        self._api = (ctypes, wintypes, k32, handle, InputRecord)

    def read_key(self) -> str | None:
        if self._api is None:
            self._setup()
        ctypes, wintypes, k32, handle, InputRecord = self._api
        pending = wintypes.DWORD(0)
        if not k32.GetNumberOfConsoleInputEvents(handle, ctypes.byref(pending)) or pending.value == 0:
            return None
        record, read = InputRecord(), wintypes.DWORD(0)
        if not k32.ReadConsoleInputW(handle, ctypes.byref(record), 1, ctypes.byref(read)) or read.value == 0:
            return None
        if record.EventType != self.KEY_EVENT or not record.Event.KeyEvent.bKeyDown:
            return None                  # key releases, mouse, focus and resize events
        key = record.Event.KeyEvent
        if key.wVirtualKeyCode == 0x20 and key.dwControlKeyState & self.CTRL:
            return "mic"
        special = self.VK.get(key.wVirtualKeyCode)
        if special == "tab":
            return "shift-tab" if key.dwControlKeyState & self.SHIFT else None
        if special:
            return special
        return _name(key.uChar) if key.uChar and key.uChar != "\x00" else None


_WIN = _WindowsConsole()


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
