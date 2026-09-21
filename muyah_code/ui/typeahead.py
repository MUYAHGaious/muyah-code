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
PASTE_START, PASTE_END = "\x1b[200~", "\x1b[201~"


class PasteAssembler:
    """Raw input units -> keys. A paste becomes ONE key, "paste:<text>", instead of a flood of keys where
    every newline is Enter (that queued each pasted line as its own message, and Ctrl+C got lost in it).

    Units are single characters (including "\x1b" and "\r") or named keys ("up", "backspace"...).
    Pastes are recognized two ways:
      * bracketed paste: the terminal wraps it in ESC[200~ ... ESC[201~ (turned on while it works;
        Windows Terminal then also skips its "paste anyway?" warning),
      * a burst: several characters and a newline arriving in the same read (terminals without it)."""

    def __init__(self):
        self.pending = ""            # a possible ESC[200~ being read
        self.paste: list[str] | None = None

    def feed(self, units: list[str]) -> list[str]:
        keys: list[str] = []
        burst = self.paste is None and not self.pending and _looks_like_paste(units)
        if burst:
            text = "".join("\n" if u in ("\r", "\n", "enter") else u for u in units if len(u) == 1 or u == "enter")
            return [f"paste:{text}"]
        for u in units:
            if self.paste is not None:
                self.paste.append("\n" if u in ("\r", "enter") else u if len(u) == 1 else "")
                joined = "".join(self.paste)
                if joined.endswith(PASTE_END):
                    keys.append("paste:" + joined[: -len(PASTE_END)])
                    self.paste = None
                continue
            if len(u) == 1 and (self.pending or u == "\x1b"):
                self.pending += u
                if self.pending == PASTE_START:
                    self.paste, self.pending = [], ""
                elif not PASTE_START.startswith(self.pending):
                    keys += self._flush_pending()
                continue
            keys.append(_name(u) if len(u) == 1 else u)
        if self.pending and self.paste is None:
            keys += self._flush_pending()      # a lone Esc (nothing followed it in this read)
        return [k for k in keys if k]

    def _flush_pending(self) -> list[str]:
        out = [_name(ch) for ch in self.pending]
        self.pending = ""
        return [k for k in out if k]


def _looks_like_paste(units: list[str]) -> bool:
    chars = sum(1 for u in units if len(u) == 1 and u.isprintable())
    newlines = sum(1 for u in units if u in ("\r", "\n", "enter"))
    return newlines >= 1 and chars >= 2 and len(units) >= 4 and "\x1b" not in units


class KeyReader:
    """Reads single keys from the terminal in a background thread and hands them to `on_key`.

    Keys are delivered as text; special keys as names: "enter", "backspace", "esc", "shift-tab", "up", "down",
    "delete", "mic"."""

    def __init__(self, on_key: Callable[[str], None]):
        self.on_key = on_key
        self.assembler = PasteAssembler()
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
        _bracketed_paste(True)
        self._thread = threading.Thread(target=self._run, name="muyah-typeahead", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1)
        self._thread = None
        _bracketed_paste(False)
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
        _bracketed_paste(True)       # a menu in between (prompt_toolkit) turned it off when it closed
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
                units = reader()
            except (OSError, ValueError):
                units = []
            self._idle.set()
            if units:
                for key in self.assembler.feed(units):
                    try:
                        self.on_key(key)
                    except Exception:  # a display problem must not kill the reader
                        continue
            else:
                time.sleep(POLL)

    def _read_windows(self) -> list[str]:
        # msvcrt.getwch() only returns characters, so Shift+Tab arrives as a plain Tab. Console input
        # records carry the key code and the Shift state. Everything waiting is read at once (a paste).
        return _WIN.read_units()

    def _read_posix(self) -> list[str]:
        units = self._read_posix_one()
        if units is None:
            return []
        return units if isinstance(units, list) else [units]

    def _read_posix_one(self):
        import select

        fd = sys.stdin.fileno()
        ready, _, _ = select.select([fd], [], [], POLL)
        if not ready:
            return None
        data = os.read(fd, 4096)
        if self.assembler.paste is not None or PASTE_START.encode() in data or len(data) > 8:
            return list(data.decode("utf-8", errors="ignore"))   # a paste: the assembler sorts it out
        if data == b"\x1b[Z":
            return "shift-tab"
        named = {b"\x1b[D": "left", b"\x1bOD": "left", b"\x1b[C": "right", b"\x1bOC": "right",
                 b"\x1b[H": "home", b"\x1bOH": "home", b"\x1b[1~": "home", b"\x1b[F": "end", b"\x1bOF": "end",
                 b"\x1b[4~": "end", b"\x1b[I": "focus-in", b"\x1b[O": "focus-out", b"\x1b[A": "up", b"\x1bOA": "up", b"\x1b[B": "down", b"\x1bOB": "down", b"\x1b[3~": "delete"}
        if data in named:
            return named[data]
        if data.startswith(b"\x1b") and len(data) > 1:
            return None               # an escape sequence (arrow keys...), not a bare Esc
        return list(data.decode("utf-8", errors="ignore"))

    def _enter_cbreak(self) -> None:
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            if self._saved_tty is None:
                self._saved_tty = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            sys.stdout.write("\x1b[?1004h")    # ask the terminal to report focus changes (for notifications)
            sys.stdout.flush()
        except (ImportError, OSError, ValueError):
            self._saved_tty = None

    def _leave_cbreak(self) -> None:
        if self._saved_tty is None:
            return
        try:
            import termios

            sys.stdout.write("\x1b[?1004l")
            sys.stdout.flush()
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved_tty)
        except (ImportError, OSError, ValueError):
            pass
        self._saved_tty = None


class _WindowsConsole:
    """Key presses from the Windows console input buffer (ReadConsoleInputW), with modifier state."""

    KEY_EVENT = 0x0001
    FOCUS_EVENT = 0x0010
    SHIFT = 0x0010
    CTRL = 0x0004 | 0x0008   # left / right ctrl
    VK = {0x09: "tab", 0x0D: "enter", 0x08: "backspace", 0x1B: "esc", 0x26: "up", 0x28: "down", 0x2E: "delete",
          0x25: "left", 0x27: "right", 0x24: "home", 0x23: "end"}

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

    MAX_BATCH = 4096

    def read_units(self) -> list[str]:
        """Every input event waiting right now, as units for PasteAssembler: characters (Esc and Enter
        as "\x1b" and "\r", so an ESC[200~ paste marker can be recognized) and named keys."""
        if self._api is None:
            self._setup()
        ctypes, wintypes, k32, handle, InputRecord = self._api
        pending = wintypes.DWORD(0)
        if not k32.GetNumberOfConsoleInputEvents(handle, ctypes.byref(pending)) or pending.value == 0:
            return []
        count = min(pending.value, self.MAX_BATCH)
        records, read = (InputRecord * count)(), wintypes.DWORD(0)
        if not k32.ReadConsoleInputW(handle, records, count, ctypes.byref(read)) or read.value == 0:
            return []
        units = []
        for record in records[: read.value]:
            unit = self._unit(record)
            if unit:
                units.append(unit)
        return units

    def _unit(self, record) -> str | None:
        if record.EventType == self.FOCUS_EVENT:   # FOCUS_EVENT_RECORD.bSetFocus sits where bKeyDown does
            return "focus-in" if record.Event.KeyEvent.bKeyDown else "focus-out"
        if record.EventType != self.KEY_EVENT or not record.Event.KeyEvent.bKeyDown:
            return None                  # key releases, mouse and resize events
        key = record.Event.KeyEvent
        if key.wVirtualKeyCode == 0x20 and key.dwControlKeyState & self.CTRL:
            return "mic"
        special = self.VK.get(key.wVirtualKeyCode)
        if special == "tab":
            return "shift-tab" if key.dwControlKeyState & self.SHIFT else "\t"
        if special == "esc":
            return "\x1b"
        if special == "enter":
            return "\r"
        if special:
            return special
        return key.uChar if key.uChar and key.uChar != "\x00" else None


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


def _bracketed_paste(on: bool) -> None:
    """Ask the terminal to mark pastes (ESC[200~ ... ESC[201~) so they arrive as one piece."""
    try:
        if sys.stdout is not None and sys.stdout.isatty():
            sys.stdout.write("\x1b[?2004h" if on else "\x1b[?2004l")
            sys.stdout.flush()
    except (OSError, ValueError):
        pass     # not a terminal (piped): nothing to ask
