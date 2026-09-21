"""Talk instead of typing.

On Windows, the mic key opens Windows voice typing (the same as pressing Win+H yourself): it is free, needs
no API key, is very accurate, and types what you say straight into the focused input - MUYAH-CODE's prompt,
or its "type while it works" box. The key is pressed in the terminal, so the terminal is the focused window.

The shortcut is sent only after you let go of Ctrl (and Shift/Alt): pressed while Ctrl+Space is still held,
Windows would see Ctrl+Win+H, which does nothing.

Other systems have no equivalent built in; there the mic uses a speech-to-text service instead (see the
voice extra in the README).
"""

from __future__ import annotations

import sys
import threading
import time

VK_LWIN = 0x5B
VK_H = 0x48
MODIFIERS = (0x11, 0x10, 0x12)       # Ctrl, Shift, Alt
KEYEVENTF_KEYUP = 0x0002
INPUT_KEYBOARD = 1
RELEASE_WAIT = 2.0                   # seconds to wait for you to let go of Ctrl+Space


def windows_dictation_available() -> bool:
    if sys.platform != "win32":
        return False
    return sys.getwindowsversion().major >= 10


def _send_win_h() -> bool:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    ULONG_PTR = ctypes.c_size_t

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class MOUSEINPUT(ctypes.Structure):     # the union's largest member: INPUT must have the right size
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    # wait until Ctrl/Shift/Alt are up: Ctrl+Win+H is not the voice typing shortcut
    end = time.monotonic() + RELEASE_WAIT
    while time.monotonic() < end and any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in MODIFIERS):
        time.sleep(0.02)

    def key(vk: int, up: bool) -> INPUT:
        return INPUT(INPUT_KEYBOARD, _U(ki=KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP if up else 0, 0, 0)))

    seq = (INPUT * 4)(key(VK_LWIN, False), key(VK_H, False), key(VK_H, True), key(VK_LWIN, True))
    sent = user32.SendInput(4, seq, ctypes.sizeof(INPUT))
    return sent == 4


def start_windows_dictation() -> bool:
    """Press Win+H for the user (once Ctrl is released): voice typing opens over the terminal."""
    if not windows_dictation_available():
        return False
    threading.Thread(target=_send_win_h, name="muyah-voice", daemon=True).start()
    return True


def start_dictation() -> tuple[bool, str]:
    """Start talking. Returns (started, what to tell the user)."""
    if start_windows_dictation():
        return True, ("Listening (Windows voice typing): speak, and your words appear in the prompt. If nothing "
                      "opens, turn on voice typing in Windows Settings > Time & language > Speech.")
    return False, ("Voice input here uses Windows voice typing (Win+H), which this system does not have. "
                   "On macOS use the system dictation shortcut (press Fn twice or the mic key).")
