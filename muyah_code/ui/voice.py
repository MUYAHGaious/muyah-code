"""Talk instead of typing.

On Windows, the mic key opens Windows voice typing (the same as pressing Win+H yourself): it is free, needs
no API key, is very accurate, and types what you say straight into the focused input - MUYAH-CODE's prompt,
or its "type while it works" box. The key is pressed in the terminal, so the terminal is the focused window.

Other systems have no equivalent built in; there the mic uses a speech-to-text service instead (see the
voice extra in the README).
"""

from __future__ import annotations

import sys
import time

VK_LWIN = 0x5B
VK_H = 0x48
KEYEVENTF_KEYUP = 0x0002


def windows_dictation_available() -> bool:
    if sys.platform != "win32":
        return False
    return sys.getwindowsversion().major >= 10


def start_windows_dictation() -> bool:
    """Press Win+H for the user: Windows voice typing opens over the focused window (the terminal)."""
    if not windows_dictation_available():
        return False
    import ctypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    press = user32.keybd_event
    press(VK_LWIN, 0, 0, 0)
    press(VK_H, 0, 0, 0)
    time.sleep(0.02)
    press(VK_H, 0, KEYEVENTF_KEYUP, 0)
    press(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
    return True


def start_dictation() -> tuple[bool, str]:
    """Start talking. Returns (started, what to tell the user)."""
    if start_windows_dictation():
        return True, "Listening (Windows voice typing): speak, and your words appear in the prompt."
    return False, ("Voice input here uses Windows voice typing (Win+H), which this system does not have. "
                   "On macOS use the system dictation shortcut (press Fn twice or the mic key).")
