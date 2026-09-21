"""Tell you when MUYAH-CODE is done or needs you, while you are in another window.

Setting "notify": auto (default) | osc9 | osc777 | osc99 | bell | native | off.

auto picks what your terminal understands:
  * iTerm2, WezTerm, Ghostty: OSC 9 (a desktop notification from the terminal itself)
  * kitty: OSC 99
  * GNOME Terminal / other VTE terminals, foot: OSC 777
  * Windows: a Windows toast notification with a soft chime ("notify_sound": im, default, mail, reminder, silent)
  * macOS Terminal.app: a notification through osascript
  * anything else: the terminal bell

Only when the terminal is not focused. The terminal reports focus changes while a turn runs (Windows
console focus events; elsewhere the terminal's focus reporting, CSI ?1004h). If it never reports focus,
you are notified only for turns longer than notify_after seconds (default 20).

Every notification also runs Notification hooks: {"message": ..., "kind": "done" | "attention" | "budget"}.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

MODES = ("auto", "osc9", "osc777", "osc99", "bell", "native", "off")
# "notify_sound": the chime of a system notification. Windows toast sounds; macOS uses a system sound name.
WINDOWS_SOUNDS = {"im": "ms-winsoundevent:Notification.IM", "default": "ms-winsoundevent:Notification.Default",
                  "mail": "ms-winsoundevent:Notification.Mail", "reminder": "ms-winsoundevent:Notification.Reminder",
                  "sms": "ms-winsoundevent:Notification.SMS"}
MAC_SOUNDS = {"im": "Glass", "default": "Ping", "mail": "Pop", "reminder": "Hero", "sms": "Tink"}


def auto_mode(env: dict | None = None, platform: str | None = None) -> str:
    env = os.environ if env is None else env
    platform = platform or sys.platform
    program = (env.get("TERM_PROGRAM") or "").lower()
    if program in ("iterm.app", "wezterm", "ghostty") or env.get("WEZTERM_PANE") or env.get("GHOSTTY_RESOURCES_DIR"):
        return "osc9"
    if env.get("KITTY_WINDOW_ID") or "kitty" in (env.get("TERM") or ""):
        return "osc99"
    if env.get("VTE_VERSION") or "foot" in (env.get("TERM") or ""):
        return "osc777"
    if platform == "win32" or platform == "darwin":
        return "native"
    return "bell"


def sequence(mode: str, title: str, message: str) -> str:
    """The escape sequence a terminal turns into a notification ('' for native / off)."""
    clean = message.replace("\x07", " ").replace("\x1b", " ").replace(";", ",")
    if mode == "osc9":
        return f"\x1b]9;{title}: {clean}\x07"
    if mode == "osc777":
        return f"\x1b]777;notify;{title};{clean}\x07"
    if mode == "osc99":
        return f"\x1b]99;;{title}: {clean}\x1b\\"
    if mode == "bell":
        return "\x07"
    return ""                  # native: the system notification brings its own sound (no terminal beep)


def _native(title: str, message: str, sound: str = "im") -> bool:
    """A system notification without extra packages. Runs in the background; returns False if unavailable."""
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    try:
        if sys.platform == "win32":
            ps = shutil.which("powershell") or shutil.which("pwsh")
            if not ps:
                return False
            t, m = title.replace("'", "''"), message.replace("'", "''")
            script = (
                "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null;"
                "$x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
                f"$n = $x.GetElementsByTagName('text'); $n.Item(0).AppendChild($x.CreateTextNode('{t}')) > $null;"
                f"$n.Item(1).AppendChild($x.CreateTextNode('{m}')) > $null;"
                "$a = $x.CreateElement('audio');"
                + ("$a.SetAttribute('silent', 'true');" if sound == "silent" else
                   f"$a.SetAttribute('src', '{WINDOWS_SOUNDS.get(sound, WINDOWS_SOUNDS['im'])}');")
                + "$x.DocumentElement.AppendChild($a) > $null;"
                "$id = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe';"
                "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($id).Show("
                "[Windows.UI.Notifications.ToastNotification]::new($x))")
            subprocess.Popen([ps, "-NoProfile", "-NonInteractive", "-Command", script], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, **flags)
            return True
        if sys.platform == "darwin" and shutil.which("osascript"):
            esc = message.replace('"', "'")
            chime = "" if sound == "silent" else f' sound name "{MAC_SOUNDS.get(sound, "Glass")}"'
            subprocess.Popen(["osascript", "-e", f'display notification "{esc}" with title "{title}"{chime}'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        if shutil.which("notify-send"):
            subprocess.Popen(["notify-send", title, message], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    except OSError:
        return False
    return False


class Notifier:
    def __init__(self, cfg, write, hooks=None, native=_native):
        """write(text): raw output to the terminal. hooks: the HookRunner (Notification hooks)."""
        self.mode = str(cfg.get("notify", "auto") or "auto").lower()
        if self.mode not in MODES:
            self.mode = "auto"
        if self.mode == "auto":
            self.mode = auto_mode()
        self.after = float(cfg.get("notify_after", 20) or 20)
        self.sound = str(cfg.get("notify_sound", "im") or "im").lower()
        self.write = write
        self.hooks = hooks
        self.native = native
        self.sent: list[tuple[str, str]] = []

    def should(self, focused: bool | None, seconds: float) -> bool:
        if self.mode == "off":
            return False
        if focused is None:          # the terminal never said: only for long waits
            return seconds >= self.after
        return not focused

    def notify(self, kind: str, message: str, title: str = "MUYAH-CODE") -> None:
        if self.hooks is not None and self.hooks.has("Notification"):
            self.hooks.run("Notification", {"message": message, "kind": kind, "title": title})
        if self.mode == "off":
            return
        self.sent.append((kind, message))
        seq = sequence(self.mode, title, message)
        if seq:
            self.write(seq)
        if self.mode == "native" and not self.native(title, message, self.sound):
            self.write("\x07")    # no notification system here: at least the bell
