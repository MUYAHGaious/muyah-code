"""Make `muyah` a command you can type anywhere.

`pip install --user` (and some Python installs) put the `muyah` launcher in a scripts folder that is not on
PATH; pip only prints a warning that is easy to miss, and then `muyah` is "not recognized". `muyah path`
(or `python -m muyah_code path`, which always works) finds the folder that holds the launcher and adds it
to your user PATH:

  * Windows: the user Path in the registry (HKCU\\Environment), then tells running programs it changed;
  * macOS / Linux: one marked line in your shell start-up files (~/.profile, plus ~/.bashrc / ~/.zshrc
    when you have them).

Nothing is changed when the folder is already on PATH; running it twice adds nothing twice.
"""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path

MARKER = "# added by muyah path"
LAUNCHER = "muyah.exe" if os.name == "nt" else "muyah"


def launcher_dirs() -> list[Path]:
    """Folders where pip puts console scripts for this Python (the normal one, then the per-user one)."""
    schemes = [None, f"{os.name}_user"]
    if sys.platform == "darwin":
        schemes.append("osx_framework_user")
    out: list[Path] = []
    for scheme in schemes:
        try:
            d = Path(sysconfig.get_path("scripts", scheme) if scheme else sysconfig.get_path("scripts"))
        except KeyError:
            continue
        if d not in out:
            out.append(d)
    return out


def find_launcher_dir() -> Path | None:
    """The folder that actually holds the `muyah` launcher, if any."""
    for d in launcher_dirs():
        if (d / LAUNCHER).exists():
            return d
    return None


def on_path(folder: Path, path_value: str | None = None) -> bool:
    value = os.environ.get("PATH", "") if path_value is None else path_value
    want = os.path.normcase(os.path.normpath(str(folder)))
    return any(os.path.normcase(os.path.normpath(os.path.expandvars(p))) == want
               for p in value.split(os.pathsep) if p.strip())


def command_available() -> bool:
    return shutil.which("muyah") is not None


def add_to_path(folder: Path, home: Path | None = None) -> str:
    """Add `folder` to the user's PATH for new terminals. Returns what was done, in a sentence."""
    if os.name == "nt":
        return _add_windows(folder)
    return _add_posix(folder, home or Path.home())


def _add_windows(folder: Path) -> str:
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                        winreg.KEY_READ | winreg.KEY_WRITE) as key:
        try:
            current, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            current, kind = "", winreg.REG_EXPAND_SZ
        if on_path(folder, current):
            return f"{folder} is already on your user PATH, but this window started without it."
        parts = [p for p in current.split(";") if p.strip()]
        parts.append(str(folder))
        winreg.SetValueEx(key, "Path", 0, kind if kind in (winreg.REG_SZ, winreg.REG_EXPAND_SZ)
                          else winreg.REG_EXPAND_SZ, ";".join(parts))
    _broadcast_environment_change()
    return f"Added {folder} to your user PATH."


def _broadcast_environment_change() -> None:
    """Tell Explorer (and so new terminals) that the environment changed, as the Settings dialog does."""
    import ctypes

    HWND_BROADCAST, WM_SETTINGCHANGE, SMTO_ABORTIFHUNG = 0xFFFF, 0x001A, 0x0002
    result = ctypes.c_size_t()
    ctypes.windll.user32.SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment",
                                             SMTO_ABORTIFHUNG, 2000, ctypes.byref(result))


def _add_posix(folder: Path, home: Path) -> str:
    line = f'export PATH="{folder}:$PATH"  {MARKER}\n'
    files = [home / ".profile"] + [home / name for name in (".bashrc", ".zshrc") if (home / name).exists()]
    if sys.platform == "darwin" and not (home / ".zshrc").exists():
        files.append(home / ".zshrc")          # the default shell on macOS reads .zshrc, not .profile
    changed = []
    for f in files:
        text = f.read_text(encoding="utf-8") if f.exists() else ""
        if str(folder) in text:
            continue
        with f.open("a", encoding="utf-8") as out:
            out.write(("" if not text or text.endswith("\n") else "\n") + line)
        changed.append(f.name)
    if not changed:
        return f"{folder} is already in your shell start-up files."
    return f"Added {folder} to PATH in ~/{', ~/'.join(changed)}."


def fix(console) -> int:
    """`muyah path`: put the launcher's folder on PATH (if it is not) and say what to do next."""
    folder = find_launcher_dir()
    if folder is None:
        console.print("[yellow]Could not find the muyah launcher.[/] Reinstall with "
                      "[bold]pipx install git+https://github.com/MUYAHGaious/muyah-code[/], or keep using "
                      "[bold]python -m muyah_code[/].")
        return 1
    if on_path(folder) and command_available():
        console.print(f"[green]✓[/] `muyah` already works from any folder ({folder}).")
        return 0
    try:
        message = add_to_path(folder)
    except OSError as e:
        console.print(f"[red]Could not change PATH:[/] {e}. Add this folder to PATH yourself: {folder}")
        return 1
    console.print(f"[green]✓[/] {message}")
    if os.name == "nt":
        # a terminal keeps the PATH it started with, and terminals inside an app (VS Code, Cursor, Windows
        # Terminal) copy the app's: give the one line that reloads it in this window, with no restart
        console.print("[dim]To use it in this window now (PowerShell), run:[/]")
        console.print("  [bold]$env:Path = [Environment]::GetEnvironmentVariable('Path','User') + ';' + "
                      "[Environment]::GetEnvironmentVariable('Path','Machine')[/]", soft_wrap=True)
        console.print("[dim]New windows find it by themselves once the app they run in (VS Code, Cursor, "
                      "Windows Terminal...) has been fully closed and opened again.[/]")
    else:
        console.print(f"[dim]To use it in this window now, run:[/] [bold]export PATH=\"{folder}:$PATH\"[/]",
                      soft_wrap=True)
        console.print("[dim]New terminals find it by themselves.[/]")
    return 0


def startup_hint() -> str:
    """One line for when you started it as `python -m muyah_code` and `muyah` would not have worked."""
    if command_available():
        return ""
    folder = find_launcher_dir()
    if folder is None:
        return ""
    return ("Tip: `muyah` is not on your PATH yet, so it only starts as `python -m muyah_code`. "
            "Run `python -m muyah_code path` once to fix that.")
