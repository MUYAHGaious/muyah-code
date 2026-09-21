"""Record the README's terminal demo: the real `muyah` CLI in a real pseudo-console, typed into like a person.

MUYAH-CODE runs exactly as installed (python -m muyah_code) inside a Windows pseudo-console (pywinpty); a
scripted model (tests/fakeserver.py) answers, so every run is the same. The screen is sampled 20 times a
second through a virtual terminal (pyte), each distinct frame is drawn at 1.5x by scripts/termshot.py, and
ffmpeg turns them into an animated WebP (for the README) and an MP4 (full quality).

    pip install pywinpty pyte pillow         # and ffmpeg on PATH
    python scripts/record_cli_demo.py docs/images/demo.webp --mp4 docs/videos/demo.mp4

Windows only (pywinpty); the paths shown are tidied to ~\\projects\\shop-api, so no user name appears.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "scripts")]

import pyte  # noqa: E402
from fakeserver import FakeOpenAI, reply  # noqa: E402
from termshot import render_screen  # noqa: E402

COLS, ROWS = 100, 34
DISPLAY_PATH = r"~\projects\shop-api"
SAMPLE = 0.05            # seconds between screen samples
MAX_HOLD = 2.2           # a frame never stays longer than this (long waits are shortened)

CART = "def total(prices, tax=0.0):\n    subtotal = sum(prices[1:])\n    return round(subtotal * (1 + tax), 2)\n"
TEST = "from cart import total\n\n\ndef test_total():\n    assert total([10, 5], tax=0.1) == 16.5\n"
SCRIPT = [
    reply("", [{"name": "TodoWrite", "arguments": {"todos": [
        {"content": "Reproduce the failing test", "status": "completed"},
        {"content": "Fix total() in cart.py", "status": "in_progress", "activeForm": "Fixing total() in cart.py"},
        {"content": "Run the test suite", "status": "pending"}]}},
        {"name": "Read", "arguments": {"file_path": "cart.py"}}], think=1.6),
    reply("", [{"name": "Edit", "arguments": {"file_path": "cart.py", "old_string": "sum(prices[1:])",
                                              "new_string": "sum(prices)"}}], think=1.2),
    reply("", [{"name": "Bash", "arguments": {"command": "python -m pytest -q -p no:cacheprovider"}}], think=0.9),
    reply("Fixed **`total()`** in `cart.py:2`: it skipped the first price (`prices[1:]`), so every cart was "
          "under-charged.\n\n- Changed it to `sum(prices)`.\n- Verified: `pytest -q` passes (1 passed).",
          think=1.2),
]


class Snapshot:
    """A frozen copy of the pyte screen (what render_screen needs), with real paths shown tidied."""

    def __init__(self, screen: pyte.Screen, replacements: list[tuple[str, str]]):
        self.columns, self.lines = screen.columns, screen.lines
        self.buffer = [_tidy([screen.buffer[y][x] for x in range(screen.columns)], replacements)
                       for y in range(screen.lines)]
        self.display = ["".join(c.data for c in row) for row in self.buffer]
        self.cursor = None if screen.cursor.hidden else (screen.cursor.x, screen.cursor.y)

    def key(self):
        return tuple(tuple(row) for row in self.buffer), self.cursor


def _tidy(row: list, replacements: list[tuple[str, str]]) -> list:
    """Replace a real path in a screen row by the display path, keeping the styles; the row stays as wide."""
    for real, shown in replacements:
        text = "".join(c.data for c in row)
        i = text.find(real)
        while i >= 0:
            style = row[i]
            new = [style._replace(data=ch) for ch in shown]
            blank = row[-1]._replace(data=" ")
            row = row[:i] + new + row[i + len(real):]
            row += [blank] * (len(text) - len(row))
            text = "".join(c.data for c in row)
            i = text.find(real)
    return row


class Recorder:
    def __init__(self, proc, replacements: list[tuple[str, str]]):
        self.proc = proc
        self.replacements = replacements
        self.screen = pyte.Screen(COLS, ROWS)
        self.stream = pyte.Stream(self.screen)
        self.lock = threading.Lock()
        self.frames: list[tuple[float, Snapshot]] = []
        self.stop = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._sample, daemon=True).start()

    def _read(self) -> None:
        while not self.stop.is_set():
            try:
                data = self.proc.read(4096)
            except EOFError:
                return
            with self.lock:
                self.stream.feed(data)

    def _sample(self) -> None:
        while not self.stop.is_set():
            with self.lock:
                snap = Snapshot(self.screen, self.replacements)
            if not self.frames or snap.key() != self.frames[-1][1].key():
                self.frames.append((time.monotonic(), snap))
            time.sleep(SAMPLE)

    def text(self) -> str:
        with self.lock:
            return "\n".join(self.screen.display)

    def wait_for(self, needle: str, timeout: float = 60) -> None:
        end = time.monotonic() + timeout
        while needle not in self.text():
            if time.monotonic() > end:
                raise SystemExit(f"timed out waiting for {needle!r}:\n{self.text()}")
            time.sleep(0.05)


def _long(path: Path) -> str:
    """The long form of a path (temp folders often come as 8.3 short names like MUYAHG~1)."""
    import ctypes

    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetLongPathNameW(str(path), buf, 1024)
    return buf.value if n else str(path)


def type_like_a_person(proc, text: str, delay: float = 0.045) -> None:
    for ch in text:
        proc.write(ch)
        time.sleep(delay)


def encode(frames: list[tuple[float, Snapshot]], end: float, out: str, mp4: str | None, scale: float,
           width: int) -> None:
    work = Path(tempfile.mkdtemp(prefix="muyah-demo-frames-"))
    try:
        lines = ["ffconcat version 1.0"]
        cache: dict = {}
        for i, (t, snap) in enumerate(frames):
            nxt = frames[i + 1][0] if i + 1 < len(frames) else end
            hold = min(MAX_HOLD, max(SAMPLE, nxt - t))
            key = snap.key()
            if key not in cache:
                path = work / f"f{len(cache):05d}.png"
                render_screen(snap, path, title="muyah", rows=ROWS, scale=scale, cursor=snap.cursor)
                cache[key] = path.name
            lines += [f"file '{cache[key]}'", f"duration {hold:.3f}"]
        lines.append(f"file '{cache[frames[-1][1].key()]}'")      # the last frame's duration needs this
        (work / "list.ffconcat").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{len(frames)} frames, {len(cache)} distinct")
        base = ["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(work / "list.ffconcat")]
        if mp4:
            subprocess.run(base + ["-vf", "fps=30,pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-crf", "14",
                                   "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4],
                           check=True)
            print(f"wrote {mp4}: {Path(mp4).stat().st_size / 1e6:.1f} MB")
        subprocess.run(base + ["-vf", f"fps=20,scale={width}:-1:flags=lanczos", "-c:v", "libwebp_anim",
                               "-q:v", "92", "-compression_level", "6", "-loop", "0", out], check=True)
        print(f"wrote {out}: {Path(out).stat().st_size / 1e6:.1f} MB")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="animated WebP to write")
    ap.add_argument("--mp4", help="also write an MP4 here")
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--width", type=int, default=1400, help="width of the WebP")
    a = ap.parse_args()

    from winpty import PtyProcess
    from winpty.enums import Backend

    from muyah_code.pathfix import find_launcher_dir

    from muyah_code.trust import trust

    home = Path(tempfile.mkdtemp(prefix="muyah-demo-home-"))
    project = Path(tempfile.mkdtemp(prefix="demo-")) / "shop-api"
    (project / ".muyah").mkdir(parents=True)
    (project / "cart.py").write_text(CART, encoding="utf-8")
    (project / "test_cart.py").write_text(TEST, encoding="utf-8")
    trust(home, project)
    with FakeOpenAI(SCRIPT, chunk_delay=0.035) as srv:
        (home / "settings.json").write_text(json.dumps({
            "viz": {"autostart": False}, "notify": "off", "theme": "teal", "provider": "openrouter",
            "base_url": srv.url, "model": "qwen3-coder-480b", "api_key": "demo"}), encoding="utf-8")
        launcher = find_launcher_dir()
        env = {**os.environ, "MUYAH_HOME": str(home), "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8",
               "COLORTERM": "truecolor", "WT_SESSION": "demo",
               "PATH": (str(launcher) + os.pathsep if launcher else "") + os.environ.get("PATH", "")}
        # NO_COLOR (set by some tools and CI) would record a colorless demo: people's terminals show color
        for var in ("MUYAH_BASE_URL", "MUYAH_API_KEY", "MUYAH_MODEL", "MUYAH_PROFILE", "NO_COLOR"):
            env.pop(var, None)
        proc = PtyProcess.spawn([sys.executable, "-m", "muyah_code", "--mode", "acceptEdits",
                                 "--allowedTools", "Bash(python:*)", "--no-mcp"], cwd=str(project), env=env,
                                dimensions=(ROWS, COLS), backend=Backend.ConPTY)
        replacements = sorted({(v, DISPLAY_PATH) for v in (str(project), str(project.resolve()), _long(project))},
                              key=lambda r: -len(r[0]))
        rec = Recorder(proc, replacements)
        try:
            rec.wait_for("❯")
            time.sleep(1.4)
            type_like_a_person(proc, "the cart total test is failing, fix it")
            time.sleep(0.6)
            proc.write("\r")
            rec.wait_for("Worked", timeout=90)
            time.sleep(2.8)
            for _ in range(5):                      # Shift+Tab through the modes, back to edit
                proc.write("\x1b[Z")
                time.sleep(1.0)
            time.sleep(0.4)
            type_like_a_person(proc, "/", 0.1)
            time.sleep(1.6)
            type_like_a_person(proc, "mo", 0.18)
            time.sleep(2.0)
            proc.write("\x15")                      # clear the line
            time.sleep(1.2)
            end = time.monotonic()
        finally:
            rec.stop.set()
            try:
                proc.write("\x03\x03")
                time.sleep(0.5)
            finally:
                proc.terminate(force=True)
    frames = [f for f in rec.frames if f[0] <= end]
    encode(frames, end, a.out, a.mp4, a.scale, a.width)
    return 0


if __name__ == "__main__":
    sys.exit(main())
