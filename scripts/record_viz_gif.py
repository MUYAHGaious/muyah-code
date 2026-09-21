"""Record the live view replaying a session, as an animated GIF for the README.

It serves the replay (muyah_code.viz.VizServer), opens it in headless Chrome, captures real frames with
the DevTools screencast, and turns them into a GIF with ffmpeg. Nothing is simulated: it is the page
exactly as a user sees it.

    python scripts/make_viz_demo.py --out demo.events.jsonl       # record a session first
    python scripts/record_viz_gif.py demo.events.jsonl docs/images/viz-demo.gif [--speed 2]

Needs Chrome (or Edge), ffmpeg on PATH, and `pip install websocket-client`.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import websocket

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from muyah_code.events import load_events  # noqa: E402
from muyah_code.viz import VizServer  # noqa: E402

BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "chromium-browser", "microsoft-edge"]
PORT = 9339


def find_browser() -> str:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    raise SystemExit("Chrome or Edge is needed to record the GIF.")


class DevTools:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=30, suppress_origin=True)
        self.next_id = 0
        self.frames: list[tuple[float, bytes]] = []
        self.lock = threading.Lock()
        self.replies: dict[int, dict] = {}
        self.closed = False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        while not self.closed:
            try:
                msg = json.loads(self.ws.recv())
            except Exception:
                return
            if msg.get("method") == "Page.screencastFrame":
                p = msg["params"]
                with self.lock:
                    self.frames.append((time.monotonic(), base64.b64decode(p["data"])))
                self.send("Page.screencastFrameAck", {"sessionId": p["sessionId"]}, wait=False)
            elif "id" in msg:
                self.replies[msg["id"]] = msg

    def send(self, method: str, params: dict | None = None, wait: bool = True) -> dict:
        self.next_id += 1
        rid = self.next_id
        self.ws.send(json.dumps({"id": rid, "method": method, "params": params or {}}))
        if not wait:
            return {}
        end = time.time() + 30
        while rid not in self.replies and time.time() < end:
            time.sleep(0.01)
        return self.replies.pop(rid, {})

    def evaluate(self, expression: str):
        res = self.send("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        return res.get("result", {}).get("result", {}).get("value")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("out")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=820)
    ap.add_argument("--gif-width", type=int, default=960)
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()

    server = VizServer(events=load_events(Path(a.events)), title="demo")
    url = server.start() + f"&speed={a.speed:g}"
    profile = tempfile.mkdtemp(prefix="muyah-gif-chrome-")
    chrome = subprocess.Popen([find_browser(), "--headless=new", f"--remote-debugging-port={PORT}",
                               f"--window-size={a.width},{a.height}", f"--user-data-dir={profile}",
                               "--hide-scrollbars", "--force-device-scale-factor=1", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames_dir = Path(tempfile.mkdtemp(prefix="muyah-gif-frames-"))
    try:
        targets = []
        for _ in range(100):
            try:
                targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=2).read())
                if any(t.get("type") == "page" for t in targets):
                    break
            except OSError:
                pass
            time.sleep(0.1)
        page = next(t for t in targets if t.get("type") == "page")
        dt = DevTools(page["webSocketDebuggerUrl"])
        dt.send("Page.enable")
        dt.send("Emulation.setDeviceMetricsOverride", {"width": a.width, "height": a.height,
                                                       "deviceScaleFactor": 1, "mobile": False})
        dt.send("Page.navigate", {"url": url})
        for _ in range(100):  # wait until the recording is loaded and playing
            if dt.evaluate("typeof adjusted !== 'undefined' && adjusted.length > 0"):
                break
            time.sleep(0.1)
        dt.evaluate("seekTo(0); playing = true;")
        dt.send("Page.startScreencast", {"format": "jpeg", "quality": 85, "maxWidth": a.width,
                                         "maxHeight": a.height, "everyNthFrame": 1})
        start = time.monotonic()
        while time.monotonic() - start < 180:
            done = dt.evaluate("replayPos >= adjusted.length && replayClock > adjusted[adjusted.length-1].t + 1")
            if done:
                break
            time.sleep(0.2)
        time.sleep(1.0)
        dt.send("Page.stopScreencast")
        end = time.monotonic()
        dt.closed = True
        with dt.lock:
            frames = list(dt.frames)
        if not frames:
            raise SystemExit("no frames captured")
        # resample to a steady frame rate (the screencast only sends frames when something changed)
        n = 0
        t = start
        i = 0
        while t <= end:
            while i + 1 < len(frames) and frames[i + 1][0] <= t:
                i += 1
            (frames_dir / f"f{n:05d}.jpg").write_bytes(frames[i][1])
            n += 1
            t += 1 / a.fps
        palette = (f"fps={a.fps},scale={a.gif_width}:-1:flags=lanczos,split[x][y];"
                   "[x]palettegen=max_colors=160:stats_mode=diff[p];[y][p]paletteuse=dither=bayer:bayer_scale=4:"
                   "diff_mode=rectangle")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(a.fps), "-i",
                        str(frames_dir / "f%05d.jpg"), "-vf", palette, "-loop", "0", a.out], check=True)
        print(f"wrote {a.out}: {n} frames, {Path(a.out).stat().st_size / 1e6:.1f} MB")
    finally:
        chrome.terminate()
        server.stop()
        shutil.rmtree(frames_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
