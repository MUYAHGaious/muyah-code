"""Regenerate the README screenshots (docs/images/*.png) from the REAL MUYAH-CODE UI.

Each scene runs the actual code - the trust check, the REPL, a full agent turn against a scripted local
model server (tests/fakeserver.py) - inside a virtual terminal (pyte), then renders the screen to PNG
with scripts/termshot.py. The images therefore always match what users see.

    pip install pyte pillow
    python scripts/make_screenshots.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "scripts")]
os.environ["MUYAH_HOME"] = tempfile.mkdtemp(prefix="muyah-shots-")
for var in ("MUYAH_BASE_URL", "MUYAH_API_KEY", "MUYAH_MODEL", "MUYAH_PROFILE"):
    os.environ.pop(var, None)

from fakeserver import FakeOpenAI, reply  # noqa: E402
from prompt_toolkit.application import create_app_session  # noqa: E402
from prompt_toolkit.data_structures import Size  # noqa: E402
from prompt_toolkit.input import create_pipe_input  # noqa: E402
from prompt_toolkit.output.vt100 import Vt100_Output  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.text import Text  # noqa: E402
from termshot import render  # noqa: E402

from muyah_code.app import App  # noqa: E402
from muyah_code.config import load_config  # noqa: E402
from muyah_code.trust import ensure_trusted  # noqa: E402
from muyah_code.ui.base import PermissionRequest  # noqa: E402
from muyah_code.ui.repl import Repl  # noqa: E402
from muyah_code.ui.select import Prompter  # noqa: E402
from muyah_code.ui.terminal import TerminalUI  # noqa: E402
from muyah_code.ui.theme import set_theme, theme  # noqa: E402

OUT = ROOT / "docs" / "images"
COLS = 100
WORKDIR = Path(tempfile.mkdtemp(prefix="demo-")) / "shop-api"
# Shown instead of the real temp folder (keeps screenshots tidy and free of the author's username).
DISPLAY_PATH = r"~\projects\shop-api"


class Screen:
    """A virtual terminal both Rich and prompt_toolkit write into."""

    def __init__(self, rows: int = 34, below: int = 8):
        self.rows = rows
        self.raw = io.StringIO()
        outer = self

        class Term(Vt100_Output):
            def get_rows_below_cursor_position(self):
                return below

        self.term = Term(self.raw, lambda: Size(rows=outer.rows, columns=COLS), term="xterm-256color",
                         enable_cpr=False)
        self.console = Console(file=self.raw, width=COLS, force_terminal=True, color_system="truecolor",
                               highlight=False)

    def save(self, name: str) -> None:
        ansi = self.raw.getvalue()
        for real in (str(WORKDIR.resolve()), str(WORKDIR)):
            ansi = ansi.replace(real, DISPLAY_PATH)
        path = render(ansi, OUT / name, cols=COLS, rows=self.rows, title="muyah")
        print("wrote", path.relative_to(ROOT))


def later(delay: float, *actions) -> None:
    def run():
        time.sleep(delay)
        for a in actions:
            a()
            time.sleep(0.4)
    threading.Thread(target=run, daemon=True).start()


def make_project() -> Path:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    (WORKDIR / ".muyah").mkdir(exist_ok=True)
    (WORKDIR / "cart.py").write_text("def total(prices, tax=0.0):\n    subtotal = sum(prices[1:])\n"
                                     "    return round(subtotal * (1 + tax), 2)\n", encoding="utf-8")
    (WORKDIR / "test_cart.py").write_text("from cart import total\n\n\ndef test_total():\n"
                                          "    assert total([10, 5], tax=0.1) == 16.5\n", encoding="utf-8")
    return WORKDIR


def scene_trust() -> None:
    s = Screen(rows=22, below=6)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=s.term):
        s.console.print(f"PS {DISPLAY_PATH}> [bold]muyah[/]\n")
        later(1.0, lambda: s.save("trust.png"), lambda: pipe.send_text("\r"))
        ensure_trusted(WORKDIR, Path(os.environ["MUYAH_HOME"]), s.console, Prompter())


def scene_home_and_menu() -> None:
    s = Screen(rows=18, below=10)
    ui = TerminalUI(s.console)
    with FakeOpenAI([]) as srv, create_pipe_input() as pipe, create_app_session(input=pipe, output=s.term):
        cfg = load_config(cwd=WORKDIR, overrides={"base_url": srv.url, "model": "claude-opus-5",
                                                  "provider": "anthropic"})
        app = App(cfg, ui, cwd=WORKDIR, mode="acceptEdits", enable_mcp=False)
        later(2.0, lambda: pipe.send_text("/pro"), lambda: time.sleep(0.6), lambda: s.save("home.png"),
              lambda: pipe.send_text("\x15/exit\r"))
        Repl(app, ui).run()
        app.shutdown()


def scene_session() -> None:
    """A full turn: plan, read, edit with a diff, run the tests, answer - then the input box again."""
    s = Screen(rows=60, below=4)
    ui = TerminalUI(s.console, animate=False)
    script = [
        reply("", [{"name": "TodoWrite", "arguments": {"todos": [
            {"content": "Reproduce the failing test", "status": "completed"},
            {"content": "Fix total() in cart.py", "status": "in_progress", "activeForm": "Fixing total() in cart.py"},
            {"content": "Run the test suite", "status": "pending"}]}},
            {"name": "Read", "arguments": {"file_path": "cart.py"}}]),
        reply("", [{"name": "Edit", "arguments": {"file_path": "cart.py", "old_string": "sum(prices[1:])",
                                                  "new_string": "sum(prices)"}}]),
        reply("", [{"name": "Bash", "arguments": {"command": "python -m pytest -q -p no:cacheprovider"}}]),
        reply("Fixed **`total()`** in `cart.py:2`: it skipped the first price (`prices[1:]`), so every cart "
              "was under-charged.\n\n```python\nsubtotal = sum(prices)\n```\n\n"
              "- Verified: `pytest -q` passes (1 passed).\n- No other callers relied on the old behaviour."),
    ]
    with FakeOpenAI(script) as srv, create_pipe_input() as pipe, create_app_session(input=pipe, output=s.term):
        cfg = load_config(cwd=WORKDIR, overrides={"base_url": srv.url, "model": "qwen3-coder-480b",
                                                  "provider": "openrouter"})
        cfg.set("learning.reflect", False)
        # as if the user had once answered "Yes, and don't ask again" for python commands
        app = App(cfg, ui, cwd=WORKDIR, mode="acceptEdits", enable_mcp=False, allowed_tools=["Bash(python:*)"])
        prompt = "the cart total test is failing, fix it"
        later(1.2, lambda: pipe.send_text(prompt + "\r"), lambda: time.sleep(4.0), lambda: s.save("session.png"),
              lambda: pipe.send_text("/exit\r"))
        Repl(app, ui).run()
        app.shutdown()


def scene_permission() -> None:
    s = Screen(rows=24, below=8)
    ui = TerminalUI(s.console, animate=False)
    ui.prompter = Prompter()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=s.term):
        s.console.print(Text("● ", style=theme().tool) + Text("Edit", style="bold") + Text("(cart.py)", style=theme().dim))
        diff = ("--- a/cart.py\n+++ b/cart.py\n@@ -1,3 +1,3 @@\n def total(prices, tax=0.0):\n"
                "-    subtotal = sum(prices[1:])\n+    subtotal = sum(prices)\n     return round(subtotal * (1 + tax), 2)")
        later(1.0, lambda: s.save("permission.png"), lambda: pipe.send_text("\x1b"))
        ui.ask_permission(PermissionRequest("Edit", "Edit(cart.py)", diff, "edits a file", "Edit(**)", "write"))


def scene_logo() -> None:
    """The mascot, large, from the same pixel grid the terminal draws (muyah_code/ui/logo.py)."""
    from PIL import Image, ImageDraw
    from termshot import _font

    from muyah_code.ui.logo import PIXELS

    px, pad = 26, 30
    t = theme()
    a = tuple(int(t.accent[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(t.accent2[i:i + 2], 16) for i in (1, 3, 5))
    w, h = len(PIXELS[0]), len(PIXELS)
    font = _font(64)
    text_w = int(font.getlength("MUYAH-CODE"))
    img = Image.new("RGBA", (w * px + 2 * pad + 40 + text_w, h * px + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for y, row in enumerate(PIXELS):
        for x, cell in enumerate(row):
            if cell == "#":
                f = x / (w - 1)
                color = tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(3))
                d.rounded_rectangle([pad + x * px, pad + y * px, pad + (x + 1) * px - 2, pad + (y + 1) * px - 2],
                                    radius=4, fill=color)
    d.text((pad + w * px + 40, pad + h * px // 2), "MUYAH-CODE", font=font, fill=a, anchor="lm")
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / "logo.png")
    print("wrote", (OUT / "logo.png").relative_to(ROOT))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    set_theme("teal")
    scene_logo()
    make_project()
    scene_trust()
    scene_home_and_menu()
    scene_session()
    scene_permission()
