"""The MUYAH mascot: an M-shaped creature with two eyes, drawn with half-block characters.

It is defined as a pixel grid (two pixel rows per terminal row) so it stays crisp in any terminal font,
and colored with the theme gradient. `play_intro` shows a short shimmer + blink (< 1 s) on real terminals.
"""

from __future__ import annotations

import time

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from muyah_code.ui.theme import theme

# '#' = filled pixel. Rows 3 hold the eyes (the two '.' holes).
PIXELS = [
    "##......##",
    "###....###",
    "####..####",
    "#.######.#",
    "##########",
    "##.#..#.##",
]
EYE_ROW = 3


def _hex(color: str) -> tuple[int, int, int] | None:
    c = color.lstrip("#")
    if len(c) != 6:
        return None
    try:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except ValueError:
        return None


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], f: float) -> str:
    f = max(0.0, min(1.0, f))
    r, g, bl = (int(a[i] + (b[i] - a[i]) * f) for i in range(3))
    return f"#{r:02x}{g:02x}{bl:02x}"


def mascot(shift: float = 0.0, blink: bool = False) -> list[Text]:
    """Three lines of the mascot. shift moves the gradient (0..1); blink closes the eyes."""
    grid = [list(row) for row in PIXELS]
    if blink:
        grid[EYE_ROW] = ["#"] * len(grid[EYE_ROW])
    t = theme()
    a, b = _hex(t.accent), _hex(t.accent2)
    width = len(grid[0])
    lines = []
    for r in range(0, len(grid), 2):
        line = Text()
        for x in range(width):
            top, bot = grid[r][x] == "#", grid[r + 1][x] == "#"
            ch = "█" if top and bot else "▀" if top else "▄" if bot else " "
            if a and b:
                pos = (x / (width - 1) + shift) % 2.0
                style = _mix(a, b, pos if pos <= 1 else 2 - pos)  # ping-pong across the gradient
            else:
                style = t.accent
            line.append(ch, style=f"bold {style}")
        lines.append(line)
    return lines


def header_renderable(lines: list[Text], shift: float = 0.0, blink: bool = False) -> Table:
    """Mascot on the left, up to three text lines on the right (like Claude Code's welcome)."""
    grid = Table.grid(padding=(0, 3))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True, overflow="ellipsis")
    art = mascot(shift, blink)
    for i in range(3):
        grid.add_row(art[i], lines[i] if i < len(lines) else Text(""))
    return grid


def play_intro(console: Console, lines: list[Text], animate: bool = True) -> None:
    """Show the header; on a real terminal, first run a short shimmer and one blink."""
    if animate and console.is_terminal:
        frames = 14
        try:
            with Live(header_renderable(lines), console=console, refresh_per_second=30, transient=True) as live:
                for i in range(frames):
                    live.update(header_renderable(lines, shift=i / frames * 2, blink=i in (9, 10)))
                    time.sleep(0.045)
        except Exception:
            pass  # never let a cosmetic animation break startup
    console.print(Group(header_renderable(lines)))
