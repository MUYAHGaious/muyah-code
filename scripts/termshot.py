"""Render terminal output (ANSI text) to a PNG screenshot, like Windows Terminal would show it.

Used by scripts/make_screenshots.py for the README images, and handy to eyeball UI changes.
Needs: pyte, Pillow, and a monospace TTF (Cascadia Mono / Consolas / DejaVu Sans Mono).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyte
from PIL import Image, ImageDraw, ImageFont

BG = (13, 17, 23)
FG = (201, 209, 217)
PALETTE = {
    "black": (13, 17, 23), "red": (242, 139, 130), "green": (142, 227, 179), "brown": (246, 209, 134),
    "yellow": (246, 209, 134), "blue": (124, 199, 232), "magenta": (199, 146, 234), "cyan": (111, 214, 201),
    "white": (201, 209, 217), "brightblack": (125, 133, 144), "brightred": (255, 161, 152),
    "brightgreen": (166, 240, 198), "brightyellow": (250, 222, 160), "brightblue": (150, 210, 240),
    "brightmagenta": (215, 170, 245), "brightcyan": (140, 230, 215), "brightwhite": (240, 246, 252),
}
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\CascadiaMono.ttf", r"C:\Windows\Fonts\consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/System/Library/Fonts/Menlo.ttc",
]


SYMBOL_FONTS = [r"C:\Windows\Fonts\seguisym.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Apple Symbols.ttf"]
# Glyphs most monospace fonts lack; real terminals draw them from a fallback font, so do the same.
SYMBOLS = set("⎿✔✓⏵⏸⚠✻✗…↑↓○▶◆")


def _font(size: int, candidates=FONT_CANDIDATES) -> ImageFont.FreeTypeFont:
    for f in candidates:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def _color(value: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value == "default":
        return default
    if value in PALETTE:
        return PALETTE[value]
    if len(value) == 6:
        try:
            return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
        except ValueError:
            pass
    return default


def render(ansi: str, path: Path, cols: int = 100, rows: int = 30, title: str = "", trim: bool = True,
           size: int = 20) -> Path:
    screen = pyte.Screen(cols, rows)
    screen.set_mode(pyte.modes.LNM)
    pyte.Stream(screen).feed(ansi)
    font = _font(size)
    symbol_font = _font(size - 2, SYMBOL_FONTS + FONT_CANDIDATES)
    cw = int(font.getlength("M"))
    ch = int(size * 1.32)
    last = rows
    if trim:
        while last > 1 and not screen.display[last - 1].strip():
            last -= 1
    pad, bar = 22, 38 if title else 0
    img = Image.new("RGB", (cols * cw + 2 * pad, last * ch + 2 * pad + bar), BG)
    d = ImageDraw.Draw(img)
    if title:  # window chrome
        d.rectangle([0, 0, img.width, bar], fill=(22, 27, 34))
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            d.ellipse([16 + i * 22, 13, 28 + i * 22, 25], fill=c)
        d.text((img.width // 2 - font.getlength(title) // 2, 8), title, font=font, fill=(139, 148, 158))
    for y in range(last):
        for x in range(cols):
            cell = screen.buffer[y][x]
            fg = _color(cell.fg, FG)
            bg = _color(cell.bg, BG)
            if cell.reverse:
                fg, bg = bg, fg
            px, py = pad + x * cw, pad + bar + y * ch
            if bg != BG:
                d.rectangle([px, py, px + cw, py + ch], fill=bg)
            if cell.data.strip():
                if cell.data in "█▀▄":  # draw blocks as solid shapes so the mascot has no font gaps
                    top = py if cell.data in "█▀" else py + ch // 2
                    bottom = py + ch if cell.data in "█▄" else py + ch // 2
                    d.rectangle([px, top, px + cw - 1, bottom - 1], fill=fg)
                elif cell.data in SYMBOLS:
                    d.text((px, py + 1), cell.data, font=symbol_font, fill=fg)
                else:
                    d.text((px, py), cell.data, font=font, fill=fg)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


if __name__ == "__main__":
    render(sys.stdin.read(), Path(sys.argv[1]), title=sys.argv[2] if len(sys.argv) > 2 else "")
