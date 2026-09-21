"""Render terminal output (ANSI text, or a live pyte screen) to a PNG, like Windows Terminal shows it.

Used by scripts/make_screenshots.py (README images) and scripts/record_cli_gif.py (the terminal demo), and
handy to eyeball UI changes. Images are drawn at `scale` times the on-screen size (2 = sharp on high-DPI
displays and when GitHub scales them down). Box-drawing characters are drawn as shapes, so borders join up
with no gaps between rows, as in a real terminal.

Needs: pyte, Pillow, and a monospace TTF (Cascadia Mono / Consolas / DejaVu Sans Mono).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyte
from PIL import Image, ImageDraw, ImageFont

BG = (13, 17, 23)
CHROME = (22, 27, 34)
BORDER = (48, 54, 61)
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
BOLD_CANDIDATES = [
    r"C:\Windows\Fonts\consolab.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]
SYMBOL_FONTS = [r"C:\Windows\Fonts\seguisym.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Apple Symbols.ttf"]
# Glyphs most monospace fonts lack; real terminals draw them from a fallback font, so do the same.
SYMBOLS = set("▣⇄⎿✔✓⏵⏸⚠✻✗…↑↓○▶◆●◼□±❯›⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
BOX = set("─━│╭╮╰╯┌┐└┘├┤")


def _font(size: int, candidates=FONT_CANDIDATES, bold: bool = False) -> ImageFont.FreeTypeFont:
    for f in candidates:
        if Path(f).exists():
            font = ImageFont.truetype(f, size)
            if bold:
                try:
                    font.set_variation_by_name("Bold")    # Cascadia Mono is a variable font
                except (OSError, ValueError, AttributeError):
                    pass
            return font
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


def _box(d: ImageDraw.ImageDraw, ch: str, x: int, y: int, cw: int, lh: int, fill, width: int) -> None:
    """Box-drawing characters as lines through the cell's centre, so neighbours always meet."""
    cx, cy = x + cw // 2, y + lh // 2
    w = width * 2 if ch == "━" else width
    left, right, up, down = {
        "─": (1, 1, 0, 0), "━": (1, 1, 0, 0), "│": (0, 0, 1, 1), "╭": (0, 1, 0, 1), "┌": (0, 1, 0, 1),
        "╮": (1, 0, 0, 1), "┐": (1, 0, 0, 1), "╰": (0, 1, 1, 0), "└": (0, 1, 1, 0), "╯": (1, 0, 1, 0),
        "┘": (1, 0, 1, 0), "├": (0, 1, 1, 1), "┤": (1, 0, 1, 1),
    }[ch]
    half = w // 2
    if ch in "╭╮╰╯":     # rounded corners: a quarter arc, then straight to the cell edges
        r = min(cw, lh) // 2
        box = {"╭": (cx, cy, cx + 2 * r, cy + 2 * r, 180, 270), "╮": (cx - 2 * r, cy, cx, cy + 2 * r, 270, 360),
               "╰": (cx, cy - 2 * r, cx + 2 * r, cy, 90, 180), "╯": (cx - 2 * r, cy - 2 * r, cx, cy, 0, 90)}[ch]
        d.arc(box[:4], box[4], box[5], fill=fill, width=w)
        if right:
            d.rectangle([cx + r, cy - half, x + cw, cy - half + w - 1], fill=fill)
        if left:
            d.rectangle([x, cy - half, cx - r, cy - half + w - 1], fill=fill)
        if down:
            d.rectangle([cx - half, cy + r, cx - half + w - 1, y + lh], fill=fill)
        if up:
            d.rectangle([cx - half, y, cx - half + w - 1, cy - r], fill=fill)
        return
    if left:
        d.rectangle([x, cy - half, cx, cy - half + w - 1], fill=fill)
    if right:
        d.rectangle([cx, cy - half, x + cw, cy - half + w - 1], fill=fill)
    if up:
        d.rectangle([cx - half, y, cx - half + w - 1, cy], fill=fill)
    if down:
        d.rectangle([cx - half, cy, cx - half + w - 1, y + lh], fill=fill)


def render_screen(screen: pyte.Screen, path: Path | None = None, title: str = "", trim: bool = True,
                  scale: float = 2.0, rows: int | None = None,
                  cursor: tuple[int, int] | None = None) -> Image.Image:
    """Draw a pyte screen as a terminal window. rows: how many rows to show (default: up to the last used).
    cursor: (column, row) to draw a bar cursor at, as a terminal does while you type."""
    cols = screen.columns
    size = int(round(19 * scale))
    font, bold_font = _font(size), _font(size, BOLD_CANDIDATES + FONT_CANDIDATES, bold=True)
    symbol_font = _font(size - int(2 * scale), SYMBOL_FONTS + FONT_CANDIDATES)
    cw = int(round(font.getlength("M")))
    lh = int(round(size * 1.3))
    last = rows or screen.lines
    if trim and rows is None:
        while last > 1 and not screen.display[last - 1].strip():
            last -= 1
    pad, bar, radius = int(26 * scale), int(40 * scale) if title else 0, int(12 * scale)
    width, height = cols * cw + 2 * pad, last * lh + 2 * pad + bar
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, width - 1, height - 1], radius=radius, fill=BG + (255,), outline=BORDER,
                        width=max(1, int(scale)))
    if title:  # window chrome
        d.rounded_rectangle([0, 0, width - 1, bar + radius], radius=radius, fill=CHROME, outline=BORDER,
                            width=max(1, int(scale)))
        d.rectangle([max(1, int(scale)), bar, width - 1 - max(1, int(scale)), bar + radius], fill=BG)
        d.line([0, bar, width, bar], fill=BORDER, width=max(1, int(scale)))
        dot = int(12 * scale)
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            x0 = int(18 * scale) + i * int(22 * scale)
            d.ellipse([x0, (bar - dot) // 2, x0 + dot, (bar - dot) // 2 + dot], fill=c)
        tfont = _font(int(15 * scale))
        d.text((width // 2, bar // 2), title, font=tfont, fill=(139, 148, 158), anchor="mm")
    line_w = max(1, int(round(1.2 * scale)))
    for y in range(last):
        row = screen.buffer[y]
        for x in range(cols):
            cell = row[x]
            fg = _color(cell.fg, FG)
            bg = _color(cell.bg, BG)
            if cell.reverse:
                fg, bg = bg, fg
            px, py = pad + x * cw, pad + bar + y * lh
            if bg != BG:
                d.rectangle([px, py, px + cw, py + lh], fill=bg)
            data = cell.data
            if not data.strip():
                continue
            if data in "█▀▄":  # blocks as solid shapes, so the mascot has no font gaps
                top = py if data in "█▀" else py + lh // 2
                bottom = py + lh if data in "█▄" else py + lh // 2
                d.rectangle([px, top, px + cw - 1, bottom - 1], fill=fg)
            elif data in BOX:
                _box(d, data, px, py, cw, lh, fg, line_w)
            elif data in SYMBOLS:
                d.text((px + cw // 2, py + lh // 2), data, font=symbol_font, fill=fg, anchor="mm")
            else:
                d.text((px, py + (lh - size) // 2), data, font=bold_font if cell.bold else font, fill=fg)
    if cursor is not None and 0 <= cursor[1] < last:
        cx, cy = pad + cursor[0] * cw, pad + bar + cursor[1] * lh
        d.rectangle([cx, cy + int(3 * scale), cx + max(2, int(2 * scale)) - 1, cy + lh - int(3 * scale)], fill=FG)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, optimize=True)
    return img


def render(ansi: str, path: Path, cols: int = 100, rows: int = 30, title: str = "", trim: bool = True,
           scale: float = 2.0) -> Path:
    screen = pyte.Screen(cols, rows)
    screen.set_mode(pyte.modes.LNM)
    pyte.Stream(screen).feed(ansi)
    render_screen(screen, path, title=title, trim=trim, scale=scale)
    return path


if __name__ == "__main__":
    render(sys.stdin.read(), Path(sys.argv[1]), title=sys.argv[2] if len(sys.argv) > 2 else "")
