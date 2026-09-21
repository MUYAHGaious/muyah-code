"""Color themes. Pick with `"theme": "<name>"` in settings or /theme <name>."""

from __future__ import annotations

import weakref
from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    accent: str       # brand color: bullets, prompt, panels
    accent2: str      # gradient partner for the banner
    ok: str
    err: str
    warn: str
    dim: str
    tool: str         # tool-call bullet
    code_theme: str   # pygments theme for code blocks
    add_bg: str       # diff backgrounds
    del_bg: str
    user_bg: str = "#23302f"   # band behind your sent prompts


THEMES = {
    # light teal palette (default): soft aqua accents, mint gradient, cool greys
    "teal": Theme("teal", "#6fd6c9", "#b8f0e6", "#8ee3b3", "#f28b82", "#f6d186", "#8aa6a3", "#7cc7e8",
                  "nord", "#123a33", "#3a1f22"),
    "muyah": Theme("muyah", "#e07b53", "#f2c14e", "#7ec27e", "#e5534b", "#e0af68", "grey58", "#7aa2f7",
                   "monokai", "#16361f", "#3d1a1a"),
    "ocean": Theme("ocean", "#4fb3d9", "#8be9fd", "#50fa7b", "#ff5555", "#f1fa8c", "grey58", "#bd93f9",
                   "dracula", "#123326", "#3a1616"),
    "forest": Theme("forest", "#8fbf5a", "#d4c86a", "#8fbf5a", "#d9534f", "#e6b450", "grey58", "#6fb3b8",
                    "gruvbox-dark", "#1f3318", "#3b1c16"),
    "mono": Theme("mono", "white", "grey70", "white", "bold white", "white", "grey50", "white",
                  "bw", "grey15", "grey11", "grey19"),
    "light": Theme("light", "#b3541e", "#c28f00", "#2e7d32", "#c62828", "#a15c00", "grey42", "#1565c0",
                   "friendly", "#e3f5e3", "#fbe3e3", "#ececec"),
}

DEFAULT_THEME = "teal"
_current = THEMES[DEFAULT_THEME]


def theme() -> Theme:
    return _current


def set_theme(name: str) -> Theme:
    global _current
    if name not in THEMES:
        raise ValueError(f"Unknown theme '{name}'. Available: {', '.join(THEMES)}")
    _current = THEMES[name]
    for console in list(_calm_consoles):
        console.pop_theme()
        console.push_theme(markdown_styles())
    return _current


_calm_consoles: weakref.WeakSet = weakref.WeakSet()   # consoles to restyle on /theme


def markdown_styles():
    """Answers in plain white: Rich's defaults color list numbers cyan and headings magenta, which made a
    reply look like a rainbow. Only inline code keeps the accent; links stay recognizable but quiet."""
    from rich.theme import Theme as RichTheme

    t = _current
    return RichTheme({
        "markdown.h1": "bold", "markdown.h2": "bold", "markdown.h3": "bold", "markdown.h4": "bold",
        "markdown.h5": "bold", "markdown.h6": "bold", "markdown.h7": "bold",
        "markdown.list": "none", "markdown.item.number": "none", "markdown.item.bullet": "none",
        "markdown.block_quote": f"italic {t.dim}", "markdown.hr": t.dim,
        "markdown.code": f"bold {t.accent}", "markdown.code_block": "none",
        "markdown.link": "underline", "markdown.link_url": f"underline {t.dim}",
        "markdown.table.border": t.dim, "markdown.table.header": "bold", "markdown.kbd": "bold",
    })


def calm_markdown(console):
    """Use the plain answer styles on this console (kept in step with /theme)."""
    console.push_theme(markdown_styles())
    _calm_consoles.add(console)
    return console
