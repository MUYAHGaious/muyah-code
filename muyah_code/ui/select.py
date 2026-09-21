"""Arrow-key menus and text questions, shared by the REPL, `muyah login` and permission prompts.

    ↑/↓ to move · Enter to choose · Esc / Ctrl+C to cancel (returns None)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import choice
from prompt_toolkit.styles import Style

from muyah_code.ui.theme import theme


def _style() -> Style:
    return Style.from_dict({"selected-option": f"bold {theme().accent}", "bottom-toolbar": "noreverse"})


def _escape_cancels() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("escape", eager=True)
    def _(event):
        event.app.exit(exception=KeyboardInterrupt(), style="class:aborting")

    return kb


class Prompter:
    """Interactive questions. Tests substitute an object with the same two methods."""

    def select(self, message: str, options: Sequence[tuple[Any, str]], default: Any = None) -> Any | None:
        if not options:
            return None
        values = [v for v, _ in options]
        try:
            return choice(message=[("bold", message)], options=list(options),
                          default=default if default in values else values[0], style=_style(),
                          symbol="❯", key_bindings=_escape_cancels(),
                          bottom_toolbar=[("fg:ansibrightblack", " ↑/↓ move · Enter choose · Esc cancel")])
        except (KeyboardInterrupt, EOFError):
            return None

    def ask(self, message: str, password: bool = False, default: str = "") -> str:
        try:
            answer = PromptSession().prompt([(f"fg:{theme().accent}", "? "), ("", message)], is_password=password,
                                            default=default)
        except (KeyboardInterrupt, EOFError):
            return ""
        return answer

    def confirm(self, message: str, default: bool = True) -> bool:
        picked = self.select(message, [(True, "Yes"), (False, "No")], default=default)
        return bool(picked)
