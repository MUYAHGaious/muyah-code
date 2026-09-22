"""Arrow-key menus and text questions, shared by the REPL, `muyah login`, the trust check and permission prompts.

The menu is a small inline prompt_toolkit application that only takes the rows it needs (the stock
`choice()` dialog stretches to the bottom of the terminal and pushes the question off screen):

    Do you trust this folder?
      No, exit
    ❯ Yes, I trust this folder

    Enter to confirm · Esc to cancel
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

from muyah_code.ui.theme import theme

MAX_VISIBLE = 12


def _menu(message: str, options: Sequence[tuple[Any, str]], default: Any = None) -> Any | None:
    values = [v for v, _ in options]
    state = {"i": values.index(default) if default in values else 0, "top": 0}
    accent = theme().accent

    def visible_range() -> tuple[int, int]:
        n = len(options)
        if n <= MAX_VISIBLE:
            return 0, n
        i = state["i"]
        top = min(max(state["top"], i - MAX_VISIBLE + 1), i)
        state["top"] = top
        return top, top + MAX_VISIBLE

    def lines():
        if get_app().is_done:           # answered: keep one quiet line, not the whole menu and its hint
            return [("bold", message + "  "), ("fg:ansibrightblack", options[state["i"]][1])] if message else []
        out: list[tuple[str, str]] = [("bold", message + "\n")] if message else []
        start, end = visible_range()
        if start > 0:
            out.append(("fg:ansibrightblack", f"  ↑ {start} more\n"))
        for idx in range(start, end):
            label = options[idx][1]
            if idx == state["i"]:
                out += [(f"fg:{accent} bold", f"❯ {label}\n")]
            else:
                out += [("", f"  {label}\n")]
        if end < len(options):
            out.append(("fg:ansibrightblack", f"  ↓ {len(options) - end} more\n"))
        out.append(("fg:ansibrightblack", "\nEnter to confirm · Esc to cancel"))
        return out

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _(event):
        state["i"] = (state["i"] - 1) % len(options)

    @kb.add("down")
    @kb.add("j")
    @kb.add("tab")
    def _(event):
        state["i"] = (state["i"] + 1) % len(options)

    @kb.add("enter")
    def _(event):
        event.app.exit(result=values[state["i"]])

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _(event):
        event.app.exit(result=None)

    for n in range(1, min(9, len(options)) + 1):  # number keys still work as a shortcut
        @kb.add(str(n))
        def _(event, n=n):
            event.app.exit(result=values[n - 1])

    control = FormattedTextControl(lines, focusable=True, show_cursor=False)
    height = (1 if message else 0) + min(len(options), MAX_VISIBLE) + 2 + (2 if len(options) > MAX_VISIBLE else 0)

    def rows() -> Dimension:
        return Dimension.exact((1 if message else 0) if get_app().is_done else height)

    app = Application(layout=Layout(HSplit([Window(control, height=rows)])),
                      key_bindings=kb, full_screen=False, erase_when_done=False)
    return app.run()


OTHER = "__other__"


def question_menu(question: str, options: Sequence[tuple[str, str]], header: str = "", multi: bool = False,
                  step: str = "") -> list[str] | str | None:
    """A question like Claude Code asks it: a label chip, the question, numbered options each with a short
    explanation, and "Type something else" last. Returns the chosen label (a list when `multi`), OTHER when
    the user wants to type, or None on Esc."""
    labels = [label for label, _ in options] + [OTHER]
    state = {"i": 0, "checked": set()}
    accent = theme().accent

    def width() -> int:
        try:
            return max(30, get_app().output.get_size().columns - 1)
        except Exception:
            return 80

    def wrapped(text: str, indent: int) -> list[str]:
        """Wrap at word boundaries (the terminal would cut words in half)."""
        import textwrap

        return textwrap.wrap(text, max(20, width() - indent)) or [""]

    def lines():
        if get_app().is_done:        # answered: one quiet line
            return []
        out: list[tuple[str, str]] = []
        if header or step:
            out += [(f"fg:{accent} reverse", f" {header or 'Question'} "), ("fg:ansibrightblack", f"  {step}\n")]
        out.append(("bold", "\n".join(wrapped(question, 0)) + "\n\n"))
        for idx, label in enumerate(labels):
            here = idx == state["i"]
            pointer = "❯ " if here else "  "
            style = f"fg:{accent} bold" if here else ""
            if label == OTHER:
                out += [(style, f"{pointer}{idx + 1}. "), (style or "fg:ansibrightblack", "Type something else…\n")]
                continue
            box = ("[x] " if idx in state["checked"] else "[ ] ") if multi else ""
            out.append((style, f"{pointer}{idx + 1}. {box}{label}\n"))
            desc = options[idx][1]
            if desc:
                out += [("fg:ansibrightblack", f"      {part}\n") for part in wrapped(desc, 6)]
        keys = "Space to choose · Enter to confirm" if multi else "Enter to select"
        out.append(("fg:ansibrightblack", f"\n{keys} · ↑/↓ to move · Esc to skip"))
        return out

    kb = KeyBindings()

    @kb.add("up")
    def _(event):
        state["i"] = (state["i"] - 1) % len(labels)

    @kb.add("down")
    @kb.add("tab")
    def _(event):
        state["i"] = (state["i"] + 1) % len(labels)

    @kb.add("space")
    def _(event):
        if multi and labels[state["i"]] != OTHER:
            state["checked"] ^= {state["i"]}

    @kb.add("enter")
    def _(event):
        label = labels[state["i"]]
        if label == OTHER:
            event.app.exit(result=OTHER)
        elif multi:
            chosen = sorted(state["checked"]) or [state["i"]]
            event.app.exit(result=[labels[i] for i in chosen])
        else:
            event.app.exit(result=label)

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _(event):
        event.app.exit(result=None)

    for n in range(1, min(9, len(labels)) + 1):
        @kb.add(str(n))
        def _(event, n=n):
            state["i"] = n - 1
            if not multi:
                event.app.exit(result=labels[n - 1] if labels[n - 1] != OTHER else OTHER)

    control = FormattedTextControl(lines, focusable=True, show_cursor=False)

    def rows() -> Dimension:
        if get_app().is_done:
            return Dimension.exact(0)
        n = (1 if header or step else 0) + len(wrapped(question, 0)) + 1 + len(labels) \
            + sum(len(wrapped(d, 6)) for _, d in options if d) + 2
        return Dimension.exact(n)

    app = Application(layout=Layout(HSplit([Window(control, height=rows)])),
                      key_bindings=kb, full_screen=False, erase_when_done=False)
    return app.run()


class Prompter:
    """Interactive questions. Tests substitute an object with the same three methods."""

    def select(self, message: str, options: Sequence[tuple[Any, str]], default: Any = None) -> Any | None:
        if not options:
            return None
        try:
            return _menu(message, list(options), default)
        except (KeyboardInterrupt, EOFError):
            return None

    def ask(self, message: str, password: bool = False, default: str = "") -> str:
        try:
            answer = PromptSession().prompt([(f"fg:{theme().accent}", "? "), ("", message)], is_password=password,
                                            default=default)
        except (KeyboardInterrupt, EOFError):
            return ""
        return answer

    def question(self, question: str, options: Sequence[tuple[str, str]], header: str = "", multi: bool = False,
                 step: str = "") -> list[str] | str | None:
        try:
            return question_menu(question, list(options), header, multi, step)
        except (KeyboardInterrupt, EOFError):
            return None

    def confirm(self, message: str, default: bool = True) -> bool:
        picked = self.select(message, [(True, "Yes"), (False, "No")], default=default)
        return bool(picked)
