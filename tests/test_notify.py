"""Notifications when a turn ends or needs you, only while you are in another window."""

import io

from rich.console import Console

from muyah_code.notify import Notifier, auto_mode, sequence
from muyah_code.ui.terminal import TerminalUI


class Cfg(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class Hooks:
    def __init__(self):
        self.runs = []

    def has(self, event):
        return event == "Notification"

    def run(self, event, payload):
        self.runs.append((event, payload))


def test_auto_picks_what_the_terminal_understands():
    assert auto_mode({"TERM_PROGRAM": "iTerm.app"}, "darwin") == "osc9"
    assert auto_mode({"TERM_PROGRAM": "WezTerm"}, "linux") == "osc9"
    assert auto_mode({"KITTY_WINDOW_ID": "1"}, "linux") == "osc99"
    assert auto_mode({"VTE_VERSION": "7200"}, "linux") == "osc777"
    assert auto_mode({"WT_SESSION": "x"}, "win32") == "native"
    assert auto_mode({}, "linux") == "bell"


def test_sequences_are_safe():
    assert sequence("osc9", "MUYAH-CODE", "Done") == "\x1b]9;MUYAH-CODE: Done\x07"
    assert sequence("osc777", "T", "a;b") == "\x1b]777;notify;T;a,b\x07"        # ; would split the fields
    assert "\x07" not in sequence("osc9", "T", "evil\x07\x1b[2J")[:-1]
    assert sequence("off", "T", "x") == ""


def test_only_when_unfocused_or_after_a_long_wait():
    out, native = [], []
    n = Notifier(Cfg(notify="osc9"), out.append, hooks=Hooks(), native=lambda t, m: native.append(m))
    assert not n.should(True, 300) and n.should(False, 1)
    assert not n.should(None, 5) and n.should(None, 25)       # unknown focus: only long turns
    n.notify("done", "Done in 30s")
    assert out == ["\x1b]9;MUYAH-CODE: Done in 30s\x07"] and n.hooks.runs[0][1]["kind"] == "done"
    w = Notifier(Cfg(notify="native"), out.append, native=lambda t, m: native.append(m))
    w.notify("attention", "Waiting for your approval")
    assert native == ["Waiting for your approval"] and out[-1] == "\x07"
    off = Notifier(Cfg(notify="off"), out.append)
    assert not off.should(False, 999)


def test_focus_events_from_the_key_reader_are_tracked_not_typed():
    ui = TerminalUI(Console(file=io.StringIO(), width=100))
    ui._on_key("focus-out")
    assert ui.focused is False and ui._draft == ""
    ui._on_key("focus-in")
    assert ui.focused is True


def test_approvals_ask_for_attention():
    from muyah_code.ui.base import PermissionRequest

    ui = TerminalUI(Console(file=io.StringIO(), width=100))
    seen = []
    ui.on_attention = seen.append
    ui.console.input = lambda *a, **k: "n"      # answer the menu
    ui.ask_permission(PermissionRequest("Bash", "Bash(npm test)", "npm test", "", "", "exec"))
    assert seen == ["Waiting for your approval: Bash(npm test)"]
