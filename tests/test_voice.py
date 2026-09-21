"""The mic key opens Windows voice typing, after Ctrl is released, without blocking the prompt."""

import threading

from muyah_code.ui import voice


def test_voice_typing_is_started_in_the_background(monkeypatch):
    started = threading.Event()
    monkeypatch.setattr(voice, "windows_dictation_available", lambda: True)
    monkeypatch.setattr(voice, "_send_win_h", lambda: started.set() or True)
    ok, message = voice.start_dictation()
    assert ok and "voice typing" in message
    assert started.wait(2)                         # sent from its own thread (it waits for Ctrl to be let go)


def test_without_windows_voice_typing_it_says_so(monkeypatch):
    monkeypatch.setattr(voice, "windows_dictation_available", lambda: False)
    ok, message = voice.start_dictation()
    assert not ok and "Win+H" in message
