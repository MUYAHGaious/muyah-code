"""Talk instead of typing: one key (F2, or Ctrl+Space) starts listening, the words land in your prompt.

Engines ("voice": {"engine": ...}):
  windows  Windows voice typing (the same as pressing Win+H yourself): free, no key, types straight into the
           focused input. The default on Windows. The shortcut is sent only after you let go of Ctrl/Shift/
           Alt (pressed while Ctrl+Space is held, Windows would see Ctrl+Win+H, which does nothing).
  whisper  record from the microphone, then transcribe with a Whisper API: Groq (whisper-large-v3-turbo, with
           the key you saved via /provider groq), OpenAI, or any OpenAI-compatible /audio/transcriptions URL
           ("voice": {"url": ..., "model": ..., "api_key": ...}). The default elsewhere when a key exists.
  local    record, then transcribe on this machine with faster-whisper (pip install "muyah-code[voice-local]").

whisper and local need the microphone extra: pip install "muyah-code[voice]". Press the key again to stop,
or just stop talking: recording ends after SILENCE_STOP seconds of quiet. The text is put in your prompt to
edit; it is never sent on its own.
"""

from __future__ import annotations

import io
import sys
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path

VK_LWIN = 0x5B
VK_H = 0x48
MODIFIERS = (0x11, 0x10, 0x12)       # Ctrl, Shift, Alt
KEYEVENTF_KEYUP = 0x0002
INPUT_KEYBOARD = 1
RELEASE_WAIT = 2.0                   # seconds to wait for you to let go of Ctrl+Space

RATE = 16000                         # what Whisper wants: 16 kHz mono
SILENCE_STOP = 2.0                   # stop after this long of quiet, once you have spoken
MAX_SECONDS = 120
SPEECH_LEVEL = 500                   # RMS of int16 samples that counts as speech
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
OPENAI_URL = "https://api.openai.com/v1/audio/transcriptions"


# ---------------------------------------------------------------------------- Windows voice typing

def windows_dictation_available() -> bool:
    if sys.platform != "win32":
        return False
    return sys.getwindowsversion().major >= 10


def _send_win_h() -> bool:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    ULONG_PTR = ctypes.c_size_t

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class MOUSEINPUT(ctypes.Structure):     # the union's largest member: INPUT must have the right size
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    # wait until Ctrl/Shift/Alt are up: Ctrl+Win+H is not the voice typing shortcut
    end = time.monotonic() + RELEASE_WAIT
    while time.monotonic() < end and any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in MODIFIERS):
        time.sleep(0.02)

    def key(vk: int, up: bool) -> INPUT:
        return INPUT(INPUT_KEYBOARD, _U(ki=KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP if up else 0, 0, 0)))

    seq = (INPUT * 4)(key(VK_LWIN, False), key(VK_H, False), key(VK_H, True), key(VK_LWIN, True))
    sent = user32.SendInput(4, seq, ctypes.sizeof(INPUT))
    return sent == 4


def start_windows_dictation() -> bool:
    """Press Win+H for the user (once Ctrl is released): voice typing opens over the terminal."""
    if not windows_dictation_available():
        return False
    threading.Thread(target=_send_win_h, name="muyah-voice", daemon=True).start()
    return True


def start_dictation() -> tuple[bool, str]:
    """Windows voice typing. Returns (started, what to tell the user)."""
    if start_windows_dictation():
        return True, ("Listening (Windows voice typing): speak, and your words appear in the prompt. If nothing "
                      "opens, turn on voice typing in Windows Settings > Time & language > Speech.")
    return False, ("Voice input here uses Windows voice typing (Win+H), which this system does not have. "
                   "On macOS use the system dictation shortcut (press Fn twice or the mic key).")


# ---------------------------------------------------------------------------- recording

class Recorder:
    """The microphone, 16 kHz mono, until stop() or SILENCE_STOP seconds of quiet after speech."""

    def __init__(self, on_auto_stop: Callable[[], None] | None = None, stream_factory=None):
        self.frames: list[bytes] = []
        self.on_auto_stop = on_auto_stop
        self.started = time.monotonic()
        self._spoke = False
        self._quiet_since: float | None = None
        self._stopped = False
        self._stream_factory = stream_factory
        self._stream = None

    def start(self) -> None:
        factory = self._stream_factory
        if factory is None:
            try:
                import sounddevice
            except ImportError as e:
                raise RuntimeError('the microphone needs the voice extra: pip install "muyah-code[voice]"') from e
            factory = sounddevice.RawInputStream
        self._stream = factory(samplerate=RATE, channels=1, dtype="int16", callback=self._callback)
        self._stream.start()

    def _callback(self, data, frames, time_info, status) -> None:
        chunk = bytes(data)
        self.frames.append(chunk)
        now = time.monotonic()
        if _rms(chunk) >= SPEECH_LEVEL:
            self._spoke, self._quiet_since = True, None
        elif self._spoke and self._quiet_since is None:
            self._quiet_since = now
        long_quiet = self._spoke and self._quiet_since is not None and now - self._quiet_since >= SILENCE_STOP
        if (long_quiet or now - self.started >= MAX_SECONDS) and not self._stopped and self.on_auto_stop:
            threading.Thread(target=self.on_auto_stop, daemon=True).start()

    def stop(self) -> bytes:
        """End recording; the audio as a WAV file."""
        self._stopped = True
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(b"".join(self.frames))
        return buf.getvalue()

    @property
    def seconds(self) -> float:
        return time.monotonic() - self.started


def _rms(chunk: bytes) -> float:
    import array

    samples = array.array("h", chunk[: len(chunk) - len(chunk) % 2])
    if not samples:
        return 0.0
    return (sum(s * s for s in samples[::4]) / max(1, len(samples[::4]))) ** 0.5


# ---------------------------------------------------------------------------- transcription

def whisper_endpoint(cfg, home: Path) -> tuple[str, str, str] | None:
    """(url, api key, model) for transcription: your voice.url, else Groq, else OpenAI (with saved keys)."""
    from muyah_code.providers import resolve_key

    if cfg.get("voice.url"):
        return str(cfg.get("voice.url")), str(cfg.get("voice.api_key") or "none"), \
            str(cfg.get("voice.model") or "whisper-1")
    groq = resolve_key(home, "groq")
    if groq:
        return GROQ_URL, groq, str(cfg.get("voice.model") or "whisper-large-v3-turbo")
    openai = resolve_key(home, "openai")
    if openai:
        return OPENAI_URL, openai, str(cfg.get("voice.model") or "gpt-4o-mini-transcribe")
    return None


def transcribe_api(wav: bytes, url: str, key: str, model: str, language: str | None = None) -> str:
    import httpx

    data = {"model": model, "response_format": "json"}
    if language:
        data["language"] = language
    resp = httpx.post(url, headers={"Authorization": f"Bearer {key}"}, data=data,
                      files={"file": ("speech.wav", wav, "audio/wav")}, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"transcription failed (HTTP {resp.status_code}): {resp.text[:200]}")
    return str(resp.json().get("text") or "").strip()


def transcribe_local(wav: bytes, model: str = "base") -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError('offline transcription needs: pip install "muyah-code[voice-local]"') from e
    segments, _ = WhisperModel(model, device="cpu", compute_type="int8").transcribe(io.BytesIO(wav))
    return " ".join(s.text.strip() for s in segments).strip()


# ---------------------------------------------------------------------------- the mic key

class VoiceInput:
    """What the mic key does. toggle() starts or stops; the text arrives through on_text."""

    def __init__(self, cfg, home: Path, recorder_factory=None, transcriber=None):
        self.cfg = cfg
        self.home = home
        self.recorder_factory = recorder_factory or Recorder
        self.transcriber = transcriber
        self.recorder: Recorder | None = None
        self.state = ""                 # "" | "recording" | "transcribing" (shown in the status line)
        self._on_text: Callable[[str], None] | None = None
        self._on_note: Callable[[str], None] | None = None
        self._lock = threading.Lock()

    def engine(self) -> str:
        chosen = str(self.cfg.get("voice.engine", "auto") or "auto").lower()
        if chosen != "auto":
            return chosen
        if windows_dictation_available():
            return "windows"
        return "whisper"

    def toggle(self, on_text: Callable[[str], None], on_note: Callable[[str], None]) -> str:
        """Returns a line to show right away."""
        engine = self.engine()
        if engine == "windows":
            started, message = start_dictation()
            return "" if started else message
        with self._lock:
            if self.recorder is not None:
                self._finish()
                return "Transcribing…"
            if engine == "whisper" and whisper_endpoint(self.cfg, self.home) is None and not self.cfg.get("voice.url"):
                return ("Voice needs a transcription service: save a Groq key with /provider groq (free tier), or an "
                        'OpenAI key, or set "voice": {"engine": "local"} to transcribe on this machine.')
            self._on_text, self._on_note = on_text, on_note
            recorder = self.recorder_factory(on_auto_stop=self._auto_stop)
            try:
                recorder.start()
            except Exception as e:      # no microphone, no permission, no extra installed: say which
                return f"Cannot record: {e}"
            self.recorder, self.state = recorder, "recording"
        return "● Listening. Press the mic key again (or just stop talking) to finish."

    def _auto_stop(self) -> None:
        with self._lock:
            if self.recorder is not None:
                self._finish()

    def _finish(self) -> None:
        recorder, self.recorder = self.recorder, None
        wav = recorder.stop()
        self.state = "transcribing"
        threading.Thread(target=self._transcribe, args=(wav,), name="muyah-transcribe", daemon=True).start()

    def _transcribe(self, wav: bytes) -> None:
        try:
            if self.transcriber is not None:
                text = self.transcriber(wav)
            elif self.engine() == "local":
                text = transcribe_local(wav, str(self.cfg.get("voice.model") or "base"))
            else:
                url, key, model = whisper_endpoint(self.cfg, self.home)
                text = transcribe_api(wav, url, key, model, self.cfg.get("voice.language"))
        except Exception as e:          # a failed transcription is reported, never raised into the prompt
            self.state = ""
            if self._on_note:
                self._on_note(f"Voice: {e}")
            return
        self.state = ""
        if text and self._on_text:
            self._on_text(text)
        elif self._on_note:
            self._on_note("Voice: nothing was heard.")
