"""Token usage across sessions, and the rate limits your provider reports.

Every model call appends one line to ~/.muyah/usage.jsonl:
    {"ts": 1758460000.0, "model": "...", "provider": "...", "in": 1234, "out": 56, "agent": "main"}

Most OpenAI-compatible providers (Groq, OpenRouter, OpenAI, Together, Fireworks, Cerebras...) and Anthropic
send their current limits as response headers: how many requests / tokens are left and when they reset.
The client keeps the last ones it saw; `/usage` shows them. Some (Gemini) send none: then a 429 message
is the only signal, and it says when to retry.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from muyah_code.appendlog import append_line, flush

USAGE_FILE = "usage.jsonl"


def record(home: Path, model: str, provider: str, prompt_tokens: int, completion_tokens: int,
           agent: str = "main") -> None:
    append_line(home / USAGE_FILE, json.dumps({
        "ts": round(time.time(), 3), "model": model, "provider": provider, "in": int(prompt_tokens),
        "out": int(completion_tokens), "agent": agent}))


@dataclass
class Totals:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict[str, list[int]] = field(default_factory=dict)   # model -> [requests, in, out]

    def add(self, model: str, tokens_in: int, tokens_out: int) -> None:
        self.requests += 1
        self.prompt_tokens += tokens_in
        self.completion_tokens += tokens_out
        row = self.by_model.setdefault(model or "?", [0, 0, 0])
        row[0] += 1
        row[1] += tokens_in
        row[2] += tokens_out


def summarize(home: Path, now: float | None = None) -> dict[str, Totals]:
    """Totals for today (local time) and the last 7 days, from the usage log."""
    path = home / USAGE_FILE
    flush(path)
    now = now or time.time()
    midnight = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    periods = {"today": (midnight, Totals()), "7 days": (now - 7 * 86400, Totals())}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    ts = float(r["ts"])
                except (ValueError, KeyError, TypeError):
                    continue
                for since, totals in periods.values():
                    if ts >= since:
                        totals.add(r.get("model", "?"), int(r.get("in") or 0), int(r.get("out") or 0))
    except OSError:
        pass
    return {name: totals for name, (_, totals) in periods.items()}


# ---------------------------------------------------------------------------------------- rate limits

@dataclass
class Limit:
    name: str                  # "requests", "tokens", "input-tokens", ...
    limit: str | None = None
    remaining: str | None = None
    reset: str | None = None   # as the provider sent it (a duration or a timestamp)


_OPENAI = re.compile(r"^x-ratelimit-(limit|remaining|reset)-(.+)$")            # x-ratelimit-remaining-requests
_ANTHROPIC = re.compile(r"^anthropic-ratelimit-(.+)-(limit|remaining|reset)$")  # anthropic-ratelimit-tokens-limit
_PLAIN = re.compile(r"^x-ratelimit-(limit|remaining|reset)$")                   # OpenRouter-style


def parse_limits(headers: dict | None) -> list[Limit]:
    """Rate-limit headers -> one row per limited resource (requests, tokens...)."""
    rows: dict[str, Limit] = {}
    for key, value in (headers or {}).items():
        k = key.lower()
        m = _OPENAI.match(k)
        if m:
            field_, name = m.group(1), m.group(2)
        elif (m := _ANTHROPIC.match(k)):
            name, field_ = m.group(1), m.group(2)
        elif (m := _PLAIN.match(k)):
            field_, name = m.group(1), "requests"
        else:
            continue
        setattr(rows.setdefault(name, Limit(name)), field_, str(value))
    return list(rows.values())


def describe_reset(value: str | None, now: float | None = None) -> str:
    """'6m0s' / '12.5s' / '2026-09-21T15:00:00Z' / '1758460000000' (ms epoch) -> 'in 12s'."""
    if not value:
        return ""
    now = now or time.time()
    v = value.strip()
    if re.fullmatch(r"(\d+(\.\d+)?(ms|h|m|s))+", v):
        seconds = 0.0
        for amount, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", v):
            seconds += float(amount) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
        return f"in {_duration(seconds)}"
    if re.fullmatch(r"\d{10,13}(\.\d+)?", v):          # epoch seconds or milliseconds
        ts = float(v) / (1000 if len(v.split(".")[0]) == 13 else 1)
        return f"in {_duration(ts - now)}"
    try:
        ts = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return f"in {_duration(ts.timestamp() - now)}"
    except ValueError:
        return v


def _duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s" if seconds >= 10 else f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h {int(seconds % 3600 // 60):02d}m"


def compact(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e4 else f"{n:,}"
