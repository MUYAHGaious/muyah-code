"""Token usage and cost across sessions, budgets, and the rate limits your provider reports.

Every model call (the agent's, sub-agents', compaction, learning, web page summaries...) appends one
line to ~/.muyah/usage.jsonl:
    {"ts": 1758460000.0, "model": "...", "provider": "...", "purpose": "main", "in": 1234, "out": 56,
     "cr": 1000, "cw": 0, "cost": 0.0031, "s": 2.4}
"in" counts every input token, cached ones included; "cr"/"cw" are the cached tokens read / written
(prompt caching makes those much cheaper); "cost" is dollars (null when the model is unpriced, see
pricing.py); "s" is how long the reply took.

Most OpenAI-compatible providers (Groq, OpenRouter, OpenAI, Together, Fireworks, Cerebras...) and Anthropic
send their current limits as response headers: how many requests / tokens are left and when they reset.
The client keeps the last ones it saw; `/usage` shows them. Some (Gemini) send none: then a 429 message
is the only signal, and it says when to retry.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from muyah_code.appendlog import append_line, flush

USAGE_FILE = "usage.jsonl"


def _int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def normalize(raw: dict | None) -> dict:
    """A provider's usage block -> {"in", "out", "cache_read", "cache_write"}, "in" counting all input.

    OpenAI / Groq / Gemini / xAI report cached tokens inside prompt_tokens (prompt_tokens_details.cached_tokens);
    DeepSeek as prompt_cache_hit_tokens; our Anthropic client already adds its cache reads/writes to
    prompt_tokens and passes cache_read / cache_write along."""
    raw = raw or {}
    details = raw.get("prompt_tokens_details") or {}
    cache_read = _int(raw.get("cache_read")) or _int(details.get("cached_tokens") if isinstance(details, dict) else 0) \
        or _int(raw.get("prompt_cache_hit_tokens"))
    return {"in": _int(raw.get("prompt_tokens")), "out": _int(raw.get("completion_tokens")),
            "cache_read": cache_read, "cache_write": _int(raw.get("cache_write"))}


@dataclass
class Call:
    """One model request."""
    model: str
    provider: str
    purpose: str            # main | subagent:<type> | compact | reflect | webfetch | btw | ...
    tokens_in: int
    tokens_out: int
    cache_read: int = 0
    cache_write: int = 0
    cost: float | None = None   # dollars; None = unpriced
    seconds: float = 0.0
    ts: float = field(default_factory=time.time)

    def line(self) -> str:
        return json.dumps({"ts": round(self.ts, 3), "model": self.model, "provider": self.provider,
                           "purpose": self.purpose, "in": self.tokens_in, "out": self.tokens_out,
                           "cr": self.cache_read, "cw": self.cache_write,
                           "cost": None if self.cost is None else round(self.cost, 6), "s": round(self.seconds, 2)})


def record_call(home: Path, call: Call) -> None:
    append_line(home / USAGE_FILE, call.line())


def purpose_group(purpose: str) -> str:
    """"subagent:explore" -> "sub-agent explore" for tables."""
    return purpose.replace("subagent:", "sub-agent ", 1) if purpose.startswith("subagent:") else purpose


class Ledger:
    """This session's calls, in memory (the file keeps every session's)."""

    def __init__(self):
        self.calls: list[Call] = []
        self._lock = threading.Lock()

    def add(self, call: Call) -> None:
        with self._lock:
            self.calls.append(call)

    def _snapshot(self) -> list[Call]:
        with self._lock:
            return list(self.calls)

    @property
    def requests(self) -> int:
        return len(self.calls)

    @property
    def tokens_in(self) -> int:
        return sum(c.tokens_in for c in self._snapshot())

    @property
    def tokens_out(self) -> int:
        return sum(c.tokens_out for c in self._snapshot())

    @property
    def cost(self) -> float:
        return sum(c.cost for c in self._snapshot() if c.cost is not None)

    @property
    def unpriced(self) -> int:
        return sum(1 for c in self._snapshot() if c.cost is None)

    @property
    def cache_hit(self) -> float | None:
        """Share of input tokens served from the provider's prompt cache."""
        calls = self._snapshot()
        total = sum(c.tokens_in for c in calls)
        return sum(c.cache_read for c in calls) / total if total else None

    @property
    def tokens_per_second(self) -> float | None:
        calls = [c for c in self._snapshot() if c.seconds > 0 and c.tokens_out > 0]
        seconds = sum(c.seconds for c in calls)
        return sum(c.tokens_out for c in calls) / seconds if seconds else None

    def by_purpose(self) -> dict[str, list]:
        """purpose -> [requests, tokens in, tokens out, cost (None if all unpriced)]."""
        rows: dict[str, list] = {}
        for c in self._snapshot():
            row = rows.setdefault(purpose_group(c.purpose), [0, 0, 0, None])
            row[0] += 1
            row[1] += c.tokens_in
            row[2] += c.tokens_out
            if c.cost is not None:
                row[3] = (row[3] or 0.0) + c.cost
        return rows


@dataclass
class Totals:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read: int = 0
    cost: float = 0.0
    unpriced: int = 0
    by_model: dict[str, list] = field(default_factory=dict)   # model -> [requests, in, out, cost]

    def add(self, model: str, tokens_in: int, tokens_out: int, cost: float | None = None, cache_read: int = 0) -> None:
        self.requests += 1
        self.prompt_tokens += tokens_in
        self.completion_tokens += tokens_out
        self.cache_read += cache_read
        row = self.by_model.setdefault(model or "?", [0, 0, 0, 0.0])
        row[0] += 1
        row[1] += tokens_in
        row[2] += tokens_out
        if cost is None:
            self.unpriced += 1
        else:
            self.cost += cost
            row[3] += cost


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
                cost = r.get("cost")
                for since, totals in periods.values():
                    if ts >= since:
                        totals.add(r.get("model", "?"), _int(r.get("in")), _int(r.get("out")),
                                   None if cost is None else float(cost), _int(r.get("cr")))
    except OSError:
        pass
    return {name: totals for name, (_, totals) in periods.items()}


# ---------------------------------------------------------------------------------------- budgets

class Budget:
    """Spending limits: budget.session_usd and budget.daily_usd (dollars; 0 or unset = no limit).

    check() before each model request: "ok", "warn" (80% reached, said once per limit) or "over".
    When over, the agent asks whether to go on (the terminal) or stops the turn (headless)."""

    WARN_AT = 0.8

    def __init__(self, ledger: Ledger, home: Path, session_usd: float = 0.0, daily_usd: float = 0.0):
        self.ledger = ledger
        self.home = home
        self.session_usd = float(session_usd or 0)
        self.daily_usd = float(daily_usd or 0)
        self._day = time.strftime("%Y-%m-%d")
        self._before_session = summarize(home)["today"].cost if self.daily_usd else 0.0
        self.warned: set[str] = set()

    @property
    def active(self) -> bool:
        return self.session_usd > 0 or self.daily_usd > 0

    def spent_today(self) -> float:
        today = time.strftime("%Y-%m-%d")
        if today != self._day:              # past midnight: today's spending starts from this session's calls
            self._day, self._before_session = today, 0.0
            midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            self._before_session = -sum(c.cost for c in self.ledger.calls if c.cost is not None and c.ts < midnight)
        return self._before_session + self.ledger.cost

    def limits(self) -> list[tuple[str, float, float]]:
        """(name, spent, limit) for each limit that is set."""
        out = []
        if self.session_usd > 0:
            out.append(("session", self.ledger.cost, self.session_usd))
        if self.daily_usd > 0:
            out.append(("today", self.spent_today(), self.daily_usd))
        return out

    def check(self) -> tuple[str, str]:
        level, message = "ok", ""
        for name, spent, limit in self.limits():
            if spent >= limit:
                return "over", f"{'Session' if name == 'session' else 'Daily'} budget reached: ${spent:.2f} of ${limit:.2f}"
            if spent >= limit * self.WARN_AT and name not in self.warned:
                self.warned.add(name)
                level = "warn"
                message = f"{spent / limit:.0%} of the {'session' if name == 'session' else 'daily'} budget used " \
                          f"(${spent:.2f} of ${limit:.2f})"
        return level, message

    def raise_limits(self, factor: float = 1.5) -> None:
        """"Continue": every limit that is reached goes up by half (at least past what is spent)."""
        for name, spent, limit in self.limits():
            if spent >= limit:
                new = max(limit * factor, spent + 0.01)
                if name == "session":
                    self.session_usd = new
                else:
                    self.daily_usd = new
                self.warned.discard(name)


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


def reset_seconds(value: str | None, now: float | None = None) -> float | None:
    """Seconds until a limit resets, from any of the header formats; None if unknown."""
    text = describe_reset(value, now)
    if not text.startswith("in "):
        return None
    total = 0.0
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)(h|m|s)", text):
        total += float(amount) * {"h": 3600, "m": 60, "s": 1}[unit]
    return round(total, 1)


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
