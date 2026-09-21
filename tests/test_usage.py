"""/usage: the usage log, the summaries, and the rate limits providers report in their headers."""

import io
import json
import re
import time

from fakeserver import FakeOpenAI, reply
from rich.console import Console
from test_agent_e2e import make_app

from muyah_code import usage
from muyah_code.ui.commands import render_usage


def test_rate_limit_headers_from_each_provider_style():
    groq = usage.parse_limits({"x-ratelimit-limit-requests": "14400", "x-ratelimit-remaining-requests": "14370",
                               "x-ratelimit-reset-requests": "2m59.56s", "x-ratelimit-limit-tokens": "6000",
                               "x-ratelimit-remaining-tokens": "5997", "x-ratelimit-reset-tokens": "7.66s"})
    rows = {r.name: r for r in groq}
    assert rows["requests"].remaining == "14370" and rows["tokens"].limit == "6000"
    assert usage.describe_reset(rows["tokens"].reset) == "in 7.7s"
    assert usage.describe_reset(rows["requests"].reset) == "in 2m 59s"
    anthropic = usage.parse_limits({"anthropic-ratelimit-input-tokens-remaining": "39000",
                                    "anthropic-ratelimit-input-tokens-limit": "40000",
                                    "anthropic-ratelimit-requests-reset": "2026-09-21T15:00:00Z"})
    names = {r.name for r in anthropic}
    assert names == {"input-tokens", "requests"}
    openrouter = usage.parse_limits({"X-RateLimit-Limit": "20", "X-RateLimit-Remaining": "17",
                                     "X-RateLimit-Reset": str(int((time.time() + 30) * 1000))})
    assert openrouter[0].name == "requests" and openrouter[0].remaining == "17"
    assert re.fullmatch(r"in (29|30)s", usage.describe_reset(openrouter[0].reset))
    assert usage.parse_limits({"content-type": "json"}) == []


def test_usage_log_sums_today_and_last_week(tmp_path):
    now = time.time()
    lines = [{"ts": now - 60, "model": "a", "in": 100, "out": 10},
             {"ts": now - 3 * 86400, "model": "b", "in": 1000, "out": 100},
             {"ts": now - 30 * 86400, "model": "c", "in": 5, "out": 5}]
    (tmp_path / "usage.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
    periods = usage.summarize(tmp_path, now)
    assert (periods["today"].requests, periods["today"].prompt_tokens) == (1, 100)
    assert (periods["7 days"].requests, periods["7 days"].completion_tokens) == (2, 110)
    assert set(periods["7 days"].by_model) == {"a", "b"}


def test_usage_shows_session_totals_and_live_provider_limits(project):
    limits = {"x-ratelimit-limit-requests": "30", "x-ratelimit-remaining-requests": "29",
              "x-ratelimit-reset-requests": "2s"}
    with FakeOpenAI([reply("", [{"name": "Glob", "arguments": {"pattern": "*"}}]), reply("done")],
                    headers=limits) as srv:
        app = make_app(srv, project)
        app.run_prompt("hi")
        app.shutdown()
    assert app.llm.limits["x-ratelimit-remaining-requests"] == "29"
    out = io.StringIO()
    render_usage(Console(file=out, width=100, color_system=None), app.home, llm=app.llm)
    text = out.getvalue()
    assert re.search(r"This session\s+2\s", text) and re.search(r"Today\s+2\s", text)
    assert "Provider limits" in text and re.search(r"requests\s+29\s+30\s+in 2\.0s", text)
    assert "fake-model 2 req" in text                     # by model, last 7 days
