"""Phase B: every model call is accounted (whoever makes it), priced honestly, and budgets are enforced."""

import io
import json
import re

from fakeserver import FakeOpenAI, reply
from rich.console import Console
from test_agent_e2e import RecUI, make_app

from muyah_code import usage
from muyah_code.pricing import FREE, Price, Pricing, is_self_hosted, money
from muyah_code.ui.usage_view import render_usage

PRICED = {"fake-model": {"input": 2.0, "output": 10.0, "cache_read": 0.2}}   # $ per 1M tokens


def log_lines(home):
    usage.summarize(home)   # flushes the background writer
    return [json.loads(x) for x in (home / "usage.jsonl").read_text(encoding="utf-8").splitlines()]


def test_usage_blocks_from_each_provider_are_normalized():
    openai = usage.normalize({"prompt_tokens": 1000, "completion_tokens": 50,
                              "prompt_tokens_details": {"cached_tokens": 800}})
    assert openai == {"in": 1000, "out": 50, "cache_read": 800, "cache_write": 0}
    deepseek = usage.normalize({"prompt_tokens": 500, "completion_tokens": 5, "prompt_cache_hit_tokens": 300})
    assert deepseek["cache_read"] == 300
    anthropic = usage.normalize({"prompt_tokens": 1200, "completion_tokens": 9, "cache_read": 1000,
                                 "cache_write": 150})
    assert anthropic == {"in": 1200, "out": 9, "cache_read": 1000, "cache_write": 150}
    assert usage.normalize(None) == {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}


def test_cached_tokens_are_priced_at_the_cache_rate():
    claude = Price(5e-6, 25e-6, 0.5e-6, 6.25e-6)
    # 1000 input = 150 written to the cache + 800 read from it + 50 fresh
    assert abs(claude.cost(1000, 100, cache_read=800, cache_write=150)
               - (50 * 5e-6 + 800 * 0.5e-6 + 150 * 6.25e-6 + 100 * 25e-6)) < 1e-12
    no_cache_price = Price(1e-6, 2e-6)
    assert no_cache_price.cost(1000, 0, cache_read=800) == 1000 * 1e-6   # no discount without a cache price


def test_prices_come_from_the_table_by_provider_and_are_never_guessed(tmp_path):
    p = Pricing(tmp_path, refresh=False)
    assert p.price("claude-opus-5", "anthropic", "https://api.anthropic.com").input == 5e-6
    assert p.price("gemini-2.5-flash", "gemini", "https://generativelanguage.googleapis.com").output == 2.5e-6
    assert p.price("totally-made-up-model", "groq", "https://api.groq.com/openai/v1") is None
    assert p.price("anything", "", "http://127.0.0.1:8000/v1") is FREE                  # your own server
    assert p.price("qwen", "", "https://abc-def.trycloudflare.com/v1") is FREE        # your Colab tunnel
    assert p.supports_vision("claude-opus-5", "anthropic") is True
    own = Pricing(tmp_path, overrides={"proxied": {"input": 1, "output": 3}}, refresh=False)
    assert own.price("proxied", "", "http://localhost:4000").output == 3e-6          # a local proxy can bill
    assert money(None) == "unpriced" and money(0) == "$0" and money(0.0042) == "$0.0042" and money(1.5) == "$1.50"
    assert is_self_hosted("http://192.168.1.20:8080/v1") and not is_self_hosted("https://api.openai.com/v1")


def test_openrouter_uses_its_live_prices(tmp_path):
    p = Pricing(tmp_path, refresh=False)
    p.load_openrouter(fetch=lambda: {"data": [{"id": "qwen/qwen3-coder", "pricing": {
        "prompt": "0.0000003", "completion": "0.0000012", "input_cache_read": "0.00000003"}}]})
    price = p.price("qwen/qwen3-coder", "openrouter", "https://openrouter.ai/api/v1")
    assert (price.source, price.input, price.cache_read) == ("openrouter", 3e-7, 3e-8)


def test_every_call_is_recorded_with_its_purpose_and_cost(project):
    script = [
        reply("", [{"name": "Agent", "arguments": {"prompt": "look around", "subagent_type": "explore"}}],
              cached_tokens=100),
        reply("", [{"name": "Glob", "arguments": {"pattern": "*"}}]),   # the sub-agent
        reply("nothing here"),                                            # its report
        reply("Done."),
        reply("Summary: looked around, found nothing."),                   # compaction
    ]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project, pricing={"update": False, "models": PRICED})
        app.run_prompt("explore")
        assert app.context._summarize(app.agent.messages[1:4], app.llm, "").startswith("Summary")
        app.shutdown()
    purposes = [c.purpose for c in app.ledger.calls]
    assert purposes == ["main", "subagent:explore", "subagent:explore", "main", "compact"]
    first = app.ledger.calls[0]
    assert first.cache_read == 100
    assert abs(first.cost - ((first.tokens_in - 100) * 2e-6 + 100 * 0.2e-6 + 10 * 10e-6)) < 1e-12
    lines = log_lines(app.home)
    assert [x["purpose"] for x in lines] == purposes and lines[0]["cr"] == 100 and lines[0]["cost"] > 0
    assert abs(app.ledger.cost - sum(c.cost for c in app.ledger.calls)) < 1e-12


def test_usage_shows_cost_where_tokens_went_and_the_context(project):
    with FakeOpenAI([reply("", [{"name": "Glob", "arguments": {"pattern": "*"}}], cached_tokens=50),
                     reply("done")], headers={"x-ratelimit-remaining-requests": "29",
                                              "x-ratelimit-limit-requests": "30"}) as srv:
        app = make_app(srv, project, pricing={"update": False, "models": PRICED},
                       budget={"session_usd": 1.0})
        app.run_prompt("hi")
        app.shutdown()
    out = io.StringIO()
    render_usage(Console(file=out, width=110, color_system=None), app.home, llm=app.llm, app=app)
    text = out.getvalue()
    assert re.search(r"This session\s+\$0\.\d{4} · 2 requests · [\d,.k]+ in / 20 out · cache hit \d+%", text)
    assert "Where the tokens went" in text and re.search(r"main\s+█+·*\s+100%\s+2", text)
    assert "Context now" in text and "tool definitions" in text
    assert "session budget" in text and "of $1.00" in text
    assert re.search(r"Today\s+2\s", text) and "requests      29    30" in text


def test_unpriced_models_say_so(project):
    with FakeOpenAI([reply("ok")]) as srv:
        app = make_app(srv, project)
        app.llm.base_url = "https://api.example.com/v1"   # not self-hosted, not in any price list
        app.run_prompt("hi")
        app.shutdown()
    out = io.StringIO()
    render_usage(Console(file=out, width=110, color_system=None), app.home, app=app)
    assert "This session  unpriced · 1 request" in out.getvalue()
    assert "not in the price list" in out.getvalue()


def test_headless_stops_at_the_budget(project):
    script = [reply("", [{"name": "Glob", "arguments": {"pattern": "*"}}], prompt_tokens=200_000), reply("never")]
    with FakeOpenAI(script) as srv:
        app = make_app(srv, project, headless=True, pricing={"update": False, "models": PRICED},
                       budget={"session_usd": 0.1})
        app.ui.headless = True
        res = app.run_prompt("go")
        app.shutdown()
    assert res.status == "budget" and "Session budget reached: $0.40 of $0.10" in res.error
    assert len(srv.requests) == 1                        # the second request was never sent


class BudgetUI(RecUI):
    def __init__(self, answer):
        super().__init__()
        self.answer = answer

    def ask_user(self, question, options):
        self.events.append(("question", question))
        return self.answer


def test_the_terminal_asks_before_going_over_budget(project):
    script = [reply("", [{"name": "Glob", "arguments": {"pattern": "*"}}], prompt_tokens=45_000),
              reply("", [{"name": "Glob", "arguments": {"pattern": "*"}}], prompt_tokens=10_000), reply("finished")]
    with FakeOpenAI(script) as srv:
        ui = BudgetUI("Continue (raise the limit by half)")
        app = make_app(srv, project, ui=ui, pricing={"update": False, "models": PRICED},
                       budget={"session_usd": 0.1})
        res = app.run_prompt("go")
        app.shutdown()
    assert ("warn", "90% of the session budget used ($0.09 of $0.10)") in ui.events
    assert any(e[0] == "question" and "Session budget reached" in e[1] for e in ui.events)
    assert res.status == "ok" and res.text == "finished" and app.budget.session_usd > 0.1

    with FakeOpenAI(script) as srv:
        ui = BudgetUI("Stop this turn")
        app = make_app(srv, project, ui=ui, pricing={"update": False, "models": PRICED},
                       budget={"session_usd": 0.05})
        res = app.run_prompt("go")
        app.shutdown()
    assert res.status == "budget" and len(srv.requests) == 1
