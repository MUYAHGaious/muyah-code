"""/usage (and `muyah usage`): what this session cost and where the tokens went.

Only what matters, in this order:
  1. this session: cost, tokens in/out, prompt-cache hit rate, output speed, budget
  2. where the tokens went: the agent, each sub-agent, compaction, learning, web page summaries...
  3. the context window right now: system prompt, conversation, tool results, tool definitions
  4. your provider's rate limits: what is left and when it resets
  5. today and the last 7 days (with `--all`: per model)
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.markup import escape
from rich.table import Table
from rich.text import Text

from muyah_code import usage
from muyah_code.pricing import money

BAR = 16


def _bar(share: float) -> str:
    filled = max(0, min(BAR, round(share * BAR)))
    return "█" * filled + "·" * (BAR - filled)


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def render_usage(console, home: Path, llm=None, app=None, full: bool = False) -> None:
    ledger = getattr(app, "ledger", None)
    if ledger is not None:
        _session(console, app, ledger)
        if ledger.requests:
            _purposes(console, ledger)
        _context(console, app)
    _periods(console, home, full)
    if llm is not None:
        _limits(console, llm, ledger)
    if full and app is not None and getattr(app.pricing, "update_error", ""):
        console.print(f"[dim]{escape(app.pricing.update_error)}[/]")


def _session(console, app, ledger) -> None:
    head = Text("This session  ", style="bold")
    if not ledger.requests:
        console.print(head + Text("no model requests yet", style="dim"))
        return
    parts = [money(ledger.cost) if ledger.requests > ledger.unpriced else "unpriced",
             f"{ledger.requests} request{'s' if ledger.requests != 1 else ''}",
             f"{usage.compact(ledger.tokens_in)} in / {usage.compact(ledger.tokens_out)} out"]
    if ledger.cache_hit is not None:
        parts.append(f"cache hit {_pct(ledger.cache_hit)}")
    if ledger.tokens_per_second:
        parts.append(f"{ledger.tokens_per_second:.0f} tokens/s")
    console.print(head + Text(" · ".join(parts)))
    eq, eq_to = ledger.equivalent
    if eq_to:
        console.print(f"[dim]  Self-hosted, so it costs you nothing per token. On a paid API ({escape(eq_to)}) the same "
                      f"work would be about [/][bold]{money(eq)}[/][dim]. Compare with another model: "
                      'pricing.compare_to, e.g. "deepseek:deepseek-v4-pro".[/]')
    if ledger.unpriced and ledger.requests > ledger.unpriced:
        console.print(f"[dim]  {ledger.unpriced} request(s) are unpriced (the model is not in the price list; "
                      "set pricing.models to add it)[/]")
    elif ledger.unpriced:
        console.print("[dim]  This model is not in the price list, so no cost is shown. Set your own prices with "
                      'pricing.models, e.g. {"my-model": {"input": 0.5, "output": 1.5}} ($ per 1M tokens).[/]')
    budget = getattr(app, "budget", None)
    for name, spent, limit in (budget.limits() if budget is not None else []):
        share = spent / limit if limit else 0
        style = "red" if share >= 1 else "yellow" if share >= usage.Budget.WARN_AT else "dim"
        console.print(Text(f"  {name} budget  {_bar(share)} ${spent:.2f} of ${limit:.2f}", style=style))


def _purposes(console, ledger) -> None:
    rows = ledger.by_purpose()
    total = sum(r[1] + r[2] for r in rows.values()) or 1
    t = Table(show_header=True, box=None, padding=(0, 2), header_style="bold", title="Where the tokens went",
              title_justify="left", title_style="bold")
    t.add_column("")
    t.add_column("share")
    t.add_column("requests", justify="right")
    t.add_column("in", justify="right")
    t.add_column("out", justify="right")
    t.add_column("cost", justify="right")
    for name, (req, tin, tout, cost) in sorted(rows.items(), key=lambda kv: -(kv[1][1] + kv[1][2])):
        share = (tin + tout) / total
        t.add_row(escape(name), f"{_bar(share)} {share:>4.0%}", str(req), usage.compact(tin), usage.compact(tout),
                  money(cost))
    console.print()
    console.print(t)


def _context(console, app) -> None:
    agent, ctx = getattr(app, "agent", None), getattr(app, "context", None)
    if agent is None or ctx is None:
        return
    from muyah_code.agent.context import is_tool_result_message

    msgs = agent.messages
    tools = None if agent.text_mode else app.registry.schemas()
    parts = [("system prompt", ctx.count(msgs[:1])),
             ("conversation", ctx.count([m for m in msgs[1:] if not is_tool_result_message(m)])),
             ("tool results", ctx.count([m for m in msgs[1:] if is_tool_result_message(m)])),
             ("tool definitions", ctx.count(msgs[:1], tools) - ctx.count(msgs[:1]) if tools else 0)]
    used = sum(n for _, n in parts)
    console.print()
    console.print(Text(f"Context now  {usage.compact(used)} of {usage.compact(ctx.usable)} usable tokens "
                       f"({used / max(1, ctx.usable):.0%})", style="bold"))
    for name, n in parts:
        console.print(Text(f"  {name:<17}{_bar(n / max(1, ctx.usable))} {usage.compact(n)}", style="dim"))


def _periods(console, home: Path, full: bool) -> None:
    periods = usage.summarize(home)
    t = Table(show_header=True, box=None, padding=(0, 2), header_style="bold")
    t.add_column("")
    t.add_column("requests", justify="right")
    t.add_column("tokens in", justify="right")
    t.add_column("tokens out", justify="right")
    t.add_column("cached", justify="right")
    t.add_column("cost", justify="right")
    for name, label in (("today", "Today"), ("7 days", "Last 7 days")):
        p = periods[name]
        cost = money(p.cost) + (" +unpriced" if p.unpriced and p.cost else "") if p.requests > p.unpriced \
            else ("unpriced" if p.requests else "-")
        if p.equivalent:
            cost += f" (≈ {money(p.equivalent)} if paid)"
        t.add_row(label, str(p.requests), usage.compact(p.prompt_tokens), usage.compact(p.completion_tokens),
                  _pct(p.cache_read / p.prompt_tokens if p.prompt_tokens else None), cost)
    console.print()
    console.print(t)
    week = periods["7 days"]
    if not week.by_model:
        return
    rows = sorted(week.by_model.items(), key=lambda kv: -(kv[1][1] + kv[1][2]))
    if full:
        mt = Table(show_header=True, box=None, padding=(0, 2), header_style="bold", title="By model, last 7 days",
                   title_justify="left", title_style="bold")
        for col in ("model", "requests", "in", "out", "cost"):
            mt.add_column(col, justify="left" if col == "model" else "right")
        for model, (req, tin, tout, cost) in rows:
            mt.add_row(escape(model), str(req), usage.compact(tin), usage.compact(tout), money(cost))
        console.print()
        console.print(mt)
    else:
        top = rows[:3]
        console.print("[dim]Top models, 7 days: " + " · ".join(
            f"{escape(m)} {usage.compact(r[1] + r[2])} tokens {money(r[3])}" for m, r in top)
            + (" · /usage --all for everything" if len(rows) > 3 else "") + "[/]")


def _limits(console, llm, ledger) -> None:
    limits = usage.parse_limits(getattr(llm, "limits", {}))
    console.print()
    requests = ledger.requests if ledger is not None else llm.total_usage.get("requests", 0)
    if not limits:
        console.print(f"[dim]{escape(getattr(llm, 'model', ''))}: the provider has not reported its limits"
                      + (" yet (they arrive with the first reply)." if not requests else
                         ". Some providers (e.g. Gemini) never do; if you hit one, the error says when to retry.")
                      + "[/]")
        return
    age = time.time() - getattr(llm, "limits_at", 0)
    lt = Table(show_header=True, box=None, padding=(0, 2), header_style="bold",
               title=f"Provider limits ({escape(getattr(llm, 'model', ''))}, as of {age:.0f}s ago)",
               title_justify="left", title_style="bold")
    lt.add_column("")
    lt.add_column("left", justify="right")
    lt.add_column("of", justify="right")
    lt.add_column("resets")
    for row in limits:
        lt.add_row(row.name.replace("-", " "), row.remaining or "?", row.limit or "?", usage.describe_reset(row.reset))
    console.print(lt)
    retry = (getattr(llm, "limits", {}) or {}).get("retry-after")
    if retry:
        console.print(f"[yellow]Rate limited: retry after {escape(str(retry))}s.[/]")
