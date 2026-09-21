"""Token budgeting and compaction.

Strategy when the conversation passes `threshold` of the usable window:
  1. Prune: shrink old tool outputs (the bulk of any coding session), keeping the most recent ones intact.
  2. Summarize: replace the older part of the conversation with an LLM-written structured summary,
     keeping the recent tail verbatim. The cut always lands on a real user message so tool-call
     pairs are never split.
  3. Fallback: if the summary call fails, use a mechanical digest instead of failing the turn.
"""

from __future__ import annotations

import json

from muyah_code.llm.client import estimate_tokens
from muyah_code.llm.content import IMAGE_TOKENS, count_images, strip_images, text_of

PRUNED_MARK = "[output pruned to save context"
SUMMARY_PROMPT = """\
You are compacting a coding session so work can continue in a fresh context window. Write a precise summary \
that lets another engineer continue without the transcript. Use these sections:

1. User requests: every explicit request and constraint, in the user's words where possible.
2. Work done: files created/modified (paths) and what changed; commands run and their outcomes.
3. Key facts: important code locations, APIs, decisions, and conventions discovered.
4. Errors and fixes: what failed and how it was resolved (or not).
5. Current state: what was in progress at the end, and the exact next steps.

Be specific (paths, function names, commands). No filler.{focus}"""


def is_tool_result_message(m: dict) -> bool:
    if m.get("role") == "tool" or m.get("_images"):
        return True
    content = m.get("content")
    return m.get("role") == "user" and isinstance(content, str) and content.startswith("<tool_result")


def is_real_user_message(m: dict) -> bool:
    return m.get("role") == "user" and not is_tool_result_message(m)


class ContextManager:
    """Adapts to any window size (8k..1M+) and self-calibrates against the server's real token counts."""

    def __init__(self, window: int, max_output: int, threshold: float = 0.8):
        self.threshold = threshold
        self.ratio = 1.0  # real tokens / estimated tokens, learned from server usage reports
        self.calibrated = False
        self.set_window(window, max_output)

    def set_window(self, window: int, max_output: int) -> None:
        self.window = max(2048, int(window))
        # completion reserve scales with the window: tiny models must not spend a third of it on output
        self.max_output = max(512, min(int(max_output), self.window // 4))

    @property
    def usable(self) -> int:
        return self.window - self.max_output

    @property
    def keep_recent_results(self) -> int:
        return 3 if self.window <= 16384 else 6 if self.window <= 65536 else 12

    def tool_output_chars(self, configured: int) -> int:
        """Cap a single tool result at ~20% of the usable window."""
        return max(3000, min(int(configured), int(self.usable * 0.2 * 3.5)))

    def raw_count(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        total = sum(estimate_tokens(text_of(m.get("content") or "")) + 6 +
                    IMAGE_TOKENS * count_images(m.get("content")) for m in messages)
        for m in messages:
            if m.get("tool_calls"):
                total += estimate_tokens(m["tool_calls"])
        if tools:
            total += estimate_tokens(tools)
        return total

    def count(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        return int(self.raw_count(messages, tools) * self.ratio)

    def calibrate(self, messages: list[dict], tools: list[dict] | None, actual_prompt_tokens: int) -> None:
        """Learn this model's tokenizer density from the usage the server reported for a request."""
        raw = self.raw_count(messages, tools)
        if actual_prompt_tokens <= 0 or raw < 200:
            return
        observed = min(3.0, max(0.4, actual_prompt_tokens / raw))
        self.ratio = observed if not self.calibrated else 0.6 * self.ratio + 0.4 * observed
        self.calibrated = True

    def completion_budget(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        """max_tokens that still fits the window (servers reject prompt + max_tokens > window)."""
        left = self.window - self.count(messages, tools) - 64
        return max(256, min(self.max_output, left))

    def needs_compaction(self, messages: list[dict], tools: list[dict] | None = None) -> bool:
        return self.count(messages, tools) > self.usable * self.threshold

    # ------------------------------------------------------------------ step 1

    def prune(self, messages: list[dict], keep_recent: int | None = None, max_chars: int = 400) -> int:
        """Shrink old tool outputs in place. Returns number of messages pruned."""
        if keep_recent is None:
            keep_recent = self.keep_recent_results
        idxs = [i for i, m in enumerate(messages) if is_tool_result_message(m)]
        pruned = 0
        for i in idxs[:-keep_recent] if keep_recent else idxs:
            content = messages[i].get("content") or ""
            if count_images(content):             # old screenshots are the most expensive thing to keep
                messages[i] = {**messages[i], "content": strip_images(content)}
                pruned += 1
                continue
            if not isinstance(content, str) or len(content) <= max_chars or PRUNED_MARK in content:
                continue
            messages[i] = {**messages[i], "content": content[:max_chars] + f"\n{PRUNED_MARK}; re-run the tool if needed]"}
            pruned += 1
        return pruned

    # ------------------------------------------------------------------ step 2

    def split_point(self, messages: list[dict], tail_budget: int) -> int:
        """Index where the verbatim tail starts: a real user message, tail within budget when possible."""
        user_idxs = [i for i, m in enumerate(messages) if i > 0 and is_real_user_message(m)]
        if not user_idxs:
            return len(messages)
        chosen = user_idxs[-1]
        for i in reversed(user_idxs):
            if self.count(messages[i:]) <= tail_budget:
                chosen = i
            else:
                break
        return chosen

    def compact(self, messages: list[dict], llm=None, focus: str = "", todos: list[dict] | None = None,
                tools: list[dict] | None = None, emergency: bool = False) -> tuple[list[dict], str]:
        """Return (new_messages, description). messages[0] must be the system message.

        emergency=True is used after the server rejected a prompt as too long: our estimate was low,
        so the ratio is bumped and the target is stricter.
        """
        self.last: dict = {}                     # what this compaction did, for the one-line report
        if emergency:
            self.ratio = min(3.0, self.ratio * 1.25)
        work = [dict(m) for m in messages]
        before = self.count(work, tools)
        target = int(self.usable * (0.35 if emergency else 0.5))
        pruned = self.prune(work, keep_recent=1 if emergency else None)
        # In an emergency the server just proved our estimate too low, so never stop at the cheap step.
        if not focus and not emergency and self.count(work, tools) <= target:
            self.last = {"kind": "pruned", "count": pruned}
            return work, f"pruned {pruned} old tool outputs ({before} -> {self.count(work, tools)} tokens)"

        split = self.split_point(work, tail_budget=int(self.usable * 0.3))
        head, tail = work[1:split], work[split:]
        if not head:
            # a single enormous turn: shrink every tool result, harder in an emergency
            keep = 0 if emergency else 2
            self.prune(work, keep_recent=keep, max_chars=1000 if emergency else 1500)
            self.last = {"kind": "pruned", "count": 0}
            return work, f"pruned tool outputs inside the current turn ({before} -> {self.count(work, tools)} tokens)"

        summary = self._summarize(head, llm, focus)
        mechanical = summary is None
        if mechanical:
            summary = mechanical_digest(head)
        if todos:
            summary += "\n\nCurrent todo list:\n" + "\n".join(f"- [{t['status']}] {t['content']}" for t in todos)
        new = [
            work[0],
            {"role": "user", "content": "[Summary of the earlier conversation - context was compacted]\n\n" + summary},
            {"role": "assistant", "content": "Understood. I have the summary and will continue from the current state."},
            *tail,
        ]
        after = self.count(new, tools)
        how = "mechanical digest" if mechanical else "LLM summary"
        desc = f"compacted {len(head)} messages into a {how}"
        if after > target:
            # the kept (current) turn is itself too big, e.g. a few large file reads in a small window:
            # shrink its tool outputs too, or the next step compacts again and gets nowhere
            if self.prune(new, keep_recent=1 if emergency else 2, max_chars=1000 if emergency else 1500):
                desc += " and shortened large tool outputs in the current turn"
            after = self.count(new, tools)
        if after >= before:
            # summarizing cost more than it saved: keep the conversation, with its tool outputs shortened
            self.prune(work, keep_recent=1, max_chars=1000)
            new, desc = work, "shortened large tool outputs (a summary would not have been smaller)"
            self.last = {"kind": "pruned", "count": 0}
            after = self.count(new, tools)
        if self.last.get("kind") != "pruned":
            self.last = {"kind": "summary", "count": len(head),
                         "yours": sum(1 for m in head if is_real_user_message(m))}
        return new, f"{desc} ({before} -> {after} tokens)"

    def _summarize(self, head: list[dict], llm, focus: str) -> str | None:
        if llm is None:
            return None
        budget_chars = int(self.usable * 0.55 * 3.5)
        transcript = render_transcript(head, per_message=1500)
        if len(transcript) > budget_chars:
            first_user = next((render_transcript([m], 1500) for m in head if is_real_user_message(m)), "")
            transcript = first_user + "\n...\n" + transcript[-(budget_chars - len(first_user)):]
        focus_txt = f"\n\nThe user asked to focus on: {focus}" if focus else ""
        messages = [
            {"role": "system", "content": SUMMARY_PROMPT.format(focus=focus_txt)},
            {"role": "user", "content": f"Transcript to summarize:\n\n{transcript}"},
        ]
        try:
            out = llm.chat(messages, max_tokens=min(2000, self.window // 6), temperature=0, purpose="compact")
        except Exception:
            return None
        text = (out.content or "").strip()
        return text or None


def render_transcript(messages: list[dict], per_message: int = 1500) -> str:
    out = []
    for m in messages:
        role = m.get("role", "?")
        content = text_of(m.get("content") or "").strip()
        if len(content) > per_message:
            content = content[: per_message // 2] + " ... " + content[-per_message // 2:]
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "")
                if isinstance(args, dict):
                    args = json.dumps(args)
                calls.append(f"{fn.get('name')}({args[:200]})")
            content = (content + "\n" if content else "") + "-> tool calls: " + "; ".join(calls)
        label = "TOOL RESULT" if is_tool_result_message(m) else role.upper()
        if content:
            out.append(f"{label}: {content}")
    return "\n\n".join(out)


def mechanical_digest(messages: list[dict]) -> str:
    requests, actions = [], []
    for m in messages:
        if is_real_user_message(m):
            requests.append("- " + " ".join(text_of(m.get("content") or "").split())[:300])
        elif m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                actions.append(f"- {fn.get('name')}({str(fn.get('arguments', ''))[:120]})")
            if m.get("content"):
                actions.append("- said: " + " ".join(text_of(m["content"]).split())[:200])
    return ("User requests so far:\n" + "\n".join(requests[-10:]) +
            "\n\nRecent actions:\n" + "\n".join(actions[-25:]))
