"""The agent turn loop.

    user prompt -> [compact if needed] -> LLM -> tool calls? -> hooks -> permission -> run -> results -> LLM ...
                                                   no calls  -> Stop hook -> final answer

Robustness features for open-weights models:
  * native function calling with automatic fallback to a strict text protocol
  * malformed/invalid calls are answered with a precise error so the model can correct itself
  * repeated identical calls are detected and broken
  * truncated outputs (finish_reason=length) are continued or explained
  * context overflow triggers emergency compaction and a retry
  * Ctrl+C leaves the transcript consistent
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from muyah_code.agent.context import ContextManager
from muyah_code.hooks import HookRunner
from muyah_code.agent.context import is_real_user_message
from muyah_code.llm.client import (
    AssistantMessage,
    ContextOverflowError,
    LLMClient,
    ToolCall,
    ToolsUnsupportedError,
)
from muyah_code.llm.toolcall_parser import parse_tool_calls
from muyah_code.permissions import PermissionManager, suggest_rule
from muyah_code.tools.base import READ, Tool, ToolContext, ToolResult
from muyah_code.tools.registry import ToolRegistry
from muyah_code.ui.base import UI, PermissionRequest

REPEAT_WARN = 3
REPEAT_STOP = 5
MAX_CONTINUATIONS = 2


@dataclass
class TurnResult:
    text: str = ""
    status: str = "ok"  # ok | max_steps | interrupted | error | blocked | loop | denied
    steps: int = 0
    tool_calls: int = 0
    error: str | None = None
    signals: list[dict] = field(default_factory=list)
    duration: float = 0.0


STOCK_OPENER = re.compile(
    r"^\s*(?:(?:you(?:'re| are) (?:absolutely|completely|totally|so|entirely) (?:right|correct)|"
    r"you(?:'re| are) right(?: to [^.!?\n]{1,60})?(?=[.!,:;\u2014-])|"
    r"(?:great|excellent|good|fantastic) (?:question|catch|point|idea|call)|absolutely right|"
    r"i apologi[sz]e for (?:the|any) confusion|sorry for (?:the|any) confusion)[.!,:;]*\s*)+",
    re.IGNORECASE)


def _k(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def compact_summary(last: dict, before: int, after: int) -> str:
    """One clean line: what was compacted and what it saved."""
    saved = f"{_k(before)} → {_k(after)} tokens" + (f" (−{100 - 100 * after // max(1, before)}%)" if after < before else "")
    if last.get("kind") == "summary":
        n, yours = last.get("count", 0), last.get("yours", 0)
        return (f"Compacted {n} older message{'s' if n != 1 else ''} ({yours} of yours, {n - yours} from the AI and "
                f"its tools) into a summary · {saved}")
    count = last.get("count") or 0
    what = f"{count} old tool output{'s' if count != 1 else ''}" if count else "the largest tool outputs"
    return f"Shortened {what} · {saved}"


def strip_stock_opener(text: str) -> str:
    """Drop flattering / reflexive openers from the start of an answer; the rest stays as written."""
    stripped = STOCK_OPENER.sub("", text, count=1)
    if stripped != text and stripped[:1].islower():
        stripped = stripped[:1].upper() + stripped[1:]
    return stripped


class _OpenerFilter:
    """Hold back the first words of a streamed answer until it is clear whether they are a stock opener."""

    WINDOW = 90

    def __init__(self, sink: Callable[[str], None], on_strip: Callable[[], None] | None = None):
        self.sink = sink
        self.on_strip = on_strip
        self.buf = ""
        self.decided = False

    def feed(self, chunk: str) -> None:
        if self.decided:
            self.sink(chunk)
            return
        self.buf += chunk
        if len(self.buf) >= self.WINDOW or "\n" in self.buf.strip():
            self._decide()

    def _decide(self) -> None:
        self.decided = True
        text = strip_stock_opener(self.buf)
        if text != self.buf and self.on_strip:
            self.on_strip()
        self.buf = ""
        if text:
            self.sink(text)

    def flush(self) -> None:
        if not self.decided:
            self._decide()


class _TagHider:
    """Hide <tool_call>...</tool_call> blocks from the live text stream (text protocol mode)."""

    OPEN, CLOSE = "<tool_call>", "</tool_call>"

    def __init__(self, sink: Callable[[str], None]):
        self.sink = sink
        self.buf = ""
        self.inside = False

    def feed(self, chunk: str) -> None:
        self.buf += chunk
        while self.buf:
            tag = self.CLOSE if self.inside else self.OPEN
            idx = self.buf.find(tag)
            if idx == -1:
                keep = len(tag) - 1
                if len(self.buf) > keep:
                    emit, self.buf = self.buf[:-keep], self.buf[-keep:]
                    if not self.inside:
                        self.sink(emit)
                return
            if not self.inside:
                self.sink(self.buf[:idx])
            self.buf = self.buf[idx + len(tag):]
            self.inside = not self.inside

    def flush(self) -> None:
        if not self.inside and self.buf:
            self.sink(self.buf)
        self.buf = ""




class _Cancelled(Exception):
    """Raised inside an abandoned model request so it stops at its next chunk."""


def interruptible_call(chat, messages, on_text=None, on_reasoning=None, cancel=None, **kwargs):
    """Run a model request on a worker thread and wait in short slices.

    A blocked network read cannot be interrupted in Python, so Ctrl+C / Esc used to take effect only when
    the server sent its next bytes (seconds, while a model thinks). Waiting here instead lets the interrupt
    land within ~0.1 s; the abandoned request stops at its next chunk and its output is dropped."""
    cancelled = threading.Event()

    def guard(callback):
        if callback is None:
            return None

        def wrapped(chunk):
            if cancelled.is_set():
                raise _Cancelled()
            callback(chunk)
        return wrapped

    box: dict = {}

    def work():
        try:
            box["result"] = chat(messages, on_text=guard(on_text), on_reasoning=guard(on_reasoning), **kwargs)
        except BaseException as e:  # handed to the waiting thread below
            box["error"] = e

    worker = threading.Thread(target=work, name="muyah-llm", daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.1)
            if cancel is not None and cancel.is_set():   # cancelled from elsewhere (an editor over ACP)
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        cancelled.set()
        raise
    if "error" in box:
        raise box["error"]
    return box["result"]


LONG_ARG_KEYS = ("content", "new_string", "old_string", "edits", "prompt")


def brief_args(args: dict | None) -> dict:
    """Tool arguments for the live view: short values as they are, long text as a size."""
    out: dict = {}
    for k, v in (args or {}).items():
        if isinstance(v, str):
            out[k] = f"({len(v):,} chars)" if k in LONG_ARG_KEYS and len(v) > 200 else v[:400]
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        else:
            text = json.dumps(v, default=str)
            out[k] = text if len(text) <= 300 else f"({len(text):,} chars)"
    return out


FILE_CONTENT_LIMIT = 12000   # characters of a file / diff the live view gets per event


def file_detail(name: str, args: dict | None) -> dict:
    """For the live view's file panels: what is being written, or the edit (old -> new), as it starts."""
    args = args or {}
    if name == "Write":
        return {"file": {"path": args.get("file_path", ""), "content": str(args.get("content", ""))[:FILE_CONTENT_LIMIT]}}
    if name in ("Edit", "MultiEdit"):
        edits = args.get("edits") if isinstance(args.get("edits"), list) else [args]
        return {"file": {"path": args.get("file_path", ""), "edits": [
            {"old": str(e.get("old_string", ""))[:4000], "new": str(e.get("new_string", ""))[:4000]}
            for e in edits[:20] if isinstance(e, dict)]}}
    if name == "Read":
        return {"file": {"path": args.get("file_path", "")}}
    return {}


def result_detail(name: str, res: ToolResult) -> dict:
    """What the live view shows about a finished tool: output (tail for commands), file text, diff, exit code."""
    detail: dict = {}
    content = res.content or ""
    if name in ("Bash", "BashOutput") or name.startswith("mcp__"):
        detail["output"] = content[-3000:]
    elif name == "Read":
        detail["output"] = content[:FILE_CONTENT_LIMIT]     # the file as the model saw it (line-numbered)
    elif name == "Skill":
        detail["output"] = content[:6000]                   # the skill's instructions
    elif name not in ("Write", "Edit", "MultiEdit"):
        detail["output"] = content[:2000]
    if res.display:
        detail["diff"] = res.display[:FILE_CONTENT_LIMIT]
    if "exit_code" in res.meta:
        detail["exit_code"] = res.meta["exit_code"]
    if res.display:
        lines = res.display.splitlines()
        detail["added"] = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
        detail["removed"] = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
    return detail


class Agent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        registry: ToolRegistry,
        permissions: PermissionManager,
        ctx: ToolContext,
        ui: UI,
        context: ContextManager,
        system_prompt: Callable[[bool], str],
        turn_context: Callable[[str], str] | None = None,
        hooks: HookRunner | None = None,
        session=None,
        tool_mode: str = "auto",
        max_steps: int = 60,
        max_tool_output_chars: int = 24000,
        is_subagent: bool = False,
        session_id: str = "",
    ):
        self.llm = llm
        self.registry = registry
        self.permissions = permissions
        self.ctx = ctx
        self.ui = ui
        self.context = context
        self.system_prompt = system_prompt
        self.turn_context = turn_context
        self.hooks = hooks
        self.session = session
        self.tool_mode = tool_mode
        self.text_mode = tool_mode == "text"
        self.max_steps = max_steps
        self.max_tool_output_chars = max_tool_output_chars
        self.is_subagent = is_subagent
        self.session_id = session_id
        self.messages: list[dict] = [{"role": "system", "content": ""}]
        self.last_user_prompt = ""
        self._prompt_dirty = True
        self.events = None       # muyah_code.events.EventBus (feeds /viz); None = no events
        self.label = "main"      # which agent emitted an event (sub-agents use their type name)
        from muyah_code.agent.escalation import Escalation

        self.escalation = Escalation()
        self.summarizer = None   # the model that writes compaction summaries (models.summarize; None = self.llm)
        self.cancel_event = None  # threading.Event: set from another thread to stop the turn (ACP session/cancel)
        # escalate(client) -> (bigger client, "from" label, "to" label) or None (set by the app: models.py ladder)
        self.escalate = None
        self._tool_seq = 0

    # ------------------------------------------------------------------ events (for /viz)

    @property
    def purpose(self) -> str:
        """What this agent's model calls are for, in the usage log."""
        return "main" if self.label == "main" else f"subagent:{self.label}"

    def _emit(self, type_: str, **data) -> None:
        if self.events is not None:
            self.events.emit(type_, agent=self.label, **data)

    def emit_context(self) -> None:
        if self.events is None:
            return
        from muyah_code.agent.context import is_tool_result_message

        tools_msgs = [m for m in self.messages[1:] if is_tool_result_message(m)]
        convo = [m for m in self.messages[1:] if not is_tool_result_message(m)]
        c = self.context
        self._emit("context", used=c.count(self.messages), usable=c.usable, window=c.window,
                   parts={"system": c.count(self.messages[:1]), "conversation": c.count(convo),
                          "tools": c.count(tools_msgs)})

    def _new_tool_id(self) -> int:
        self._tool_seq += 1
        return self._tool_seq

    def _emit_tool_request(self, tool_id: int, name: str, title: str, args: dict | None) -> None:
        """The model asked for this tool (nothing has run yet)."""
        self._emit("tool_request", id=tool_id, name=name, title=title[:200], args=brief_args(args))

    def _emit_tool_start(self, tool_id: int, name: str, title: str, args: dict | None) -> None:
        """The tool is actually starting to run, right now."""
        self._emit("tool_start", id=tool_id, name=name, title=title[:200], args=brief_args(args),
                   **file_detail(name, args))

    def _emit_tool_end(self, tool_id: int, name: str, res: ToolResult, duration: float) -> None:
        self._emit("tool_end", id=tool_id, name=name, ok=not res.is_error, summary=(res.summary or "")[:160],
                   duration=round(duration, 3), chars=len(res.content or ""), **result_detail(name, res))

    def _show_tool_start(self, title: str, name: str, tool_id: int | None = None, args: dict | None = None) -> int:
        tool_id = tool_id if tool_id is not None else self._new_tool_id()
        self.ui.tool_start(title)
        self._emit_tool_start(tool_id, name, title, args)
        return tool_id

    def _show_tool_end(self, tool_id: int, title: str, name: str, res: ToolResult, duration: float) -> None:
        self.ui.tool_end(title, res)
        self._emit_tool_end(tool_id, name, res, duration)

    # ------------------------------------------------------------------ public

    def load_history(self, messages: list[dict]) -> None:
        self.messages = [{"role": "system", "content": ""}] + [m for m in messages if m.get("role") != "system"]

    def clear(self) -> None:
        self.messages = [{"role": "system", "content": ""}]
        self.ctx.todos.clear()
        self._new_epoch()
        if self.session:
            self.session.log_replace([], "clear")

    def refresh_system_prompt(self) -> None:
        """Rebuild the system prompt. It is kept stable between turns on purpose: servers with prefix
        caching (vLLM, colibri, llama.cpp) then reuse the whole conversation prefix instead of re-reading it.
        Per-turn material (relevant lessons, mode reminders) goes into the user message instead."""
        self.messages[0] = {"role": "system", "content": self.system_prompt(self.text_mode)}
        self._prompt_dirty = False

    def invalidate_system_prompt(self) -> None:
        self._prompt_dirty = True

    def compact(self, focus: str = "", emergency: bool = False) -> str:
        if self.hooks and self.hooks.has("PreCompact"):
            self.hooks.run("PreCompact", self._hook_base() | {"trigger": "manual" if focus else "auto"})
        tools = None if self.text_mode else self.registry.schemas()
        before = self.context.count(self.messages, tools)
        convo = self.messages[1:]
        yours = sum(1 for m in convo if is_real_user_message(m))
        self.ui.compact_started(len(convo), yours, before)
        new, detail = self.context.compact(self.messages, self.summarizer or self.llm, focus=focus,
                                           todos=self.ctx.todos, tools=tools, emergency=emergency,
                                           on_progress=self.ui.compact_progress)
        after = self.context.count(new, tools)
        desc = compact_summary(getattr(self.context, "last", {}) or {}, before, after)
        self.ui.compact_finished(desc)
        self.messages = new
        self._new_epoch()
        if self.session:
            self.session.log_replace(self.messages[1:], "compact")
        self._emit("compact", description=desc, emergency=emergency)
        self.emit_context()
        return desc

    def run(self, prompt: str, images: list[tuple[str, str, str]] | None = None) -> TurnResult:
        """images: (media type, base64, label) attached to the prompt (@image.png mentions)."""
        self._prompt_images = images or []
        start = time.time()
        result = TurnResult()
        self._emit("subagent_start" if self.is_subagent else "turn_start", prompt=prompt[:300],
                   model=getattr(self.llm, "model", ""))
        try:
            self._run(prompt, result)
        except KeyboardInterrupt:
            self._repair_after_interrupt()
            result.status = "interrupted"
            self.ui.warn("Interrupted. Tell MUYAH-CODE what to do instead.")
        result.duration = time.time() - start
        self._emit("subagent_end" if self.is_subagent else "turn_end", status=result.status,
                   duration=round(result.duration, 2), tool_calls=result.tool_calls)
        self.emit_context()
        return result

    # ------------------------------------------------------------------ internals

    def _hook_base(self) -> dict:
        return {
            "session_id": self.session_id,
            "transcript_path": str(self.session.path) if self.session else "",
            "permission_mode": self.permissions.mode,
        }

    def _append(self, msg: dict) -> None:
        self.messages.append(msg)
        if self.session:
            self.session.log_message(msg)

    def _run(self, prompt: str, result: TurnResult) -> None:
        if self.hooks and self.hooks.has("UserPromptSubmit"):
            out = self.hooks.run("UserPromptSubmit", self._hook_base() | {"prompt": prompt})
            for w in out.warnings:
                self.ui.warn(w)
            if out.blocked:
                result.status = "blocked"
                result.text = out.reason
                self.ui.error(f"Prompt blocked by hook: {out.reason}")
                return
            if out.additional_context:
                prompt = f"{prompt}\n\n<context source=\"hook\">\n{out.additional_context}\n</context>"

        self.last_user_prompt = prompt
        if self._prompt_dirty or not self.messages[0].get("content"):
            self.refresh_system_prompt()
        if self.turn_context:
            extra = self.turn_context(prompt)
            if extra:
                prompt = f"{prompt}\n\n{extra}"
        images, self._prompt_images = self._prompt_images, []
        if images and self.sees_images():
            from muyah_code.llm.content import image_part

            parts = [{"type": "text", "text": prompt}]
            for media, b64, label in images:
                parts += [{"type": "text", "text": f"[{label}]"}, image_part(media, b64)]
            self._append({"role": "user", "content": parts})
        else:
            if images:
                prompt += "\n\n" + self._no_vision_note(", ".join(label for _, _, label in images))
            self._append({"role": "user", "content": prompt})

        seen: dict[str, int] = {}
        failures: dict[str, str] = {}
        continuations = 0
        empty_nudged = False
        stop_hook_rounds = 0

        for step in range(1, self.max_steps + 1):
            result.steps = step
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise KeyboardInterrupt
            if step > 1:
                self._take_queued_messages()
                if self._prompt_dirty:        # e.g. the mode changed (Shift+Tab) while it worked
                    self.refresh_system_prompt()
            if not self._within_budget():
                result.status = "budget"
                result.error = self._last_error
                return
            resp = self._call_llm()
            if resp is None:
                result.status = "error"
                result.error = self._last_error
                return

            calls = list(resp.tool_calls)
            parsed_from_text = False
            if not calls and resp.content and (self.text_mode or "<tool_call>" in resp.content
                                               or "<function=" in resp.content):
                calls, cleaned = parse_tool_calls(resp.content, self.registry.resolve)
                parsed_from_text = bool(calls)
                if parsed_from_text and not self.text_mode:
                    resp.content = cleaned

            if not calls:
                if not resp.content.strip():
                    if resp.finish_reason == "length" or empty_nudged:
                        result.status = "error"
                        result.error = "The model returned an empty response."
                        self.ui.error(result.error + " Try again, /compact, or a larger max_tokens.")
                        return
                    empty_nudged = True
                    self._append({"role": "user", "content": "(Your last reply was empty. Continue the task, "
                                                             "or give your final answer.)"})
                    continue
                # to_message() keeps provider-private data (e.g. Claude's thinking blocks) for replay
                self._append({"role": "assistant", "content": resp.content} if self.text_mode else resp.to_message())
                if resp.finish_reason == "length" and continuations < MAX_CONTINUATIONS:
                    continuations += 1
                    self._append({"role": "user", "content": "Your reply was cut off by the output limit. "
                                                             "Continue exactly where you stopped."})
                    continue
                hook_hint = self._stop_hook(stop_hook_rounds)
                if hook_hint:
                    stop_hook_rounds += 1
                    self._append({"role": "user", "content": hook_hint})
                    continue
                result.text = resp.content
                return

            # --- record the assistant message in the right shape for the protocol in use
            if self.text_mode:
                self._append({"role": "assistant", "content": resp.content})
            else:
                resp.tool_calls = calls
                self._append(resp.to_message())

            outputs = self._execute(calls, resp, result, seen, failures)
            if outputs is None:  # user denied without feedback: stop and hand control back
                result.status = "denied"
                return
            self._append_results(calls, outputs)
            self.escalation.step(len(outputs), sum(1 for o in outputs if o is not None and o.is_error))
            self._maybe_escalate()
            if result.status == "loop":
                self.ui.warn("Stopped: the model kept repeating the same tool call.")
                return
        result.status = "max_steps"
        self.ui.warn(f"Reached the step limit ({self.max_steps}). Say 'continue' to keep going.")

    _last_error: str | None = None

    def _within_budget(self) -> bool:
        """Before each model request: warn at 80% of a budget; at 100% ask whether to go on (headless: stop)."""
        budget = self.ctx.service("budget")
        if budget is None or not budget.active:
            return True
        level, message = budget.check()
        if level == "warn":
            self.ui.warn(message)
        if level != "over":
            return True
        self._emit("budget", state="over", message=message)
        if self.ui.headless:
            self._last_error = message + ". Raise budget.session_usd / budget.daily_usd (or --max-cost) to go on."
            self.ui.warn(self._last_error)
            return False
        answer = self.ui.ask_user(f"{message}. Keep going?", ["Continue (raise the limit by half)", "Stop this turn"])
        if answer.startswith("Continue"):
            budget.raise_limits()
            self.ui.info("Budget raised for this session: " + ", ".join(
                f"{name} ${limit:.2f}" for name, _, limit in budget.limits()))
            return True
        self._last_error = message + ". Stopped before the next request."
        return False

    def _maybe_escalate(self) -> None:
        """Hard failure signals -> the next bigger model takes over (it sees the failed attempts)."""
        reason = self.escalation.take_reason()
        if not reason or self.escalate is None:
            return
        step = self.escalate(self.llm)
        if step is None:
            return
        target, before, after = step
        self.llm = target
        self.ui.warn(f"Escalated {before} → {after}: {reason}")
        self._emit("escalate", source=before, target=after, reason=reason)

    def _call_llm(self) -> AssistantMessage | None:
        """One model request. If the provider fails after its own retries (rate limit, server error, network),
        the next `fallback` model answers this request instead, and the terminal says so."""
        original = self.llm
        tried: list[str] = []
        try:
            while True:
                self._fallback_error = None
                resp = self._request()
                if resp is not None or self._fallback_error is None:
                    return resp
                target = self._next_fallback(tried, original)
                if target is None:
                    return None
                self.llm = target
        finally:
            self.llm = original

    _fallback_error: BaseException | None = None

    def _next_fallback(self, tried: list[str], original):
        pool = self.ctx.service("models")
        if pool is None:
            return None
        from muyah_code.config import ConfigError

        for spec in pool.fallbacks():
            if spec in tried:
                continue
            tried.append(spec)
            try:
                client = pool.get(spec)
            except (ConfigError, ValueError) as e:
                self.ui.warn(f"Fallback {spec} skipped: {e}")
                continue
            if client is original or client is self.llm:
                continue
            kind = getattr(self._fallback_error, "kind", "error").replace("_", " ")
            self.ui.warn(f"Using the fallback model {pool.label(client)} for this request "
                         f"({kind} from {getattr(original, 'model', 'the main model')}).")
            self._emit("fallback", source=getattr(original, "model", ""), target=getattr(client, "model", ""),
                       reason=kind)
            return client
        return None

    def _request(self) -> AssistantMessage | None:
        overflow_retries = 0
        while True:
            if self.context.needs_compaction(self.messages, None if self.text_mode else self.registry.schemas()):
                self.compact()
            tools = None if self.text_mode else self.registry.schemas()
            max_tokens = self.context.completion_budget(self.messages, tools)
            self.ui.assistant_start()
            hider = _TagHider(self.ui.text) if self.text_mode else None
            opener = _OpenerFilter(hider.feed if hider else self.ui.text,
                                   lambda: self._emit("stock_opener_removed"))
            on_text = opener.feed
            on_reasoning = self.ui.reasoning
            meter = None
            if self.events is not None:
                from muyah_code.events import TokenMeter

                meter = TokenMeter(self.events, self.label)
                self.emit_context()
                self._emit("llm_start", model=getattr(self.llm, "model", ""))
                show_text, show_reasoning = on_text, on_reasoning

                def on_text(chunk, _show=show_text, _meter=meter):
                    _meter.feed(chunk)
                    _show(chunk)

                def on_reasoning(chunk, _show=show_reasoning, _meter=meter):
                    _meter.feed(chunk, thinking=True)
                    _show(chunk)
            started = time.time()
            previous_status = getattr(self.llm, "on_status", None)
            if meter is not None and hasattr(self.llm, "on_status"):
                def on_status(state, _ui=self.ui):
                    self._emit("llm_status", state=state)
                    _ui.model_status(state)
                self.llm.on_status = on_status
            try:
                resp = interruptible_call(self.llm.chat, self.messages, tools=tools, on_text=on_text,
                                          on_reasoning=on_reasoning, max_tokens=max_tokens, purpose=self.purpose,
                                          cancel=self.cancel_event)
                opener.flush()
                if hider:
                    hider.flush()
                if resp.content:
                    resp.content = strip_stock_opener(resp.content)
                if meter is not None:
                    meter.flush()
                    self._emit("llm_end", prompt_tokens=int(resp.usage.get("prompt_tokens") or 0),
                               completion_tokens=int(resp.usage.get("completion_tokens") or 0),
                               duration=round(time.time() - started, 3), calls=[c.name for c in resp.tool_calls])
                    self._emit_limits()
            except ToolsUnsupportedError as e:
                self.ui.assistant_end()
                self._llm_failed(meter, started, e)
                if self.tool_mode == "native":
                    self._last_error = f"Server rejected native tool calling: {e}"
                    self.ui.error(self._last_error)
                    return None
                self.ui.warn("Server has no native tool calling; switching to the text tool protocol.")
                self.text_mode = True
                self.refresh_system_prompt()
                continue
            except ContextOverflowError as e:
                self.ui.assistant_end()
                self._llm_failed(meter, started, e)
                overflow_retries += 1
                if overflow_retries > 2:
                    self._last_error = f"Context still too large after compaction: {e}"
                    self.ui.error(self._last_error)
                    return None
                self.ui.warn("The model's context window is full; compacting and retrying...")
                self.compact(emergency=True)
                continue
            except KeyboardInterrupt as e:
                self.ui.assistant_end()
                self._llm_failed(meter, started, e)
                raise
            except Exception as e:  # LLMError and anything unexpected from the SDK
                self.ui.assistant_end()
                self._llm_failed(meter, started, e)
                self._last_error = str(e)
                self.ui.error(f"Model request failed: {e}")
                from muyah_code.llm.pool import FALLBACK_KINDS

                if getattr(e, "kind", "") in FALLBACK_KINDS:
                    self._fallback_error = e
                return None
            finally:
                if hasattr(self.llm, "on_status"):
                    self.llm.on_status = previous_status
            self.ui.assistant_end()
            prompt_tokens = int(resp.usage.get("prompt_tokens") or 0)
            if prompt_tokens:
                self.context.calibrate(self.messages, tools, prompt_tokens)
            return resp

    def _emit_limits(self) -> None:
        """The provider's rate limits as of this reply (only when it reports them): for the live view."""
        limits = getattr(self.llm, "limits", None)
        if not limits:
            return
        from muyah_code.usage import parse_limits, reset_seconds

        rows = [{"name": r.name, "remaining": r.remaining, "limit": r.limit, "reset_s": reset_seconds(r.reset)}
                for r in parse_limits(limits)]
        if rows:
            self._emit("limits", model=getattr(self.llm, "model", ""), rows=rows)

    def _new_epoch(self) -> None:
        rewind = self.ctx.service("rewind")
        if rewind is not None and not self.is_subagent:
            rewind.new_epoch()

    def _take_queued_messages(self) -> None:
        """Messages you typed while it worked arrive between steps, as soon as the current step is done."""
        take = getattr(self.ui, "take_queued", None)
        messages = take() if callable(take) else []
        for msg in messages:
            self._emit("user_message", text=msg[:2000], queued=True)
            self._append({"role": "user", "content": msg})

    def _llm_failed(self, meter, started: float, error: BaseException) -> None:
        if meter is not None:
            meter.flush()
            self._emit("llm_end", prompt_tokens=0, completion_tokens=0, duration=round(time.time() - started, 3),
                       calls=[], error=(str(error) or type(error).__name__)[:300])

    def _stop_hook(self, rounds: int) -> str | None:
        event = "SubagentStop" if self.is_subagent else "Stop"
        if not self.hooks or not self.hooks.has(event) or rounds >= 3:
            return None
        out = self.hooks.run(event, self._hook_base() | {"stop_hook_active": rounds > 0})
        for w in out.warnings:
            self.ui.warn(w)
        if out.blocked:
            return f"<stop-hook>\n{out.reason}\n</stop-hook>\nAddress this before finishing."
        return None

    # ------------------------------------------------------------------ tool execution

    def _execute(self, calls: list[ToolCall], resp: AssistantMessage, result: TurnResult,
                 seen: dict[str, int], failures: dict[str, str]) -> list[ToolResult] | None:
        outputs: list[ToolResult | None] = [None] * len(calls)
        prepared: list[tuple[int, Tool, dict]] = []
        for i, call in enumerate(calls):
            result.tool_calls += 1
            if call.parse_error or call.name == "__invalid__":
                self.escalation.malformed_call()
                hint = ""
                if resp.finish_reason == "length":
                    hint = (" Your reply hit the output token limit, so the call was cut off. Write large files in "
                            "smaller pieces: create the file with part of the content, then append with Edit.")
                outputs[i] = ToolResult.error(f"Error: could not parse your call to {call.name}: {call.parse_error}."
                                              f"{hint} Send the call again with valid JSON arguments.")
                continue
            tool, args, errors = self.registry.validate(call.name, call.arguments)
            if tool is None or errors:
                self.escalation.malformed_call()
                schema = json.dumps(tool.parameters, separators=(",", ":"))[:1200] if tool else ""
                outputs[i] = ToolResult.error(
                    f"Error: invalid call to {call.name}: {'; '.join(errors)}." +
                    (f" Expected parameters schema: {schema}" if schema else ""))
                continue
            call.name = tool.name
            sig = tool.name + ":" + json.dumps(args, sort_keys=True, default=str)
            seen[sig] = seen.get(sig, 0) + 1
            if seen[sig] >= REPEAT_STOP:
                result.status = "loop"
                result.signals.append({"type": "loop", "tool": tool.name, "args": _short(args)})
                outputs[i] = ToolResult.error("Error: this exact call was repeated too many times; stopping.")
                continue
            prepared.append((i, tool, args))

        # every call the model made is visible as "requested" before anything runs
        ids: dict[int, int] = {}
        for i, call in enumerate(calls):
            ids[i] = self._new_tool_id()
            tool = self.registry.get(call.name)
            title = tool.title(call.arguments or {}) if tool and outputs[i] is None else call.name
            self._emit_tool_request(ids[i], call.name, title, call.arguments)
            if outputs[i] is not None:  # rejected before running (bad JSON, invalid arguments, loop)
                self._emit_tool_end(ids[i], call.name, outputs[i], 0.0)

        # Parallelize when every call is a pure read (no prompts, no side effects, no UI of its own).
        parallel = len(prepared) > 1 and all(
            t.kind == READ and t.is_read_only(a) and self.permissions.check(t, a, self.ctx).action == "allow"
            and not (self.hooks and self.hooks.has("PreToolUse"))
            for _, t, a in prepared
        )
        if parallel:
            def timed(i, t, a):
                self._emit_tool_start(ids[i], t.name, t.title(a), a)   # the moment it really starts
                t0 = time.time()
                res = self.registry.execute(t, a, self.ctx, self._max_output())
                took = time.time() - t0
                self._emit_tool_end(ids[i], t.name, res, took)          # and the moment it really ends
                return res, took

            with ThreadPoolExecutor(max_workers=min(6, len(prepared))) as pool:
                futures = {i: pool.submit(timed, i, t, a) for i, t, a in prepared}
                for i, t, a in prepared:
                    res, took = futures[i].result()
                    self.ui.tool_start(t.title(a))
                    self.ui.tool_end(t.title(a), res)
                    outputs[i] = res
        else:
            for n, (i, tool, args) in enumerate(prepared):
                res = self._run_one(tool, args, ids[i])
                if res is None:  # denied without feedback: skip the rest
                    for j, t, _ in prepared[n:]:
                        outputs[j] = ToolResult.error("Skipped: the user rejected a previous action in this batch.")
                        if j != i:
                            self._emit_tool_end(ids[j], t.name, outputs[j], 0.0)
                    self._append_results(calls, [o or ToolResult.error("Skipped.") for o in outputs])
                    return None
                outputs[i] = res

        for (i, tool, args) in prepared:
            res = outputs[i]
            assert res is not None
            sig = tool.name + ":" + json.dumps(args, sort_keys=True, default=str)
            if seen.get(sig, 0) >= REPEAT_WARN and result.status != "loop":
                res.content += (f"\n\n[Note: you have made this exact {tool.name} call {seen[sig]} times and the "
                                "result will not change. Try a different approach.]")
            nested = self.ctx.service("nested_instructions")
            if nested is not None and not res.is_error and args.get("file_path"):
                extra = nested.for_path(self.ctx.resolve(str(args["file_path"])))
                if extra:
                    res.content += extra
            self._track_signal(tool, args, res, result, failures)
            self.escalation.tool_result(tool.name, args, res.is_error)
        return [o if o is not None else ToolResult.error("Error: no result") for o in outputs]

    def _handoff(self, tool: Tool, args: dict, tool_id: int) -> ToolResult:
        """A delete: never executed. The user sees the exact command (and what it would remove) to run in
        their own terminal; the model is told so and must not try another way."""
        from muyah_code.risk import delete_targets

        command = args.get("command", "")
        targets = delete_targets(command, self.ctx.cwd)
        self.ui.handoff(command, targets)
        self._emit("tool_permission", id=tool_id, name=tool.name, state="handoff",
                   reason="deletes are left to the user", command=command[:500], targets=targets[:50])
        res = ToolResult(
            "Not run. MUYAH-CODE never deletes files or folders itself: the exact command was shown to the user "
            "to run in their own terminal if they want it. Do not try to delete another way (no scripts, other "
            "commands or tools). Continue with the rest of the task; if the deletion matters for it, tell the "
            "user what to run.", summary="delete handed to you to run", meta={"handoff": True})
        self._emit_tool_end(tool_id, tool.name, res, 0.0)
        return res

    def _refuse(self, title: str, name: str, message: str, tool_id: int | None = None) -> ToolResult:
        """A call that never ran (blocked, denied): shown as a failed line; the live view never shows it running."""
        tool_id = tool_id if tool_id is not None else self._new_tool_id()
        res = ToolResult.error(message)
        self.ui.tool_start(title)
        self.ui.tool_end(title, res)
        self._emit_tool_end(tool_id, name, res, 0.0)
        return res

    def _max_output(self) -> int:
        return self.context.tool_output_chars(self.max_tool_output_chars)

    def _run_one(self, tool: Tool, args: dict, tool_id: int | None = None) -> ToolResult | None:
        tool_id = tool_id if tool_id is not None else self._new_tool_id()
        title = tool.title(args)
        forced: str | None = None
        if self.hooks and self.hooks.has("PreToolUse"):
            out = self.hooks.run("PreToolUse", self._hook_base() | {"tool_name": tool.name, "tool_input": args},
                                 tool_name=tool.name)
            for w in out.warnings:
                self.ui.warn(w)
            if out.blocked:
                return self._refuse(title, tool.name, f"Blocked by PreToolUse hook: {out.reason}", tool_id)
            if out.updated_input:
                args = {**args, **out.updated_input}
                title = tool.title(args)
            forced = out.permission

        decision = self.permissions.check(tool, args, self.ctx)
        if forced == "allow" and decision.action != "deny":
            decision.action = "allow"
        elif forced == "ask" and decision.action == "allow":
            decision.action = "ask"

        if decision.action == "handoff":
            return self._handoff(tool, args, tool_id)
        if decision.action == "deny":
            self._emit("tool_permission", id=tool_id, name=tool.name, state="denied", reason=decision.reason)
            return self._refuse(title, tool.name, f"Permission denied: {decision.reason}.", tool_id)
        if decision.action == "ask":
            if self.ctx.headless:
                self._emit("tool_permission", id=tool_id, name=tool.name, state="denied",
                           reason="needs approval in a non-interactive run")
                return self._refuse(title, tool.name,
                                    f"Permission denied: {tool.name} needs approval, and this is a non-interactive "
                                    "run. It was not executed. Use a different approach or report what you would run.",
                                    tool_id)
            self._emit("tool_permission", id=tool_id, name=tool.name, state="asking", reason=decision.reason)
            rule = suggest_rule(tool, args, self.ctx)
            preview = tool.preview(args, self.ctx) or tool.permission_subject(args, self.ctx)
            reply = self.ui.ask_permission(PermissionRequest(tool.name, title, preview or "", decision.reason,
                                                             rule, tool.kind))
            self._emit("tool_permission", id=tool_id, name=tool.name,
                       state="denied" if reply.choice == "no" else "approved", choice=reply.choice)
            if reply.choice == "no":
                res = ToolResult.error(f"The user rejected this action and said: {reply.feedback}"
                                       if reply.feedback else "The user rejected this action.")
                self._emit_tool_end(tool_id, tool.name, res, 0.0)
                if not reply.feedback:
                    return None
                return res
            if reply.choice in ("always", "project"):
                self.permissions.add("allow", rule)
                if reply.choice == "project":
                    path = self.ctx.config.append_rule("allow", rule, scope="local")
                    self.ui.info(f"Saved rule {rule} to {path}")

        rewind = self.ctx.service("rewind")
        if rewind is not None and not tool.is_read_only(args):
            rewind.before_change()   # the turn's snapshot must be done before anything changes
        tid = self._show_tool_start(title, tool.name, tool_id, args)
        t0 = time.time()
        res = self.registry.execute(tool, args, self.ctx, self._max_output())
        self._show_tool_end(tid, title, tool.name, res, time.time() - t0)

        if self.hooks:
            event = "PostToolUseFailure" if res.is_error else "PostToolUse"
            if self.hooks.has(event):
                out = self.hooks.run(event, self._hook_base() | {
                    "tool_name": tool.name, "tool_input": args,
                    "tool_response": {"content": res.content[:4000], "is_error": res.is_error}}, tool_name=tool.name)
                for w in out.warnings:
                    self.ui.warn(w)
                if out.blocked:
                    res.content += f"\n\n[PostToolUse hook feedback]\n{out.reason}"
                if out.additional_context:
                    res.content += f"\n\n[hook context]\n{out.additional_context}"
        return res

    _prompt_images: list = []

    def sees_images(self) -> bool:
        """Can this agent's current model look at images? (the `vision` setting, the price table, the name)"""
        check = self.ctx.service("vision")
        return bool(check(self.llm)) if callable(check) else False

    def _no_vision_note(self, what: str) -> str:
        return (f"[{what}: not sent, because {getattr(self.llm, 'model', 'this model')} cannot see images. "
                'If it can, set "vision": true in settings.]')

    def _append_results(self, calls: list[ToolCall], outputs: list[ToolResult]) -> None:
        sees = self.sees_images()
        images: list[tuple[str, str, str]] = []
        texts = []
        for call, res in zip(calls, outputs, strict=True):
            text = res.content
            if res.images:
                if sees:
                    images += [(media, b64, f"image {i} from {call.name}") for i, (media, b64) in
                               enumerate(res.images, len(images) + 1)]
                else:
                    text += "\n" + self._no_vision_note(f"{len(res.images)} image(s)")
            texts.append(text)
        if self.text_mode:
            blocks = []
            for call, res, text in zip(calls, outputs, texts, strict=True):
                status = ' status="error"' if res.is_error else ""
                blocks.append(f'<tool_result name="{call.name}"{status}>\n{text}\n</tool_result>')
            self._append({"role": "user", "content": "\n\n".join(blocks)})
        else:
            for call, text in zip(calls, texts, strict=True):
                self._append({"role": "tool", "tool_call_id": call.id, "content": text})
        if images:
            from muyah_code.llm.content import image_part

            parts = [{"type": "text", "text": "Images returned by the tool calls above:"}]
            for media, b64, label in images:
                parts += [{"type": "text", "text": f"[{label}]"}, image_part(media, b64)]
            # "_images": part of the tool results for pruning and compaction (keys with "_" are never sent)
            self._append({"role": "user", "content": parts, "_images": True})

    def _repair_after_interrupt(self) -> None:
        """Make sure every native tool call has a result so the history stays valid."""
        if len(self.messages) < 2:
            return
        # find the last assistant message with tool_calls
        for idx in range(len(self.messages) - 1, 0, -1):
            m = self.messages[idx]
            if m.get("role") == "assistant" and m.get("tool_calls"):
                answered = {x.get("tool_call_id") for x in self.messages[idx + 1:] if x.get("role") == "tool"}
                for tc in m["tool_calls"]:
                    if tc["id"] not in answered:
                        self._append({"role": "tool", "tool_call_id": tc["id"],
                                      "content": "Interrupted by the user before this tool finished."})
                break
            if m.get("role") == "user":
                break
        self._append({"role": "user", "content": "[The user interrupted the previous response.]"})
        self._append({"role": "assistant", "content": "Stopped. Waiting for your next instruction."})

    def _track_signal(self, tool: Tool, args: dict, res: ToolResult, result: TurnResult,
                      failures: dict[str, str]) -> None:
        """Collect learning signals: failures, and failures that were later fixed."""
        key = _signal_key(tool, args)
        if res.is_error:
            result.signals.append({"type": "tool_error", "tool": tool.name, "args": _short(args),
                                   "error": res.content[:400]})
            failures[key] = res.content[:400]
        elif key in failures:
            result.signals.append({"type": "recovered", "tool": tool.name, "args": _short(args),
                                   "previous_error": failures.pop(key)})


def _signal_key(tool: Tool, args: dict) -> str:
    if tool.name == "Bash":
        words = str(args.get("command", "")).split()
        return "Bash:" + " ".join(words[:2])
    if "file_path" in args:
        return f"{tool.name}:{args['file_path']}"
    return tool.name


def _short(args: dict) -> str:
    s = json.dumps(args, default=str)
    return s if len(s) <= 300 else s[:297] + "..."
