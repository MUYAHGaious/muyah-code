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
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from muyah_code.agent.context import ContextManager
from muyah_code.hooks import HookRunner
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


def interruptible_call(chat, messages, on_text=None, on_reasoning=None, **kwargs):
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


def result_detail(name: str, res: ToolResult) -> dict:
    """What the live view shows about a finished tool: output (tail for commands), exit code, diff size."""
    detail: dict = {}
    content = res.content or ""
    if name in ("Bash", "BashOutput") or name.startswith("mcp__"):
        detail["output"] = content[-1500:]
    elif name not in ("Read", "Write", "Edit"):
        detail["output"] = content[:1200]
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
        self._tool_seq = 0

    # ------------------------------------------------------------------ events (for /viz)

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
        self._emit("tool_start", id=tool_id, name=name, title=title[:200], args=brief_args(args))

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
        new, desc = self.context.compact(self.messages, self.llm, focus=focus, todos=self.ctx.todos,
                                         tools=tools, emergency=emergency)
        self.messages = new
        if self.session:
            self.session.log_replace(self.messages[1:], "compact")
        self._emit("compact", description=desc, emergency=emergency)
        self.emit_context()
        return desc

    def run(self, prompt: str) -> TurnResult:
        start = time.time()
        result = TurnResult()
        self._emit("subagent_start" if self.is_subagent else "turn_start", prompt=prompt[:300])
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
        checkpoints = self.ctx.service("checkpoints")
        if checkpoints is not None:
            checkpoints.begin_turn(self.last_user_prompt)
        self._append({"role": "user", "content": prompt})

        seen: dict[str, int] = {}
        failures: dict[str, str] = {}
        continuations = 0
        empty_nudged = False
        stop_hook_rounds = 0

        for step in range(1, self.max_steps + 1):
            result.steps = step
            if step > 1:
                self._take_queued_messages()
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
            if result.status == "loop":
                self.ui.warn("Stopped: the model kept repeating the same tool call.")
                return
        result.status = "max_steps"
        self.ui.warn(f"Reached the step limit ({self.max_steps}). Say 'continue' to keep going.")

    _last_error: str | None = None

    def _call_llm(self) -> AssistantMessage | None:
        overflow_retries = 0
        while True:
            if self.context.needs_compaction(self.messages, None if self.text_mode else self.registry.schemas()):
                self.ui.info("Context is getting full; compacting...")
                self.ui.info(self.compact())
            tools = None if self.text_mode else self.registry.schemas()
            max_tokens = self.context.completion_budget(self.messages, tools)
            self.ui.assistant_start()
            hider = _TagHider(self.ui.text) if self.text_mode else None
            on_text = hider.feed if hider else self.ui.text
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
                self.llm.on_status = lambda state: self._emit("llm_status", state=state)
            try:
                resp = interruptible_call(self.llm.chat, self.messages, tools=tools, on_text=on_text,
                                          on_reasoning=on_reasoning, max_tokens=max_tokens)
                if hider:
                    hider.flush()
                if meter is not None:
                    meter.flush()
                    self._emit("llm_end", prompt_tokens=int(resp.usage.get("prompt_tokens") or 0),
                               completion_tokens=int(resp.usage.get("completion_tokens") or 0),
                               duration=round(time.time() - started, 3), calls=[c.name for c in resp.tool_calls])
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
                self.ui.info(self.compact(emergency=True))
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
                return None
            finally:
                if hasattr(self.llm, "on_status"):
                    self.llm.on_status = previous_status
            self.ui.assistant_end()
            prompt_tokens = int(resp.usage.get("prompt_tokens") or 0)
            if prompt_tokens:
                self.context.calibrate(self.messages, tools, prompt_tokens)
            return resp

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
                hint = ""
                if resp.finish_reason == "length":
                    hint = (" Your reply hit the output token limit, so the call was cut off. Write large files in "
                            "smaller pieces: create the file with part of the content, then append with Edit.")
                outputs[i] = ToolResult.error(f"Error: could not parse your call to {call.name}: {call.parse_error}."
                                              f"{hint} Send the call again with valid JSON arguments.")
                continue
            tool, args, errors = self.registry.validate(call.name, call.arguments)
            if tool is None or errors:
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
            self._track_signal(tool, args, res, result, failures)
        return [o if o is not None else ToolResult.error("Error: no result") for o in outputs]

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

    def _append_results(self, calls: list[ToolCall], outputs: list[ToolResult]) -> None:
        if self.text_mode:
            blocks = []
            for call, res in zip(calls, outputs, strict=True):
                status = ' status="error"' if res.is_error else ""
                blocks.append(f'<tool_result name="{call.name}"{status}>\n{res.content}\n</tool_result>')
            self._append({"role": "user", "content": "\n\n".join(blocks)})
            return
        for call, res in zip(calls, outputs, strict=True):
            self._append({"role": "tool", "tool_call_id": call.id, "content": res.content})

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
