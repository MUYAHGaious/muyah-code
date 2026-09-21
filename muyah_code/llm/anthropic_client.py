"""Native Claude adapter (official `anthropic` SDK, Messages API).

MUYAH-CODE keeps its conversation in OpenAI chat format; this adapter translates to and from the
Messages API so Claude gets its full feature set: adaptive thinking, prompt caching, refusal fallbacks.

Translation rules
  * system messages -> top-level `system`
  * assistant tool_calls -> `tool_use` blocks; consecutive `tool` messages -> ONE user message of
    `tool_result` blocks (splitting them would teach Claude to stop making parallel calls)
  * Claude's raw content (incl. thinking blocks) is stored on the assistant message under the private key
    `_anthropic_content` and replayed unchanged on the next request, as the API requires between tool calls.
    Keys starting with "_" are stripped before messages are sent to any other provider.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from muyah_code.llm.client import AssistantMessage, ContextOverflowError, LLMError, ToolCall
from muyah_code.llm.models import is_billing_error, is_context_overflow

RAW_KEY = "_anthropic_content"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
NO_SAMPLING_PARAMS = ("claude-opus-5", "claude-sonnet-5", "claude-fable", "claude-mythos", "claude-opus-4-7",
                      "claude-opus-4-8")
EFFORT_MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7")
FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5-1")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content or "")


def _safe_blocks(blocks: list[dict]) -> list[dict]:
    """After a mid-output fallback, drop thinking/tool_use blocks that precede the last `fallback` marker."""
    idx = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=-1)
    if idx < 0:
        return blocks
    drop = {"thinking", "redacted_thinking", "tool_use"}
    return [b for i, b in enumerate(blocks) if i > idx or (b.get("type") not in drop)]


def to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """OpenAI-format history -> (system, Messages API messages)."""
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system_parts.append(_content_text(m.get("content")))
            continue
        if role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""),
                     "content": _content_text(m.get("content")) or "(no output)"}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) and \
                    all(b.get("type") == "tool_result" for b in out[-1]["content"]):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue
        if role == "assistant":
            raw = m.get(RAW_KEY)
            if raw:
                blocks = _safe_blocks([dict(b) for b in raw])
            else:
                blocks = []
                text = _content_text(m.get("content"))
                if text.strip():
                    blocks.append({"type": "text", "text": text})
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append({"type": "tool_use", "id": tc.get("id"), "name": fn.get("name"),
                                   "input": args if isinstance(args, dict) else {}})
            if not blocks:
                blocks = [{"type": "text", "text": "(no response)"}]
            out.append({"role": "assistant", "content": blocks})
            continue
        # user
        text = _content_text(m.get("content"))
        if out and out[-1]["role"] == "user":
            prev = out[-1]["content"]
            if isinstance(prev, str):
                out[-1]["content"] = [{"type": "text", "text": prev}]
            out[-1]["content"].append({"type": "text", "text": text})
        else:
            out.append({"role": "user", "content": text})
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(conversation start)"})
    return "\n\n".join(p for p in system_parts if p.strip()), out


def to_anthropic_tools(tools: list[dict] | None) -> list[dict]:
    result = []
    for t in tools or []:
        fn = t.get("function", t)
        result.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            # stream large tool inputs (file contents) as generated; MUYAH validates every input itself
            "eager_input_streaming": True,
        })
    return result


def _block_dict(block) -> dict:
    d = block.model_dump(mode="json", exclude_none=True) if hasattr(block, "model_dump") else dict(block)
    d.pop("parsed_output", None)
    return d


FINISH = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length",
          "model_context_window_exceeded": "length", "refusal": "refusal", "pause_turn": "stop"}


class AnthropicClient:
    """Same interface as LLMClient (chat / list_models / probe_context_window / total_usage)."""

    api = "anthropic"

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.anthropic.com",
                 max_tokens: int = 16000, timeout: float = 600, extra_headers: dict | None = None,
                 effort: str | None = None, fallbacks: bool = True, max_retries: int = 3):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - dependency is declared in pyproject
            raise LLMError("Claude support needs the 'anthropic' package: pip install anthropic") from e
        self._anthropic = anthropic
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = None
        self.effort = effort
        self.fallbacks = fallbacks
        self.stream = True
        self.on_status: Callable[[str], None] | None = None  # "sent" / "first_token" (live view)
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url, max_retries=max_retries,
                                           timeout=None if timeout <= 0 else timeout,
                                           default_headers={**(extra_headers or {})})
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}

    @classmethod
    def from_config(cls, cfg) -> AnthropicClient:
        return cls(model=cfg["model"], api_key=cfg.get("api_key", ""), base_url=cfg["base_url"],
                   max_tokens=int(cfg.get("max_tokens", 16000)), timeout=float(cfg.get("request_timeout", 600)),
                   extra_headers=cfg.get("extra_headers") or {}, effort=cfg.get("effort"),
                   fallbacks=bool(cfg.get("fallbacks", True)))

    # ------------------------------------------------------------------ discovery

    def list_models(self, timeout: float = 15.0) -> list[dict]:
        page = self._client.with_options(timeout=timeout).models.list()
        return [{"id": m.id, "max_input_tokens": getattr(m, "max_input_tokens", None)} for m in page]

    def probe_context_window(self, timeout: float = 10.0) -> int | None:
        try:
            info = self._client.with_options(timeout=timeout).models.retrieve(self.model)
        except Exception:
            return None
        val = getattr(info, "max_input_tokens", None)
        return int(val) if isinstance(val, int) and val > 0 else None

    # ------------------------------------------------------------------ chat

    def _params(self, messages, tools, max_tokens) -> dict:
        system, msgs = to_anthropic(messages)
        params: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens or self.max_tokens,
                                  "messages": msgs, "cache_control": {"type": "ephemeral"}}
        if system:
            params["system"] = system
        if tools:
            params["tools"] = to_anthropic_tools(tools)
        effort = self.effort or ("xhigh" if self.model.startswith(EFFORT_MODELS) else None)
        if effort:
            params["output_config"] = {"effort": effort}
        return params

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             on_text: Callable[[str], None] | None = None, on_reasoning: Callable[[str], None] | None = None,
             max_tokens: int | None = None, temperature: float | None = None) -> AssistantMessage:
        a = self._anthropic
        params = self._params(messages, tools, max_tokens)
        use_fallbacks = self.fallbacks and self.model.startswith(FALLBACK_MODELS)
        try:
            if use_fallbacks:
                cm = self._client.beta.messages.stream(**params, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                cm = self._client.messages.stream(**params)
            if self.on_status:
                self.on_status("sent")
            with cm as stream:
                first = True
                for event in stream:
                    if first and self.on_status:
                        first = False
                        self.on_status("first_token")
                    if event.type == "text" and on_text:
                        on_text(event.text)
                    elif (event.type == "content_block_delta" and on_reasoning
                          and getattr(event.delta, "type", "") == "thinking_delta"):
                        on_reasoning(event.delta.thinking)
                final = stream.get_final_message()
        except ValueError as e:  # the SDK could not parse a streamed tool input at all
            raise LLMError(f"Claude returned an unparseable tool call: {e}") from e
        except a.BadRequestError as e:
            msg = str(e)
            if is_billing_error(msg):
                raise LLMError("Your Anthropic credit balance is too low (the key itself is fine). Add credits at "
                               "https://console.anthropic.com/settings/billing, then retry.") from e
            if is_context_overflow(msg) or "prompt is too long" in msg.lower():
                raise ContextOverflowError(msg) from e
            raise LLMError(f"Claude rejected the request: {getattr(e, 'message', msg)}") from e
        except a.AuthenticationError as e:
            raise LLMError("Claude API key was rejected. Run /provider to set a new key.") from e
        except a.PermissionDeniedError as e:
            raise LLMError(f"This API key cannot use {self.model}: {getattr(e, 'message', e)}") from e
        except a.NotFoundError as e:
            raise LLMError(f"Model '{self.model}' was not found. Try /models or /provider.") from e
        except a.RateLimitError as e:
            raise LLMError("Claude rate limit reached; wait a moment and retry.") from e
        except a.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {getattr(e, 'message', e)}") from e
        except a.APIConnectionError as e:
            raise LLMError(f"Cannot reach the Claude API: {e}") from e
        return self._convert(final)

    def _convert(self, final) -> AssistantMessage:
        blocks = [_block_dict(b) for b in final.content]
        stop = final.stop_reason or "end_turn"
        if stop == "refusal":
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            note = "[Claude declined this request.]"
            return AssistantMessage(content=f"{text}\n\n{note}".strip() if text else note, finish_reason="refusal",
                                    usage=self._usage(final))
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        reasoning = "\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking").strip()
        calls = []
        for b in blocks:
            if b.get("type") != "tool_use":
                continue
            args = b.get("input")
            call = ToolCall(name=b.get("name", ""), arguments=args if isinstance(args, dict) else None,
                            raw_arguments=json.dumps(args if isinstance(args, dict) else {}))
            call.id = b.get("id") or call.id
            if stop == "max_tokens":
                call.parse_error = "the tool input was cut off by the output limit"
            elif not isinstance(args, dict):
                call.parse_error = "tool input is not a JSON object"
            calls.append(call)
        msg = AssistantMessage(content=text, reasoning=reasoning, tool_calls=calls,
                               finish_reason=FINISH.get(stop, "stop"), usage=self._usage(final))
        msg.raw = {RAW_KEY: blocks}
        return msg

    def _usage(self, final) -> dict:
        u = final.usage
        prompt = int((u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0)
                     + (getattr(u, "cache_creation_input_tokens", 0) or 0))
        out = int(u.output_tokens or 0)
        self.total_usage["requests"] += 1
        self.total_usage["prompt_tokens"] += prompt
        self.total_usage["completion_tokens"] += out
        return {"prompt_tokens": prompt, "completion_tokens": out}
