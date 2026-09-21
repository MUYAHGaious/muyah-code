"""OpenAI-compatible chat client: streaming, retries, tool-call assembly, capability probing.

Works with vLLM, Ollama, LM Studio, llama.cpp server, OpenRouter, Groq, DeepSeek, OpenAI...
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import openai
from openai import OpenAI

from muyah_code.llm.models import is_billing_error, is_context_overflow, is_tools_unsupported

THINK_RE = re.compile(r"<think>([\s\S]*?)(?:</think>|$)")


def limit_headers(headers) -> dict:
    """The rate-limit related headers of a response (what /usage shows about the provider's limits)."""
    out = {}
    for k, v in dict(headers or {}).items():
        low = k.lower()
        if "ratelimit" in low or low == "retry-after":
            out[low] = v
    return out


ENDPOINT_KEY = "_endpoint"  # on assistant messages whose tool calls carry provider fields


def _extra_fields(obj) -> dict:
    """Fields the OpenAI SDK model did not define (the server's own additions), without empty values."""
    extra = getattr(obj, "model_extra", None) or {}
    return {k: v for k, v in extra.items() if v is not None}


def _merge(into: dict, new: dict) -> None:
    """Merge streamed pieces of provider fields (nested dicts merge; strings from later chunks append)."""
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(into.get(k), dict):
            _merge(into[k], v)
        elif isinstance(v, str) and isinstance(into.get(k), str) and into[k] != v:
            into[k] += v
        else:
            into[k] = v


class LLMError(Exception):
    pass


class ToolsUnsupportedError(LLMError):
    """The server rejected the `tools` parameter (e.g. vLLM without --enable-auto-tool-choice)."""


class ContextOverflowError(LLMError):
    """The prompt does not fit the model's context window."""


@dataclass
class ToolCall:
    name: str
    arguments: dict | None
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:12])
    raw_arguments: str = ""
    parse_error: str | None = None
    # provider fields to send back unchanged with this call, e.g. Gemini's
    # {"extra_content": {"google": {"thought_signature": "..."}}} (required for its next request)
    extra: dict = field(default_factory=dict)

    def signature(self) -> str:
        return self.name + ":" + json.dumps(self.arguments, sort_keys=True, default=str)


@dataclass
class AssistantMessage:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    # provider-private data to replay with this message (keys start with "_"; stripped for other providers)
    raw: dict = field(default_factory=dict)

    def to_message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        msg.update(self.raw)
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.raw_arguments or json.dumps(tc.arguments or {}),
                    },
                    **tc.extra,
                }
                for tc in self.tool_calls
            ]
        return msg


def estimate_tokens(obj: Any) -> int:
    """Cheap, model-agnostic token estimate (~3.5 chars/token for code-heavy text)."""
    if isinstance(obj, str):
        text = obj
    else:
        text = json.dumps(obj, default=str, ensure_ascii=False)
    return int(len(text) / 3.5) + 1


def parse_arguments(raw: str) -> tuple[dict | None, str | None]:
    raw = (raw or "").strip()
    if not raw:
        return {}, None
    try:
        val = json.loads(raw)
    except json.JSONDecodeError as e:
        # Some models wrap arguments in code fences or add trailing junk; take the first JSON object.
        try:
            start = raw.index("{")
            val, _ = json.JSONDecoder().raw_decode(raw[start:])
        except (ValueError, json.JSONDecodeError):
            return None, f"arguments are not valid JSON: {e}"
    if isinstance(val, str):  # double-encoded
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            return None, "arguments are a string, expected a JSON object"
    if not isinstance(val, dict):
        return None, f"arguments must be a JSON object, got {type(val).__name__}"
    return val, None


def split_think(text: str) -> tuple[str, str]:
    """Separate <think>...</think> reasoning (Qwen3, DeepSeek-R1) from the visible answer."""
    if "<think>" not in text:
        return text, ""
    reasoning = "\n".join(m.strip() for m in THINK_RE.findall(text))
    visible = THINK_RE.sub("", text).strip()
    return visible, reasoning


class _ThinkFilter:
    """Streams visible text while holding back <think> blocks."""

    def __init__(self, on_text: Callable[[str], None] | None, on_reasoning: Callable[[str], None] | None):
        self.on_text = on_text
        self.on_reasoning = on_reasoning
        self.buf = ""
        self.inside = False

    def feed(self, chunk: str) -> None:
        self.buf += chunk
        while self.buf:
            tag = "</think>" if self.inside else "<think>"
            idx = self.buf.find(tag)
            if idx == -1:
                # keep a tail that could be the start of a tag
                keep = len(tag) - 1
                emit, self.buf = (self.buf[:-keep], self.buf[-keep:]) if len(self.buf) > keep else ("", self.buf)
                self._emit(emit)
                return
            self._emit(self.buf[:idx])
            self.buf = self.buf[idx + len(tag):]
            self.inside = not self.inside

    def flush(self) -> None:
        self._emit(self.buf)
        self.buf = ""

    def _emit(self, text: str) -> None:
        if not text:
            return
        cb = self.on_reasoning if self.inside else self.on_text
        if cb:
            cb(text)


RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)


def make_client(cfg):
    """The right client for the configured API: native Claude, or any OpenAI-compatible server."""
    if (cfg.get("api") or "openai") == "anthropic":
        from muyah_code.llm.anthropic_client import AnthropicClient

        return AnthropicClient.from_config(cfg)
    return LLMClient.from_config(cfg)


class LLMClient:
    api = "openai"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "none",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 300,
        stream: bool = True,
        extra_headers: dict | None = None,
        max_retries: int = 3,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or "none"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.stream = stream
        self.extra_headers = extra_headers or {}
        self.max_retries = max_retries
        # called with "sent" when a request leaves and "first_token" when the reply starts (live view)
        self.on_status: Callable[[str], None] | None = None
        self.limits: dict = {}        # rate-limit headers of the last response (see /usage)
        self.limits_at = 0.0
        # called after every successful request: (client, purpose, usage dict, seconds) -> usage accounting
        self.on_call: Callable[[Any, str, dict, float], None] | None = None
        # timeout <= 0 means "wait as long as it takes": huge models on CPU/NVMe streaming (e.g. colibri)
        # can spend many minutes on prefill before the first token arrives.
        http_timeout = httpx.Timeout(None, connect=30.0) if timeout <= 0 else httpx.Timeout(timeout, connect=30.0)
        self._client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
            timeout=http_timeout,
            max_retries=0,
            default_headers={"X-Title": "MUYAH-CODE", **self.extra_headers},
        )
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}

    @classmethod
    def from_config(cls, cfg) -> LLMClient:
        return cls(
            base_url=cfg["base_url"],
            model=cfg["model"],
            api_key=cfg.get("api_key", "none"),
            temperature=float(cfg.get("temperature", 0.2)),
            max_tokens=int(cfg.get("max_tokens", 4096)),
            timeout=float(cfg.get("request_timeout", 300)),
            stream=bool(cfg.get("stream", True)),
            extra_headers=cfg.get("extra_headers") or {},
        )

    # ------------------------------------------------------------------ discovery

    def list_models(self, timeout: float = 15.0) -> list[dict]:
        resp = self._client.with_options(timeout=timeout).models.list()
        out = []
        for m in resp.data:
            d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
            out.append(d)
        return out

    def probe_context_window(self, timeout: float = 10.0) -> int | None:
        """vLLM reports max_model_len, OpenRouter context_length, llama.cpp n_ctx."""
        try:
            models = self.list_models(timeout=timeout)
        except Exception:
            return None
        for m in models:
            if m.get("id") != self.model:
                continue
            for key in ("max_model_len", "context_length", "context_window", "n_ctx", "max_context_length"):
                val = m.get(key) or (m.get("meta") or {}).get(key)
                if isinstance(val, int) and val > 0:
                    return val
        return None

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        purpose: str = "main",
    ) -> AssistantMessage:
        """purpose says what the call is for (main, subagent:<type>, compact, reflect...) in the usage log."""
        started = time.monotonic()
        messages = [self._outgoing(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        attempt = 0
        shrunk = False
        while True:
            attempt += 1
            try:
                if self.stream:
                    result = self._chat_stream(kwargs, on_text, on_reasoning)
                else:
                    result = self._chat_once(kwargs, on_text, on_reasoning)
                self.total_usage["requests"] += 1
                for k in ("prompt_tokens", "completion_tokens"):
                    self.total_usage[k] += int(result.usage.get(k) or 0)
                if self.on_call is not None:
                    self.on_call(self, purpose, result.usage, time.monotonic() - started)
                return result
            except openai.BadRequestError as e:
                msg = str(e).lower()
                if is_billing_error(msg):
                    raise LLMError(self._billing_message(e)) from e
                if is_context_overflow(msg):
                    if kwargs.get("max_tokens", 0) > 1024 and not shrunk:
                        # Often prompt + max_tokens > window: a smaller completion budget fits.
                        kwargs["max_tokens"] = max(1024, kwargs["max_tokens"] // 2)
                        shrunk = True
                        continue
                    raise ContextOverflowError(self._describe(e)) from e
                if tools and is_tools_unsupported(msg):
                    raise ToolsUnsupportedError(str(e)) from e
                raise LLMError(self._describe(e)) from e
            except openai.NotFoundError as e:
                raise LLMError(self._describe(e) + self._model_hint()) from e
            except openai.AuthenticationError as e:
                raise LLMError(f"Authentication failed ({e.status_code}). Check api_key for {self.base_url}.") from e
            except RETRYABLE as e:
                self._remember_limits(getattr(getattr(e, "response", None), "headers", None))
                if is_billing_error(str(e)):  # e.g. OpenAI 429 insufficient_quota: retrying cannot help
                    raise LLMError(self._billing_message(e)) from e
                if attempt > self.max_retries:
                    raise LLMError(self._describe(e)) from e
                time.sleep(min(2 ** attempt, 20))
            except openai.APIStatusError as e:
                if e.status_code == 402 or is_billing_error(str(e)):
                    raise LLMError(self._billing_message(e)) from e
                raise LLMError(self._describe(e)) from e
            except httpx.HTTPError as e:
                if attempt > self.max_retries:
                    raise LLMError(f"HTTP error talking to {self.base_url}: {e}") from e
                time.sleep(min(2 ** attempt, 20))

    def _create(self, kwargs):
        """chat.completions.create without the SDK's request "transform": it walks every message and
        tool schema in pure Python on each request (about a second per request on a long conversation,
        and it grows with every message). Our messages are already plain JSON, so they go in the body
        as they are."""
        kwargs = dict(kwargs)
        body = dict(kwargs.pop("extra_body", None) or {})
        body["messages"] = kwargs.pop("messages")
        if "tools" in kwargs:
            body["tools"] = kwargs.pop("tools")
        if self.on_status:
            self.on_status("sent")
        raw = self._client.chat.completions.with_raw_response.create(messages=[], extra_body=body, **kwargs)
        self._remember_limits(raw.headers)
        return raw.parse()

    def _remember_limits(self, headers) -> None:
        found = limit_headers(headers)
        if found:
            self.limits, self.limits_at = found, time.time()

    def _chat_once(self, kwargs, on_text, on_reasoning) -> AssistantMessage:
        resp = self._create(kwargs)
        if not resp.choices:
            raise LLMError("Server returned no choices")
        choice = resp.choices[0]
        msg = choice.message
        content = msg.content or ""
        visible, think = split_think(content)
        reasoning = (getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or "") + think
        if on_reasoning and reasoning:
            on_reasoning(reasoning)
        if on_text and visible:
            on_text(visible)
        calls = []
        for tc in msg.tool_calls or []:
            args, err = parse_arguments(tc.function.arguments)
            call = ToolCall(name=tc.function.name, arguments=args,
                            raw_arguments=tc.function.arguments or "", parse_error=err, extra=_extra_fields(tc))
            if tc.id:
                call.id = tc.id
            calls.append(call)
        usage = resp.usage.model_dump() if resp.usage else {}
        return self._tag(AssistantMessage(visible, reasoning, calls, choice.finish_reason, usage))

    def _chat_stream(self, kwargs, on_text, on_reasoning) -> AssistantMessage:
        kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        try:
            stream = self._create(kwargs)
        except openai.BadRequestError as e:
            if "stream_options" in str(e):
                kwargs.pop("stream_options")
                stream = self._create(kwargs)
            else:
                raise
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        pending: dict[int, dict] = {}
        finish = None
        usage: dict = {}

        def rec_reasoning(t: str) -> None:
            reasoning_parts.append(t)
            if on_reasoning:
                on_reasoning(t)

        def rec_text(t: str) -> None:
            text_parts.append(t)
            if on_text:
                on_text(t)

        filt = _ThinkFilter(rec_text, rec_reasoning)
        try:
            first = True
            for chunk in stream:
                if first and self.on_status:
                    first = False
                    self.on_status("first_token")
                if getattr(chunk, "usage", None):
                    usage = chunk.usage.model_dump()
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if choice.finish_reason:
                    finish = choice.finish_reason
                if delta is None:
                    continue
                r = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if r:
                    rec_reasoning(r)
                if delta.content:
                    filt.feed(delta.content)
                for tcd in delta.tool_calls or []:
                    idx = tcd.index if tcd.index is not None else len(pending)
                    slot = pending.setdefault(idx, {"id": None, "name": "", "args": "", "extra": {}})
                    if tcd.id:
                        slot["id"] = tcd.id
                    _merge(slot["extra"], _extra_fields(tcd))
                    if tcd.function:
                        if tcd.function.name:
                            slot["name"] += tcd.function.name
                        if tcd.function.arguments:
                            slot["args"] += tcd.function.arguments
        finally:
            filt.flush()
            close = getattr(stream, "close", None)
            if close:
                close()

        calls = []
        for idx in sorted(pending):
            slot = pending[idx]
            if not slot["name"]:
                continue
            args, err = parse_arguments(slot["args"])
            tc = ToolCall(name=slot["name"], arguments=args, raw_arguments=slot["args"], parse_error=err,
                          extra=slot["extra"])
            if slot["id"]:
                tc.id = slot["id"]
            calls.append(tc)
        return self._tag(AssistantMessage("".join(text_parts).strip(), "".join(reasoning_parts), calls, finish,
                                          usage))

    # ------------------------------------------------------------------ helpers

    def _tag(self, msg: AssistantMessage) -> AssistantMessage:
        """Remember which endpoint produced provider-specific tool-call fields (see `_outgoing`)."""
        if any(tc.extra for tc in msg.tool_calls):
            msg.raw[ENDPOINT_KEY] = self.base_url
        return msg

    def _outgoing(self, message: dict) -> dict:
        """The message as this server should see it: without provider-private keys ("_..."), and without
        another provider's tool-call fields (after /provider or /model switches mid-conversation)."""
        out = {k: v for k, v in message.items() if not k.startswith("_")}
        if out.get("tool_calls") and message.get(ENDPOINT_KEY) != self.base_url:
            out["tool_calls"] = [{k: v for k, v in tc.items() if k in ("id", "type", "function")}
                                 for tc in out["tool_calls"]]
        return out

    def _describe(self, e: Exception) -> str:
        status = getattr(e, "status_code", None)
        body = getattr(e, "body", None)
        detail = ""
        if isinstance(body, dict):
            err = body.get("error")
            detail = body.get("message") or (err.get("message", "") if isinstance(err, dict) else str(err or ""))
        detail = detail or str(e)
        prefix = f"HTTP {status}: " if status else ""
        if status in (524, 522, 504):
            return (f"HTTP {status}: the tunnel/proxy gave up waiting for the model. Cloudflare quick tunnels "
                    "drop requests that stay silent for ~100s (long prefill on big models). Use a tunnel without "
                    "that limit (the Pinggy URL printed by the MUYAH server notebook) or a smaller prompt/model.")
        if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
            return f"Cannot reach {self.base_url} ({e.__class__.__name__}). Is the server/tunnel up? Try /model or /config."
        return prefix + detail[:800]

    def _billing_message(self, e: Exception) -> str:
        text = str(e)
        low = text.lower()
        if any(s in low for s in ("free_tier", "free tier", "retrydelay", "per minute", "perminute", "per day")):
            m = re.search(r"retry(?:delay)?['\": ]+(\d+(?:\.\d+)?)s", text, re.IGNORECASE)
            wait = f" Try again in about {float(m.group(1)):.0f}s," if m else " Wait a minute and try again,"
            return (f"Rate limit reached for {self.model}: this key is on a free tier with a small number of "
                    f"requests per minute/day, and they are used up.{wait} use a different model or key, or "
                    f"enable billing in the provider's console. (The key itself works.)")
        return (f"The account behind this API key has no credits or quota left ({self.base_url}). The key itself "
                f"is fine: add credits / a payment method in the provider's console, then retry. "
                f"Details: {self._describe(e)[:200]}")

    def _model_hint(self) -> str:
        try:
            ids = [m.get("id") for m in self.list_models(timeout=10)]
        except Exception:
            return ""
        if not ids:
            return ""
        return f"\nModel '{self.model}' not found. Available: {', '.join(ids[:10])}. Switch with /model <name>."
