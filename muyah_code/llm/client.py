"""OpenAI-compatible chat client: streaming, retries, tool-call assembly, capability probing.

Works with vLLM, Ollama, LM Studio, llama.cpp server, OpenRouter, Groq, DeepSeek, OpenAI...

It speaks the protocol directly over httpx (the same HTTP library the OpenAI SDK uses) instead of through
the SDK: importing the SDK took ~1.7 s and its lazily imported resources another ~1 s on the first request,
most of MUYAH-CODE's startup time and a good part of its memory, for what is one JSON POST and a stream of
server-sent events.
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


STANDARD_TOOL_CALL_KEYS = {"index", "id", "type", "function"}


def _extra_fields(tool_call: dict) -> dict:
    """A tool call's non-standard fields (the server's own additions, e.g. Gemini's thought signature)."""
    return {k: v for k, v in (tool_call or {}).items() if k not in STANDARD_TOOL_CALL_KEYS and v is not None}


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
    """kind: rate_limit | server | network | timeout (worth trying another provider) | other."""

    def __init__(self, message: str = "", kind: str = "other"):
        super().__init__(message)
        self.kind = kind


class HTTPStatusFailure(Exception):
    """The server answered with an error status (or sent an error inside the stream)."""

    def __init__(self, status_code: int, body, headers=None):
        self.status_code = status_code
        self.body = body if isinstance(body, dict) else {"message": str(body or "")}
        self.headers = headers or {}
        err = self.body.get("error")
        self.message = self.body.get("message") or (err.get("message", "") if isinstance(err, dict) else str(err or ""))
        super().__init__(f"HTTP {status_code}: {self.message or self.body}")


def error_kind(e: BaseException) -> str:
    if isinstance(e, HTTPStatusFailure):
        return "rate_limit" if e.status_code == 429 else "server" if e.status_code >= 500 else "other"
    if isinstance(e, httpx.TimeoutException):
        return "timeout"
    if isinstance(e, httpx.HTTPError):
        return "network"
    return "other"


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


def _retryable(e: BaseException) -> bool:
    return isinstance(e, httpx.HTTPError) or (isinstance(e, HTTPStatusFailure) and
                                             (e.status_code == 429 or e.status_code >= 500))


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
        self._http = httpx.Client(timeout=http_timeout, headers={
            "Authorization": f"Bearer {self.api_key}", "X-Title": "MUYAH-CODE", "User-Agent": "MUYAH-CODE",
            **self.extra_headers})
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

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + "/" + path

    def list_models(self, timeout: float = 15.0) -> list[dict]:
        resp = self._http.get(self._url("models"), timeout=timeout)
        if resp.status_code >= 400:
            raise HTTPStatusFailure(resp.status_code, _json_or_text(resp), resp.headers)
        data = resp.json()
        return [m for m in (data.get("data") if isinstance(data, dict) else data) or [] if isinstance(m, dict)]

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
            except (HTTPStatusFailure, httpx.HTTPError) as e:
                status = getattr(e, "status_code", None)
                if status == 400:
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
                if status == 404:
                    raise LLMError(self._describe(e) + self._model_hint()) from e
                if status == 401:
                    raise LLMError(f"Authentication failed ({status}). Check api_key for {self.base_url}.") from e
                if status == 402 or is_billing_error(str(e)):   # e.g. OpenAI 429 insufficient_quota
                    raise LLMError(self._billing_message(e)) from e
                if _retryable(e):
                    if attempt > self.max_retries:
                        raise LLMError(self._describe(e), kind=error_kind(e)) from e
                    time.sleep(min(2 ** attempt, 20))
                    continue
                raise LLMError(self._describe(e), kind=error_kind(e)) from e

    def _post(self, body: dict) -> dict:
        """One non-streaming chat completion: the parsed JSON body."""
        if self.on_status:
            self.on_status("sent")
        resp = self._http.post(self._url("chat/completions"), json=body)
        self._remember_limits(resp.headers)
        data = _json_or_text(resp)
        if resp.status_code >= 400:
            raise HTTPStatusFailure(resp.status_code, data, resp.headers)
        if isinstance(data, dict) and data.get("error") and not data.get("choices"):
            raise HTTPStatusFailure(500, data, resp.headers)
        return data if isinstance(data, dict) else {}

    def _events(self, body: dict):
        """A streaming chat completion: yields each server-sent event's JSON."""
        if self.on_status:
            self.on_status("sent")
        with self._http.stream("POST", self._url("chat/completions"), json=body) as resp:
            self._remember_limits(resp.headers)
            if resp.status_code >= 400:
                resp.read()
                raise HTTPStatusFailure(resp.status_code, _json_or_text(resp), resp.headers)
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                if not payload:
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("error") and not event.get("choices"):
                    err = event["error"]
                    code = err.get("code") if isinstance(err, dict) else None
                    raise HTTPStatusFailure(code if isinstance(code, int) and code >= 400 else 500, event)
                yield event

    def _remember_limits(self, headers) -> None:
        found = limit_headers(headers)
        if found:
            self.limits, self.limits_at = found, time.time()

    def _chat_once(self, kwargs, on_text, on_reasoning) -> AssistantMessage:
        resp = self._post(kwargs)
        choices = resp.get("choices") or []
        if not choices:
            raise LLMError("Server returned no choices")
        choice = choices[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        visible, think = split_think(content)
        reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "") + think
        if on_reasoning and reasoning:
            on_reasoning(reasoning)
        if on_text and visible:
            on_text(visible)
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw = fn.get("arguments") or ""
            raw = raw if isinstance(raw, str) else json.dumps(raw)
            args, err = parse_arguments(raw)
            call = ToolCall(name=fn.get("name") or "", arguments=args, raw_arguments=raw, parse_error=err,
                            extra=_extra_fields(tc))
            if tc.get("id"):
                call.id = tc["id"]
            calls.append(call)
        return self._tag(AssistantMessage(visible, reasoning, calls, choice.get("finish_reason"),
                                          resp.get("usage") or {}))

    def _chat_stream(self, kwargs, on_text, on_reasoning) -> AssistantMessage:
        kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        stream = self._events(kwargs)
        try:
            first_event = next(stream, None)
        except HTTPStatusFailure as e:
            if e.status_code == 400 and "stream_options" in str(e):   # an older server without usage in streams
                kwargs.pop("stream_options")
                stream = self._events(kwargs)
                first_event = next(stream, None)
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

        def chunks():
            if first_event is not None:
                yield first_event
            yield from stream

        try:
            first = True
            for chunk in chunks():
                if first and self.on_status:
                    first = False
                    self.on_status("first_token")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta")
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                if not delta:
                    continue
                r = delta.get("reasoning_content") or delta.get("reasoning")
                if r:
                    rec_reasoning(r)
                if delta.get("content"):
                    filt.feed(delta["content"])
                for tcd in delta.get("tool_calls") or []:
                    idx = tcd.get("index") if tcd.get("index") is not None else len(pending)
                    slot = pending.setdefault(idx, {"id": None, "name": "", "args": "", "extra": {}})
                    if tcd.get("id"):
                        slot["id"] = tcd["id"]
                    _merge(slot["extra"], _extra_fields(tcd))
                    fn = tcd.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"])
        finally:
            filt.flush()
            stream.close()

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
        detail = getattr(e, "message", "") or str(e)
        prefix = f"HTTP {status}: " if status else ""
        if status in (524, 522, 504):
            return (f"HTTP {status}: the tunnel/proxy gave up waiting for the model. Cloudflare quick tunnels "
                    "drop requests that stay silent for ~100s (long prefill on big models). Use a tunnel without "
                    "that limit (the Pinggy URL printed by the MUYAH server notebook) or a smaller prompt/model.")
        if isinstance(e, httpx.HTTPError):
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


def _json_or_text(resp: httpx.Response):
    try:
        return resp.json()
    except ValueError:
        return {"message": resp.text[:2000]}
