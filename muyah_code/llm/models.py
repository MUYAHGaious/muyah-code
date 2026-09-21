"""Context-window knowledge for when the server does not report it.

Resolution order used by the app (first hit wins):
    1. explicit config  "context_window" (global, per profile, or --context-window)
    2. server probe     /v1/models (vLLM max_model_len, OpenRouter context_length, llama.cpp n_ctx...)
    3. this table       matched against the model id
    4. DEFAULT_WINDOW   conservative fallback
"""

from __future__ import annotations

import re

DEFAULT_WINDOW = 16384

# (regex on lowercased model id, window). First match wins, so specific patterns go first.
KNOWN_WINDOWS: list[tuple[str, int]] = [
    (r"claude-(fable|mythos)|claude-(opus|sonnet)-5|claude-opus-4-[678]|claude-sonnet-4-6", 1_000_000),
    (r"claude", 200_000),
    (r"gemini-(1\.5|2|2\.5|3)", 1_000_000),
    (r"gpt-4\.1", 1_000_000),
    (r"gpt-5", 400_000),
    (r"gpt-4o|gpt-4-turbo|o1|o3|o4", 128_000),
    (r"gpt-4", 8_192),
    (r"gpt-3\.5", 16_385),
    (r"deepseek.*(v3|r1|chat|reasoner|coder-v2)", 128_000),
    (r"deepseek-coder", 16_384),
    (r"qwen3-coder", 262_144),
    (r"qwen3", 131_072),
    (r"qwen2\.5-coder-(0\.5|1\.5|3)b", 32_768),
    (r"qwen2\.5", 32_768),
    (r"qwq", 131_072),
    (r"llama-?3\.[1-3]|llama-?4", 131_072),
    (r"llama-?3", 8_192),
    (r"codellama", 16_384),
    (r"mistral-large|mistral-medium|mistral-small-3|codestral", 131_072),
    (r"mixtral", 32_768),
    (r"mistral", 32_768),
    (r"gemma-?3", 131_072),
    (r"gemma-?2", 8_192),
    (r"phi-?4", 16_384),
    (r"phi-?3.*128k", 131_072),
    (r"kimi|moonshot", 131_072),
    (r"glm-4", 131_072),
    (r"grok", 131_072),
    (r"starcoder2", 16_384),
]
_WINDOW_SUFFIX = re.compile(r"(\d+)k\b")


def known_context_window(model: str) -> int | None:
    m = (model or "").lower()
    # explicit hints in names like "...-128k" or ":32k"
    hint = _WINDOW_SUFFIX.search(m)
    if hint and int(hint.group(1)) in (4, 8, 16, 32, 64, 128, 200, 256, 1000):
        return int(hint.group(1)) * 1024
    for pattern, window in KNOWN_WINDOWS:
        if re.search(pattern, m):
            return window
    return None


def is_context_overflow(message: str) -> bool:
    msg = message.lower()
    return any(s in msg for s in (
        "maximum context length", "context length", "context_length_exceeded", "context window",
        "too many tokens", "prompt is too long", "input is too long", "reduce the length",
        "exceeds the model", "tokens exceed", "max_model_len",
    ))


BILLING_PATTERNS = ("credit balance", "insufficient_quota", "exceeded your current quota", "billing",
                    "payment required", "insufficient balance", "insufficient credits", "out of credits",
                    "quota exceeded", "requires more credits", "top up")


def is_billing_error(message: str) -> bool:
    """The account cannot pay (no credits / quota). Retrying or switching models will not help."""
    low = message.lower()
    return any(p in low for p in BILLING_PATTERNS)
