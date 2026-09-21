"""Strict parser for tool calls written as text (servers without native function calling).

Accepted forms, and ONLY these:
  1. <tool_call>{"name": "...", "arguments": {...}}</tool_call>          (Hermes / Qwen2.5)
  2. <tool_call><function=Name><parameter=k>v</parameter></function></tool_call>   (Qwen3-Coder XML)
  3. ```tool_call / ```json fenced blocks holding {"name": ..., "arguments": ...}
  4. A bare JSON object {"name": <registered tool>, "arguments"|"parameters": {...}}

Deliberately NOT accepted: ordinary code blocks, "File: x.py" headers, or anything else
that merely *looks* like content. The old agent turned those into write_file calls and
littered projects with junk; a tool call must be an explicit, named, registered call.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from muyah_code.llm.client import ToolCall

Resolver = Callable[[str], "str | None"]

_DECODER = json.JSONDecoder(strict=False)  # tolerate raw newlines inside strings (file contents)
TOOL_CALL_TAG = re.compile(r"<tool_call>([\s\S]*?)(?:</tool_call>|$)")
FUNCTION_XML = re.compile(r"<function=([\w.\-]+)>([\s\S]*?)(?:</function>|$)")
PARAM_XML = re.compile(r"<parameter=([\w.\-]+)>\n?([\s\S]*?)\n?</parameter>")
FENCE = re.compile(r"```(tool_call|tool_code|json)?[ \t]*\n([\s\S]*?)```")


def _loads(text: str):
    text = text.strip()
    if not text:
        raise ValueError("empty")
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object")
    try:
        obj, _ = _DECODER.raw_decode(text[start:])
        return obj
    except json.JSONDecodeError:
        # Common small-model slip: trailing commas.
        cleaned = re.sub(r",\s*([}\]])", r"\1", text[start:])
        obj, _ = _DECODER.raw_decode(cleaned)
        return obj


def _coerce_xml_value(v: str):
    s = v.strip()
    if s in ("true", "false"):
        return s == "true"
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if s[:1] in "[{":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return v


def _call_from_obj(obj, resolve: Resolver, require_args_key: bool) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    args = None
    if not name and isinstance(obj.get("function"), dict):
        fn = obj["function"]
        name = fn.get("name")
        args = fn.get("arguments", fn.get("parameters"))
    if not isinstance(name, str):
        return None
    canonical = resolve(name)
    if canonical is None:
        return None
    has_key = "arguments" in obj or "parameters" in obj
    if args is None:
        if require_args_key and not has_key:
            return None
        args = obj.get("arguments", obj.get("parameters"))
        if args is None:
            args = {k: v for k, v in obj.items() if k not in ("name", "type", "id")}
    err = None
    if isinstance(args, str):
        try:
            args = json.loads(args, strict=False)
        except json.JSONDecodeError as e:
            args, err = None, f"arguments are not valid JSON: {e}"
    if args is not None and not isinstance(args, dict):
        args, err = None, "arguments must be a JSON object"
    return ToolCall(name=canonical, arguments=args, raw_arguments=json.dumps(args) if args else "", parse_error=err)


def _xml_calls(body: str, resolve: Resolver) -> list[ToolCall]:
    calls = []
    for fname, inner in FUNCTION_XML.findall(body):
        canonical = resolve(fname)
        if canonical is None:
            continue
        args = {k: _coerce_xml_value(v) for k, v in PARAM_XML.findall(inner)}
        calls.append(ToolCall(name=canonical, arguments=args, raw_arguments=json.dumps(args)))
    return calls


def parse_tool_calls(text: str, resolve: Resolver) -> tuple[list[ToolCall], str]:
    """Return (calls, text_with_calls_removed)."""
    if not text:
        return [], ""
    calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []

    # 1 + 2: <tool_call> blocks
    for m in TOOL_CALL_TAG.finditer(text):
        body = m.group(1)
        found = _xml_calls(body, resolve) if "<function=" in body else []
        if not found:
            try:
                c = _call_from_obj(_loads(body), resolve, require_args_key=False)
            except (ValueError, json.JSONDecodeError):
                c = None
            if c is None and body.strip():
                # Explicit tool_call tag but unparseable: surface as an error so the model can retry.
                c = ToolCall(name="__invalid__", arguments=None, raw_arguments=body.strip()[:500],
                             parse_error="could not parse <tool_call> body as JSON {\"name\":..., \"arguments\":{...}}")
            found = [c] if c else []
        if found:
            calls.extend(found)
            spans.append(m.span())
    if calls:
        return calls, _strip(text, spans)

    # 2b: bare Qwen3-Coder <function=...> blocks
    if "<function=" in text:
        calls = _xml_calls(text, resolve)
        if calls:
            return calls, FUNCTION_XML.sub("", text).strip()

    # 3: fenced blocks
    for m in FENCE.finditer(text):
        lang = m.group(1) or ""
        try:
            obj = _loads(m.group(2))
        except (ValueError, json.JSONDecodeError):
            continue
        objs = obj if isinstance(obj, list) else [obj]
        found = [c for o in objs if (c := _call_from_obj(o, resolve, require_args_key=lang != "tool_call"))]
        if found:
            calls.extend(found)
            spans.append(m.span())
    if calls:
        return calls, _strip(text, spans)

    # 4: bare JSON objects naming a registered tool, with an explicit arguments key
    pos = 0
    while True:
        idx = text.find("{", pos)
        if idx == -1:
            break
        try:
            obj, length = _DECODER.raw_decode(text[idx:])
        except json.JSONDecodeError:
            pos = idx + 1
            continue
        c = _call_from_obj(obj, resolve, require_args_key=True)
        if c:
            calls.append(c)
            spans.append((idx, idx + length))
        pos = idx + length
    return calls, _strip(text, spans)


def _strip(text: str, spans: list[tuple[int, int]]) -> str:
    out, last = [], 0
    for s, e in sorted(spans):
        out.append(text[last:s])
        last = e
    out.append(text[last:])
    return "".join(out).strip()


TEXT_PROTOCOL_INSTRUCTIONS = """\
# How to call tools
This server has no native function calling, so you call tools by writing a tool_call block:

<tool_call>
{"name": "Read", "arguments": {"file_path": "src/app.py"}}
</tool_call>

Rules:
- The block must contain one JSON object with "name" and "arguments". Use valid JSON (escape quotes and backslashes).
- You may emit several <tool_call> blocks in one reply; they run in order.
- After your tool calls, STOP and wait. Results come back in <tool_result> blocks in the next message.
- Never invent tool results. Code blocks in normal text are NOT executed and do NOT create files; use Write/Edit.
- When the task is done, reply with plain text and no tool_call block.
"""
