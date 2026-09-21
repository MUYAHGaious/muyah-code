from muyah_code.agent.context import ContextManager, mechanical_digest
from muyah_code.llm.client import AssistantMessage
from muyah_code.llm.models import is_context_overflow, known_context_window


def convo(turns: int, tool_chars: int = 4000):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"request {i}"})
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * tool_chars})
        msgs.append({"role": "assistant", "content": f"done {i}"})
    return msgs


class FakeLLM:
    def __init__(self, text="SUMMARY", fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    def chat(self, messages, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("down")
        return AssistantMessage(content=self.text)


def test_window_scaling():
    small = ContextManager(8192, 4096)
    assert small.max_output == 2048  # never more than a quarter of a small window
    big = ContextManager(200_000, 8192)
    assert big.max_output == 8192
    assert small.keep_recent_results < big.keep_recent_results
    assert small.tool_output_chars(24000) < big.tool_output_chars(24000)


def test_needs_compaction_threshold():
    cm = ContextManager(8192, 1024)
    assert not cm.needs_compaction(convo(1, 100))
    assert cm.needs_compaction(convo(10, 4000))


def test_prune_only_old_tool_outputs():
    cm = ContextManager(8192, 1024)
    msgs = convo(8, 4000)
    n = cm.prune(msgs)
    tools = [m for m in msgs if m["role"] == "tool"]
    assert n == 8 - cm.keep_recent_results
    assert all(len(m["content"]) < 1000 for m in tools[:n])
    assert all(len(m["content"]) == 4000 for m in tools[n:])


def test_compact_summarizes_and_keeps_tool_pairs_intact():
    cm = ContextManager(8192, 1024)
    msgs = convo(12, 6000)
    llm = FakeLLM()
    new, desc = cm.compact(msgs, llm, focus="keep API notes")
    assert llm.calls == 1 and "LLM summary" in desc
    assert new[0]["content"] == "sys"
    assert "SUMMARY" in new[1]["content"]
    # every tool message still follows an assistant message that issued it
    ids = set()
    for m in new:
        if m.get("tool_calls"):
            ids = {tc["id"] for tc in m["tool_calls"]}
        if m["role"] == "tool":
            assert m["tool_call_id"] in ids
    assert cm.count(new) < cm.count(msgs)


def test_compact_falls_back_to_mechanical_digest():
    cm = ContextManager(8192, 1024)
    new, desc = cm.compact(convo(12, 6000), FakeLLM(fail=True), focus="x")
    assert "mechanical" in desc and "User requests so far" in new[1]["content"]


def test_calibration_learns_tokenizer_density():
    cm = ContextManager(32768, 2048)
    msgs = convo(3, 3000)
    raw = cm.raw_count(msgs)
    cm.calibrate(msgs, None, raw * 2)
    assert 1.9 < cm.ratio < 2.1
    assert cm.count(msgs) > raw * 1.9


def test_emergency_compaction_is_stricter():
    cm = ContextManager(8192, 1024)
    new, _ = cm.compact(convo(12, 6000), FakeLLM(), emergency=True)
    assert cm.ratio > 1.0
    assert cm.count(new) < cm.usable * 0.8


def test_completion_budget_never_exceeds_window():
    cm = ContextManager(8192, 4096)
    msgs = convo(4, 5000)
    assert cm.count(msgs) + cm.completion_budget(msgs) <= cm.window or cm.completion_budget(msgs) == 256


def test_known_windows_and_overflow_detection():
    assert known_context_window("Qwen/Qwen2.5-Coder-32B-Instruct-AWQ") == 32768
    assert known_context_window("deepseek-v4-colibri") is None or known_context_window("deepseek-chat") == 128000
    assert known_context_window("llama3.1:8b") == 131072
    assert known_context_window("my-model-128k") == 131072
    assert is_context_overflow("This model's maximum context length is 16384 tokens. However, you requested 17000")
    assert not is_context_overflow("invalid api key")


def test_mechanical_digest_mentions_requests():
    assert "request 0" in mechanical_digest(convo(2)[1:])


def test_compaction_never_grows_the_context_when_the_current_turn_is_the_big_part():
    """A small window where the latest turn's file reads are most of it: summarizing the 2 older messages
    used to make it bigger (11,289 -> 11,499) and compaction ran again on every step."""
    class Summarizer:
        def chat(self, messages, **kw):
            class R:
                content = "Summary: the user wants a notes CLI; cli.py and storage.py exist. " * 8
            return R()

    ctx = ContextManager(16384, 4096)
    msgs = [{"role": "system", "content": "s" * 12000}, {"role": "user", "content": "build a notes cli"},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "review it"}]
    for i in range(3):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "Read", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 9000})
    before = ctx.count(msgs)
    new, desc = ctx.compact(msgs, Summarizer())
    assert ctx.count(new) < before * 0.9, desc
    assert "shortened large tool outputs" in desc
    assert [m["role"] for m in new if m["role"] == "tool"]              # every tool result still answered


def test_the_compaction_report_is_one_clear_line():
    from muyah_code.agent.loop import compact_summary

    line = compact_summary({"kind": "summary", "count": 34, "yours": 12}, 11400, 3200)
    assert line == ("Compacted 34 older messages (12 of yours, 22 from the AI and its tools) into a summary · "
                    "11.4k → 3.2k tokens (−72%)")
    assert compact_summary({"kind": "pruned", "count": 5}, 9000, 6000) == "Shortened 5 old tool outputs · 9.0k → 6.0k tokens (−34%)"
