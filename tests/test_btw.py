"""/btw side questions (never added to the conversation) and editing queued messages while it works."""

import io

from fakeserver import FakeOpenAI, reply
from rich.console import Console
from test_repl import run_session

from muyah_code.app import consistent_prefix
from muyah_code.ui.terminal import TerminalUI


def test_btw_answers_without_changing_the_conversation(project):
    script = [reply("Done."), reply("We are on step 2 of 3."), reply("Next answer.")]
    code, out, app = run_session(project, ["do the thing", "/btw where are we?", "and now?", "/exit"], script)
    assert "We are on step 2 of 3." in out and "not added to the conversation" in out
    user_texts = [m["content"] for m in app.agent.messages if m["role"] == "user"]
    assert not any("where are we?" in (t if isinstance(t, str) else "") for t in user_texts)
    assert not any("We are on step 2" in (m.get("content") or "") for m in app.agent.messages
                   if m["role"] == "assistant")
    assert [c.purpose for c in app.ledger.calls] == ["main", "btw", "main"]


def test_btw_reuses_the_prompt_cache_and_calls_no_tools(project):
    script = [reply("Done."), reply("It is the parser.")]
    with FakeOpenAI(script) as srv:
        from test_agent_e2e import make_app

        app = make_app(srv, project)
        app.run_prompt("look at the parser")
        answer = app.btw("which file was it?")
        app.shutdown()
    main_req, btw_req = srv.requests
    assert answer == "It is the parser."
    assert btw_req["messages"][0] == main_req["messages"][0]            # same system prompt: cached prefix
    assert btw_req["tools"] == main_req["tools"] and btw_req["tool_choice"] == "none"
    assert "which file was it?" in btw_req["messages"][-1]["content"]


def test_consistent_prefix_drops_tool_calls_still_running():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "tool_call_id": "a", "content": "done"}]
    assert consistent_prefix(msgs) == msgs[:2]
    msgs.append({"role": "tool", "tool_call_id": "b", "content": "done"})
    assert consistent_prefix(msgs) == msgs


def keys(ui, *seq):
    for k in seq:
        if len(k) > 1 and k in ("enter", "up", "down", "delete", "esc", "backspace"):
            ui._on_key(k)
        else:
            for ch in k:
                ui._on_key(ch)


def test_queued_messages_can_be_picked_edited_and_removed():
    ui = TerminalUI(Console(file=io.StringIO(), width=100))
    keys(ui, "first", "enter", "second", "enter", "third", "enter")
    assert ui._queued == ["first", "second", "third"]
    keys(ui, "up", "up")                         # select "second"
    assert ui._selected == 1
    keys(ui, "delete")
    assert ui._queued == ["first", "third"] and ui._selected == 1
    keys(ui, "up", "enter")                      # pull "first" back into the box
    assert ui._draft == "first" and ui._queued == ["third"]
    keys(ui, " please", "enter")
    assert ui._queued == ["third", "first please"]
    keys(ui, "up", "esc")                        # esc leaves the queue instead of interrupting
    assert ui._selected is None and ui._queued == ["third", "first please"]


def test_btw_typed_while_it_works_is_answered_not_queued():
    ui = TerminalUI(Console(file=io.StringIO(), width=100))
    asked = []
    ui.on_btw = asked.append
    keys(ui, "/btw what model is this?", "enter")
    assert asked == ["what model is this?"] and ui._queued == []


def test_after_esc_queued_messages_run_one_at_a_time_in_order(project, monkeypatch):
    """Esc stops the running turn; the queued messages then run as separate turns, oldest first, and the ones
    still waiting stay queued (not fed into the running turn), so each Esc moves on to the next."""
    real_end = TerminalUI.end_typing
    calls = {"n": 0}

    def end_typing(self):
        queued, draft = real_end(self)
        calls["n"] += 1
        if calls["n"] == 1:                 # as if "second" and "third" were typed during the first turn
            return ["second", "third"], draft
        return queued, draft

    monkeypatch.setattr(TerminalUI, "end_typing", end_typing)
    script = [reply("one"), reply("two"), reply("three")]
    code, out, app = run_session(project, ["first", "/exit"], script)
    prompts = [m["content"].split("\n\n")[0] for m in app.agent.messages if m["role"] == "user"]
    assert prompts == ["first", "second", "third"]            # three separate turns, in order


def test_carried_messages_wait_for_their_own_turn():
    ui = TerminalUI(Console(file=io.StringIO(), width=100))
    ui.begin_typing(carried=["third"])
    assert ui._queued == ["third"] and ui.take_queued() == []   # visible, but not injected into this turn
    assert ui.end_typing() == (["third"], "")


def test_the_input_box_stays_while_the_answer_streams():
    from rich.text import Text

    ui = TerminalUI(Console(file=io.StringIO(), width=100))
    ui._reader = object()                       # listening for keys, as during a turn
    ui._queued = ["next question"]
    ui._stream_tail = Text("● the answer so far")
    shown = io.StringIO()
    Console(file=shown, width=100, color_system=None).print(ui._StatsRenderable(ui))
    screen = shown.getvalue()
    assert screen.index("the answer so far") < screen.index("next question") < screen.index("❯ Press up")


def test_up_with_nothing_queued_walks_back_through_earlier_prompts():
    ui = TerminalUI(Console(file=io.StringIO(), width=100))
    ui.history = lambda: ["fix the parser", "run the tests"]
    keys(ui, "up")
    assert ui._draft == "run the tests"
    keys(ui, "up", "up")
    assert ui._draft == "fix the parser"          # stops at the oldest
    keys(ui, "down", "down")
    assert ui._draft == ""                        # back to what you were typing
    keys(ui, "up", " please", "enter")
    assert ui._queued == ["run the tests please"]


def test_history_is_capped(tmp_path):
    from muyah_code.ui.repl import HISTORY_LIMIT, RecentHistory

    h = RecentHistory(str(tmp_path / "history"))
    for i in range(HISTORY_LIMIT + 50):
        h.store_string(f"prompt {i}")
    recent = h.recent()
    assert len(recent) == HISTORY_LIMIT and recent[-1] == f"prompt {HISTORY_LIMIT + 49}"
