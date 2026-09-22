"""First run (a local model, a free key, the free trial, your own key) and questions asked the Claude Code way."""

import io

from rich.console import Console

from muyah_code import onboarding
from muyah_code.backends import Endpoint
from muyah_code.config import load_config
from muyah_code.providers import TRIAL, build_provider_profile, get_provider
from muyah_code.tools.meta import AskUserTool, _questions


class QuestionPrompter:
    """Answers questions in order; records what was asked."""

    def __init__(self, *picks):
        self.picks = list(picks)
        self.asked = []

    def question(self, question, options, header="", multi=False, step=""):
        self.asked.append((question, [label for label, _ in options], header))
        want = self.picks.pop(0)
        return next(label for label, _ in options if label.startswith(want))

    def select(self, *a, **k):
        raise AssertionError("the plain menu should not be used when question() exists")

    def ask(self, *a, **k):
        return ""

    def confirm(self, *a, **k):
        return True


def test_needs_setup_only_when_nothing_says_which_model(project, monkeypatch):
    for var in onboarding.ENV_SETUP:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(onboarding, "probe", lambda *a, **k: Endpoint("x", "u", error="down"))
    cfg = load_config(cwd=project)
    assert onboarding.needs_setup(cfg)
    assert not onboarding.needs_setup(cfg, explicit=True)                 # --model / --base-url given
    monkeypatch.setenv("MUYAH_BASE_URL", "http://localhost:9999/v1")
    assert not onboarding.needs_setup(cfg)
    monkeypatch.delenv("MUYAH_BASE_URL")
    monkeypatch.setattr(onboarding, "probe", lambda *a, **k: Endpoint("x", "u"))
    assert not onboarding.needs_setup(cfg)                                 # your own server at the default address
    cfg.set("provider", "groq")
    assert not onboarding.needs_setup(cfg)


def test_a_running_local_model_is_offered_first(project, monkeypatch):
    monkeypatch.setattr(onboarding, "scan_local", lambda timeout=1.0: [
        Endpoint("ollama", "http://localhost:11434/v1", models=["qwen2.5-coder:14b"]),
        Endpoint("lmstudio", "http://localhost:1234/v1", error="down")])
    calls = []
    monkeypatch.setattr("muyah_code.provider_setup.setup_provider",
                        lambda cfg, console, ui, choice=None, key=None, model=None, test=True: calls.append((choice, model)) or choice)
    ui = QuestionPrompter("Ollama")
    got = onboarding.first_run(load_config(cwd=project), Console(file=io.StringIO()), ui)
    labels = ui.asked[0][1]
    assert labels[0].startswith("Ollama (local): qwen2.5-coder:14b") and labels[0].endswith("(Recommended)")
    assert got == "ollama" and calls == [("ollama", "qwen2.5-coder:14b")]


def test_free_key_opens_the_key_page_and_the_trial_needs_no_key(project, monkeypatch):
    monkeypatch.setattr(onboarding, "scan_local", lambda timeout=1.0: [])
    opened, calls = [], []
    monkeypatch.setattr(onboarding.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr("muyah_code.provider_setup.setup_provider",
                        lambda cfg, console, ui, choice=None, key=None, model=None, test=True: calls.append(choice) or choice)
    cfg = load_config(cwd=project)
    ui = QuestionPrompter("Get a free key", "Groq")
    assert ui and onboarding.first_run(cfg, Console(file=io.StringIO()), ui) == "groq"
    assert ui.asked[0][1][0].endswith("(Recommended)") and opened == ["https://console.groq.com/keys"]
    assert onboarding.first_run(cfg, Console(file=io.StringIO()), QuestionPrompter("Try it now")) == TRIAL

    trial = get_provider(TRIAL)
    profile = build_provider_profile(trial, trial.default_model)
    assert profile["api_key"] == "none" and "credential" not in profile       # no key to look up
    assert profile["min_request_interval"] >= 15                               # paced, never rate-limited


def test_ask_user_takes_claude_code_style_questions_and_the_old_form():
    qs = _questions({"questions": [
        {"header": "Sign-in", "question": "How should buyers sign in?", "options": [
            {"label": "Phone (Recommended)", "description": "Most buyers are on phones."}, {"label": "Email"}]},
        {"question": "Which payments?", "multiSelect": True, "options": ["Mobile money", "Cards"]}]})
    assert qs[0]["header"] == "Sign-in" and qs[0]["options"][0]["description"] == "Most buyers are on phones."
    assert qs[1]["multiSelect"] and [o["label"] for o in qs[1]["options"]] == ["Mobile money", "Cards"]
    old = _questions({"question": "Which database?", "options": ["SQLite", "Postgres"]})
    assert old[0]["question"] == "Which database?" and len(old[0]["options"]) == 2


def test_ask_user_reports_every_answer_to_the_model(ctx):
    class UI:
        def ask_questions(self, questions):
            return ["Phone (Recommended)", ["Mobile money", "Cards"]]

    ctx.services["ui"] = UI()
    res = AskUserTool().run({"questions": [
        {"question": "How should buyers sign in?", "options": [{"label": "Phone (Recommended)"}, {"label": "Email"}]},
        {"question": "Which payments?", "multiSelect": True,
         "options": [{"label": "Mobile money"}, {"label": "Cards"}]}]}, ctx)
    assert "How should buyers sign in? → Phone (Recommended)" in res.content
    assert "Which payments? → Mobile money, Cards" in res.content


def test_the_client_waits_out_a_services_request_gap(monkeypatch):
    from muyah_code.llm import client as client_mod
    from muyah_code.llm.client import LLMClient

    slept, now = [], [100.0]
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: slept.append(round(s, 1)))
    c = LLMClient("http://x/v1", "m", min_request_interval=16)
    c._pace()                    # first request: no wait
    now[0] += 4
    c._pace()                    # 4 s later: wait the other 12
    assert slept == [12.0]
