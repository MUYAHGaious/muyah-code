import json

import pytest
from fakeanthropic import FakeAnthropic, turn
from fakeserver import FakeOpenAI, error, reply

import muyah_code.provider_setup as provider_setup
from muyah_code.app import App
from muyah_code.config import load_config
from muyah_code.llm.anthropic_client import RAW_KEY, to_anthropic, to_anthropic_tools
from muyah_code.providers import (
    PROVIDERS,
    Provider,
    build_provider_profile,
    get_provider,
    load_credentials,
    looks_wrong,
    rank_models,
    remove_credential,
    resolve_key,
    save_credential,
)
from muyah_code.tools import default_registry


class Answers:
    """Scripted user: answers the Prompter questions (select / ask / confirm) in order.
    None means "just press Enter" (the default). Records every question and whether input was hidden."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = []

    def _next(self):
        return self.answers.pop(0) if self.answers else None

    def select(self, message, options, default=None):
        self.asked.append(("select", message, [v for v, _ in options]))
        picked = self._next()
        return default if picked is None else picked

    def ask(self, message, password=False, default=""):
        self.asked.append(("ask", message, password))
        picked = self._next()
        return default if picked is None else picked

    def confirm(self, message, default=True):
        self.asked.append(("confirm", message, None))
        picked = self._next()
        return default if picked is None else picked


# ---------------------------------------------------------------------------------------------- catalog

def test_catalog_has_the_major_providers():
    ids = {p.id for p in PROVIDERS}
    assert {"anthropic", "openai", "gemini", "openrouter", "groq", "deepseek", "mistral", "xai", "ollama"} <= ids
    assert len(ids) == len(PROVIDERS)  # unique
    assert get_provider("anthropic").api == "anthropic"


@pytest.mark.parametrize("query,expected", [("1", "anthropic"), ("claude", "anthropic"), ("google", "gemini"),
                                            ("grok", "xai"), ("kimi", "moonshot"), ("OpenAI", "openai")])
def test_get_provider_by_number_or_name(query, expected):
    assert get_provider(query).id == expected


def test_rank_models_prefers_coding_flagships_and_drops_noise():
    p = get_provider("openai")
    ranked = rank_models(p, ["text-embedding-3-large", "whisper-1", "gpt-4.1-mini", "gpt-5", "babbage-002"])
    assert ranked == ["gpt-5", "gpt-4.1-mini"]  # non-chat models are removed, newest generation first
    assert rank_models(get_provider("gemini"), ["models/gemini-2.5-flash"]) == ["gemini-2.5-flash"]


def test_key_sanity_hints():
    p = get_provider("anthropic")
    assert looks_wrong(p, "sk-ant-abc") is None
    assert "sk-ant-" in looks_wrong(p, "AIzaXYZ")
    assert "spaces" in looks_wrong(p, "sk-ant-a b")


def test_credential_store_and_env_fallback(tmp_path, monkeypatch):
    save_credential(tmp_path, "openai", " sk-saved \n")
    assert load_credentials(tmp_path) == {"openai": "sk-saved"}
    assert resolve_key(tmp_path, "openai") == "sk-saved"
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    assert resolve_key(tmp_path, "groq") == "gsk_env"
    assert remove_credential(tmp_path, "openai") and resolve_key(tmp_path, "openai") is None


def test_profile_references_key_but_never_stores_it():
    prof = build_provider_profile(get_provider("anthropic"), "claude-opus-5")
    assert prof["credential"] == "anthropic" and prof["api_key"] == "none" and prof["api"] == "anthropic"
    assert prof["max_tokens"] == 32000


def test_config_resolves_saved_key_for_active_profile(project, isolated_home, monkeypatch):
    save_credential(isolated_home, "deepseek", "sk-deep")
    cfg = load_config(cwd=project)
    cfg.persist("profiles.deepseek", build_provider_profile(get_provider("deepseek"), "deepseek-chat"))
    cfg.persist("profile", "deepseek")
    fresh = load_config(cwd=project)
    assert fresh["api_key"] == "sk-deep" and fresh["context_window"] == 131072
    assert "sk-deep" not in (isolated_home / "settings.json").read_text()
    monkeypatch.setenv("MUYAH_API_KEY", "sk-override")
    assert load_config(cwd=project)["api_key"] == "sk-override"


# ---------------------------------------------------------------------------------------------- setup flow

def _test_provider(url):
    return Provider("testprov", "Test Provider", url, "fake-model", ("TESTPROV_API_KEY",), key_hint="sk-t")


def test_setup_flow_paste_key_pick_model_and_activate(project, isolated_home, monkeypatch):
    from rich.console import Console

    with FakeOpenAI([reply("PONG")], models=[{"id": "fake-model"}, {"id": "other-model"}],
                    require_key="sk-test-123") as srv:
        monkeypatch.setattr(provider_setup, "get_provider", lambda _: _test_provider(srv.url))
        ask = Answers("sk-test-123", "other-model")  # paste key, pick a model
        cfg = load_config(cwd=project)
        name = provider_setup.setup_provider(cfg, Console(quiet=True), ask, choice="testprov")
    assert name == "testprov"
    assert ask.asked[0] == ("ask", "Paste your Test Provider API key: ", True)  # hidden input
    assert load_credentials(isolated_home)["testprov"] == "sk-test-123"
    fresh = load_config(cwd=project)
    assert fresh.get("profile") == "testprov" and fresh["model"] == "other-model"
    assert fresh["api_key"] == "sk-test-123"


def test_setup_flow_rejected_key_then_retry(project, isolated_home, monkeypatch):
    from rich.console import Console

    with FakeOpenAI([reply("PONG")], require_key="sk-test-good") as srv:
        monkeypatch.setattr(provider_setup, "get_provider", lambda _: _test_provider(srv.url))
        ask = Answers("sk-test-bad", "sk-test-good", None)  # wrong key, right key, Enter
        name = provider_setup.setup_provider(load_config(cwd=project), Console(quiet=True), ask, choice="x")
    assert name == "testprov" and load_credentials(isolated_home)["testprov"] == "sk-test-good"


def test_setup_flow_offers_env_key(project, isolated_home, monkeypatch):
    from rich.console import Console

    monkeypatch.setenv("TESTPROV_API_KEY", "sk-test-env")
    with FakeOpenAI([reply("PONG")], require_key="sk-test-env") as srv:
        monkeypatch.setattr(provider_setup, "get_provider", lambda _: _test_provider(srv.url))
        ask = Answers(None, None)  # accept env key, Enter on the model menu
        assert provider_setup.setup_provider(load_config(cwd=project), Console(quiet=True), ask, choice="x")
    assert ask.asked[0][0] == "confirm" and "TESTPROV_API_KEY" in ask.asked[0][1]
    assert "testprov" not in load_credentials(isolated_home)  # env keys are not copied to disk


# ---------------------------------------------------------------------------------------------- Claude adapter

def test_to_anthropic_translation():
    history = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "Read", "arguments": '{"file_path": "x"}'}},
            {"id": "b", "type": "function", "function": {"name": "LS", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "X"},
        {"role": "tool", "tool_call_id": "b", "content": "Y"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "ignored", RAW_KEY: [{"type": "thinking", "thinking": "t", "signature": "s"},
                                                          {"type": "text", "text": "kept"}]},
    ]
    system, msgs = to_anthropic(history)
    assert system == "SYS"
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    results = msgs[2]["content"]
    assert [b["type"] for b in results] == ["tool_result", "tool_result", "text"]  # grouped, then the text
    assert msgs[1]["content"][0] == {"type": "tool_use", "id": "a", "name": "Read", "input": {"file_path": "x"}}
    assert msgs[3]["content"][0]["type"] == "thinking"  # raw blocks replayed unchanged
    tools = to_anthropic_tools(default_registry().schemas())
    assert tools[0]["input_schema"]["required"] == ["file_path"] and tools[0]["eager_input_streaming"] is True


def test_claude_end_to_end_tool_loop(project, isolated_home):
    (project / "notes.txt").write_text("secret=42\n")
    script = [
        turn(thinking="Read the file first.", tools=[{"id": "toolu_1", "name": "Read",
                                                      "input": {"file_path": "notes.txt"}},
                                                     {"id": "toolu_2", "name": "LS", "input": {}}]),
        turn(thinking="Found it.", text="The secret is **42**."),
    ]
    with FakeAnthropic(script) as srv:
        save_credential(isolated_home, "anthropic", "sk-ant-test")
        cfg = load_config(cwd=project)
        cfg.persist("profiles.anthropic", {**build_provider_profile(get_provider("anthropic"), "claude-opus-5"),
                                           "base_url": srv.url})
        cfg.persist("profile", "anthropic")
        cfg = load_config(cwd=project)
        cfg.set("learning.reflect", False)
        from test_agent_e2e import RecUI

        app = App(cfg, RecUI(), cwd=project, mode="bypassPermissions", enable_mcp=False)
        res = app.run_prompt("what is the secret in notes.txt?")
        first, second = srv.requests
        beta = srv.headers[0].get("anthropic-beta", "")

    assert res.status == "ok" and res.text == "The secret is **42**."
    assert app.window == 1_000_000 and app.window_source == "server"  # from /v1/models/{id} max_input_tokens
    # request shape: no sampling params on Opus 5, prompt caching, xhigh effort, refusal fallbacks
    assert "temperature" not in first and first["cache_control"] == {"type": "ephemeral"}
    assert first["output_config"] == {"effort": "xhigh"} and first["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in beta
    assert "MUYAH-CODE" in first["system"]
    # replay: thinking + both tool_use blocks unchanged, then BOTH results in ONE user message
    assistant, results = second["messages"][1], second["messages"][2]
    assert [b["type"] for b in assistant["content"]] == ["thinking", "tool_use", "tool_use"]
    assert assistant["content"][0]["signature"] == "sig-abc"
    assert [b["tool_use_id"] for b in results["content"]] == ["toolu_1", "toolu_2"]
    assert "secret=42" in results["content"][0]["content"]
    # usage includes cached tokens so compaction timing stays right
    assert app.llm.total_usage["prompt_tokens"] == 2 * 2000


def test_claude_bad_key_is_reported_clearly(project, isolated_home):
    with FakeAnthropic([]) as srv:
        cfg = load_config(cwd=project, overrides={"api": "anthropic", "base_url": srv.url, "model": "claude-opus-5",
                                                  "api_key": "sk-ant-wrong"})
        from muyah_code.llm.anthropic_client import AnthropicClient
        from muyah_code.llm.client import LLMError

        with pytest.raises(LLMError, match="rejected"):
            AnthropicClient.from_config(cfg).chat([{"role": "user", "content": "hi"}])


def test_no_credits_keeps_the_valid_key_and_explains(project, isolated_home, monkeypatch):
    """Real case: the key lists models fine, but the account has no API credits."""
    from rich.console import Console

    import muyah_code.llm.anthropic_client as ac

    with FakeAnthropic([]) as srv:
        srv.post_error = (400, "Your credit balance is too low to access the Anthropic API. Please go to Plans & "
                               "Billing to upgrade or purchase credits.")
        real = ac.AnthropicClient.__init__

        def to_fake(self, *a, **kw):
            kw["base_url"] = srv.url
            real(self, *a, **kw)

        monkeypatch.setattr(ac.AnthropicClient, "__init__", to_fake)
        console = Console(record=True, width=120)
        ask = Answers("sk-ant-test", None)
        name = provider_setup.setup_provider(load_config(cwd=project), console, ask, choice="anthropic")
        out = console.export_text()
    assert name is None  # not activated: every request would fail
    assert load_credentials(isolated_home)["anthropic"] == "sk-ant-test"  # but the working key is kept
    assert "no credits" in out and "settings/billing" in out and "Try another model" not in out
    assert "credit balance is too low" in out


def test_retired_model_offers_the_providers_suggestion(project, isolated_home, monkeypatch):
    """Real case: Gemini lists gemini-2.5-pro but refuses it for new users and names the replacement."""
    from rich.console import Console

    retired = ('HTTP 404: [{"error": {"code": 404, "message": "This model models/gemini-2.5-pro is no longer '
               'available to new users. Please update your code to use models/gemini-3.1-pro-preview for the '
               'latest features."}}]')
    calls = []

    def fake_test(p, key, model):
        calls.append(model)
        return (False, retired) if model == "gemini-2.5-pro" else (True, "PONG")

    monkeypatch.setattr(provider_setup, "_test", fake_test)
    monkeypatch.setattr(provider_setup, "_list_models", lambda p, k: (
        ["models/gemini-2.5-pro", "models/gemini-3.1-pro-preview", "models/gemini-2.5-flash-preview-tts"], None))
    console = Console(record=True, width=140)
    ask = Answers(None, "gemini-2.5-pro", None)  # accept env key, pick the retired model, then Enter
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test")
    name = provider_setup.setup_provider(load_config(cwd=project), console, ask, choice="gemini")
    out = console.export_text()
    assert name == "gemini" and calls == ["gemini-2.5-pro", "gemini-3.1-pro-preview"]
    assert "is not available" in out and "no longer available to new users" in out
    assert "tts" not in out  # speech models are not offered for coding
    assert load_config(cwd=project)["model"] == "gemini-3.1-pro-preview"


def test_model_ranking_is_version_aware():
    gem = rank_models(get_provider("gemini"), ["models/gemini-2.5-pro", "models/gemma-4-31b-it",
                                               "models/gemini-3.1-pro-preview", "models/aqa"])
    assert gem[0] == "gemini-3.1-pro-preview" and "aqa" not in gem
    claude = rank_models(get_provider("anthropic"), ["claude-fable-5-1", "claude-opus-4-8", "claude-opus-5"])
    assert claude[0] == "claude-opus-5"  # curated order: the pricier Fable tier is not the default


def test_openai_quota_errors_are_not_retried(project):
    from muyah_code.llm.client import LLMClient, LLMError

    with FakeOpenAI([error(429, "You exceeded your current quota, please check your plan and billing details.")] * 5
                    ) as srv:
        client = LLMClient(srv.url, "fake-model")
        with pytest.raises(LLMError, match="no credits or quota"):
            client.chat([{"role": "user", "content": "hi"}])
        assert len(srv.requests) == 1  # no pointless retries with backoff


def test_private_keys_never_reach_openai_compatible_servers(project):
    with FakeOpenAI([reply("ok")]) as srv:
        from muyah_code.llm.client import LLMClient

        LLMClient(srv.url, "fake-model").chat([{"role": "user", "content": "hi"},
                                              {"role": "assistant", "content": "x", RAW_KEY: [{"type": "text"}]},
                                              {"role": "user", "content": "again"}])
        sent = json.dumps(srv.requests[0]["messages"])
    assert RAW_KEY not in sent
