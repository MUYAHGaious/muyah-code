import pytest
from fakeserver import FakeOpenAI, reply

from muyah_code.backends import (
    build_profile,
    normalize_base_url,
    probe,
    profile_name_for,
    serve_command,
    tunnel_warning,
)
from muyah_code.cli import main
from muyah_code.config import load_config


def test_normalize_base_url():
    assert normalize_base_url("abc.trycloudflare.com/") == "https://abc.trycloudflare.com"
    assert normalize_base_url("localhost:11434") == "http://localhost:11434"
    assert normalize_base_url("https://x.a.pinggy.link/v1/chat/completions") == "https://x.a.pinggy.link/v1"


def test_probe_adds_v1_when_missing():
    with FakeOpenAI() as srv:
        bare = srv.url[:-3]  # strip /v1: users often paste the tunnel root
        ep = probe(bare)
        assert ep.ok and ep.base_url.endswith("/v1") and ep.models == ["fake-model"]


def test_probe_reports_unreachable():
    ep = probe("http://127.0.0.1:9", timeout=0.5)
    assert not ep.ok and ep.error


def test_profile_names():
    assert profile_name_for("https://abc.trycloudflare.com/v1") == "colab"
    assert profile_name_for("https://api.groq.com/openai/v1") == "groq"
    assert profile_name_for("http://localhost:11434/v1") == "ollama"


def test_colibri_preset_disables_timeouts_and_reflection():
    prof = build_profile("https://x/v1", "glm-5.2-colibri", None, 65536, "colibri")
    assert prof["request_timeout"] == 0 and prof["learning"] == {"reflect": False}
    assert prof["engine"] == "colibri" and prof["context_window"] == 65536 and "api_key" not in prof


@pytest.mark.parametrize("engine,expect", [
    ("colibri", ["coli", "serve", "--model", "/m/glm", "--host", "127.0.0.1", "--port", "8000", "--model-id", "glm",
                 "--ctx", "4096"]),
    ("soup", ["soup", "serve", "--model", "/m/glm", "--port", "8000"]),
    ("llamacpp", ["llama-server", "-m", "/m/glm", "--host", "127.0.0.1", "--port", "8000", "--jinja", "-c", "4096"]),
])
def test_serve_commands(engine, expect):
    argv, _ = serve_command(engine, "/m/glm", 8000, 4096, expect[0])
    assert argv == expect


def test_ollama_serve_sets_context_length():
    argv, env = serve_command("ollama", "qwen2.5-coder:7b", 11434, 32768, "ollama")
    assert argv == ["ollama", "serve"] and env["OLLAMA_CONTEXT_LENGTH"] == "32768"


def test_tunnel_warning_only_for_cloudflare():
    assert "524" in tunnel_warning("https://a.trycloudflare.com/v1", "colibri")
    assert tunnel_warning("https://a.a.pinggy.link/v1", "colibri") is None


def test_connect_saves_engine_profile(project, monkeypatch):
    monkeypatch.chdir(project)
    with FakeOpenAI([reply("PONG")]) as srv:
        assert main(["connect", srv.url, "--engine", "colibri", "--name", "big"]) == 0
    cfg = load_config(cwd=project)
    assert cfg.get("profile") == "big"
    assert cfg["base_url"] == srv.url and cfg["model"] == "fake-model"
    assert cfg["request_timeout"] == 0 and cfg.get("learning.reflect") is False


def test_connect_fails_cleanly_when_unreachable(project, monkeypatch):
    monkeypatch.chdir(project)
    assert main(["connect", "http://127.0.0.1:9/v1", "--no-test"]) == 1
