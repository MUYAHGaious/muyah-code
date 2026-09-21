"""The built-in browser: offered in every session, started (Playwright MCP) only when the agent calls it."""

import json
import sys
from pathlib import Path

from fakeserver import FakeOpenAI, reply
from test_agent_e2e import RecUI

from muyah_code.app import App
from muyah_code.config import load_config
from muyah_code.mcp import browser
from muyah_code.mcp.client import ServerConfig

FAKE = Path(__file__).with_name("fakemcp.py")


def app_for(srv, project, **settings):
    cfg = load_config(cwd=project, overrides={"base_url": srv.url, "model": "fake-model", "api_key": "k",
                                              "vision": True, **settings})
    cfg.set("learning.reflect", False)
    return App(cfg, RecUI(), cwd=project, mode="bypassPermissions", enable_mcp=True)


def test_offered_unless_turned_off_or_replaced():
    class Cfg(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    assert browser.wanted(Cfg(), {})
    assert not browser.wanted(Cfg({"mcp.browser": False}), {})
    assert not browser.wanted(Cfg(), {"playwright": object()})      # you configured your own
    cfg = browser.browser_config()
    assert cfg.command == "npx" and "@playwright/mcp@latest" in cfg.args and "--headless" in cfg.args


def test_nothing_starts_until_the_agent_asks_then_the_real_tools_replace_it(project, monkeypatch):
    monkeypatch.setattr(browser, "node_version", lambda: 22)
    monkeypatch.setattr(browser, "browser_config",
                        lambda headless=True, output_dir=None: ServerConfig(name="browser", command=sys.executable, args=[str(FAKE)]))
    script = [reply("", [{"name": "Browser", "arguments": {}}]),
              reply("", [{"name": "mcp__browser__screenshot", "arguments": {}}]),
              reply("The page renders.")]
    with FakeOpenAI(script) as srv:
        app = app_for(srv, project)
        assert "Browser" in app.registry.names() and (app.mcp is None or "browser" not in app.mcp.clients)
        res = app.run_prompt("check the page")
        names = app.agent.registry.names()
        app.shutdown()
    first_tools = [t["function"]["name"] for t in srv.requests[0]["tools"]]
    later_tools = [t["function"]["name"] for t in srv.requests[1]["tools"]]
    assert "Browser" in first_tools and "mcp__browser__screenshot" not in first_tools
    assert "Browser" not in later_tools and "mcp__browser__screenshot" in later_tools
    started = [m for m in srv.requests[1]["messages"] if m["role"] == "tool"][-1]["content"]
    assert started.startswith("Browser tools are ready: mcp__browser__lookup_issue")
    assert res.text == "The page renders." and "Browser" not in names
    assert any(p["type"] == "image_url" for m in srv.requests[2]["messages"] if isinstance(m.get("content"), list)
               for p in m["content"])


def test_without_node_the_agent_is_told_why(project, monkeypatch):
    monkeypatch.setattr(browser, "node_version", lambda: None)
    script = [reply("", [{"name": "Browser", "arguments": {}}]), reply("No browser here.")]
    with FakeOpenAI(script) as srv:
        app = app_for(srv, project)
        app.run_prompt("check the page")
        app.shutdown()
    told = [m for m in srv.requests[1]["messages"] if m["role"] == "tool"][-1]["content"]
    assert "need Node.js 18 or newer" in told


def test_turned_off_in_settings(project):
    (project / ".muyah" / "settings.json").write_text(json.dumps({"mcp": {"browser": False}}))
    with FakeOpenAI([]) as srv:
        app = app_for(srv, project)
        app.shutdown()
    assert "Browser" not in app.registry.names()
