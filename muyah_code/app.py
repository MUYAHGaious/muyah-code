"""Application wiring: builds every subsystem from config and runs turns with learning around them."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from muyah_code import __version__
from muyah_code.agent.context import ContextManager
from muyah_code.agent.loop import Agent, TurnResult
from muyah_code.agent.prompts import PromptInputs, build_system_prompt, git_snapshot, turn_context_block
from muyah_code.config import Config, load_config
from muyah_code.events import EventBus
from muyah_code.hooks import HookRunner
from muyah_code.learning.lessons import Lesson, LessonStore, Reflector, is_correction
from muyah_code.llm.client import LLMClient, make_client
from muyah_code.llm.models import DEFAULT_WINDOW, known_context_window
from muyah_code.llm.toolcall_parser import TEXT_PROTOCOL_INSTRUCTIONS
from muyah_code.memory import InstructionFile, load_instructions
from muyah_code.permissions import PermissionManager
from muyah_code.session import Session, sessions_dir
from muyah_code.skills.loader import SkillRegistry
from muyah_code.subagents import AgentDef, AgentTool, SubagentManager, SubagentUI, load_agent_defs
from muyah_code.tools import builtin_tools
from muyah_code.tools.base import ToolContext
from muyah_code.tools.registry import ToolRegistry
from muyah_code.tools.shell import JobManager, detect_shell
from muyah_code.ui.base import UI

FILE_MENTION = re.compile(r"(?<![\w/\\])@([\w./\\:~-]+\.[\w]+|[\w./\\-]+/)")
MAX_MENTION_CHARS = 40000


@dataclass
class TurnRecord:
    prompt: str
    result: TurnResult
    lesson_ids: list[str] = field(default_factory=list)
    start_index: int = 0


def resolve_context_window(cfg: Config, llm: LLMClient) -> tuple[int, str]:
    explicit = int(cfg.get("context_window") or 0)
    if explicit > 0:
        return explicit, "config"
    probed = llm.probe_context_window()
    if probed:
        return probed, "server"
    known = known_context_window(llm.model)
    if known:
        return known, "model table"
    return DEFAULT_WINDOW, "default"


BTW_NOTE = ("(A side question while you work. Answer briefly from what this conversation already shows. Do not "
            "call tools. This exchange is not added to the conversation.)")


def consistent_prefix(messages: list[dict]) -> list[dict]:
    """The longest start of the history a provider accepts: no assistant tool calls without their results
    (mid-turn, the newest calls may still be running)."""
    good, pending = 0, set()
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            pending = {tc.get("id") for tc in m.get("tool_calls") or []}
        elif m.get("role") == "tool":
            pending.discard(m.get("tool_call_id"))
        if not pending:
            good = i + 1
    return messages[:good]


def describe_rewind(point, result: dict, rewind) -> str:
    """One readable summary of a rewind, for /undo and /rewind."""
    lines = [f"Rewound to before turn {point.turn}: {point.prompt[:80]}"]
    code = result.get("code") or []
    if code:
        verbs = {"M": "restored", "A": "brought back", "D": "removed"}
        lines += [f"  {verbs.get(st, st)} {path}" for st, path in code[:30]]
        if len(code) > 30:
            lines.append(f"  … and {len(code) - 30} more files")
    elif "code" in result and not result.get("conversation"):
        lines.append("  no file changes to undo")
    if result.get("conversation"):
        lines.append("  the conversation is back to that point too")
    if result.get("fallback"):
        lines.append("  (only files edited with Write/Edit could be restored: " + rewind.disabled_reason + ")")
    lines.append("  Changed your mind? /rewind → \"Undo the last rewind\".")
    return "\n".join(lines)


class App:
    def __init__(
        self,
        cfg: Config,
        ui: UI,
        cwd: Path | None = None,
        mode: str | None = None,
        headless: bool = False,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        resume: str | None = None,
        continue_last: bool = False,
        enable_mcp: bool = True,
        persist_session: bool = True,
        llm: LLMClient | None = None,
    ):
        self.cfg = cfg
        self.ui = ui
        self.cwd = (cwd or Path.cwd()).resolve()
        self.headless = headless
        self.home = cfg.home
        self.root = cfg.project_root

        from muyah_code import usage
        from muyah_code.pricing import Pricing

        self.pricing = Pricing(cfg.home, cfg.get("pricing.models") or {}, bool(cfg.get("pricing.update", True)))
        self.ledger = usage.Ledger()            # every model call of this session, with its cost
        self.budget = usage.Budget(self.ledger, cfg.home, cfg.get("budget.session_usd") or 0,
                                   cfg.get("budget.daily_usd") or 0)
        self.llm = llm or make_client(cfg)
        self._watch_calls(self.llm)
        from muyah_code.llm.pool import ModelPool, RoleClient

        self.models = ModelPool(cfg, self.llm, watch=self._watch_calls)   # roles, escalation, fallback
        self.summarizer = RoleClient(self.models, "summarize", warn=ui.warn)
        self.btw_model = RoleClient(self.models, "btw", warn=ui.warn)
        self.window, self.window_source = resolve_context_window(cfg, self.llm)
        self.context = ContextManager(self.window, int(cfg.get("max_tokens", 4096)),
                                      float(cfg.get("compact_threshold", 0.8)))
        self.prompt_profile = self._choose_profile()

        self.permissions = PermissionManager.from_config(cfg, mode_override=mode)
        for rule in allowed_tools or []:
            self.permissions.add("allow", rule)
        for rule in disallowed_tools or []:
            self.permissions.add("deny", rule)
        self.hooks = HookRunner(cfg.get("hooks") or {}, self.cwd, cfg.get("shell", "auto"))
        self.skills = SkillRegistry.load(self.root, self.home, bool(cfg.get("compat.claude_skills", True)))
        self.agent_defs, self.agent_def_errors = load_agent_defs(self.root, self.home,
                                                                 bool(cfg.get("compat.claude_skills", True)))
        if not cfg.get("models.edit") and self.agent_defs.get("editor") and self.agent_defs["editor"].source == "bundled":
            del self.agent_defs["editor"]      # only useful with a cheaper "edit" model to hand work to
        self.instructions = load_instructions(self.root, self.cwd, self.home, bool(cfg.get("compat.claude_md", True)))
        self.shell = detect_shell(cfg.get("shell", "auto"))
        self.git_status = git_snapshot(self.cwd)
        self.jobs = JobManager()

        self.learning_enabled = bool(cfg.get("learning.enabled", True))
        project_lessons = self.root / ".muyah" / "lessons.jsonl"
        self.lessons = LessonStore(self.home / "lessons.jsonl", project_lessons)
        self.reflector = Reflector(self.summarizer, self.lessons)
        self.last_turn: TurnRecord | None = None
        self.verify_note = ""        # the last /verify result, told to the model with the next prompt
        self._current_lessons: list[Lesson] = []

        self.mcp = None
        self.mcp_tools = []
        self._browser_offered = False
        self._browser_lock = threading.Lock()
        if enable_mcp:
            from muyah_code.mcp.browser import wanted
            from muyah_code.mcp.client import MCPManager, load_server_configs

            configs = load_server_configs(self.root, self.home)
            self._browser_offered = wanted(cfg, configs)
            if configs:
                self.mcp = MCPManager(configs, self.home / "logs")
                self.mcp_tools = self.mcp.connect_all()

        sdir = sessions_dir(self.home, self.root)
        history: list[dict] = []
        resume_meta: dict = {}
        if resume or continue_last:
            if continue_last and not resume:
                infos = Session.list_sessions(sdir, limit=1)
                resume = infos[0].id if infos else None
            if resume:
                self.session, history, resume_meta = Session.resume(sdir, resume)
            else:
                self.session = Session(sdir)
        else:
            self.session = Session(sdir) if persist_session else None
        if self.session:
            self.session.start({"cwd": str(self.cwd), "model": self.llm.model, "version": __version__})
        from muyah_code.rewind import Rewind

        self.rewind = Rewind(self.home, self.root, self.session)
        self.checkpoints = self.rewind.files  # per-file snapshots (the fallback without git)
        if resume_meta.get("rewind"):
            self.rewind.load(resume_meta["rewind"])
        # Live event stream for /viz; also recorded next to the session file so it can be replayed.
        self.events = EventBus(record_to=self.session.path.with_suffix(".events.jsonl") if self.session else None)
        self.hooks.events = self.events

        self.registry = ToolRegistry(builtin_tools() + list(self.mcp_tools))
        if self._browser_offered:
            from muyah_code.mcp.browser import BrowserTool

            self.registry.register(BrowserTool(self._start_browser))
        self.subagents = SubagentManager(self.agent_defs, self._make_subagent, depth=0, roles=self._roles)
        self.registry.register(AgentTool(self.subagents))

        self.ctx = self._make_ctx(ui, depth=0)
        self.agent = self._make_agent(self._fit(self.registry), self.ctx, ui, session=self.session)
        self.resumed = bool(history)
        self.viz = None  # the /viz web server, started on demand
        if history:
            self.agent.load_history(history)
        self.emit_session()

        if self.hooks.has("SessionStart"):
            out = self.hooks.run("SessionStart", {"session_id": self.session_id, "source":
                                                  "resume" if history else "startup"})
            if out.additional_context:
                self.instructions.append(InstructionFile(Path("SessionStart hook"), out.additional_context))

    # ------------------------------------------------------------------ construction helpers

    @property
    def session_id(self) -> str:
        return self.session.id if self.session else ""

    def _watch_calls(self, llm, provider_id: str | None = None) -> None:
        llm.on_call = self._on_call
        llm.provider_id = (self.cfg.get("provider") or "") if provider_id is None else provider_id

    def _choose_profile(self) -> str:
        from muyah_code.agent.lean import choose_profile
        from muyah_code.pricing import is_self_hosted
        from muyah_code.providers import BY_ID

        prov = BY_ID.get(self.cfg.get("provider") or "")
        local = is_self_hosted(getattr(self.llm, "base_url", "") or "", bool(prov and prov.local))
        return choose_profile(self.cfg.get("prompt_profile", "auto"), self.window, self.llm.model, local)

    def _fit(self, registry: ToolRegistry) -> ToolRegistry:
        """Lean mode: the 6 core tools with short descriptions; the rest one find_tools call away."""
        if self.prompt_profile != "lean":
            return registry
        from muyah_code.agent.lean import lean_registry

        return lean_registry(registry)

    def sees_images(self, llm) -> bool:
        from muyah_code.llm.content import model_sees_images

        model = getattr(llm, "model", "")
        return model_sees_images(model, self.pricing.supports_vision(model, getattr(llm, "provider_id", "")),
                                 self.cfg.get("vision", "auto"))

    def mention_images(self, prompt: str) -> list[tuple[str, str, str]]:
        """@image.png mentions -> (media type, base64, label) to attach to the prompt."""
        from muyah_code.llm.content import describe, is_image_path, load_image

        found = []
        for m in FILE_MENTION.finditer(prompt):
            p = Path(m.group(1)).expanduser()
            p = p if p.is_absolute() else (self.cwd / p)
            if not (p.is_file() and is_image_path(p)):
                continue
            try:
                media, b64, data = load_image(p)
            except (OSError, ValueError) as e:
                self.ui.warn(f"{m.group(1)} was not attached: {e}")
                continue
            found.append((media, b64, describe(m.group(1), data)))
        return found

    def _start_browser(self) -> list[str]:
        """The Browser tool's first call: start the Playwright MCP server and swap in its real tools."""
        from muyah_code.mcp.browser import browser_config
        from muyah_code.mcp.client import MCPManager
        from muyah_code.tools.base import ToolError

        with self._browser_lock:
            if self.mcp is not None and "browser" in self.mcp.clients:
                return [t.name for t in self.mcp_tools if t.name.startswith("mcp__browser__")]
            if self.mcp is None:
                self.mcp = MCPManager({}, self.home / "logs")
            try:
                tools = self.mcp.add(browser_config(headless=bool(self.cfg.get("mcp.browser_headless", True)),
                                                    output_dir=self.home / "browser"))
            except Exception as e:
                raise ToolError(f"Could not start the browser tools: {e}. Details in {self.home / 'logs'}.") from e
            self.mcp_tools.extend(tools)
            for registry in {id(r): r for r in (self.registry, self.agent.registry)}.values():
                registry.unregister("Browser")
                for tool in tools:
                    registry.register(tool)
            self.emit_session()
            return [t.name for t in tools]

    def btw(self, question: str) -> str:
        """A side question: answered from the conversation so far, never added to it. Uses the same system
        prompt and tools as the main agent (so the provider's prompt cache is reused) but no tool calls."""
        messages = consistent_prefix(list(self.agent.messages))
        messages.append({"role": "user", "content": f"{question}\n\n{BTW_NOTE}"})
        tools = None if self.agent.text_mode else self.agent.registry.schemas()
        resp = self.btw_model.chat(messages, tools=tools, tool_choice="none" if tools else None,
                                   max_tokens=1500, purpose="btw")
        return (resp.content or "").strip() or "(No answer without tools: ask it in the conversation instead.)"

    def _roles(self) -> list[str]:
        """Model roles a sub-agent can be run on (the ones set in settings, plus main)."""
        return ["main", *[r for r in self.models.configured() if r not in ("btw", "verify")]]

    def _escalate(self, client):
        from muyah_code.config import ConfigError

        try:
            target = self.models.step_up(client)
        except (ConfigError, ValueError) as e:
            self.ui.warn(f"Could not escalate: {e}")
            return None
        if target is None:
            return None
        return target, self.models.label(client), self.models.label(target)

    def _on_call(self, client, purpose: str, raw_usage: dict, seconds: float) -> None:
        """Every model call, whoever made it: priced, added to this session's ledger and ~/.muyah/usage.jsonl."""
        from muyah_code import usage
        from muyah_code.providers import BY_ID

        provider_id = getattr(client, "provider_id", "") or ""
        prov = BY_ID.get(provider_id)
        model, base_url = getattr(client, "model", ""), getattr(client, "base_url", "") or ""
        price = self.pricing.price(model, provider_id, base_url, local=bool(prov and prov.local))
        if price is not None and price.source == "litellm":
            self.pricing.refresh_in_background()
        n = usage.normalize(raw_usage)
        cost = price.cost(n["in"], n["out"], n["cache_read"], n["cache_write"]) if price else None
        call = usage.Call(model, prov.name if prov else base_url, purpose, n["in"], n["out"], n["cache_read"],
                          n["cache_write"], cost, seconds)
        self.ledger.add(call)
        usage.record_call(self.home, call)
        events = getattr(self, "events", None)
        if events is not None:
            events.emit("cost", purpose=purpose, model=model, cost=cost, session_cost=round(self.ledger.cost, 6),
                        cache_read=n["cache_read"], tokens_in=n["in"], tokens_out=n["out"],
                        unpriced=self.ledger.unpriced,
                        budget=[{"name": name, "spent": round(s, 4), "limit": lim}
                                for name, s, lim in self.budget.limits()])

    def _mcp_inventory(self) -> list[dict]:
        if not self.mcp:
            return []
        return [{"name": name, "status": status,
                 "tools": [t.get("name") for t in (self.mcp.clients[name].tools if name in self.mcp.clients else [])]}
                for name, status in self.mcp.status.items()]

    def emit_session(self) -> None:
        from muyah_code.providers import BY_ID

        prov = BY_ID.get(self.cfg.get("provider") or "")
        self.events.emit("session", model=self.llm.model, provider=prov.name if prov else self.llm.base_url,
                         window=self.window, cwd=str(self.cwd), mode=self.permissions.mode,
                         session_id=self.session_id, tools=self.registry.names(), mcp=self._mcp_inventory(),
                         lessons_path={k: str(v) for k, v in self.lessons.paths.items() if v is not None},
                         memory_files=[str(f.path) for f in self.instructions],
                         skills=[{"name": s.name, "description": s.description[:160], "source": s.source}
                                 for s in self.skills.all()],
                         agents=[{"name": d.name, "description": d.description[:160]}
                                 for d in self.agent_defs.values()],
                         hooks=[{"event": event, "matcher": group.get("matcher") or "*",
                                 "command": str(h.get("command", ""))[:160]}
                                for event, groups in (self.hooks.config or {}).items()
                                for group in groups or [] for h in group.get("hooks") or []])
        self.agent.emit_context()

    def _make_ctx(self, ui: UI, depth: int) -> ToolContext:
        ctx = ToolContext(cwd=self.cwd, project_root=self.root, config=self.cfg, depth=depth, headless=self.headless)
        ctx.services.update({
            "ui": ui, "llm": self.summarizer, "skills": self.skills, "jobs": self.jobs, "budget": self.budget,
            "models": self.models, "vision": self.sees_images,
            "checkpoints": self.rewind, "rewind": self.rewind, "context_window": self.window, "events": self.events,
        })
        return ctx

    def _prompt_builder(self, registry: ToolRegistry, extra: str = "", subagent: bool = False):
        def build(text_mode: bool) -> str:
            return build_system_prompt(PromptInputs(
                cwd=self.cwd, project_root=self.root, model=self.llm.model, shell_kind=self.shell.kind,
                permission_mode=self.permissions.mode, tool_names=registry.names(),
                instructions=self.instructions, skills_index=self.skills.index_text(),
                text_protocol=TEXT_PROTOCOL_INSTRUCTIONS if text_mode else "",
                git_status=self.git_status, extra=extra, subagent=subagent,
                lean=self.prompt_profile == "lean", skill_names=self.skills.names(),
            ))
        return build

    def _turn_context(self, prompt: str) -> str:
        self._current_lessons = []
        pushback = len(self.agent.messages) > 2 and is_correction(prompt)
        note, self.verify_note = self.verify_note, ""
        if not self.learning_enabled:
            return turn_context_block("", pushback=pushback, verify=note)
        k = int(self.cfg.get("learning.max_lessons_in_prompt", 5))
        found = self.lessons.search(prompt, k=k)
        self._current_lessons = found
        self.lessons.record_use(found)
        if found:
            self.events.emit("lessons", items=[x.render()[2:][:160] for x in found])
        return turn_context_block("\n".join(x.render() for x in found), pushback=pushback, verify=note)

    def _make_agent(self, registry: ToolRegistry, ctx: ToolContext, ui: UI, session=None, extra: str = "",
                    subagent: bool = False, max_steps: int | None = None, llm=None, context=None) -> Agent:
        agent = Agent(
            llm=llm or self.llm, registry=registry, permissions=self.permissions, ctx=ctx, ui=ui,
            context=context or self.context,
            system_prompt=self._prompt_builder(registry, extra, subagent),
            turn_context=None if subagent else self._turn_context,
            hooks=self.hooks, session=session, tool_mode=self.cfg.get("tool_mode", "auto"),
            max_steps=max_steps or int(self.cfg.get("max_steps", 60)),
            max_tool_output_chars=int(self.cfg.get("max_tool_output_chars", 24000)),
            is_subagent=subagent, session_id=self.session_id,
        )
        agent.events = self.events
        agent.summarizer = self.summarizer
        agent.escalate = self._escalate
        return agent

    def _subagent_client(self, d: AgentDef, model: str | None):
        """The model a sub-agent runs on: the Agent call's `model`, its definition's `model:`, its role
        (explore -> models.explore, editor -> models.edit), else the main model."""
        from muyah_code.config import ConfigError
        from muyah_code.llm.pool import agent_model

        wanted = model or agent_model(d.model, self.cfg.get("api") or "openai") or \
            {"explore": "explore", "editor": "edit"}.get(d.name)
        if not self.models.resolve(wanted):
            return None, None
        try:
            client = self.models.get(wanted)
        except (ConfigError, ValueError) as e:
            self.ui.warn(f"[{d.name}] its model ({wanted}) is not usable: {e}. Using the main model.")
            return None, None
        if client is self.llm:
            return None, None
        window, _ = self.models.window_for(client)
        return client, ContextManager(window, int(self.cfg.get("max_tokens", 4096)),
                                      float(self.cfg.get("compact_threshold", 0.8)))

    def _make_subagent(self, d: AgentDef, depth: int, model: str | None = None) -> Agent:
        exclude = ["AskUser"] + list(d.disallowed_tools)
        if depth >= 2:
            exclude.append("Agent")
        registry = self.registry.subset(d.tools, exclude=exclude)
        if depth < 2 and "Agent" in registry.names():
            registry.unregister("Agent")
            registry.register(AgentTool(SubagentManager(self.agent_defs, self._make_subagent, depth=depth,
                                                        roles=self._roles)))
        registry = self._fit(registry)
        sub_ui = SubagentUI(self.ui, d.name)
        ctx = self._make_ctx(sub_ui, depth)
        ctx.headless = self.headless
        extra = (f"# Sub-agent role: {d.name}\n{d.prompt}\n\nYou are running as a sub-agent. Nobody can answer "
                 "questions; work autonomously and end with a concise report.")
        worktree = None
        if d.isolation == "worktree":
            import uuid

            from muyah_code.worktree import WorktreeError, create

            name = f"agent-{d.name}-{uuid.uuid4().hex[:6]}"
            try:
                path, _ = create(self.cwd, name)
                worktree = (path, name)
                ctx.cwd = ctx.project_root = path
                extra += (f"\n\nYou work in a separate git worktree: {path} (branch muyah/{name}). Your changes "
                          "stay there; commit them there when you are done.")
            except WorktreeError as e:
                sub_ui.warn(f"no worktree ({e}); working in the project folder")
        client, context = self._subagent_client(d, model)
        agent = self._make_agent(registry, ctx, sub_ui, session=None, extra=extra, subagent=True,
                                 max_steps=d.max_steps, llm=client, context=context)
        agent.label = d.name
        if worktree is not None:
            from muyah_code.worktree import finish

            agent.on_done = lambda: finish(*worktree)
        if d.permission_mode:
            # a read-only helper gets its own permission view without changing the parent's mode
            child_perms = PermissionManager(d.permission_mode, project_root=self.root)
            child_perms.allow, child_perms.ask, child_perms.deny = (self.permissions.allow, self.permissions.ask,
                                                                    self.permissions.deny)
            agent.permissions = child_perms
        return agent

    # ------------------------------------------------------------------ runtime operations

    def set_model(self, model: str) -> str:
        self.llm.model = model
        self.cfg.set("model", model)
        return "Model: " + self._refresh_window(explicit=False)

    def _install_llm(self, llm) -> None:
        """Swap the model client everywhere that holds a reference to it."""
        self._watch_calls(llm)
        self.llm = llm
        self.agent.llm = llm
        self.models.set_main(llm)

    def _refresh_window(self, explicit: bool) -> str:
        if not explicit:
            self.cfg.set("context_window", 0)
        self.window, self.window_source = resolve_context_window(self.cfg, self.llm)
        self.context.set_window(self.window, int(self.cfg.get("max_tokens", 4096)))
        self.context.ratio, self.context.calibrated = 1.0, False
        self.ctx.services["context_window"] = self.window
        profile = self._choose_profile()
        if profile != self.prompt_profile:
            self.prompt_profile = profile
            self.agent.registry = self._fit(self.registry)
            self.ui.info(f"Prompt profile: {profile}" + (" (short prompt, 6 core tools, find_tools for the rest)"
                                                         if profile == "lean" else ""))
        self.agent.invalidate_system_prompt()
        return f"{self.llm.model} (context window {self.window:,} tokens, from {self.window_source})"

    def set_endpoint(self, base_url: str, api_key: str | None = None, model: str | None = None) -> str:
        self.cfg.set("base_url", base_url)
        self.cfg.set("api", "openai")
        if api_key is not None:
            self.cfg.set("api_key", api_key)
        if model:
            self.cfg.set("model", model)
        self._install_llm(make_client(self.cfg))
        return "Model: " + self._refresh_window(explicit=False)

    def switch_profile(self, name: str) -> str:
        """Activate a saved profile for this session: reload settings cleanly (no leftovers from the old one)."""
        fresh = load_config(cwd=self.cwd, overrides={"profile": name})
        self.cfg.data = fresh.data
        self.cfg.sources = fresh.sources
        self._install_llm(make_client(self.cfg))
        msg = f"Profile {name}: " + self._refresh_window(explicit=bool(self.cfg.get("context_window")))
        self.emit_session()
        return msg

    def set_mode(self, mode: str) -> str:
        m = self.permissions.set_mode(mode)
        self.agent.invalidate_system_prompt()
        return m

    def expand_mentions(self, prompt: str) -> str:
        """Inline @file mentions (and list @dir/ mentions) so the model starts with the content."""
        blocks = []
        for m in FILE_MENTION.finditer(prompt):
            raw = m.group(1)
            p = Path(raw).expanduser()
            p = p if p.is_absolute() else (self.cwd / p)
            try:
                p = p.resolve()
            except OSError:
                continue
            if p.is_file():
                from muyah_code.llm.content import is_image_path

                if is_image_path(p):            # attached as an image (mention_images), never as text
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if len(text) > MAX_MENTION_CHARS:
                    text = text[:MAX_MENTION_CHARS] + "\n... [truncated; use Read with offset for the rest]"
                self.ctx.file_state[str(p)] = p.stat().st_mtime_ns
                blocks.append(f'<file path="{raw}">\n{text}\n</file>')
            elif p.is_dir():
                names = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())[:200]
                blocks.append(f'<directory path="{raw}">\n' + "\n".join(names) + "\n</directory>")
        if not blocks:
            return prompt
        return prompt + "\n\n" + "\n\n".join(blocks)

    def run_prompt(self, prompt: str) -> TurnResult:
        self._learn_from_previous(prompt)
        start_index = len(self.agent.messages)
        self.agent.llm = self.llm                     # an escalation lasts one turn
        self.agent.escalation.reset()
        self.rewind.begin_turn(prompt, start_index)   # snapshot before anything in this turn changes
        result = self.agent.run(self.expand_mentions(prompt), images=self.mention_images(prompt))
        self.last_turn = TurnRecord(prompt, result, [x.id for x in self._current_lessons], start_index)
        if self.learning_enabled and self.cfg.get("learning.reflect", True) and \
                Reflector.worth_reflecting(result.signals, result.status):
            self._reflect(prompt, result.signals, result.status, start_index)
        return result

    def _learn_from_previous(self, prompt: str) -> None:
        prev = self.last_turn
        if not prev or not self.learning_enabled:
            return
        if is_correction(prompt):
            self.lessons.record_outcome(prev.lesson_ids, success=False)
            if self.cfg.get("learning.reflect", True):
                self._reflect(prev.prompt, prev.result.signals + [{"type": "user_correction", "text": prompt[:500]}],
                              prev.result.status, prev.start_index)
        elif prev.result.status == "ok":
            self.lessons.record_outcome(prev.lesson_ids, success=True)
        self.last_turn = None

    def feedback(self, positive: bool, note: str = "") -> str:
        prev = self.last_turn
        if prev is None:
            return "Nothing to rate yet: run a prompt first."
        self.lessons.record_outcome(prev.lesson_ids, success=positive)
        msg = f"Recorded {'positive' if positive else 'negative'} feedback."
        if note or not positive:
            stored = self._reflect(prev.prompt, prev.result.signals + [{
                "type": "user_feedback", "positive": positive, "note": note[:500]}], prev.result.status,
                prev.start_index)
            if stored:
                msg += f" Learned {len(stored)} lesson(s)."
        self.last_turn = None if not positive else prev
        return msg

    def _reflect(self, prompt: str, signals: list[dict], status: str, start_index: int) -> list:
        messages = self.agent.messages
        transcript = messages[start_index:] if 0 < start_index < len(messages) else messages[-20:]
        self.ui.busy("Learning from this turn")
        stored = self.reflector.reflect(prompt, signals, transcript, status)
        for lesson, merged in stored:
            self.ui.info(f"{'Reinforced' if merged else 'Learned'} [{lesson.id}] {lesson.render()[2:]}")
        return stored

    def compact(self, focus: str = "") -> str:
        return self.agent.compact(focus=focus)

    def context_usage(self) -> tuple[int, int]:
        tools = None if self.agent.text_mode else self.registry.schemas()
        return self.context.count(self.agent.messages, tools), self.context.usable

    def undo(self) -> str:
        """Rewind code and conversation to before the last turn."""
        turns = self.rewind.turns()
        if not turns:
            return "Nothing to undo."
        point = turns[-1]
        result = self.rewind.restore(point, code=True, conversation=True, agent=self.agent)
        return describe_rewind(point, result, self.rewind)

    def shutdown(self) -> None:
        self.events.close()
        if self.session is not None:
            self.session.close()
        if self.viz is not None:
            self.viz.stop()
            self.viz = None
        if self.hooks.has("SessionEnd"):
            self.hooks.run("SessionEnd", {"session_id": self.session_id})
        self.jobs.shutdown()
        if self.mcp:
            self.mcp.shutdown()
