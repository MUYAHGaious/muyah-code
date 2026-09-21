"""Application wiring: builds every subsystem from config and runs turns with learning around them."""

from __future__ import annotations

import re
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
from muyah_code.session import Checkpoints, Session, sessions_dir
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

        self.llm = llm or make_client(cfg)
        self.window, self.window_source = resolve_context_window(cfg, self.llm)
        self.context = ContextManager(self.window, int(cfg.get("max_tokens", 4096)),
                                      float(cfg.get("compact_threshold", 0.8)))

        self.permissions = PermissionManager.from_config(cfg, mode_override=mode)
        for rule in allowed_tools or []:
            self.permissions.add("allow", rule)
        for rule in disallowed_tools or []:
            self.permissions.add("deny", rule)
        self.hooks = HookRunner(cfg.get("hooks") or {}, self.cwd, cfg.get("shell", "auto"))
        self.skills = SkillRegistry.load(self.root, self.home, bool(cfg.get("compat.claude_skills", True)))
        self.agent_defs, self.agent_def_errors = load_agent_defs(self.root, self.home,
                                                                 bool(cfg.get("compat.claude_skills", True)))
        self.instructions = load_instructions(self.root, self.cwd, self.home, bool(cfg.get("compat.claude_md", True)))
        self.shell = detect_shell(cfg.get("shell", "auto"))
        self.git_status = git_snapshot(self.cwd)
        self.jobs = JobManager()
        self.checkpoints = Checkpoints()

        self.learning_enabled = bool(cfg.get("learning.enabled", True))
        project_lessons = self.root / ".muyah" / "lessons.jsonl"
        self.lessons = LessonStore(self.home / "lessons.jsonl", project_lessons)
        self.reflector = Reflector(self.llm, self.lessons)
        self.last_turn: TurnRecord | None = None
        self._current_lessons: list[Lesson] = []

        self.mcp = None
        self.mcp_tools = []
        if enable_mcp:
            from muyah_code.mcp.client import MCPManager, load_server_configs

            configs = load_server_configs(self.root, self.home)
            if configs:
                self.mcp = MCPManager(configs, self.home / "logs")
                self.mcp_tools = self.mcp.connect_all()

        sdir = sessions_dir(self.home, self.root)
        history: list[dict] = []
        if resume or continue_last:
            if continue_last and not resume:
                infos = Session.list_sessions(sdir, limit=1)
                resume = infos[0].id if infos else None
            if resume:
                self.session, history, _ = Session.resume(sdir, resume)
            else:
                self.session = Session(sdir)
        else:
            self.session = Session(sdir) if persist_session else None
        if self.session:
            self.session.start({"cwd": str(self.cwd), "model": self.llm.model, "version": __version__})
        # Live event stream for /viz; also recorded next to the session file so it can be replayed.
        self.events = EventBus(record_to=self.session.path.with_suffix(".events.jsonl") if self.session else None)
        self.hooks.events = self.events
        self.events.subscribe(self._record_usage)

        self.registry = ToolRegistry(builtin_tools() + list(self.mcp_tools))
        self.subagents = SubagentManager(self.agent_defs, self._make_subagent, depth=0)
        self.registry.register(AgentTool(self.subagents))

        self.ctx = self._make_ctx(ui, depth=0)
        self.agent = self._make_agent(self.registry, self.ctx, ui, session=self.session)
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

    def _record_usage(self, event: dict) -> None:
        """Every model call goes into ~/.muyah/usage.jsonl (see /usage)."""
        if event.get("type") != "llm_end" or event.get("error"):
            return
        from muyah_code import usage
        from muyah_code.providers import BY_ID

        prov = BY_ID.get(self.cfg.get("provider") or "")
        usage.record(self.home, getattr(self.llm, "model", ""), prov.name if prov else self.llm.base_url,
                     event.get("prompt_tokens") or 0, event.get("completion_tokens") or 0,
                     event.get("agent") or "main")

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
            "ui": ui, "llm": self.llm, "skills": self.skills, "jobs": self.jobs,
            "checkpoints": self.checkpoints, "context_window": self.window, "events": self.events,
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
            ))
        return build

    def _turn_context(self, prompt: str) -> str:
        self._current_lessons = []
        if not self.learning_enabled:
            return ""
        k = int(self.cfg.get("learning.max_lessons_in_prompt", 5))
        found = self.lessons.search(prompt, k=k)
        self._current_lessons = found
        self.lessons.record_use(found)
        if found:
            self.events.emit("lessons", items=[x.render()[2:][:160] for x in found])
        return turn_context_block("\n".join(x.render() for x in found))

    def _make_agent(self, registry: ToolRegistry, ctx: ToolContext, ui: UI, session=None, extra: str = "",
                    subagent: bool = False, max_steps: int | None = None) -> Agent:
        agent = Agent(
            llm=self.llm, registry=registry, permissions=self.permissions, ctx=ctx, ui=ui, context=self.context,
            system_prompt=self._prompt_builder(registry, extra, subagent),
            turn_context=None if subagent else self._turn_context,
            hooks=self.hooks, session=session, tool_mode=self.cfg.get("tool_mode", "auto"),
            max_steps=max_steps or int(self.cfg.get("max_steps", 60)),
            max_tool_output_chars=int(self.cfg.get("max_tool_output_chars", 24000)),
            is_subagent=subagent, session_id=self.session_id,
        )
        agent.events = self.events
        return agent

    def _make_subagent(self, d: AgentDef, depth: int) -> Agent:
        exclude = ["AskUser"] + list(d.disallowed_tools)
        if depth >= 2:
            exclude.append("Agent")
        registry = self.registry.subset(d.tools, exclude=exclude)
        if depth < 2 and "Agent" in registry.names():
            registry.unregister("Agent")
            registry.register(AgentTool(SubagentManager(self.agent_defs, self._make_subagent, depth=depth)))
        sub_ui = SubagentUI(self.ui, d.name)
        ctx = self._make_ctx(sub_ui, depth)
        ctx.headless = self.headless
        extra = (f"# Sub-agent role: {d.name}\n{d.prompt}\n\nYou are running as a sub-agent. Nobody can answer "
                 "questions; work autonomously and end with a concise report.")
        agent = self._make_agent(registry, ctx, sub_ui, session=None, extra=extra, subagent=True,
                                 max_steps=d.max_steps)
        agent.label = d.name
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
        self.llm = llm
        self.agent.llm = llm
        self.reflector.llm = llm
        self.ctx.services["llm"] = llm

    def _refresh_window(self, explicit: bool) -> str:
        if not explicit:
            self.cfg.set("context_window", 0)
        self.window, self.window_source = resolve_context_window(self.cfg, self.llm)
        self.context.set_window(self.window, int(self.cfg.get("max_tokens", 4096)))
        self.context.ratio, self.context.calibrated = 1.0, False
        self.ctx.services["context_window"] = self.window
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
        result = self.agent.run(self.expand_mentions(prompt))
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
        res = self.checkpoints.undo()
        if res is None:
            return "Nothing to undo."
        label, restored = res
        for key in list(self.ctx.file_state):
            self.ctx.file_state.pop(key, None)  # files changed on disk: force a re-read before editing
        return f"Undid changes from: {label}\n" + "\n".join(f"  {r}" for r in restored)

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
