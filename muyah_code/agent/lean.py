"""Lean mode: a much smaller prompt and tool list for small models and small context windows.

A 7B-14B model with a 16k-32k window loses a lot to a 4k-token preamble: less room for the code, and more
text it has to follow. Lean mode keeps what such models actually use:

  * a ~400-token system prompt (who it is, the working habits that matter most, honesty in one line),
  * six core tools (Read, Edit, Write, Bash, Grep, Glob) with short descriptions,
  * every other tool (web, todo list, skills, sub-agents, MCP...) one `find_tools` call away: it looks tools
    up by what they do and switches the matching ones on for the rest of the session,
  * skill names only, and project instructions capped at 4,000 characters.

`prompt_profile` = auto (the default) picks lean when the context window is under 32k, or the model runs on
your own machine and its name says it has 14B parameters or fewer. Set "full" or "lean" to decide yourself.
"""

from __future__ import annotations

import re

from muyah_code.tools.base import META, Tool, ToolContext, ToolResult
from muyah_code.tools.registry import ToolRegistry

LEAN_WINDOW = 32768          # auto: lean below this window
SMALL_PARAMS_B = 14
CORE = ("Read", "Edit", "Write", "Bash", "Grep", "Glob")
INSTRUCTIONS_CAP = 4000

LEAN_PROMPT = """\
You are MUYAH-CODE, a coding agent in the user's terminal. You change files on disk with tools; showing code \
in chat is not the same as writing it.

# How you work
1. Look before you act: Grep/Glob to find code, Read a file before you Edit or overwrite it.
2. Edit precisely: copy old_string exactly from the file. Write complete code, never placeholders.
3. Check your work: run the tests or the program with Bash and read the output before saying it works.
4. When something fails, read the error and fix the cause. After 3 failed tries, change approach.
5. Do only what was asked. Never delete files yourself; give the user the command instead.
6. Be brief: lead with the result, then how you checked it.
7. Be honest: check claims before agreeing; if the evidence says otherwise, say so.
8. Need another kind of tool (web search, todo list, skills, sub-agents...)? Call find_tools first."""

LEAN_DESCRIPTIONS = {
    "Read": "Read a file (with line numbers). Args: file_path, optional offset/limit for big files.",
    "Edit": "Replace old_string with new_string in a file. old_string must match the file exactly and be unique "
            "(or set replace_all). Read the file first.",
    "Write": "Create or overwrite a file with content. Read an existing file first.",
    "Bash": "Run a shell command in the project folder and get its output and exit code.",
    "Grep": "Search file contents with a regex. output_mode: files_with_matches (default) or content.",
    "Glob": "Find files by name pattern, e.g. **/*.py.",
}


def param_billions(model: str) -> float | None:
    """'qwen2.5-coder:7b' -> 7, 'Qwen3-Coder-30B-A3B' -> 30 (total size), 'gpt-4.1' -> None."""
    m = re.search(r"(?:^|[^a-z0-9.])(\d+(?:\.\d+)?)\s*b(?![a-z])", (model or "").lower())
    return float(m.group(1)) if m else None


def choose_profile(setting: str, window: int, model: str, self_hosted: bool) -> str:
    setting = (setting or "auto").lower()
    if setting in ("lean", "full"):
        return setting
    if window and window < LEAN_WINDOW:
        return "lean"
    size = param_billions(model)
    if self_hosted and size is not None and size <= SMALL_PARAMS_B:
        return "lean"
    return "full"


def _trim_params(schema: dict) -> dict:
    """Keep names, types, enums and required; drop the long per-parameter descriptions."""
    out = {k: v for k, v in schema.items() if k != "properties"}
    props = {}
    for name, spec in (schema.get("properties") or {}).items():
        if isinstance(spec, dict):
            props[name] = {k: v for k, v in spec.items() if k != "description"}
        else:
            props[name] = spec
    out["properties"] = props
    return out


class LeanTool(Tool):
    """A core tool with a short description (it behaves exactly like the full one)."""

    def __init__(self, inner: Tool):
        self.inner = inner
        self.name = inner.name
        self.kind = inner.kind
        self.aliases = inner.aliases
        self.description = LEAN_DESCRIPTIONS.get(inner.name, inner.description.split("\n")[0][:200])
        self.parameters = _trim_params(inner.parameters)

    def run(self, args, ctx):
        return self.inner.run(args, ctx)

    def permission_subject(self, args, ctx):
        return self.inner.permission_subject(args, ctx)

    def title(self, args):
        return self.inner.title(args)

    def is_read_only(self, args):
        return self.inner.is_read_only(args)

    def preview(self, args, ctx):
        return self.inner.preview(args, ctx)


class FindToolsTool(Tool):
    name = "find_tools"
    kind = META
    description = ("Look up more tools by what you need (e.g. 'search the web', 'todo list', 'run a skill', "
                   "'delegate to a sub-agent', or an MCP server's name). Matching tools are switched on and can be "
                   "called from your next reply. An empty query lists everything available.")
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}, "required": []}

    def __init__(self, registry: ToolRegistry, hidden: list[Tool]):
        self.registry = registry
        self.hidden = {t.name: t for t in hidden}

    def title(self, args):
        return f"find_tools({args.get('query') or ''})"

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        query = str(args.get("query") or "").lower().strip()
        waiting = [t for t in self.hidden.values() if self.registry.get(t.name) is None]
        if not waiting:
            return ToolResult("Every tool is already switched on.")
        if not query:
            return ToolResult("Available (call find_tools with a word from the one you need):\n" + "\n".join(
                f"- {t.name}: {t.description.splitlines()[0][:120]}" for t in waiting))
        words = [w for w in re.split(r"\W+", query) if len(w) > 2] or [query]

        def score(t: Tool) -> int:
            text = f"{t.name} {t.description}".lower()
            return sum(text.count(w) for w in words) + (5 if any(w in t.name.lower() for w in words) else 0)

        ranked = sorted((t for t in waiting if score(t) > 0), key=score, reverse=True)[:3]
        if not ranked:
            return ToolResult("No tool matches that. Available: " + ", ".join(t.name for t in waiting))
        for t in ranked:
            self.registry.register(t)
        return ToolResult("Switched on (call them from your next reply):\n\n" + "\n\n".join(
            f"## {t.name}\n{t.description}" for t in ranked), summary=", ".join(t.name for t in ranked))


def lean_registry(full: ToolRegistry) -> ToolRegistry:
    core = [LeanTool(t) for name in CORE if (t := full.get(name)) is not None]
    hidden = [t for t in full.tools() if t.name not in CORE]
    reg = ToolRegistry(core)
    if hidden:
        reg.register(FindToolsTool(reg, hidden))
    return reg


def lean_instructions(instructions: list, cap: int = INSTRUCTIONS_CAP) -> str:
    """Project instructions within `cap` characters (the nearest files first: they are the most specific)."""
    out, used = [], 0
    for f in reversed(instructions):
        text = f.text.strip()
        room = cap - used
        if room <= 200:
            break
        if len(text) > room:
            text = text[:room - 40].rstrip() + "\n[... cut to fit lean mode ...]"
        out.append(f"## From {f.path}\n{text}")
        used += len(text)
    return "\n\n".join(reversed(out))
