"""Planning and interaction tools: TodoWrite, AskUser, Skill."""

from __future__ import annotations

from muyah_code.tools.base import META, Tool, ToolError, ToolResult

STATUSES = ("pending", "in_progress", "completed")
MARK = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def render_todos(todos: list[dict]) -> str:
    return "\n".join(f"{MARK.get(t['status'], '[ ]')} {t['content']}" for t in todos) or "(empty)"


class TodoWriteTool(Tool):
    name = "TodoWrite"
    kind = META
    description = (
        "Create or update your task list for the current work. Use it for any task with 3+ steps: write all "
        "steps first, keep exactly ONE item in_progress, and mark items completed IMMEDIATELY when done. "
        "Send the full list every time (it replaces the previous one). Skip it for trivial one-step requests."
    )
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The complete, updated todo list",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Imperative task, e.g. 'Run the tests'"},
                        "status": {"type": "string", "enum": list(STATUSES)},
                        "activeForm": {"type": "string", "description": "Present tense, e.g. 'Running the tests'"},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    }

    def title(self, args):
        todos = args.get("todos") or []
        done = sum(1 for t in todos if isinstance(t, dict) and t.get("status") == "completed")
        return f"TodoWrite({done}/{len(todos)} done)"

    def run(self, args, ctx):
        raw = args["todos"]
        clean = []
        for i, t in enumerate(raw, 1):
            if isinstance(t, str):
                t = {"content": t, "status": "pending"}
            if not isinstance(t, dict) or not str(t.get("content", "")).strip():
                raise ToolError(f"todo #{i} must be an object with 'content' and 'status'")
            status = str(t.get("status", "pending")).lower().replace(" ", "_").replace("-", "_")
            if status in ("done", "complete"):
                status = "completed"
            if status in ("active", "doing", "in_process"):
                status = "in_progress"
            if status not in STATUSES:
                raise ToolError(f"todo #{i} has invalid status '{t.get('status')}'; use {', '.join(STATUSES)}")
            clean.append({"content": str(t["content"]).strip(), "status": status,
                          "activeForm": str(t.get("activeForm") or t["content"]).strip()})
        ctx.todos[:] = clean
        ui = ctx.service("ui")
        if ui is not None:
            ui.on_todos(clean)
        events = ctx.service("events")
        if events is not None:
            events.emit("todos", items=[{"content": t["content"], "status": t["status"]} for t in clean])
        active = sum(1 for t in clean if t["status"] == "in_progress")
        note = ""
        if active > 1:
            note = "\nWarning: more than one task is in_progress; keep exactly one active."
        elif clean and all(t["status"] == "completed" for t in clean):
            note = "\nAll tasks completed. Verify the result (run tests/commands) before reporting done."
        return ToolResult(f"Todo list updated:\n{render_todos(clean)}{note}",
                          summary=f"{sum(t['status'] == 'completed' for t in clean)}/{len(clean)} done")


class AskUserTool(Tool):
    name = "AskUser"
    kind = META
    description = (
        "Ask the user 1-4 questions when you are blocked on decisions only they can make (unclear requirements, "
        "choosing between valid approaches). Each question has a short header (a chip, max 12 characters), 2-4 "
        "options with a label (1-5 words) and a one-line description of what it means or costs; put your "
        "recommended option first and add \" (Recommended)\" to its label. The user can always type another "
        "answer. multiSelect: true when several options can apply. Do not ask for permission to proceed or for "
        "things you can find out yourself."
    )
    parameters = {
        "type": "object",
        "properties": {
            "questions": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The full question, ending with ?"},
                    "header": {"type": "string", "description": "Very short label, e.g. \"Auth method\""},
                    "options": {"type": "array", "minItems": 2, "maxItems": 4, "items": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}, "description": {"type": "string"}},
                        "required": ["label"]}},
                    "multiSelect": {"type": "boolean"},
                },
                "required": ["question", "options"]}},
            "question": {"type": "string", "description": "Short form for one question (with `options` as strings)"},
            "options": {"type": "array", "items": {"type": "string"}},
        },
    }

    def title(self, args):
        qs = _questions(args)
        return f"AskUser({qs[0]['question'][:60]})" if qs else "AskUser"

    def run(self, args, ctx):
        questions = _questions(args)
        if not questions:
            raise ToolError("Give at least one question: questions=[{question, header, options:[{label, description}]}]")
        ui = ctx.service("ui")
        if ctx.headless or ui is None:
            raise ToolError("No user is available (headless run). Make the most reasonable assumption, state it "
                            "explicitly in your final answer, and continue.")
        answers = ui.ask_questions(questions)
        lines = []
        for q, a in zip(questions, answers, strict=False):
            text = ", ".join(a) if isinstance(a, list) else (a or "")
            lines.append(f"- {q['question']} → {text or '(skipped: use your best judgement and say what you assumed)'}")
        summary = "; ".join((", ".join(a) if isinstance(a, list) else a) for a in answers if a)
        return ToolResult("The user answered:\n" + "\n".join(lines), summary=summary[:80] or "no answer")


def _questions(args: dict) -> list[dict]:
    """Both forms: questions=[...] (with header, described options, multiSelect), or question + options."""
    out = []
    for q in args.get("questions") or []:
        if isinstance(q, dict) and q.get("question"):
            opts = [o if isinstance(o, dict) else {"label": str(o)} for o in (q.get("options") or [])]
            out.append({"question": str(q["question"]), "header": str(q.get("header", "") or "")[:24],
                        "options": [{"label": str(o.get("label", "")), "description": str(o.get("description", "") or "")}
                                    for o in opts if o.get("label")],
                        "multiSelect": bool(q.get("multiSelect"))})
    if not out and args.get("question"):
        out.append({"question": str(args["question"]), "header": "",
                    "options": [{"label": str(o), "description": ""} for o in (args.get("options") or [])],
                    "multiSelect": False})
    return out


class SkillTool(Tool):
    name = "Skill"
    kind = META
    description = (
        "Load a skill: a proven step-by-step workflow for a kind of task (debugging, planning, TDD, verification, "
        "code review...). When a listed skill matches your task, load it BEFORE starting and follow it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "description": "Skill name from the available skills list"},
            "args": {"type": "string", "description": "Optional arguments for the skill"},
        },
        "required": ["skill"],
    }

    def title(self, args):
        return f"Skill({args.get('skill', '')})"

    def run(self, args, ctx):
        skills = ctx.service("skills")
        if skills is None:
            raise ToolError("Skills are not available.")
        skill = skills.get(args["skill"])
        if skill is None:
            raise ToolError(f"Unknown skill '{args['skill']}'. Available: {', '.join(skills.names()) or 'none'}")
        if skill.disable_model_invocation:
            raise ToolError(f"Skill '{skill.name}' can only be started by the user (/{skill.name}).")
        text = skill.render(args.get("args", ""))
        perms = ctx.service("permissions")
        if skill.allowed_tools and perms is not None:
            # like Claude Code: a skill's allowed-tools run without asking while you use it (this session)
            for rule in skill.allowed_tools:
                perms.add("allow", rule)
            text += ("\n\n(Allowed without asking for this session, from this skill: "
                     + ", ".join(skill.allowed_tools) + ")")
        return ToolResult(text, summary=f"Loaded skill {skill.name}")
