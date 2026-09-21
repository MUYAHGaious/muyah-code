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
        "Ask the user a question when you are blocked on a decision only they can make (ambiguous requirements, "
        "choosing between valid approaches). Provide 2-4 concrete options when possible. Do not use it for "
        "permission to proceed or for things you can find out yourself."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question"},
            "options": {"type": "array", "items": {"type": "string"}, "description": "Suggested answers"},
        },
        "required": ["question"],
    }

    def run(self, args, ctx):
        ui = ctx.service("ui")
        if ctx.headless or ui is None:
            raise ToolError("No user is available (headless run). Make the most reasonable assumption, state it "
                            "explicitly in your final answer, and continue.")
        answer = ui.ask_user(args["question"], [str(o) for o in (args.get("options") or [])])
        if not answer:
            return ToolResult("The user did not answer. Proceed with your best judgement and say what you assumed.")
        return ToolResult(f"User answered: {answer}", summary=answer[:80])


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
        return ToolResult(skill.render(args.get("args", "")), summary=f"Loaded skill {skill.name}")
