"""ExitPlanMode: the plan is ready; the user decides how to go on (like Claude Code's plan approval).

In plan mode the agent explores read-only and then calls this tool with its plan. The user sees the plan and
picks: go ahead in auto mode, go ahead accepting edits, go ahead approving each change, or keep planning
(saying what to change). On a yes the mode switches and the agent starts implementing in the same turn.
"""

from __future__ import annotations

from muyah_code.tools.base import META, Tool, ToolContext, ToolError, ToolResult

CHOICES = {
    "Yes, auto mode (runs on its own, asks only for risky actions)": "auto",
    "Yes, accept edits (asks before commands)": "acceptEdits",
    "Yes, approve each change (manual)": "default",
}
KEEP = "No, keep planning"
NAMES = {"auto": "auto", "acceptEdits": "edit", "default": "manual"}


class ExitPlanModeTool(Tool):
    name = "ExitPlanMode"
    kind = META
    description = ("Plan mode only: when the plan file is complete, call this to show it to the user. They choose how "
                   "to build it or ask you to keep planning. If approved, the mode switches and you build it right "
                   "away, following the plan file. `plan` is optional: the plan file is what is shown.")
    parameters = {"type": "object", "properties": {"plan": {"type": "string", "description": (
        "Only if you did not write the plan file: the plan, in markdown (it is saved to the plan file)")}}}

    def __init__(self, app):
        self.app = app

    def set_mode(self, mode):
        return self.app.set_mode(mode)

    def title(self, args):
        return "ExitPlanMode(plan ready)"

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        perms = ctx.service("permissions")
        if perms is None or perms.mode != "plan":
            raise ToolError("You are not in plan mode: go ahead with the task (no approval step is needed).")
        path = getattr(perms, "plan_file", None)
        plan = str(args.get("plan") or "").strip()
        if path is not None and path.exists():
            written = path.read_text(encoding="utf-8").strip()
            plan = written or plan                  # the file (the user may have edited it) is the plan
        elif plan and path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(plan + "\n", encoding="utf-8")
        if not plan:
            raise ToolError("Write the plan to the plan file first (see your turn notes), then call ExitPlanMode.")
        if path is not None:
            plan += f"\n\n*Saved in `{ctx.rel(path)}`: edit it there if you like, then choose.*"
        ui = ctx.service("ui")
        if ctx.headless:
            ui.show_plan(plan)
            return ToolResult("Plan shown. This run is not interactive: stay in plan mode and end with the plan.",
                              summary="plan shown")
        answer = ui.approve_plan(plan, [*CHOICES, KEEP])
        mode = CHOICES.get(answer)
        if mode:
            self.app.active_plan = path
            self.set_mode(mode)                   # saved to the session with the approved plan
            where = f" in {ctx.rel(path)}" if path is not None else ""
            return ToolResult(f"The user approved the plan{where}. The mode is now {NAMES[mode]}. Build it now, step "
                              "by step: start with a TodoWrite list of its steps, and keep to the plan.",
                              summary=f"approved · {NAMES[mode]}")
        said = answer if answer and answer != KEEP else ""
        return ToolResult("The user wants to keep planning. " + (f"They said: {said}. " if said else
                          "Ask what they would like to change. ") + "Stay in plan mode and revise the plan file.",
                          summary="keep planning")
