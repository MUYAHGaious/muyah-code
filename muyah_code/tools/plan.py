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
    description = ("Plan mode only: when your plan is complete, present it with this tool (markdown: the context, the "
                   "files to change and how, how you will verify). The user chooses how to proceed or asks you to keep "
                   "planning. If approved, the mode switches and you implement the plan right away.")
    parameters = {"type": "object", "properties": {"plan": {"type": "string", "description": "The plan, in markdown"}},
                  "required": ["plan"]}

    def __init__(self, set_mode):
        self.set_mode = set_mode

    def title(self, args):
        return "ExitPlanMode(plan ready)"

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        perms = ctx.service("permissions")
        if perms is None or perms.mode != "plan":
            raise ToolError("You are not in plan mode: go ahead with the task (no approval step is needed).")
        plan = str(args.get("plan") or "").strip()
        if not plan:
            raise ToolError("Put the plan itself in `plan` (markdown).")
        ui = ctx.service("ui")
        if ctx.headless:
            ui.show_plan(plan)
            return ToolResult("Plan shown. This run is not interactive: stay in plan mode and end with the plan.",
                              summary="plan shown")
        answer = ui.approve_plan(plan, [*CHOICES, KEEP])
        mode = CHOICES.get(answer)
        if mode:
            self.set_mode(mode)
            return ToolResult(f"The user approved the plan. The mode is now {NAMES[mode]}. Implement the plan now, "
                              "step by step: start with a TodoWrite list of its steps.", summary=f"approved · {NAMES[mode]}")
        said = answer if answer and answer != KEEP else ""
        return ToolResult("The user wants to keep planning. " + (f"They said: {said}. " if said else
                          "Ask what they would like to change. ") + "Stay in plan mode and revise the plan.",
                          summary="keep planning")
