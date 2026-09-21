---
name: planning
description: Use before multi-file or multi-step work (new features, refactors, migrations) to turn a request into a concrete, verifiable plan before touching code.
---
# Planning

## 1. Understand
- Restate the goal in one sentence. List explicit requirements and constraints from the user.
- Explore the relevant code first (Glob/Grep/Read, or an explore sub-agent for broad searches). Find existing
  helpers and patterns to reuse - do not reinvent what the codebase already has.
- If a requirement is genuinely ambiguous and changes the design, ask ONE focused question (AskUser).

## 2. Design
- Choose the simplest approach that fully meets the requirements. Note the main alternative only if the
  trade-off matters.
- Identify every file to create or change and what changes in each.
- Decide how you will verify it: which tests to add/run, which command proves it works.

## 3. Break it down
Write the steps with TodoWrite. Each step should be small (one file or one behavior), ordered so the code
works after each step, and end with verification. Example:
1. Add `parse_duration()` to utils/time.py with unit tests
2. Use it in cli.py for the --timeout flag
3. Run the full test suite and a manual `app --timeout 5m` check

## 4. Execute
Work through the list: exactly one item in_progress, mark completed immediately, verify as you go. If you
discover the plan is wrong, update the todo list instead of silently drifting.

In plan mode: stop after step 3 and present the plan (context, files, steps, verification) for approval.
