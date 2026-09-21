---
name: code-review
description: Use to review a diff, branch, or set of files for bugs before merging, or when the user asks for a review. Focuses on real defects, ranked by severity.
---
# Code review

## Gather
- `git diff` (or `git diff main...HEAD`, or the files named by the user). Read each changed file around the
  hunks - a diff alone hides context.
- Understand the intent: commit messages, the task description, related tests.

## Look for real defects, in this order
1. **Correctness** - wrong logic, off-by-one, unhandled None/empty/error cases, wrong conditions, broken
   invariants, race conditions, resource leaks.
2. **Security** - injection (SQL/shell/path), secrets in code, missing auth checks, unsafe deserialization.
3. **Data loss / compatibility** - destructive migrations, changed public APIs, config changes.
4. **Tests** - is the new behavior tested? Would the tests catch a regression?
5. **Clarity** - only when it would genuinely mislead the next reader.

Verify each suspected bug before reporting it: trace the code path, or write a quick check. Drop anything you
cannot substantiate. Do not pad the review with style nitpicks.

## Report
For each finding: `path:line` - what is wrong - a concrete failing scenario - the suggested fix.
Most severe first. If nothing significant is found, say so plainly.
