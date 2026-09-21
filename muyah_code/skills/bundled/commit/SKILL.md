---
name: commit
description: Use when the user asks to commit changes - reviews what is staged, writes a clear message in the repo's style, and commits safely.
---
# Commit

Only commit when the user asked for it.

1. Run `git status`, `git diff` (and `git diff --staged`), and `git log --oneline -10` to see the changes and
   the repository's message style.
2. Make sure nothing unintended is included: secrets (.env, keys, tokens), large binaries, build output, debug
   prints, temp files. Warn the user and leave those out.
3. If on the default branch (main/master) and the user did not say otherwise, suggest creating a branch first.
4. Stage the relevant files explicitly by path (avoid `git add -A` when unrelated changes exist).
5. Write the message: a concise summary line (<= 72 chars, imperative mood, matching the repo's convention),
   a blank line, then WHY the change was made if it is not obvious.
6. Commit. If a pre-commit hook fails, fix the issue and create a NEW commit - never bypass hooks with
   --no-verify, and never amend or force-push unless asked.
7. Show the result with `git log --oneline -1` and `git status`.
