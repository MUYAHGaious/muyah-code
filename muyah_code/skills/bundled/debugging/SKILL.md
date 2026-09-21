---
name: debugging
description: Use when anything fails - a bug, an error, a failing test, unexpected behavior - BEFORE proposing a fix. Finds the root cause systematically instead of guessing.
---
# Systematic debugging

Never fix a symptom you do not understand. Guess-and-check wastes time and creates new bugs.

## 1. Reproduce
- Run the failing command/test yourself with Bash and read the COMPLETE error output, bottom to top.
- If you cannot reproduce it, find out why before changing code (environment, input, order, timing).

## 2. Locate
- Read the stack trace: the first frame in project code is usually where to look.
- Read the code on that path (Read with offset/limit around the line). Grep for the function's callers.
- Check what changed recently: `git diff`, `git log -5 --stat`.

## 3. Hypothesize - one at a time
- State ONE specific hypothesis: "X is None because Y is never called when Z".
- Design the smallest check that proves or disproves it: a print, an assert, a one-line script, a narrower test.
- If disproved, discard it completely. Do not stack fixes for several hypotheses.

## 4. Fix the root cause
- Change the code where the wrong value/behavior ORIGINATES, not where it finally crashes.
- Keep the fix minimal. No unrelated refactoring.
- If a test is missing for this bug, add one that fails before the fix.

## 5. Verify
- Re-run the exact reproduction, then the surrounding test suite.
- Report: the root cause in one sentence, the fix, and the evidence (command + result).

## Escalation rule
After 3 failed fix attempts STOP. Re-read the relevant code from scratch, question your assumptions (is it the
right file? the right environment? a cached/stale build?), and try a different approach or ask the user.
