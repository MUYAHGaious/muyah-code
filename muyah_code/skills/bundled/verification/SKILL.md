---
name: verification
description: Use before claiming any work is done, fixed, or passing - and before commits. Requires running real checks and reading their output; evidence before assertions.
---
# Verification before completion

"It should work" is not evidence. Before you say done/fixed/passing:

1. **Identify the proof.** What command demonstrates the change works? (the test file, the full suite, the
   build, running the program with the relevant input, a lint/type check).
2. **Run it** with Bash - fresh, after your last edit. Do not rely on an earlier run.
3. **Read the output completely.** Check the exit code, the pass/fail counts, warnings, and that the tests you
   expected actually ran (0 tests collected is a failure).
4. **Check for side effects.** Did anything else break? Run the broader suite if you touched shared code.
   `git diff --stat` to confirm you changed only what you intended - no stray debug prints or temp files.
5. **Report honestly:**
   - Verified: "`pytest -q` -> 48 passed" (quote the key line).
   - Not verifiable here (no test setup, needs a GPU/network/credentials): say exactly that, and what the user
     should run to verify.
   - Still failing: say so, show the error, and what you tried.

Never mark a todo completed, and never write a final summary claiming success, without step 2 and 3.
