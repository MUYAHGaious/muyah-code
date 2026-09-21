---
name: tdd
description: Use when implementing a feature or bugfix where tests exist or can be added - write the failing test first, then the minimal code to pass it.
---
# Test-driven development

## The cycle (repeat per behavior)
1. **Red** - write ONE small test for the next behavior. Run it and watch it FAIL for the expected reason
   (an assertion, not an import/syntax error). A test that passes immediately proves nothing.
2. **Green** - write the minimal code that makes it pass. No extra features.
3. **Refactor** - clean up names/duplication with all tests green. Run the tests again.

## Guidelines
- Find the project's test conventions first (framework, file naming, fixtures): Glob for existing tests and
  mirror them. Find the exact command to run a single test.
- Test behavior through public interfaces, not private internals.
- One reason to fail per test; clear names like `test_parse_rejects_negative_duration`.
- For bugs: the first test reproduces the bug. It must fail before the fix and pass after.
- Keep tests fast and deterministic: no real network, fixed seeds, temp directories.
- Finish by running the full suite, not just the new test.
