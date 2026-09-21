# AGENTS.md: working on MUYAH-CODE

Read by MUYAH-CODE, Claude Code, OpenCode and other agents working in this repository.

## Project
MUYAH-CODE is a Python 3.10+ agentic coding CLI (`muyah_code/`) for OpenAI-compatible model servers.
MUYAH-CODE replaces the original single-file colab-code agent (see the first commit in git history).

## Commands
- Install (dev): `python -m pip install -e ".[dev]"` (on Windows without admin, add `--user`)
- All tests: `python -m pytest -q`
- One test: `python -m pytest -q tests/test_agent_e2e.py::test_native_tool_loop_writes_file_and_answers`
- Lint: `python -m ruff check muyah_code tests backend`
- Run: `python -m muyah_code` (interactive) or `python -m muyah_code -p "..." --base-url URL -m MODEL`

## Architecture (where things go)
- `llm/client.py`: the only place that talks HTTP to the model (streaming, retries, overflow detection).
- `llm/toolcall_parser.py`: the strict text tool protocol. Never add heuristics that turn plain code blocks
  into file writes (that is what made the old colab-code agent write junk files).
- `tools/`: one `Tool` subclass per tool. Expected failures raise `ToolError` (the message goes back to the model).
- `agent/loop.py`: the turn loop. `agent/context.py`: budgets and compaction. `agent/prompts.py`: system prompt.
- `app.py`: wires everything together. `cli.py`: argument parsing only.
- UI code only lives in `ui/`. The loop talks to the `UI` interface in `ui/base.py`, never to Rich directly.

## Rules
- Verify before claiming done: run the relevant tests and `ruff check`, and read the output.
- Every behavior change needs a test. Agent-loop behavior is tested end to end against `tests/fakeserver.py`
  (a scripted OpenAI-compatible server). No test may touch the real `~/.muyah` (the `isolated_home` fixture handles that).
- Keep the system prompt byte-stable between turns (prefix caching). Put per-turn material in `turn_context`.
- Windows is a first-class platform: use `pathlib`, `encoding="utf-8"`, preserve CRLF, and remember that the
  Bash tool may be Git Bash or PowerShell.
- No placeholders, TODO stubs, or silent `except: pass` around real work.
