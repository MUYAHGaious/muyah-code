# Changelog

All notable changes to MUYAH-CODE. Versions follow [semantic versioning](https://semver.org):
`MAJOR.MINOR.PATCH`. Bump PATCH for fixes, MINOR for new features, MAJOR for breaking changes.

## [Unreleased]

- **Fix: the spinner froze.** A new spinner was built on every redraw, and Rich spinners pick their frame from their age, so they always showed the first frame while the timer kept counting. The spinner now animates the whole time (verified: all 10 frames, hundreds of changes per turn).
- **The spinner never goes quiet during a turn.** Between steps it shows "Working…"; the learning step after a turn shows "Learning from this turn…". While thinking, the words change every few seconds.
- **Compaction shows progress:** "Compacting the conversation (N messages, X tokens)" while it runs, then the size before → after.
- **The input box stays on screen while it works**, and **Shift+Tab changes the mode mid-turn**; it applies from the agent's next action. On Windows the key reader now reads console input records, which is the only way to tell Shift+Tab from Tab.
- **Live view logs:** the Activity panel's detail view is the full log of every state change. Filter it by kind (you, model calls, tools, commands, sub-agents, hooks and lessons, errors) and open any line to see the raw event.
- README: a showcase for running huge models on your own hardware with colibri and Soup, and an accurate note on prompt caching.
- **`/usage`** (also `muyah usage`): requests and tokens for this session, today and the last 7 days, broken down by model. It also shows the **live limits your provider reports**: requests and tokens left, and when they reset. Groq, OpenRouter, OpenAI, Anthropic and others send these with every reply; if a provider sends none (Gemini), `/usage` says so. `/cost` shows the same.
- **An organized transcript**, following what Codex, Gemini CLI, OpenCode and Claude Code do:
  - Your prompt is a highlighted band, with no separator lines.
  - Each kind of action has its own glyph and verb: `$ Ran` for commands, `→ Read`, `✱ Searched`, `± Edited` (with +/- counts and the diff), `◈ Fetched`, `⇄ Called` for MCP, `◇ Using skill`, `◆ Agent`, `▣ Updated plan`.
  - Each tool prints once, when it finishes, and its glyph is green or red for the outcome. While it runs, it lives in the status area with a timer.
  - A sub-agent's steps show live under it, then collapse into one "✓ Done · N steps" line.
  - Quick lookups stack; other blocks get one blank line. Wrapped lines use hanging indents, and long paths are shortened in the middle.
  - Answers wrap at a readable width on wide terminals.
- **Type while it works.** What you type shows under the spinner. **Enter** queues it, and the model gets it at the next step (between tool calls), not after the whole turn. **Esc** stops the current step and sends it right away.
- **Interrupts are instant.** Esc and Ctrl+C now take effect in about 0.1 s even while the model is thinking; before, they waited for the server's next bytes.
- **Modes:** Shift+Tab cycles **plan → edit → manual → auto**, each with its own color (plan green, edit violet, auto yellow).
  - The new **auto** mode runs work on its own and asks only for risky actions: bulk or forced deletes, `git push`, `git reset --hard`, publishing, `sudo`, piping downloads into a shell, and edits outside the project.
  - `bypassPermissions` is only set with `--mode`, never by Shift+Tab.
- **Honest waiting.** Until the provider sends its first byte, the spinner says "Waiting for <model> · sent Ns ago", so provider queueing is not mistaken for MUYAH-CODE being stuck. Free-tier rate limits (e.g. Gemini 429) are explained as rate limits, with the retry delay, instead of "no credits".
- **The live view is a board of panels.** It works like a design canvas:
  - Pan and zoom; **Fit all**; **Follow** keeps the active panel in view.
  - Every panel shows real content: your prompt and queue, the model's true state and streaming text, the terminal with output, a panel per file (content, diffs, reads), one panel per sub-agent, skills, MCP servers, hooks, plan, lessons and context.
  - Click anything for its full detail.
  - Drag panels to arrange them and collapse them to a title line. Your layout is remembered: the view now uses a stable address, http://127.0.0.1:47433, when that port is free.
  - Files, requests, commands and skills move along the connectors as chips, only on real events.
  - No glows. Replay from 0.1× to 16×.
- **Live view:**
  - Events now fire at the real moments: requested, awaiting approval, approved/denied, running, done; for the model: sent, first token.
  - Parallel reads used to report "started" after they had finished.
  - Events carry file contents, diffs, command output, skill text and the extensions inventory (skills, agents, MCP tools, hooks).
  - At startup MUYAH-CODE offers to open the live view (Always / Never are remembered), and the status line shows when it is on.
  - `muyah viz --replay --speed` accepts 0.1 (slow motion) to 16.
- In plan mode, chained read-only commands such as `python --version && pip --version` are allowed.
- **Faster, smoother CLI** (measured in a real Windows pseudo-console, with a 2000-message conversation):
  - Enter → request reaching the model: **~1.2 s → 77 ms**. Requests no longer pass through the OpenAI SDK's per-request walk over the whole conversation, which cost about a second and grew with every message.
  - The input box no longer redraws itself twice a second while idle.
  - Transcript and event writes moved to a background writer. Each file open cost ~60 ms on Windows and blocked the main thread mid-turn.
  - Resuming a long conversation shows only its last 8 exchanges instead of reprinting everything (it took 7+ seconds).
- **Web search** uses the maintained `ddgs` package. Before, it could return 0 results and print a deprecation warning into the conversation; library warnings no longer reach the screen.
- **Live view follows your session.** `muyah viz` in a second terminal now follows the session running in the same folder, in real time. It switches to the next session when you start another. Replays moved to `muyah viz --replay [id]`.
- The live view now also shows:
  - the model's answer and thinking as it streams
  - a **Right now** list of what is running, with timers
  - **MCP** servers and their calls
  - **hooks** as they fire
  - model calls that fail
- **Fix: project root.** Your global `~/.muyah` folder no longer marks your home folder as a project. Before, every folder under your home without `.git` counted as one project rooted at your home directory: they shared sessions, and "inside the project" (acceptEdits) meant your entire home folder. Sessions saved from such folders before this fix won't show up in `/resume` from those folders.
- MCP now has an end-to-end test against a real stdio server (`tests/fakemcp.py`).
- **Fix: Gemini tool calls.** Gemini attaches a *thought signature* to every tool call and rejects the next request without it ("Function call is missing a thought_signature"). MUYAH-CODE now sends provider fields on tool calls back unchanged, and drops them if you switch to another provider mid-conversation. That error also no longer switches the session to the text tool protocol: only real "no tool support" errors do.
- **Watch it think: `/viz`** opens a live view of the agent in your browser. It shows prompt → context → model, streaming tokens, tool calls and results, sub-agents, recalled lessons, the context window by part, and a timeline. **`muyah viz [id]`** replays any recorded session, with speed control and seeking. It is served on 127.0.0.1 with a random token and has no external requests.
- **`muyah --resume`** opens an arrow-key list of this folder's recent conversations (title, age, message count). The one you pick opens with its prompts, tool calls and answers back on screen. `/resume` without an id does the same inside a session, and `muyah --resume <id>` still works.
- **Workspace trust check:** the first time you open a folder, MUYAH-CODE asks whether you trust it before loading anything from it. That folder's hooks and MCP servers can run commands, so they wait for your answer.
- **New look:** an M-shaped mascot (with a short shimmer on start) next to the name, model and folder. The input sits between two rules with the status line underneath.
- **Menus take only the space they need:** "Enter to confirm · Esc to cancel", and the number keys still work as shortcuts.
- **Permission prompts** show the command or diff once, under a clear heading ("Run command", "Edit file"), then ask "Do you want to proceed?".
- **Command rules are normalized:** `"C:\...\python.exe" -m pytest` matches `Bash(python:*)`, and that clean rule is the one suggested.
- **README** has badges and real screenshots, regenerated by `scripts/make_screenshots.py`.

- **Arrow-key menus everywhere.** Typing `/` opens the command menu with descriptions. Providers, models, permission prompts and questions are all chosen with ↑/↓ and Enter instead of typed numbers.
- **Exiting:** Ctrl+C clears the line, and pressing it twice on an empty line exits (Ctrl+D also exits).
- **Input field like Claude Code:** a thin rule with the project folder name above the `❯` prompt, and a status line directly under it (mode · model · context · `/ for commands`).
- **Compact startup header** (model, folder and one hint) in place of the banner and stats box. The details moved to the new `/status` command.
- **Model picker** ranks by newest version and hides speech, image and embedding models. When a model is retired, it offers the provider's suggested replacement.
- **Billing errors:** "no credits / quota" is explained clearly, and a valid key is kept instead of being thrown away.

## [1.0.0] - 2026-09-21

First public release. MUYAH-CODE is a from-scratch rebuild of colab-code.

- An agent loop for any OpenAI-compatible model. It uses native tool calling and falls back to a strict text protocol when a server doesn't support it.
- Native Claude support through the Anthropic SDK: adaptive thinking, prompt caching and refusal fallbacks.
- `muyah login` / `/provider`: pick one of 20 providers, paste a key, and it's ready. Keys are stored separately from settings.
- Context compaction that adapts to any window size, from 8k to 1M, and calibrates itself against the server's token counts.
- 14 tools, permission modes, hooks, MCP, sub-agents, resumable sessions and `/undo`.
- Skills (debugging, planning, verification, tdd, code-review, brainstorming, commit) and learning from its own mistakes (lessons + `muyah eval`).
- Local and remote model servers: `muyah serve` (colibri, soup, ollama, vllm, llama.cpp), `muyah connect`, and a Colab server notebook.
- A terminal UI with streaming markdown, live stats and themes (default: light teal).
