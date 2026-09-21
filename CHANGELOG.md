# Changelog

All notable changes to MUYAH-CODE. Versions follow [semantic versioning](https://semver.org):
`MAJOR.MINOR.PATCH`. Bump PATCH for fixes, MINOR for new features, MAJOR for breaking changes.

## [Unreleased]

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
