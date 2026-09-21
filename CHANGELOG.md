# Changelog

All notable changes to MUYAH-CODE. Versions follow [semantic versioning](https://semver.org):
`MAJOR.MINOR.PATCH`. Bump PATCH for fixes, MINOR for new features, MAJOR for breaking changes.

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
