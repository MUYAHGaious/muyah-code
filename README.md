# 🚀 Ultimate AI Coding Agent (2026 Edition)

A **state-of-the-art autonomous CLI coding agent** powered by open-weights LLMs running on **Google Colab (A100/L4 GPU)**. Modeled after Claude Code, Aider, and OpenHands — but free and self-hosted.

---

## ⚡ What Makes This Special

| Feature | Description |
| :--- | :--- |
| 🛠️ **17 Tools** | File ops, web search, web scraping, git, code search, sub-agents, memory |
| 🌐 **Web Research** | DuckDuckGo search + webpage scraping — no API keys needed |
| 🔀 **Git Integration** | Auto-commit, diff review, undo, branch-aware |
| 🧠 **Persistent Memory** | `AGENT.md` remembers project conventions across sessions |
| 🤖 **Sub-Agent Spawning** | Delegates research tasks to isolated contexts |
| 🗜️ **Context Compaction** | Auto-summarizes when context grows too large |
| 🔌 **Dual Tool Calling** | Native OpenAI function calling + XML/text fallback |
| 🎨 **Rich Terminal UI** | Syntax-highlighted diffs, tree views, markdown, spinners |

---

## 📁 Project Files

| File | Description |
| :--- | :--- |
| `code_agent.py` | The main CLI coding agent |
| `colab_server.ipynb` | Jupyter notebook for Google Colab |
| `colab_script.py` | Single-cell script to paste in Colab |
| `test_connection.py` | Quick test for API connectivity |
| `install_global.py` | Install `ccode` as a global command |
| `requirements.txt` | Python dependencies |

---

## ⚡ Quickstart Guide

### Step 1: Start Model Server in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Upload `colab_server.ipynb` or paste `colab_script.py` in a cell.
3. Set runtime: **A100 GPU** + **High-RAM**.
4. Run all cells. Wait for the tunnel URL:
   ```
   🎉 TUNNEL ACTIVE!
   API Base URL: https://xxxx.trycloudflare.com/v1
   ```

### Step 2: Install Local Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Agent

```bash
python code_agent.py
```

Or install globally:
```bash
python install_global.py
ccode  # Run from anywhere!
```

Paste your Colab URL when prompted, or use `/url` inside the CLI.

---

## 🎮 Commands

| Command | Description |
| :--- | :--- |
| `/url <url>` | Set Colab Cloudflare tunnel URL |
| `/model <name>` | Switch model |
| `/models` | List available models on server |
| `/auto` | Toggle auto-approve commands (YOLO mode) |
| `/compact` | Force context compaction |
| `/memory` | View AGENT.md persistent memory |
| `/stats` | Show session statistics |
| `/git` | Show git status |
| `/undo` | Undo last git commit |
| `/clear` | Clear conversation history |
| `/exit` | Quit |

---

## 🛠️ All 17 Tools

| Tool | Description |
| :--- | :--- |
| `read_file` | Read file contents with line numbers |
| `write_file` | Create or overwrite files (auto-creates directories) |
| `edit_file` | Targeted find-and-replace with diff display |
| `list_directory` | List files and folders |
| `tree` | Recursive directory tree view |
| `find_files` | Glob-based file search |
| `search_code` | Grep-like code search across workspace |
| `run_command` | Execute shell commands (with permission) |
| `web_search` | DuckDuckGo web search (no API key) |
| `fetch_webpage` | Scrape and clean a URL |
| `git_status` | Branch, modified files, recent commits |
| `git_diff` | Show code diffs |
| `git_commit` | Stage all + commit |
| `git_undo` | Undo last commit (soft reset) |
| `spawn_agent` | Spawn sub-agent for isolated tasks |
| `read_memory` | Read AGENT.md persistent memory |
| `write_memory` | Write to AGENT.md persistent memory |

---

## 🧠 AGENT.md — Persistent Memory

Create an `AGENT.md` file in your workspace root (or let the agent create it). The agent reads it at session start and can write to it. Use it for:
- Project conventions ("use pytest", "prefer async")
- Architecture notes ("auth logic in `src/auth/`")
- Known issues and gotchas
- Build/deploy instructions

---

## 💡 Example Usage

```
> Create a REST API with FastAPI that has user CRUD endpoints

🤖 Agent will:
  1. Search existing project structure (tree, list_directory)
  2. Create main.py with FastAPI app
  3. Create models.py, schemas.py, routes/
  4. Install dependencies (run_command: pip install fastapi uvicorn)
  5. Verify syntax (run_command: python -m py_compile main.py)
  6. Commit changes (git_commit)
```
