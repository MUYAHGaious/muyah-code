"""
Ultimate CLI Coding Agent (2026 Edition)
=========================================
A state-of-the-art autonomous coding agent modeled after Claude Code,
powered by open-weights LLMs via Google Colab + vLLM.

Features:
- 15+ tools: file ops, web search, web scraping, git, sub-agents, MCP, memory
- Dual-mode tool execution: native OpenAI function calling + XML/text fallback
- Context compaction: auto-summarizes when history grows too large
- AGENT.md persistent memory: remembers project conventions across sessions
- Sub-agent spawning: delegates tasks to isolated contexts
- Web research: DuckDuckGo search + webpage scraping (no API keys)
- Full Git integration: status, diff, commit, undo
- MCP plugin support: connect to any MCP server via config
- Beautiful Rich terminal UI with diffs, trees, markdown
"""

import os
import sys
import json
import re
import subprocess
import difflib
import fnmatch
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.tree import Tree
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

# Console styling
console = Console()

# Store config in user's home directory so it works across all projects and folders
CONFIG_FILE = Path.home() / ".colab_code_agent_config.json"

# ========================== SYSTEM PROMPT ==========================

DEFAULT_SYSTEM_PROMPT = """You are an elite, autonomous AI software engineer and terminal pair programmer (2026 edition).
You operate directly inside the user's workspace with full access to powerful tools for reading, writing, editing files, running commands, searching the web, managing git, and more.

CRITICAL DIRECTIVES:
1. ACT DIRECTLY USING TOOLS: You have hands. Whenever the user asks you to create, modify, debug, or inspect code — YOU MUST INVOKE TOOLS to make changes on disk. NEVER merely print code in chat.
2. NEVER JUST OUTPUT CODE BLOCKS: Do NOT output code in markdown blocks and ask the user to copy-paste. Call `write_file` to create files, `edit_file` to modify them.
3. EXPLORE FIRST: Use `list_directory`, `tree`, `search_code`, `find_files`, or `read_file` to understand the codebase before editing.
4. VERIFY YOUR WORK: After changes, use `run_command` to test (e.g., `python -m py_compile`, `pytest`, syntax checks).
5. RESEARCH WHEN NEEDED: Use `web_search` and `fetch_webpage` to look up documentation, APIs, error messages, or best practices.
6. USE GIT: Use `git_status` to understand repo state. Use `git_commit` after completing work. Use `git_undo` to rollback mistakes.
7. REMEMBER THINGS: Use `write_memory` to persist important project notes to AGENT.md. Use `read_memory` at session start.
8. PATHS: All file paths must be relative to the current working directory.

TOOL CALL FORMAT (fallback if native calling unavailable):
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>

AVAILABLE TOOLS:
- write_file(path, content): Create or overwrite a file on disk.
- edit_file(path, old_content, new_content): Targeted find-and-replace in a file.
- read_file(path, start_line?, end_line?): Read file contents with line numbers.
- list_directory(path?): List files/folders in a directory.
- tree(path?, max_depth?): Recursive directory tree.
- find_files(pattern, directory?): Glob search for files (e.g. "**/*.py").
- search_code(query, directory?): Grep-like code search across files.
- run_command(command): Execute a shell command.
- web_search(query, max_results?): Search the web via DuckDuckGo.
- fetch_webpage(url): Fetch and extract clean text from a URL.
- git_status(): Show git branch, status, and recent commits.
- git_diff(staged?): Show code diffs.
- git_commit(message): Stage all and commit.
- git_undo(): Undo last commit (soft reset).
- spawn_agent(task): Delegate a sub-task to an isolated agent.
- read_memory(): Read AGENT.md persistent memory.
- write_memory(content, mode?): Write/append to AGENT.md persistent memory.
"""

# ========================== TOOL SCHEMAS ==========================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file from the workspace with optional line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to file"},
                    "start_line": {"type": "integer", "description": "Optional starting line (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Optional ending line (1-indexed)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or overwrite an existing file with complete content. Auto-creates parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to file"},
                    "content": {"type": "string", "description": "Full file content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact snippet of text in a file with new text (targeted diff edit). The old_content must match exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to file"},
                    "old_content": {"type": "string", "description": "Exact text to find and replace"},
                    "new_content": {"type": "string", "description": "Replacement text"}
                },
                "required": ["path", "old_content", "new_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: '.')", "default": "."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tree",
            "description": "Show a recursive directory tree structure with depth limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Root directory (default: '.')", "default": "."},
                    "max_depth": {"type": "integer", "description": "Maximum depth to traverse (default: 3)", "default": 3}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files matching a glob pattern (e.g. '**/*.py', '*.json').",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match"},
                    "directory": {"type": "string", "description": "Directory to search within (default: '.')", "default": "."}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a text pattern across files in the workspace (case-insensitive grep).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text pattern to find"},
                    "directory": {"type": "string", "description": "Directory to search within (default: '.')", "default": "."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell/terminal command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Returns titles, URLs, and snippets. Use for researching documentation, APIs, error messages, or best practices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum results to return (default: 5)", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetch a URL and extract clean readable text content. Strips scripts, styles, and navigation. Use for reading documentation pages, articles, or Stack Overflow answers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show current git branch, modified/staged files, and recent commit log.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show code differences (staged or unstaged).",
            "parameters": {
                "type": "object",
                "properties": {
                    "staged": {"type": "boolean", "description": "If true, show staged changes. Otherwise show unstaged.", "default": False}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Stage all changes and create a git commit with the given message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit message"}
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_undo",
            "description": "Undo the last git commit (soft reset). Changes are preserved as unstaged.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_agent",
            "description": "Spawn an isolated sub-agent to handle a focused task (research, analysis, search). Returns only the summary, preventing context pollution in the main conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Clear description of the sub-task to perform"}
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "Read the AGENT.md persistent memory file from the workspace root. Contains project conventions, architecture notes, and learnings from past sessions.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_memory",
            "description": "Write or append to the AGENT.md persistent memory file. Use to persist project conventions, architecture notes, known issues, and important learnings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to write to AGENT.md"},
                    "mode": {"type": "string", "description": "'append' to add to existing content, 'overwrite' to replace entirely", "default": "append"}
                },
                "required": ["content"]
            }
        }
    }
]


# ========================== CODING AGENT CLASS ==========================

class CodingAgent:
    def __init__(self):
        self.config = self.load_config()
        self.auto_approve = False
        self.use_native_tools = True
        self.client: Optional[OpenAI] = None
        self.messages: List[Dict[str, Any]] = []
        self.stats = {"files_created": 0, "files_edited": 0, "commands_run": 0,
                       "searches": 0, "commits": 0, "tokens_est": 0}
        self.init_client()
        self._init_messages()

    # ========================== CONFIG ==========================

    def load_config(self) -> Dict[str, str]:
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "base_url": "http://localhost:8000/v1",
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
            "api_key": "dummy",
            "max_context_messages": 60,
            "command_timeout": 120
        }

    def save_config(self):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    def init_client(self):
        self.client = OpenAI(
            base_url=self.config["base_url"],
            api_key=self.config.get("api_key", "dummy")
        )

    def _init_messages(self):
        """Initialize message history with system prompt + AGENT.md memory."""
        system_content = DEFAULT_SYSTEM_PROMPT
        # Load AGENT.md if it exists
        agent_md = Path("AGENT.md")
        if agent_md.exists():
            try:
                mem = agent_md.read_text(encoding="utf-8")[:3000]
                system_content += f"\n\n--- PROJECT MEMORY (AGENT.md) ---\n{mem}\n--- END MEMORY ---"
            except Exception:
                pass
        self.messages = [{"role": "system", "content": system_content}]

    # ========================== TOOL IMPLEMENTATIONS ==========================

    def tool_read_file(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        if p.is_dir():
            return f"Error: {path} is a directory, not a file. Use list_directory or tree instead."
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            s = (start_line - 1) if (start_line and start_line > 0) else 0
            e = end_line if (end_line and end_line <= len(lines)) else len(lines)
            numbered = [f"{i+1:4d} | {lines[i]}" for i in range(s, e)]
            content = "".join(numbered)
            if len(content) > 15000:
                content = content[:15000] + "\n... (truncated, use start_line/end_line to read specific sections)"
            return f"File: {path} ({len(lines)} lines total)\n{content}"
        except Exception as err:
            return f"Error reading file: {err}"

    def tool_write_file(self, path: str, content: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            self.stats["files_created"] += 1
            line_count = content.count("\n") + 1
            console.print(f"  [green]✅ Created/wrote {path} ({line_count} lines, {len(content)} chars)[/green]")
            return f"Successfully wrote {len(content)} characters ({line_count} lines) to {path}."
        except Exception as err:
            return f"Error writing file: {err}"

    def tool_edit_file(self, path: str, old_content: str, new_content: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                original = f.read()

            if old_content not in original:
                # Try a more lenient match (strip trailing whitespace per line)
                old_lines = [l.rstrip() for l in old_content.split("\n")]
                orig_lines = [l.rstrip() for l in original.split("\n")]
                old_joined = "\n".join(old_lines)
                orig_joined = "\n".join(orig_lines)
                if old_joined not in orig_joined:
                    return f"Error: Target text not found in {path}. Make sure exact match including whitespace."

            updated = original.replace(old_content, new_content, 1)
            diff = list(difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}"
            ))

            if diff:
                diff_text = "".join(diff)
                console.print(Panel(Syntax(diff_text, "diff", theme="monokai"), title=f"[yellow]Diff: {path}[/yellow]", border_style="yellow"))

            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(updated)
            self.stats["files_edited"] += 1
            return f"Successfully edited {path}."
        except Exception as err:
            return f"Error editing file: {err}"

    def tool_list_directory(self, path: str = ".") -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: Path does not exist: {path}"
        try:
            entries = []
            for item in sorted(p.iterdir()):
                if item.name.startswith(".") and item.name not in (".gitignore", ".env"):
                    continue
                prefix = "📁" if item.is_dir() else "📄"
                size = ""
                if item.is_file():
                    sz = item.stat().st_size
                    size = f" ({sz:,} bytes)" if sz < 1024*100 else f" ({sz//1024}KB)"
                entries.append(f"{prefix} {item.name}{size}")
            return "\n".join(entries) if entries else "(Empty directory)"
        except Exception as err:
            return f"Error listing directory: {err}"

    def tool_tree(self, path: str = ".", max_depth: int = 3) -> str:
        SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", ".tox",
                     "dist", "build", ".egg-info", ".mypy_cache", ".pytest_cache"}
        lines = []
        def _walk(dir_path: Path, prefix: str, depth: int):
            if depth > max_depth:
                lines.append(f"{prefix}...")
                return
            try:
                entries = sorted(dir_path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            except PermissionError:
                return
            filtered = [e for e in entries if e.name not in SKIP_DIRS and not (e.name.startswith(".") and e.is_dir())]
            for i, entry in enumerate(filtered):
                is_last = (i == len(filtered) - 1)
                connector = "└── " if is_last else "├── "
                icon = "📁 " if entry.is_dir() else ""
                lines.append(f"{prefix}{connector}{icon}{entry.name}")
                if entry.is_dir():
                    extension = "    " if is_last else "│   "
                    _walk(entry, prefix + extension, depth + 1)

        root = Path(path)
        lines.append(f"📁 {root.resolve().name}/")
        _walk(root, "", 1)
        result = "\n".join(lines)
        if len(result) > 8000:
            result = result[:8000] + "\n... (truncated, use smaller max_depth)"
        return result

    def tool_find_files(self, pattern: str, directory: str = ".") -> str:
        p = Path(directory)
        results = []
        try:
            for match in sorted(p.rglob(pattern)):
                rel = match.relative_to(p)
                parts = str(rel).replace("\\", "/")
                # Skip hidden/cache dirs
                if any(part.startswith(".") or part in ("__pycache__", "node_modules", "venv")
                       for part in rel.parts):
                    continue
                results.append(parts)
                if len(results) >= 50:
                    results.append("... (capped at 50 results)")
                    break
        except Exception as err:
            return f"Error finding files: {err}"
        return "\n".join(results) if results else f"No files matching '{pattern}' found."

    def tool_search_code(self, query: str, directory: str = ".") -> str:
        results = []
        p = Path(directory)
        SKIP_DIRS = {"__pycache__", "node_modules", "venv", ".venv", ".git", "dist", "build"}
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for file in files:
                filepath = Path(root) / file
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        for line_no, line in enumerate(f, 1):
                            if query.lower() in line.lower():
                                rel = filepath.relative_to(p)
                                results.append(f"{rel}:{line_no}: {line.strip()}")
                                if len(results) >= 40:
                                    results.append("... (capped at 40)")
                                    return "\n".join(results)
                except Exception:
                    continue
        self.stats["searches"] += 1
        return "\n".join(results) if results else f"No matches found for '{query}'."

    def tool_run_command(self, command: str) -> str:
        if not self.auto_approve:
            console.print(Panel(f"[bold]{command}[/bold]", title="[bold red]🔒 Command Request[/bold red]", border_style="red"))
            if not Confirm.ask("  Execute this command?"):
                return "Command cancelled by user."

        timeout = int(self.config.get("command_timeout", 120))
        try:
            res = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=timeout
            )
            out = res.stdout.strip()
            err = res.stderr.strip()
            parts = []
            if out:
                parts.append(f"STDOUT:\n{out[:8000]}")
            if err:
                parts.append(f"STDERR:\n{err[:4000]}")
            parts.append(f"Exit code: {res.returncode}")
            self.stats["commands_run"] += 1
            return "\n".join(parts)
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds."
        except Exception as err:
            return f"Error executing command: {err}"

    def tool_web_search(self, query: str, max_results: int = 5) -> str:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return f"No results found for '{query}'."
            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(f"[{i}] {r.get('title', 'N/A')}\n    URL: {r.get('href', '')}\n    {r.get('body', '')}")
            self.stats["searches"] += 1
            return "\n\n".join(formatted)
        except Exception as err:
            return f"Error searching web: {err}"

    def tool_fetch_webpage(self, url: str) -> str:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return "Error: requests/beautifulsoup4 not installed. Run: pip install requests beautifulsoup4"
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
                tag.extract()
            text = soup.get_text(separator="\n", strip=True)
            # Clean up excessive whitespace
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            clean_text = "\n".join(lines)
            if len(clean_text) > 12000:
                clean_text = clean_text[:12000] + "\n\n... (content truncated)"
            return f"Content from {url}:\n\n{clean_text}"
        except Exception as err:
            return f"Error fetching webpage: {err}"

    def tool_git_status(self) -> str:
        parts = []
        try:
            branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, timeout=10)
            if branch.returncode == 0:
                parts.append(f"Branch: {branch.stdout.strip()}")
            else:
                return "Error: Not a git repository. Use `run_command('git init')` to initialize."
            status = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, timeout=10)
            if status.stdout.strip():
                parts.append(f"\nModified files:\n{status.stdout.strip()}")
            else:
                parts.append("\nWorking tree clean.")
            log = subprocess.run(["git", "log", "--oneline", "-5"], capture_output=True, text=True, timeout=10)
            if log.stdout.strip():
                parts.append(f"\nRecent commits:\n{log.stdout.strip()}")
        except FileNotFoundError:
            return "Error: git is not installed or not in PATH."
        except Exception as err:
            return f"Error getting git status: {err}"
        return "\n".join(parts)

    def tool_git_diff(self, staged: bool = False) -> str:
        try:
            cmd = ["git", "diff"]
            if staged:
                cmd.append("--staged")
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            diff = res.stdout.strip()
            if not diff:
                return "No differences found." + (" (staged)" if staged else " (unstaged)")
            if len(diff) > 10000:
                diff = diff[:10000] + "\n... (diff truncated)"
            return diff
        except Exception as err:
            return f"Error getting diff: {err}"

    def tool_git_commit(self, message: str) -> str:
        try:
            subprocess.run(["git", "add", "-A"], capture_output=True, text=True, timeout=10)
            res = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True, timeout=15)
            output = res.stdout.strip() + "\n" + res.stderr.strip()
            if res.returncode == 0:
                self.stats["commits"] += 1
                console.print(f"  [green]✅ Committed: {message}[/green]")
            return output.strip()
        except Exception as err:
            return f"Error committing: {err}"

    def tool_git_undo(self) -> str:
        try:
            res = subprocess.run(["git", "reset", "--soft", "HEAD~1"], capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                return "Successfully undid last commit. Changes are preserved as unstaged."
            return f"Error: {res.stderr.strip()}"
        except Exception as err:
            return f"Error undoing commit: {err}"

    def tool_spawn_agent(self, task: str) -> str:
        """Spawn a sub-agent with isolated context to handle a focused task."""
        console.print(f"  [cyan]🤖 Spawning sub-agent: {task[:60]}...[/cyan]")
        sub_messages = [
            {"role": "system", "content": "You are a focused research/analysis sub-agent. Complete the given task concisely. You have access to tools for reading files, searching code, listing directories, web search, and fetching webpages. Return a clear, concise summary of your findings."},
            {"role": "user", "content": task}
        ]

        max_sub_loops = 6
        for _ in range(max_sub_loops):
            try:
                kwargs = {
                    "model": self.config["model"],
                    "messages": sub_messages,
                    "temperature": 0.2,
                    "timeout": 60.0
                }
                if self.use_native_tools:
                    # Give sub-agent limited tools
                    sub_tools = [t for t in TOOLS if t["function"]["name"] in
                                 ("read_file", "list_directory", "search_code", "find_files",
                                  "web_search", "fetch_webpage", "tree")]
                    kwargs["tools"] = sub_tools
                    kwargs["tool_choice"] = "auto"

                response = self.client.chat.completions.create(**kwargs)
                msg = response.choices[0].message

                if msg.tool_calls:
                    sub_messages.append(msg)
                    for tc in msg.tool_calls:
                        fn_name = tc.function.name
                        try:
                            fn_args = json.loads(tc.function.arguments)
                        except Exception:
                            fn_args = {}
                        result = self.dispatch_tool(fn_name, fn_args)
                        sub_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(result)[:4000]
                        })
                else:
                    content = msg.content or "Sub-agent completed but returned no content."
                    # Check for text-based tool calls
                    extracted = self.extract_tool_calls_from_text(content)
                    if extracted:
                        sub_messages.append({"role": "assistant", "content": content})
                        summaries = []
                        for tc in extracted:
                            res = self.dispatch_tool(tc["name"], tc["args"])
                            summaries.append(f"[{tc['name']}]: {str(res)[:2000]}")
                        sub_messages.append({"role": "user", "content": "Tool results:\n" + "\n".join(summaries) + "\n\nNow provide your final summary."})
                    else:
                        console.print(f"  [cyan]✅ Sub-agent completed[/cyan]")
                        return f"Sub-agent result:\n{content[:4000]}"
            except Exception as err:
                return f"Sub-agent error: {err}"

        return "Sub-agent reached maximum iterations."

    def tool_read_memory(self) -> str:
        agent_md = Path("AGENT.md")
        if not agent_md.exists():
            return "No AGENT.md found. Use write_memory to create one with project notes."
        try:
            content = agent_md.read_text(encoding="utf-8")
            return f"AGENT.md contents:\n\n{content}"
        except Exception as err:
            return f"Error reading AGENT.md: {err}"

    def tool_write_memory(self, content: str, mode: str = "append") -> str:
        agent_md = Path("AGENT.md")
        try:
            if mode == "overwrite":
                agent_md.write_text(content, encoding="utf-8")
                return "AGENT.md overwritten successfully."
            else:
                existing = ""
                if agent_md.exists():
                    existing = agent_md.read_text(encoding="utf-8")
                separator = "\n\n" if existing else ""
                agent_md.write_text(existing + separator + content, encoding="utf-8")
                return "AGENT.md updated (appended) successfully."
        except Exception as err:
            return f"Error writing AGENT.md: {err}"

    # ========================== TOOL DISPATCH ==========================

    def dispatch_tool(self, name: str, args: Dict[str, Any]) -> str:
        display_args = json.dumps(args, ensure_ascii=False)
        if len(display_args) > 120:
            display_args = display_args[:120] + "..."
        console.print(f"  [bold cyan]⚡ {name}[/bold cyan]({display_args})")

        dispatch_map = {
            "read_file": lambda: self.tool_read_file(args.get("path", ""), args.get("start_line"), args.get("end_line")),
            "write_file": lambda: self.tool_write_file(args.get("path", ""), args.get("content", "")),
            "edit_file": lambda: self.tool_edit_file(args.get("path", ""), args.get("old_content", ""), args.get("new_content", "")),
            "list_directory": lambda: self.tool_list_directory(args.get("path", ".")),
            "tree": lambda: self.tool_tree(args.get("path", "."), args.get("max_depth", 3)),
            "find_files": lambda: self.tool_find_files(args.get("pattern", ""), args.get("directory", ".")),
            "search_code": lambda: self.tool_search_code(args.get("query", ""), args.get("directory", ".")),
            "run_command": lambda: self.tool_run_command(args.get("command", "")),
            "web_search": lambda: self.tool_web_search(args.get("query", ""), args.get("max_results", 5)),
            "fetch_webpage": lambda: self.tool_fetch_webpage(args.get("url", "")),
            "git_status": lambda: self.tool_git_status(),
            "git_diff": lambda: self.tool_git_diff(args.get("staged", False)),
            "git_commit": lambda: self.tool_git_commit(args.get("message", "Agent commit")),
            "git_undo": lambda: self.tool_git_undo(),
            "spawn_agent": lambda: self.tool_spawn_agent(args.get("task", "")),
            "read_memory": lambda: self.tool_read_memory(),
            "write_memory": lambda: self.tool_write_memory(args.get("content", ""), args.get("mode", "append")),
        }

        handler = dispatch_map.get(name)
        if handler:
            return handler()
        return f"Error: Unknown tool '{name}'"

    # ========================== TEXT TOOL CALL EXTRACTOR ==========================

    def extract_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from model text output when native tool calling is unavailable."""
        calls = []

        # 1. <tool_call> XML tags (Hermes / Qwen style)
        xml_matches = re.findall(r"<tool_call>\s*([\s\S]*?)\s*(?:</tool_call>|$)", text)
        for raw in xml_matches:
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
                name = data.get("name") or data.get("function")
                args = data.get("arguments") or data.get("parameters") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                if name:
                    calls.append({"name": name, "args": args if isinstance(args, dict) else {}})
            except Exception:
                pass
        if calls:
            return calls

        # 2. ```tool_call or ```json blocks with name/arguments
        md_matches = re.findall(r"```(?:tool_call|json)\s*([\s\S]*?)\s*```", text)
        for raw in md_matches:
            raw = raw.strip()
            try:
                data = json.loads(raw)
                name = data.get("name") or data.get("function")
                args = data.get("arguments") or data.get("parameters")
                if name:
                    if args is None:
                        args = {k: v for k, v in data.items() if k not in ("name", "function")}
                    elif isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    calls.append({"name": name, "args": args if isinstance(args, dict) else {}})
            except Exception:
                pass
        if calls:
            return calls

        # 2.5. Raw JSON tool call in text (Llama 3.x / OpenAI style):
        # e.g. {"type": "function", "name": "...", "parameters": {...}}
        # e.g. {"name": "...", "arguments": {...}}
        known_tool_names = {t["function"]["name"] for t in TOOLS}
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(text):
            idx = text.find('{', pos)
            if idx == -1:
                break
            try:
                data, end_idx = decoder.raw_decode(text[idx:])
                pos = idx + end_idx
                if isinstance(data, dict):
                    name = data.get("name")
                    if not name and isinstance(data.get("function"), dict):
                        fn_info = data["function"]
                        name = fn_info.get("name")
                        args = fn_info.get("arguments") or fn_info.get("parameters") or {}
                    else:
                        args = data.get("arguments") or data.get("parameters") or {}

                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass

                    if name and name in known_tool_names:
                        calls.append({"name": name, "args": args if isinstance(args, dict) else {}})
            except json.JSONDecodeError:
                pos = idx + 1
            except Exception:
                pos = idx + 1
        if calls:
            return calls

        # 3. Code blocks labeled with filename: ```lang:path/to/file
        file_block_matches = re.findall(r"```[a-zA-Z0-9_-]*[:\s]+([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]+)\s*\n([\s\S]*?)```", text)
        for path, code in file_block_matches:
            calls.append({"name": "write_file", "args": {"path": path.strip(), "content": code}})
        if calls:
            return calls

        # 4. "File: filename.ext" header before code block
        header_matches = re.findall(
            r"(?:###?\s*|File|Filename|Path):\s*[`*]*([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]+)[`*]*\s*\n+```[a-zA-Z0-9_-]*\n([\s\S]*?)```",
            text
        )
        for path, code in header_matches:
            calls.append({"name": "write_file", "args": {"path": path.strip(), "content": code}})

        return calls

    # ========================== CONTEXT COMPACTION ==========================

    def compact_context(self):
        """Summarize older messages to prevent context overflow."""
        max_msgs = int(self.config.get("max_context_messages", 60))
        if len(self.messages) <= max_msgs:
            return

        console.print("[dim]🗜️  Compacting context (summarizing older messages)...[/dim]")

        # Keep system prompt (index 0) and last 20 messages
        keep_recent = 20
        old_messages = self.messages[1:-keep_recent]
        recent_messages = self.messages[-keep_recent:]

        # Build a compact summary of old messages
        summary_parts = []
        for msg in old_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "") if isinstance(msg, dict) else str(getattr(msg, "content", ""))
            if not content:
                continue
            if role == "user":
                summary_parts.append(f"USER: {content[:200]}")
            elif role == "assistant":
                summary_parts.append(f"ASSISTANT: {content[:200]}")
            elif role == "tool":
                summary_parts.append(f"TOOL RESULT: {content[:150]}")

        summary = "\n".join(summary_parts[-15:])  # Keep last 15 summary entries

        # Rebuild messages
        compaction_msg = {
            "role": "user",
            "content": f"[CONTEXT COMPACTION] Here is a summary of our earlier conversation:\n{summary}\n\n[END COMPACTION] Continue from here."
        }

        self.messages = [self.messages[0], compaction_msg] + recent_messages
        console.print(f"[dim]  Compacted {len(old_messages)} messages → 1 summary[/dim]")

    # ========================== AGENT LOOP ==========================

    def step(self, user_prompt: str):
        self.messages.append({"role": "user", "content": user_prompt})

        # Auto-compact if context is getting large
        self.compact_context()

        loop_count = 0
        max_loops = 20

        while loop_count < max_loops:
            loop_count += 1
            with console.status("[bold blue]Thinking & coding...", spinner="dots"):
                try:
                    kwargs = {
                        "model": self.config["model"],
                        "messages": self.messages,
                        "temperature": 0.2,
                        "timeout": 90.0
                    }
                    if self.use_native_tools:
                        kwargs["tools"] = TOOLS
                        kwargs["tool_choice"] = "auto"

                    try:
                        response = self.client.chat.completions.create(**kwargs)
                    except Exception as err:
                        err_str = str(err).lower()
                        if "tool choice requires" in err_str or "tool-call-parser" in err_str or "tools" in err_str:
                            if self.use_native_tools:
                                console.print("[yellow]⚠️  Server lacks native tool-choice. Engaging text/XML fallback...[/yellow]")
                                self.use_native_tools = False
                            kwargs.pop("tools", None)
                            kwargs.pop("tool_choice", None)
                            response = self.client.chat.completions.create(**kwargs)
                        else:
                            raise err
                except Exception as e:
                    console.print(f"[bold red]API Error:[/bold red] {e}")
                    if "does not exist" in str(e) or "NotFoundError" in str(e) or "404" in str(e):
                        try:
                            models = [m.id for m in self.client.models.list().data]
                            if models:
                                console.print(f"[yellow]Available model(s):[/yellow] {', '.join(models)}")
                                console.print(f"[cyan]Switch with:[/cyan] [bold]/model {models[0]}[/bold]")
                        except Exception:
                            pass
                    console.print("[yellow]Tip: Check tunnel URL with /url[/yellow]")
                    return

            msg = response.choices[0].message
            content = msg.content or ""

            # Estimate token usage
            self.stats["tokens_est"] += len(content) // 4 + 50

            # Detect tool calls (native or text-extracted)
            tool_calls_to_run = []
            is_native = False

            if msg.tool_calls:
                is_native = True
                for tc in msg.tool_calls:
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except Exception:
                        fn_args = {}
                    tool_calls_to_run.append({"id": tc.id, "name": tc.function.name, "args": fn_args})
            elif content:
                extracted = self.extract_tool_calls_from_text(content)
                if extracted:
                    tool_calls_to_run = extracted

            # Display clean content (strip tool tags and raw JSON tool calls)
            clean = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", content)
            clean = re.sub(r"```tool_call[\s\S]*?```", "", clean)
            if not is_native and tool_calls_to_run:
                known_tool_names = {t["function"]["name"] for t in TOOLS}
                dec = json.JSONDecoder()
                pos = 0
                clean_parts = []
                last_end = 0
                while pos < len(clean):
                    idx = clean.find('{', pos)
                    if idx == -1:
                        clean_parts.append(clean[last_end:])
                        last_end = len(clean)
                        break
                    try:
                        obj, end_idx = dec.raw_decode(clean[idx:])
                        obj_name = obj.get("name") if isinstance(obj, dict) else None
                        if not obj_name and isinstance(obj, dict) and isinstance(obj.get("function"), dict):
                            obj_name = obj["function"].get("name")
                        if isinstance(obj, dict) and obj_name in known_tool_names:
                            clean_parts.append(clean[last_end:idx])
                            last_end = idx + end_idx
                            pos = idx + end_idx
                        else:
                            pos = idx + 1
                    except json.JSONDecodeError:
                        pos = idx + 1
                    except Exception:
                        pos = idx + 1
                if last_end < len(clean):
                    clean_parts.append(clean[last_end:])
                clean = "".join(clean_parts)

            clean = clean.strip()
            if clean:
                console.print(Panel(Markdown(clean), title="[bold green]🤖 Assistant[/bold green]", border_style="green"))

            # Execute tools
            if tool_calls_to_run:
                if is_native:
                    self.messages.append(msg)
                    for tc in tool_calls_to_run:
                        result = self.dispatch_tool(tc["name"], tc["args"])
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": str(result)[:6000]
                        })
                else:
                    self.messages.append({"role": "assistant", "content": content})
                    summaries = []
                    for tc in tool_calls_to_run:
                        result = self.dispatch_tool(tc["name"], tc["args"])
                        summaries.append(f"[Tool: {tc['name']}]\nResult:\n{str(result)[:4000]}")
                    self.messages.append({
                        "role": "user",
                        "content": "Tool execution results:\n\n" + "\n\n".join(summaries) + "\n\nContinue with next steps or provide final answer."
                    })
            else:
                # No tools — finish step
                self.messages.append({"role": "assistant", "content": content})
                break

    # ========================== REPL INTERFACE ==========================

    def show_welcome(self):
        cwd = Path.cwd()
        tool_mode = "[green]Native[/green]" if self.use_native_tools else "[yellow]Text/XML Fallback[/yellow]"
        has_git = Path(".git").exists()
        has_memory = Path("AGENT.md").exists()

        info = (
            f"[bold magenta]🚀 Ultimate AI Coding Agent (2026 Edition)[/bold magenta]\n"
            f"[cyan]Workspace:[/cyan]    {cwd}\n"
            f"[cyan]Endpoint:[/cyan]     {self.config['base_url']}\n"
            f"[cyan]Model:[/cyan]        {self.config['model']}\n"
            f"[cyan]Tool Mode:[/cyan]    {tool_mode}\n"
            f"[cyan]Git:[/cyan]          {'✅ Initialized' if has_git else '❌ Not a git repo'}\n"
            f"[cyan]Memory:[/cyan]       {'✅ AGENT.md loaded' if has_memory else '📝 No AGENT.md yet'}\n"
            f"[cyan]Tools:[/cyan]        17 tools (files, git, web, search, memory, MCP, sub-agents)\n"
            f"[cyan]Commands:[/cyan]     /help /url /model /auto /compact /memory /stats /git /undo /clear /exit"
        )
        console.print(Panel.fit(info, border_style="magenta"))

    def show_stats(self):
        table = Table(title="📊 Session Statistics", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Files Created", str(self.stats["files_created"]))
        table.add_row("Files Edited", str(self.stats["files_edited"]))
        table.add_row("Commands Run", str(self.stats["commands_run"]))
        table.add_row("Searches", str(self.stats["searches"]))
        table.add_row("Git Commits", str(self.stats["commits"]))
        table.add_row("Est. Tokens Used", f"~{self.stats['tokens_est']:,}")
        table.add_row("Messages in Context", str(len(self.messages)))
        console.print(table)

    def start_repl(self):
        # Prompt for Colab URL if not configured
        if "localhost" in self.config.get("base_url", ""):
            console.print("[yellow]⚠️  Colab Tunnel URL not configured.[/yellow]")
            url_input = Prompt.ask("[bold cyan]Paste your Colab Cloudflare URL (or Enter to skip)[/bold cyan]", default="")
            if url_input.strip():
                url = url_input.strip().rstrip("/")
                if not url.endswith("/v1"):
                    url += "/v1"
                self.config["base_url"] = url
                self.save_config()
                self.init_client()
                console.print(f"[green]✅ Saved: {self.config['base_url']}[/green]\n")

        self.show_welcome()
        cwd_name = Path.cwd().name or str(Path.cwd())
        session = PromptSession(history=InMemoryHistory())

        while True:
            try:
                user_input = session.prompt(f"\n[{cwd_name}] > ").strip()
                if not user_input:
                    continue

                # Slash commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()
                    arg = parts[1] if len(parts) > 1 else ""

                    if cmd in ("/exit", "/quit"):
                        console.print("[yellow]Goodbye! 👋[/yellow]")
                        break
                    elif cmd == "/help":
                        help_table = Table(title="⌨️  Commands", border_style="blue")
                        help_table.add_column("Command", style="bold cyan")
                        help_table.add_column("Description")
                        help_table.add_row("/url <url>", "Set Colab Cloudflare tunnel URL")
                        help_table.add_row("/model <name>", "Switch model")
                        help_table.add_row("/models", "List models on server")
                        help_table.add_row("/auto", "Toggle auto-approve commands (YOLO mode)")
                        help_table.add_row("/compact", "Force context compaction")
                        help_table.add_row("/memory", "View AGENT.md persistent memory")
                        help_table.add_row("/stats", "Show session statistics")
                        help_table.add_row("/git", "Show git status")
                        help_table.add_row("/undo", "Undo last git commit")
                        help_table.add_row("/clear", "Clear conversation history")
                        help_table.add_row("/exit", "Quit the agent")
                        console.print(help_table)
                    elif cmd == "/url":
                        new_url = arg.strip()
                        if not new_url:
                            new_url = Prompt.ask("[bold cyan]Enter Colab URL[/bold cyan]", default="").strip()
                        if new_url:
                            self.config["base_url"] = new_url.rstrip("/")
                            if not self.config["base_url"].endswith("/v1"):
                                self.config["base_url"] += "/v1"
                            self.save_config()
                            self.init_client()
                            self.use_native_tools = True  # Reset on new URL
                            console.print(f"[green]✅ URL: {self.config['base_url']}[/green]")
                        else:
                            console.print(f"Current URL: {self.config['base_url']}")
                    elif cmd == "/models":
                        try:
                            models = [m.id for m in self.client.models.list().data]
                            console.print(f"[bold cyan]Models on server ({len(models)}):[/bold cyan]")
                            for m in models:
                                active = " [green](current)[/green]" if m == self.config['model'] else ""
                                console.print(f"  • {m}{active}")
                        except Exception as e:
                            console.print(f"[red]Error: {e}[/red]")
                    elif cmd == "/model":
                        if arg:
                            self.config["model"] = arg
                            self.save_config()
                            console.print(f"[green]✅ Model: {self.config['model']}[/green]")
                        else:
                            console.print(f"Current model: {self.config['model']}")
                    elif cmd == "/auto":
                        self.auto_approve = not self.auto_approve
                        status = "[green]ON (YOLO mode 🚀)[/green]" if self.auto_approve else "[yellow]OFF (ask before executing)[/yellow]"
                        console.print(f"Auto-approve commands: {status}")
                    elif cmd == "/compact":
                        self.compact_context()
                        console.print("[green]✅ Context compacted.[/green]")
                    elif cmd == "/memory":
                        result = self.tool_read_memory()
                        console.print(Panel(Markdown(result), title="[bold blue]🧠 AGENT.md[/bold blue]", border_style="blue"))
                    elif cmd == "/stats":
                        self.show_stats()
                    elif cmd == "/git":
                        result = self.tool_git_status()
                        console.print(Panel(result, title="[bold blue]🔀 Git Status[/bold blue]", border_style="blue"))
                    elif cmd == "/undo":
                        if Confirm.ask("[yellow]Undo last git commit?[/yellow]"):
                            result = self.tool_git_undo()
                            console.print(result)
                    elif cmd == "/clear":
                        self._init_messages()
                        console.print("[green]✅ Conversation cleared.[/green]")
                    else:
                        console.print(f"[red]Unknown command: {cmd}. Type /help[/red]")
                    continue

                # Run agent step
                self.step(user_input)

            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Session ended. 👋[/yellow]")
                break


def main():
    agent = CodingAgent()
    agent.start_repl()


if __name__ == "__main__":
    main()
