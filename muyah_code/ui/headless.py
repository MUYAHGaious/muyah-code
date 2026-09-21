"""Non-interactive output for `muyah -p`: text, json, or stream-json (one JSON event per line)."""

from __future__ import annotations

import json
import sys

from muyah_code.tools.base import ToolResult
from muyah_code.ui.base import UI


class HeadlessUI(UI):
    headless = True

    def __init__(self, output_format: str = "text", verbose: bool = False, out=None, err=None):
        self.format = output_format
        self.verbose = verbose
        self.out = out or sys.stdout
        self.err = err or sys.stderr
        self.final_chunks: list[str] = []

    def _event(self, **data) -> None:
        self.out.write(json.dumps(data, ensure_ascii=False) + "\n")
        self.out.flush()

    def _log(self, msg: str) -> None:
        if self.verbose or self.format == "text":
            self.err.write(msg + "\n")
            self.err.flush()

    def assistant_start(self) -> None:
        self.final_chunks = []

    def text(self, chunk: str) -> None:
        self.final_chunks.append(chunk)
        if self.format == "stream-json":
            self._event(type="text", text=chunk)

    def tool_start(self, title: str) -> None:
        if self.format == "stream-json":
            self._event(type="tool_use", title=title)
        elif self.verbose:
            self._log(f"> {title}")

    def tool_end(self, title: str, result: ToolResult) -> None:
        if self.format == "stream-json":
            self._event(type="tool_result", title=title, is_error=result.is_error, summary=result.summary or "")
        elif self.verbose:
            self._log(f"  < {'ERROR ' if result.is_error else ''}{result.summary or ''}")

    def handoff(self, command: str, targets: list[str]) -> None:
        if self.format == "stream-json":
            self._event(type="handoff", command=command, targets=targets)
        else:
            self._log(f"Delete not run (MUYAH-CODE never deletes). To do it yourself: {command}")

    def on_todos(self, todos: list[dict]) -> None:
        if self.format == "stream-json":
            self._event(type="todos", todos=todos)

    def info(self, msg: str) -> None:
        if self.format == "stream-json":
            self._event(type="info", message=msg)
        elif self.verbose:
            self._log(msg)

    def warn(self, msg: str) -> None:
        if self.format == "stream-json":
            self._event(type="warning", message=msg)
        else:
            self._log(f"warning: {msg}")

    def error(self, msg: str) -> None:
        if self.format == "stream-json":
            self._event(type="error", message=msg)
        else:
            self.err.write(f"error: {msg}\n")
            self.err.flush()
