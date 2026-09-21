"""Tool primitives shared by every built-in, MCP and plugin tool."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from muyah_code.config import Config

# Permission classes. read/meta never prompt; write prompts unless acceptEdits; exec/network prompt.
READ, WRITE, EXEC, NETWORK, META = "read", "write", "exec", "network", "meta"


class ToolError(Exception):
    """Expected, user-facing failure. The message is returned to the model verbatim."""


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    # Optional richer rendering for the terminal (diff, table...). Never sent to the model.
    display: str | None = None
    summary: str | None = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def error(cls, message: str) -> ToolResult:
        return cls(content=message, is_error=True, summary=message.splitlines()[0][:160] if message else "error")


@dataclass
class ToolContext:
    cwd: Path
    project_root: Path
    config: Config
    # absolute path -> mtime_ns observed when the file was last Read/Written by the agent
    file_state: dict[str, int] = field(default_factory=dict)
    todos: list[dict] = field(default_factory=list)
    services: dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    headless: bool = False

    def resolve(self, path: str) -> Path:
        p = Path(str(path).strip().strip("'\"")).expanduser()
        if not p.is_absolute():
            p = self.cwd / p
        return p.resolve()

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.cwd).as_posix() or "."
        except ValueError:
            return str(path)

    def service(self, name: str):
        return self.services.get(name)


class Tool:
    name: str = ""
    description: str = ""
    parameters: dict = {"type": "object", "properties": {}}
    kind: str = READ
    aliases: tuple[str, ...] = ()

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError

    def permission_subject(self, args: dict, ctx: ToolContext) -> str:
        """The string permission rules match against (a path, a command, a URL)."""
        return ""

    def title(self, args: dict) -> str:
        """Short one-line label for the UI, e.g. Read(src/app.py)."""
        subj = ""
        for key in ("file_path", "path", "command", "pattern", "url", "query", "skill", "description"):
            if args.get(key):
                subj = str(args[key])
                break
        subj = subj.replace("\n", " ")
        if len(subj) > 90:
            subj = subj[:87] + "..."
        return f"{self.name}({subj})"

    def is_read_only(self, args: dict) -> bool:
        return self.kind in (READ, META)

    def preview(self, args: dict, ctx: ToolContext) -> str | None:
        """What the user sees when asked to approve the call (a diff, the command...)."""
        return None

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def truncate_middle(text: str, limit: int, label: str = "output") -> str:
    """Keep the head and tail: errors usually appear at the end, context at the start."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.4)
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n... [{omitted} characters of {label} omitted] ...\n\n{text[-tail:]}"


def validate_args(schema: dict, args: Any) -> tuple[dict, list[str]]:
    """Light JSON-schema validation with safe coercions small models need ("5" -> 5, "true" -> True)."""
    if args is None:
        return {}, ["arguments missing or not a JSON object"]
    if not isinstance(args, dict):
        return {}, [f"arguments must be an object, got {type(args).__name__}"]
    props = schema.get("properties", {})
    out = dict(args)
    errors: list[str] = []
    for req in schema.get("required", []):
        if req not in out or out[req] is None:
            errors.append(f"missing required parameter '{req}'")
    for key, val in list(out.items()):
        spec = props.get(key)
        if spec is None:
            if not schema.get("additionalProperties", True):
                errors.append(f"unknown parameter '{key}'")
            continue
        typ = spec.get("type")
        if val is None:
            del out[key]
            continue
        try:
            out[key] = _coerce(val, typ, spec)
        except (TypeError, ValueError) as e:
            errors.append(f"parameter '{key}': {e}")
            continue
        if "enum" in spec and out[key] not in spec["enum"]:
            errors.append(f"parameter '{key}' must be one of {spec['enum']}, got {out[key]!r}")
    return out, errors


def _coerce(val: Any, typ: str | None, spec: dict) -> Any:
    if typ is None:
        return val
    if typ == "string":
        if isinstance(val, str):
            return val
        if isinstance(val, (int, float, bool)):
            return str(val)
        raise TypeError(f"expected string, got {type(val).__name__}")
    if typ == "integer":
        if isinstance(val, bool):
            raise TypeError("expected integer, got boolean")
        if isinstance(val, int):
            return val
        if isinstance(val, float) and val.is_integer():
            return int(val)
        if isinstance(val, str) and val.strip().lstrip("-").isdigit():
            return int(val.strip())
        raise TypeError(f"expected integer, got {val!r}")
    if typ == "number":
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return val
        if isinstance(val, str):
            return float(val)
        raise TypeError(f"expected number, got {val!r}")
    if typ == "boolean":
        if isinstance(val, bool):
            return val
        if isinstance(val, str) and val.lower() in ("true", "false"):
            return val.lower() == "true"
        if val in (0, 1):
            return bool(val)
        raise TypeError(f"expected boolean, got {val!r}")
    if typ == "array":
        if isinstance(val, str):
            val = json.loads(val)
        if not isinstance(val, list):
            raise TypeError(f"expected array, got {type(val).__name__}")
        return val
    if typ == "object":
        if isinstance(val, str):
            val = json.loads(val)
        if not isinstance(val, dict):
            raise TypeError(f"expected object, got {type(val).__name__}")
        return val
    return val
