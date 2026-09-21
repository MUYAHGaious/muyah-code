"""Shared helpers for eval task checks. Each check.py runs with the task workspace as its cwd."""

import hashlib
import subprocess
import sys
from pathlib import Path


def unchanged(path: str, digest: str) -> None:
    """Fail if a file the task forbids editing was modified (line endings normalized)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != digest:
        sys.exit(f"FAIL: {path} was modified (the task forbids editing it)")


def run(*args: str) -> subprocess.CompletedProcess:
    """Run a Python script/module in the workspace with the same interpreter."""
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=120)


def pytest(path: str) -> int:
    result = run("-m", "pytest", "-q", "-p", "no:cacheprovider", path)
    print(result.stdout[-500:])
    return result.returncode
