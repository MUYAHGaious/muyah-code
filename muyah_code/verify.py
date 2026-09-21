"""/verify: check the agent's work, at the depth you choose, only when you ask.

Checking costs time (and, for e2e, tokens), so it never runs on its own unless you turn on
`verify.auto` ("quick" or "full"; the default is "off"). Depths:

  quick  lint the files the last turn changed, and run the tests that belong to them
  full   the project's whole test suite and linters
  e2e    the agent runs the app the way a user would (a browser for web UIs, when available)

quick and full run the commands directly (no model involved) and report the real exit codes.
Commands are detected from the project (pyproject.toml, package.json, Cargo.toml, go.mod); set
`verify.test` / `verify.lint` (a command or a list) in .muyah/settings.json to use your own.

Also here: a cheap, model-free check for *weakened tests* (assertions removed, skip/xfail added,
test files deleted) that the turn footer always shows.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

DEPTHS = ("quick", "full", "e2e")
AUTO = ("off", "quick", "full")
SKIP_DIRS = {".git", ".muyah", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", "target",
             ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".next"}
OUTPUT_TAIL = 20


@dataclass
class Check:
    kind: str          # test | lint
    command: str


@dataclass
class CheckResult:
    check: Check
    exit_code: int | None
    seconds: float
    output: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def tail(self, lines: int = OUTPUT_TAIL) -> str:
        return "\n".join(self.output.rstrip().splitlines()[-lines:])


# --------------------------------------------------------------------------- detecting the commands

def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _as_list(value) -> list[str]:
    if not value:
        return []
    return [value] if isinstance(value, str) else [str(v) for v in value if v]


def _node_runner(root: Path) -> str:
    for lock, runner in (("pnpm-lock.yaml", "pnpm"), ("yarn.lock", "yarn"), ("bun.lockb", "bun"), ("bun.lock", "bun")):
        if (root / lock).exists():
            return runner
    return "npm"


def _package_json(root: Path) -> dict:
    try:
        data = json.loads(_read(root / "package.json") or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _python_tests(root: Path) -> bool:
    pyproject = _read(root / "pyproject.toml")
    return ("[tool.pytest" in pyproject or (root / "pytest.ini").exists() or (root / "conftest.py").exists()
            or "[tool:pytest]" in _read(root / "setup.cfg") or "[pytest]" in _read(root / "tox.ini")
            or (root / "tests").is_dir() or (root / "test").is_dir())


def _ruff(root: Path) -> bool:
    return ("[tool.ruff" in _read(root / "pyproject.toml") or (root / "ruff.toml").exists()
            or (root / ".ruff.toml").exists())


def detect(root: Path) -> list[Check]:
    """The project's full test and lint commands, found from its config files."""
    checks: list[Check] = []
    if _python_tests(root):
        checks.append(Check("test", "python -m pytest -q"))
    if _ruff(root):
        checks.append(Check("lint", "python -m ruff check ."))
    elif (root / ".flake8").exists() or "[flake8]" in _read(root / "setup.cfg"):
        checks.append(Check("lint", "python -m flake8"))
    pkg = _package_json(root)
    scripts = pkg.get("scripts") if isinstance(pkg.get("scripts"), dict) else {}
    runner = _node_runner(root)
    test_script = str(scripts.get("test") or "")
    if test_script and "no test specified" not in test_script:
        checks.append(Check("test", f"{runner} test"))
    for name in ("lint", "typecheck", "type-check"):
        if scripts.get(name):
            checks.append(Check("lint", f"{runner} run {name}"))
    if (root / "Cargo.toml").exists():
        checks += [Check("test", "cargo test -q"), Check("lint", "cargo clippy -q")]
    if (root / "go.mod").exists():
        checks += [Check("test", "go test ./..."), Check("lint", "go vet ./...")]
    return checks


def configured(cfg, root: Path) -> tuple[list[Check], bool]:
    """(checks, from_settings): the user's verify.test / verify.lint if set, else what detect() finds."""
    tests, lints = _as_list(cfg.get("verify.test")), _as_list(cfg.get("verify.lint"))
    if tests or lints:
        return [Check("test", c) for c in tests] + [Check("lint", c) for c in lints], True
    return detect(root), False


def is_test_file(rel: str) -> bool:
    p = Path(rel)
    name = p.name.lower()
    parts = {x.lower() for x in p.parts[:-1]}
    return (name.startswith("test_") and name.endswith(".py")) or name.endswith("_test.py") \
        or name.endswith("_test.go") or bool(re.search(r"\.(test|spec)\.[cm]?[jt]sx?$", name)) \
        or ("tests" in parts and p.suffix == ".rs") or name == "conftest.py" \
        or (bool(parts & {"tests", "test", "__tests__", "spec"}) and p.suffix in (".py", ".js", ".ts", ".tsx", ".jsx"))


def _related_python_tests(root: Path, changed: list[str]) -> list[str]:
    wanted, found = set(), []
    for rel in changed:
        p = Path(rel)
        if p.suffix != ".py":
            continue
        if is_test_file(rel) and p.name != "conftest.py":
            if (root / rel).exists():
                found.append(rel)
        else:
            wanted |= {f"test_{p.stem}.py", f"{p.stem}_test.py"}
    if wanted:
        for path in _walk(root):
            if path.name in wanted:
                found.append(path.relative_to(root).as_posix())
    return sorted(set(found))


def _walk(root: Path, limit: int = 20000):
    stack, seen = [root], 0
    while stack:
        folder = stack.pop()
        try:
            entries = list(folder.iterdir())
        except OSError:
            continue
        for entry in entries:
            seen += 1
            if seen > limit:
                return
            if entry.is_dir():
                if entry.name not in SKIP_DIRS and not entry.name.startswith("."):
                    stack.append(entry)
            else:
                yield entry


def _quote(rel: str) -> str:
    return f'"{rel}"' if any(c in rel for c in " &;()'") else rel


def quick_checks(cfg, root: Path, changed: list[str]) -> tuple[list[Check], list[str]]:
    """Checks scoped to the changed files, and notes on what quick mode could not cover."""
    existing = [rel for rel in changed if (root / rel).exists()]
    checks, notes = [], []
    full, from_settings = configured(cfg, root)
    if from_settings:
        return full, ["verify.test / verify.lint are set: quick runs them as they are"]
    py = [rel for rel in existing if rel.endswith(".py")]
    if py and _ruff(root):
        checks.append(Check("lint", "python -m ruff check " + " ".join(_quote(r) for r in py)))
    if py and _python_tests(root):
        tests = _related_python_tests(root, py)
        if tests:
            checks.append(Check("test", "python -m pytest -q " + " ".join(_quote(r) for r in tests)))
        else:
            notes.append("no tests found for the changed Python files (test_<name>.py)")
    go_dirs = sorted({Path(rel).parent.as_posix() for rel in existing if rel.endswith(".go")})
    if go_dirs and (root / "go.mod").exists():
        pkgs = " ".join("./" + d if d != "." else "." for d in go_dirs)
        checks += [Check("test", f"go test {pkgs}"), Check("lint", f"go vet {pkgs}")]
    others = [c for c in full if not c.command.startswith(("python -m", "go "))]
    if others and [rel for rel in existing if not rel.endswith((".py", ".go"))]:
        notes.append("quick mode cannot narrow " + ", ".join(c.command for c in others) + ": use /verify full")
    return checks, notes


E2E_PROMPT = """Verify, end to end, that the recent changes actually work, the way a user would find out. \
Do not change any code while verifying.

1. Work out how this project is run (README, MUYAH.md/AGENTS.md, package.json scripts, entry points).
2. Run it for real: start the app / CLI / server and exercise the changed behaviour. For a web UI, open it \
with the browser tools if you have them (navigate, click through the changed flow, read the page); otherwise \
request it with curl or a short script (call the Browser tool first if the browser_* tools are not there yet). \
Stop anything you started when you are done.
3. Report: each command you ran with its exit code, what you observed, and a verdict per changed behaviour \
(works / broken / could not check, and why). If something is broken, show the evidence and ask before fixing it.
{changed}"""


def e2e_prompt(changed: list[str]) -> str:
    listing = ("\nFiles changed in the last turn:\n" + "\n".join(f"- {c}" for c in changed[:40])) if changed else ""
    return E2E_PROMPT.format(changed=listing)


# --------------------------------------------------------------------------- running

def run_checks(checks: list[Check], cwd: Path, shell, timeout: float, on_start=None, on_done=None
               ) -> list[CheckResult]:
    from muyah_code.tools.shell import run_command

    results = []
    for check in checks:
        if on_start is not None:
            on_start(check)
        t0 = time.monotonic()
        code, output, timed_out = run_command(check.command, cwd, timeout, shell)
        result = CheckResult(check, code, time.monotonic() - t0, output, timed_out)
        results.append(result)
        if on_done is not None:
            on_done(result)
    return results


def summary_for_model(depth: str, results: list[CheckResult]) -> str:
    """What the next turn is told about the last /verify (so "fix it" works without pasting)."""
    lines = [f"<verify depth=\"{depth}\">The user ran /verify {depth} after your last turn:"]
    for r in results:
        status = "timed out" if r.timed_out else f"exit {r.exit_code}"
        lines.append(f"- {r.check.command}: {status} ({'passed' if r.ok else 'FAILED'})")
        if not r.ok:
            lines.append("  " + r.tail(30).replace("\n", "\n  "))
    lines.append("</verify>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- weakened tests

_ASSERT = re.compile(r"\bassert\b|\bassert[A-Z_]\w*\s*\(|\bexpect\s*\(|\bassert_eq!|\bassert_ne!|\bassert!|"
                     r"\bt\.(?:Error|Errorf|Fatal|Fatalf)\b|\bpytest\.raises\b|\.should\b|\brequire\.\w+\(")
_SKIP = re.compile(r"@pytest\.mark\.(?:skip|skipif|xfail)\b|\bpytest\.(?:skip|xfail)\s*\(|@unittest\.skip|"
                   r"\b(?:it|test|describe)\.(?:skip|todo)\s*\(|\bx(?:it|describe|test)\s*\(|\bt\.Skip(?:f|Now)?\s*\(|"
                   r"#\[ignore\]|self\.skipTest\s*\(")


def _count(pattern: re.Pattern, text: str) -> int:
    return sum(1 for line in text.splitlines() if not line.lstrip().startswith(("#", "//")) and pattern.search(line))


def weakened_tests(changes: list[tuple[str, str]], old_text, new_root: Path) -> list[str]:
    """Plain-language warnings for test files the turn weakened. `changes` is [(status, rel)],
    `old_text(rel)` returns the file before the turn (or None)."""
    warnings = []
    for status, rel in changes:
        if not is_test_file(rel):
            continue
        if status == "D":
            warnings.append(f"{rel} deleted")
            continue
        if status != "M":          # a new test file weakens nothing that existed
            continue
        before = old_text(rel)
        if before is None:
            continue
        after = _read(new_root / rel)
        issues = []
        removed = _count(_ASSERT, before) - _count(_ASSERT, after)
        if removed > 0:
            issues.append(f"{removed} assertion{'s' if removed != 1 else ''} removed")
        skipped = _count(_SKIP, after) - _count(_SKIP, before)
        if skipped > 0:
            issues.append(f"{skipped} skip/xfail added")
        if issues:
            warnings.append(f"{rel}: " + ", ".join(issues))
    return warnings
