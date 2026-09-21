"""`muyah eval`: run benchmark tasks headless and track the pass rate over time.

A task is a directory:
    evals/tasks/<name>/task.json   {"prompt": "...", "max_steps": 30, "timeout": 600}
    evals/tasks/<name>/fixture/    files copied into a fresh temp workspace
    evals/tasks/<name>/check.py    run in the workspace after the agent finishes; exit 0 = pass
                                   (the final answer is in .muyah/answer.md)

Results are appended to ~/.muyah/evals.jsonl so you can see whether prompt/skill/lesson changes
actually make the agent better on YOUR model.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from muyah_code.config import load_config
from muyah_code.ui.headless import HeadlessUI


@dataclass
class TaskResult:
    name: str
    passed: bool
    status: str
    steps: int
    tool_calls: int
    duration: float
    detail: str = ""


def discover_tasks(tasks_dir: Path, names: list[str] | None = None) -> list[Path]:
    tasks = sorted(p for p in tasks_dir.iterdir() if (p / "task.json").is_file()) if tasks_dir.is_dir() else []
    if names:
        tasks = [t for t in tasks if t.name in names]
    return tasks


def run_task(task_dir: Path, overrides: dict, learn: bool, verbose: bool) -> TaskResult:
    from muyah_code.app import App

    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix=f"muyah-eval-{task_dir.name}-") as tmp:
        ws = Path(tmp)
        if (task_dir / "fixture").is_dir():
            shutil.copytree(task_dir / "fixture", ws, dirs_exist_ok=True)
        (ws / ".muyah").mkdir(exist_ok=True)  # marks the workspace as the project root
        cfg = load_config(cwd=ws, overrides={**overrides, "max_steps": spec.get("max_steps", 30)})
        cfg.set("learning.enabled", learn)
        cfg.set("learning.reflect", learn)
        ui = HeadlessUI("text", verbose=verbose)
        start = time.time()
        app = App(cfg, ui, cwd=ws, mode="bypassPermissions", headless=True, enable_mcp=False,
                  persist_session=False)
        try:
            result = app.run_prompt(spec["prompt"])
        finally:
            app.shutdown()
        # checks can read the final answer too (e.g. "did it push back on a false premise?")
        (ws / ".muyah" / "answer.md").write_text(result.text or "", encoding="utf-8")
        duration = time.time() - start
        check = task_dir / "check.py"
        try:
            proc = subprocess.run([sys.executable, str(check)], cwd=ws, capture_output=True, text=True,
                                  timeout=int(spec.get("check_timeout", 180)))
            passed = proc.returncode == 0
            detail = (proc.stdout + proc.stderr).strip()[-400:]
        except subprocess.TimeoutExpired:
            passed, detail = False, "check timed out"
        if result.status == "error" and result.error:  # the model call failed: say why, not just "check failed"
            detail = f"model error: {result.error[:300]}"
        return TaskResult(task_dir.name, passed, result.status, result.steps, result.tool_calls, round(duration, 1),
                          detail)


def run_eval(tasks_dir: Path, names: list[str] | None, overrides: dict, learn: bool, verbose: bool,
             console: Console) -> int:
    tasks = discover_tasks(tasks_dir, names)
    if not tasks:
        console.print(f"[red]No tasks found in {tasks_dir}[/]")
        return 2
    base_cfg = load_config(overrides=overrides)
    model = base_cfg["model"]
    console.print(f"Running {len(tasks)} eval task(s) against [bold]{model}[/] at {base_cfg['base_url']}")
    results: list[TaskResult] = []
    for t in tasks:
        console.print(f"  [dim]-> {t.name}...[/]")
        try:
            r = run_task(t, overrides, learn, verbose)
        except Exception as e:
            r = TaskResult(t.name, False, "crash", 0, 0, 0.0, f"{e.__class__.__name__}: {e}")
        results.append(r)
        mark = "[green]PASS[/]" if r.passed else "[red]FAIL[/]"
        console.print(f"     {mark} {r.status}, {r.steps} steps, {r.tool_calls} tool calls, {r.duration}s")

    passed = sum(r.passed for r in results)
    rate = passed / len(results)
    record = {"ts": time.time(), "model": model, "base_url": base_cfg["base_url"], "learning": learn,
              "pass_rate": rate, "tasks": [asdict(r) for r in results]}
    history_path = base_cfg.home / "evals.jsonl"
    previous = _previous(history_path, model)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    table = Table(title=f"Eval: {passed}/{len(results)} passed ({rate:.0%})", header_style="bold")
    for col in ("task", "result", "status", "steps", "tools", "time", "check output"):
        table.add_column(col, overflow="fold")
    for r in results:
        table.add_row(r.name, "PASS" if r.passed else "FAIL", r.status, str(r.steps), str(r.tool_calls),
                      f"{r.duration}s", r.detail[-160:] if not r.passed else "")
    console.print(table)
    if previous is not None:
        delta = rate - previous
        trend = "[green]improved[/]" if delta > 0 else "[red]regressed[/]" if delta < 0 else "unchanged"
        console.print(f"Compared with the previous run on this model ({previous:.0%}): {trend} ({delta:+.0%})")
    return 0 if passed == len(results) else 1


def _previous(path: Path, model: str) -> float | None:
    if not path.exists():
        return None
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("model") == model:
            last = rec.get("pass_rate")
    return last
