"""How much RAM and CPU MUYAH-CODE uses: startup, idle, and a 50-step session. Optionally other CLIs.

    python scripts/bench_resources.py                 # MUYAH-CODE, printed as a markdown table
    python scripts/bench_resources.py --others        # also Claude Code, OpenCode, Codex, Gemini CLI if installed
    python scripts/bench_resources.py --json out.json # machine-readable
    python scripts/bench_resources.py --check         # CI guard: fail if MUYAH-CODE exceeds the ceilings

What is measured (process tree: the CLI and everything it started):
  * startup      wall time of `--version` (median of 3): interpreter/runtime start + imports
  * ready        time until the interactive prompt is drawn, in a real pseudo-terminal, minus the time the
                 pseudo-terminal itself needs to show a trivial program's first line (about 3 s on Windows
                 ConPTY, near 0 on Linux), so the number is the CLI's own
  * idle RSS/CPU the interactive prompt sitting idle for IDLE_SECONDS
  * session      a scripted 50-step agent session (reads, searches, commands, streamed replies) run headless
                 against a local fake model server: peak RSS and average CPU. The model's own compute is not
                 included; this is what the CLI itself costs.
Other CLIs cannot be pointed at the same fake model without their accounts, so for them only startup,
ready (their first screen, which may be a login screen) and idle are measured.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

IDLE_SECONDS = 5.0
OTHERS_SETTLE = 15.0
CEILINGS = {"startup_s": 4.0, "idle_rss_mb": 250.0, "session_peak_rss_mb": 350.0, "idle_cpu_pct": 5.0}
OTHERS = {"Claude Code": "claude", "OpenCode": "opencode", "Codex": "codex", "Gemini CLI": "gemini"}


# --------------------------------------------------------------------------- measuring a process tree

def tree(proc: psutil.Process) -> list[psutil.Process]:
    try:
        return [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return []


def rss_mb(proc: psutil.Process) -> float:
    total = 0
    for p in tree(proc):
        try:
            total += p.memory_info().rss
        except psutil.Error:
            continue
    return total / 2**20


def cpu_seconds(proc: psutil.Process) -> float:
    total = 0.0
    for p in tree(proc):
        try:
            t = p.cpu_times()
            total += t.user + t.system
        except psutil.Error:
            continue
    return total


class Sampler:
    """Peak RSS and CPU% (of one core) of a process tree while something runs."""

    def __init__(self, pid: int, every: float = 0.1):
        self.proc = psutil.Process(pid)
        self.every = every
        self.peak = 0.0
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.t0 = time.monotonic()
        self.cpu0 = cpu_seconds(self.proc)
        self.cpu_last = self.cpu0      # the process may be gone when stop() runs: keep the last reading

    def _run(self):
        while not self._stop.is_set():
            mb = rss_mb(self.proc)
            cpu = cpu_seconds(self.proc)
            if cpu:
                self.cpu_last = max(self.cpu_last, cpu)
            if mb:
                self.samples.append(mb)
                self.peak = max(self.peak, mb)
            self._stop.wait(self.every)

    def start(self):
        self._thread.start()
        return self

    def stop(self) -> dict:
        cpu1 = max(cpu_seconds(self.proc), self.cpu_last)
        self._stop.set()
        self._thread.join(2)
        seconds = time.monotonic() - self.t0
        return {"peak_mb": round(self.peak, 1), "avg_mb": round(statistics.mean(self.samples), 1) if self.samples else 0,
                "cpu_pct": round(100 * max(0.0, cpu1 - self.cpu0) / max(seconds, 1e-6), 1)}


# --------------------------------------------------------------------------- a pseudo-terminal, both platforms

class Terminal:
    def __init__(self, argv: list[str], cwd: Path, env: dict):
        self.out: list[str] = []
        if os.name == "nt":
            from winpty import PtyProcess

            self._p = PtyProcess.spawn(argv, cwd=str(cwd), env=env, dimensions=(40, 120))
            self.pid = self._p.pid
            reader = self._read_winpty
        else:
            import pty

            master, slave = pty.openpty()
            self._p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=slave, stdout=slave, stderr=slave,
                                       start_new_session=True)
            os.close(slave)
            self._fd = master
            self.pid = self._p.pid
            reader = self._read_pty
        threading.Thread(target=reader, daemon=True).start()

    def _read_winpty(self):
        while True:
            try:
                self.out.append(self._p.read(4096))
            except EOFError:
                return

    def _read_pty(self):
        while True:
            try:
                data = os.read(self._fd, 4096)
            except OSError:
                return
            if not data:
                return
            self.out.append(data.decode("utf-8", errors="replace"))

    def wait_for(self, text: str, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if text in "".join(self.out):
                return True
            time.sleep(0.02)
        return False

    def close(self):
        try:
            for p in reversed(tree(psutil.Process(self.pid))):
                p.kill()
        except psutil.Error:
            pass


# --------------------------------------------------------------------------- the runs

def median_startup(argv: list[str], env: dict | None = None, runs: int = 3) -> float:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        subprocess.run(argv, capture_output=True, env=env, timeout=120)
        times.append(time.perf_counter() - t0)
    return round(statistics.median(times), 2)


_BASELINE: list[float] = []


def pty_baseline() -> float:
    """How long the pseudo-terminal takes to show the first line of a program that prints at once."""
    if not _BASELINE:
        term = Terminal([sys.executable, "-c", "print('ready-marker'); import time; time.sleep(5)"], Path.cwd(),
                        {**os.environ, "PYTHONIOENCODING": "utf-8"})
        t0 = time.monotonic()
        term.wait_for("ready-marker", 60)
        _BASELINE.append(time.monotonic() - t0)
        term.close()
    return _BASELINE[0]


def interactive(argv: list[str], cwd: Path, env: dict, ready_text: str | None, settle: float = 0.0) -> dict:
    base = pty_baseline()
    term = Terminal(argv, cwd, env)
    t0 = time.monotonic()
    try:
        # other CLIs: no marker we can wait for, so give them OTHERS_SETTLE seconds to finish starting up
        ok = term.wait_for(ready_text, 60) if ready_text else (time.sleep(OTHERS_SETTLE) or True)
        ready = round(max(0.0, time.monotonic() - t0 - base), 2) if ready_text and ok else None
        time.sleep(settle)
        s = Sampler(term.pid).start()
        time.sleep(IDLE_SECONDS)
        idle = s.stop()
    finally:
        term.close()
    return {"ready_s": ready, "idle_rss_mb": idle["avg_mb"], "idle_cpu_pct": idle["cpu_pct"]}


def session_script(steps: int):
    from fakeserver import reply

    def tool(i: int) -> dict:   # every call different (identical repeats are stopped as a loop)
        return [{"name": "Glob", "arguments": {"pattern": f"mod{i % 30}*.py"}},
                {"name": "Read", "arguments": {"file_path": f"mod{i % 30}.py", "offset": i}},
                {"name": "Grep", "arguments": {"pattern": f"def f{i}\\b", "output_mode": "content"}},
                {"name": "Bash", "arguments": {"command": f"echo step {i}"}}][i % 4]

    script = [reply("Looking at the code. " * 20, [tool(i)]) for i in range(steps - 1)]
    return script + [reply("Done: everything checked. " * 40)]


def muyah(steps: int) -> dict:
    from fakeserver import FakeOpenAI

    from muyah_code.trust import trust

    home = Path(tempfile.mkdtemp(prefix="muyah-bench-home-"))
    proj = Path(tempfile.mkdtemp(prefix="muyah-bench-proj-"))
    (proj / ".muyah").mkdir()
    for i in range(30):
        (proj / f"mod{i}.py").write_text("".join(f"def f{j}():\n    return {j}\n" for j in range(40)))
    (proj / "app.py").write_text("print('hi')\n" * 50)
    trust(home, proj)
    (home / "settings.json").write_text(json.dumps({"viz": {"autostart": False}, "learning": {"reflect": False}}))
    result: dict = {}
    with FakeOpenAI(session_script(steps), chunk_delay=0.002) as srv:
        env = {**os.environ, "MUYAH_HOME": str(home), "MUYAH_BASE_URL": srv.url, "MUYAH_MODEL": "fake-model",
               "MUYAH_API_KEY": "k", "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
        py = [sys.executable, "-m", "muyah_code"]
        result["startup_s"] = median_startup(py + ["--version"], env)
        result.update(interactive(py + ["--mode", "acceptEdits"], proj, env, "❯", settle=1.0))
        p = subprocess.Popen(py + ["-p", "check the project", "--mode", "bypassPermissions", "--output-format", "json"],
                             cwd=proj, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        s = Sampler(p.pid).start()
        t0 = time.monotonic()
        out, err = p.communicate(timeout=600)
        stats = s.stop()
        result["session_s"] = round(time.monotonic() - t0, 1)
        result["session_peak_rss_mb"] = stats["peak_mb"]
        result["session_cpu_pct"] = stats["cpu_pct"]
        payload = json.loads(out.decode("utf-8") or "{}")
        result["session_steps"] = payload.get("num_turns")
        if payload.get("status") != "ok":
            raise SystemExit(f"the benchmark session failed: {payload or err.decode('utf-8', 'replace')[-500:]}")
    return result


def other(command: str) -> dict:
    exe = shutil.which(command)
    if not exe:
        return {}
    work = Path(tempfile.mkdtemp(prefix=f"bench-{command}-"))
    argv = [exe] if os.name != "nt" or not exe.lower().endswith((".cmd", ".bat")) else ["cmd", "/c", exe]
    result = {"startup_s": median_startup(argv + ["--version"])}
    result.update(interactive(argv, work, dict(os.environ), None))
    return result


def version(command: str) -> str:
    try:
        exe = shutil.which(command)
        argv = ["cmd", "/c", exe, "--version"] if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")) \
            else [exe, "--version"]
        return subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[0][:40]
    except (OSError, IndexError, subprocess.TimeoutExpired, AttributeError):
        return "?"


def table(rows: dict[str, dict]) -> str:
    cols = [("startup_s", "startup (s)"), ("ready_s", "ready (s)"), ("idle_rss_mb", "idle RAM (MB)"),
            ("idle_cpu_pct", "idle CPU %"), ("session_peak_rss_mb", "50-step peak RAM (MB)"),
            ("session_cpu_pct", "50-step CPU %")]
    lines = ["| CLI | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in rows.items():
        cells = ["–" if r.get(k) is None else str(r[k]) for k, _ in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--others", action="store_true", help="also measure other coding CLIs that are installed")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--json", help="write the results here")
    ap.add_argument("--check", action="store_true", help="fail if MUYAH-CODE exceeds the ceilings (CI)")
    a = ap.parse_args()
    from muyah_code import __version__

    rows = {f"MUYAH-CODE {__version__}": muyah(a.steps)}
    if a.others:
        for name, command in OTHERS.items():
            measured = other(command)
            if measured:
                rows[f"{name} {version(command)}"] = measured
    machine = f"{platform.system()} {platform.release()}, {psutil.cpu_count()} CPUs, " \
              f"{psutil.virtual_memory().total / 2**30:.0f} GB RAM, Python {platform.python_version()}"
    print(f"Measured on {machine} (pseudo-terminal overhead {pty_baseline():.1f}s, subtracted)\n")
    print(table(rows))
    if a.json:
        Path(a.json).write_text(json.dumps({"machine": machine, "rows": rows}, indent=2), encoding="utf-8")
    if a.check:
        mine = next(iter(rows.values()))
        over = [f"{k} {mine[k]} > {limit}" for k, limit in CEILINGS.items() if (mine.get(k) or 0) > limit]
        if over:
            print("\nOver the ceiling: " + "; ".join(over))
            return 1
        print("\nWithin the ceilings: " + ", ".join(f"{k} ≤ {v}" for k, v in CEILINGS.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
