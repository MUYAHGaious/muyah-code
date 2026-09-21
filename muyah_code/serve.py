"""`muyah serve <engine>`: run a model server locally in the background and connect to it.

    muyah serve colibri --model /nvme/glm52_i4 --ctx 65536
    muyah serve soup --model ./output
    muyah serve ollama --model qwen2.5-coder:14b --ctx 32768
    muyah serve vllm --model Qwen/Qwen2.5-Coder-7B-Instruct
    muyah serve llamacpp --model ./qwen2.5-coder-7b-q4_k_m.gguf --ctx 32768
    muyah serve --status | --stop
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from rich.console import Console

from muyah_code.backends import ENGINE_PRESETS, build_profile, probe, serve_command, smoke_test
from muyah_code.config import Config
from muyah_code.tools.shell import IS_WINDOWS


def _state_path(cfg: Config) -> Path:
    return cfg.home / "serve.json"


def _alive(pid: int) -> bool:
    if IS_WINDOWS:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def status(cfg: Config, console: Console) -> int:
    path = _state_path(cfg)
    if not path.exists():
        console.print("No server started by MUYAH-CODE.")
        return 0
    st = json.loads(path.read_text(encoding="utf-8"))
    alive = _alive(st["pid"])
    console.print(f"{st['engine']} (pid {st['pid']}) {'[green]running[/]' if alive else '[red]not running[/]'} "
                  f"at {st['base_url']}  model {st['model']}  log {st['log']}")
    return 0 if alive else 1


def stop(cfg: Config, console: Console) -> int:
    path = _state_path(cfg)
    if not path.exists():
        console.print("Nothing to stop.")
        return 0
    st = json.loads(path.read_text(encoding="utf-8"))
    if _alive(st["pid"]):
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(st["pid"])], capture_output=True)
        else:
            try:
                os.killpg(os.getpgid(st["pid"]), signal.SIGTERM)
            except OSError:
                pass
        console.print(f"Stopped {st['engine']} (pid {st['pid']}).")
    path.unlink()
    return 0


def _tail(path: Path, n: int = 12) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def serve(cfg: Config, console: Console, engine: str, model: str, port: int | None, ctx: int | None,
          binary: str | None, wait: int, profile_name: str | None) -> int:
    preset = ENGINE_PRESETS.get(engine)
    if preset is None:
        console.print(f"[red]Unknown engine '{engine}'.[/] Known: {', '.join(ENGINE_PRESETS)}")
        return 2
    exe = binary or shutil.which(preset.binary)
    if not exe:
        console.print(f"[red]'{preset.binary}' was not found on PATH.[/]\nInstall: {preset.install}\n"
                      "Or pass --bin <path to the executable>.")
        return 1
    port = port or preset.port
    base_url = f"http://127.0.0.1:{port}/v1"
    existing = probe(base_url, timeout=1.5)
    if existing.ok:
        console.print(f"[yellow]Something already serves {base_url}[/] (models: {', '.join(existing.models[:5])}). "
                      "Connecting to it instead of starting another server.")
    else:
        argv, env = serve_command(engine, model, port, ctx, exe)
        log = cfg.home / "logs" / f"serve-{engine}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        with open(log, "ab") as lf:
            proc = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    env={**os.environ, **env}, **kwargs)
        _state_path(cfg).write_text(json.dumps({"engine": engine, "pid": proc.pid, "base_url": base_url,
                                                "model": model, "log": str(log), "started": time.time()}),
                                    encoding="utf-8")
        console.print(f"Started {engine} (pid {proc.pid}): {' '.join(argv)}\nLog: {log}")
        start = time.time()
        with console.status("Waiting for the server (loading a model can take minutes)...") as st:
            while True:
                if proc.poll() is not None:
                    console.print(f"[red]{engine} exited with code {proc.returncode}.[/] Last log lines:\n"
                                  f"{_tail(log)}")
                    _state_path(cfg).unlink(missing_ok=True)
                    return 1
                if probe(base_url, timeout=2).ok:
                    break
                elapsed = int(time.time() - start)
                if elapsed > wait:
                    console.print(f"[red]Not ready after {wait}s.[/] It may still be loading; check {log} and run "
                                  f"`muyah connect {base_url}` later, or `muyah serve --stop`.")
                    return 1
                last = _tail(log, 1)
                st.update(f"Waiting for {engine}... {elapsed}s  [dim]{last[-90:]}[/]")
                time.sleep(2)
        if engine == "ollama" and model:
            with console.status(f"Pulling {model} (first time only)..."):
                subprocess.run([exe, "pull", model], env={**os.environ, **env}, capture_output=True)

    ep = probe(base_url, timeout=10)
    served = model if engine in ("ollama", "vllm") else None
    chosen = served if served and (not ep.models or served in ep.models) else (ep.models[0] if ep.models else model)
    with console.status("Testing a completion..."):
        ok, out = smoke_test(base_url, chosen, timeout=900 if engine == "colibri" else 300)
    if not ok:
        console.print(f"[yellow]Server is up but the test completion failed: {out}[/]")
    else:
        console.print(f"[green]Ready.[/] {chosen} replied {out!r}")
    name = profile_name or f"{engine}-local"
    cfg.persist(f"profiles.{name}", build_profile(base_url, chosen, None, ctx, engine), scope="user")
    cfg.persist("profile", name, scope="user")
    console.print(f"Profile [bold]{name}[/] saved and active. Run [bold]muyah[/] to start coding; "
                  "`muyah serve --stop` to shut the server down.")
    note = ENGINE_PRESETS[engine].notes
    if note:
        console.print(f"[dim]{note}[/]")
    return 0


def main(cfg: Config, argv: list[str], console: Console) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="muyah serve", description="Run a model server locally and connect to it.")
    p.add_argument("engine", nargs="?", choices=list(ENGINE_PRESETS))
    p.add_argument("-m", "--model", help="Model path/id (colibri: model dir; soup: output dir; llamacpp: .gguf)")
    p.add_argument("--port", type=int)
    p.add_argument("--ctx", type=int, help="Context window to serve with")
    p.add_argument("--bin", dest="binary", help="Path to the engine executable")
    p.add_argument("--wait", type=int, default=1800, help="Seconds to wait for readiness (default 1800)")
    p.add_argument("--name", help="Profile name (default <engine>-local)")
    p.add_argument("--status", action="store_true")
    p.add_argument("--stop", action="store_true")
    a = p.parse_args(argv)
    if a.stop:
        return stop(cfg, console)
    if a.status:
        return status(cfg, console)
    if not a.engine:
        p.print_help()
        return 2
    if not a.model and a.engine not in ("lmstudio",):
        console.print("[red]--model is required[/]")
        return 2
    return serve(cfg, console, a.engine, a.model or "", a.port, a.ctx, a.binary, a.wait, a.name)


__all__ = ["main", "serve", "stop", "status"]
