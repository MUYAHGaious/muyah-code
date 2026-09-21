"""Rewind: go back to the state before any earlier turn, for the code, the conversation, or both.

Code is snapshotted in a *shadow* git repository that lives outside your project
(~/.muyah/rewind/<project hash>). It points its work tree at your folder but never touches your own
.git, index, branches or hooks. A snapshot is taken at the start of every turn, so everything is
covered: files the agent wrote or edited, and files changed by commands it ran (the next snapshot sees
the whole folder). Each checkpoint (turn -> commit, message index) is written to the session log, so
rewind still works after a restart or `--resume`.

Restoring takes a safety snapshot first, so a rewind can itself be undone. Things outside the folder
(databases, network calls, installed packages) cannot be rewound, and the UI says so.

Without git on the machine it falls back to per-file snapshots of what Write/Edit changed
(session.Checkpoints): commands' side effects are then not covered.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from muyah_code.session import Checkpoints

MAX_FILE_BYTES = 10 * 1024 * 1024       # bigger files are not snapshotted (and cannot be rewound)
SLOW_SNAPSHOT_SECONDS = 20.0            # a snapshot this slow turns shadow snapshots off (huge folders)
EXCLUDE = [".git/", ".muyah/", "node_modules/", ".venv/", "venv/", "__pycache__/", "dist/", "build/",
           "target/", ".mypy_cache/", ".pytest_cache/", ".ruff_cache/", ".tox/", ".next/", ".cache/"]


@dataclass
class Point:
    """The state right before one of your turns."""
    turn: int
    prompt: str
    ts: float
    msg_index: int              # len(agent.messages) before the prompt was added
    epoch: int                  # conversation epoch: /clear and compaction start a new one
    sha: str | None = None      # shadow commit (None: shadow repo unavailable)
    kind: str = "turn"          # turn | safety (taken just before a rewind: undo-the-rewind)
    messages: list = field(default_factory=list)  # safety points: the conversation to go back to

    def event(self) -> dict:
        d = asdict(self)
        d.pop("messages")
        return {"t": "checkpoint", **d}


class ShadowRepo:
    """One shadow repository per project, shared by every session there. Each session has its own index
    and its own ref (refs/muyah/<session>) and makes commits with write-tree + commit-tree, so several
    sessions in the same folder never race for git's index lock or a shared HEAD."""

    ACTIVE_SECONDS = 600   # a session counts as "working here" if it said so in the last 10 minutes

    def __init__(self, home: Path, project: Path, session_key: str = "default"):
        self.project = project.resolve()
        digest = hashlib.sha1(str(self.project).lower().encode()).hexdigest()[:16]
        self.git_dir = home / "rewind" / digest     # (~/.muyah/history is the prompt history file)
        self.hooks_dir = home / "rewind" / "_no_hooks"
        self.key = re.sub(r"[^A-Za-z0-9._-]", "_", session_key or "default")
        self.index = self.git_dir / f"index-{self.key}"
        self.ref = f"refs/muyah/{self.key}"
        self._last: tuple[str, str] | None = None   # (tree, commit) of this session's last snapshot
        self.skipped: list[str] = []   # large files left out of the last snapshot

    @staticmethod
    def usable(project: Path, home: Path) -> bool:
        """git is installed and the folder is a sensible project (not a whole drive or home)."""
        if not shutil.which("git"):
            return False
        p = project.resolve()
        return p.parent != p and p != Path.home().resolve() and p != home.resolve()

    def _git(self, *args: str, check: bool = True, timeout: float = 120) -> subprocess.CompletedProcess:
        cmd = ["git", f"--git-dir={self.git_dir}", f"--work-tree={self.project}",
               "-c", f"core.hooksPath={self.hooks_dir}", "-c", "commit.gpgsign=false",
               "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "-c", "core.longpaths=true",
               "-c", "user.name=muyah", "-c", "user.email=muyah@localhost", "-c", "gc.auto=0",
               "-c", "core.quotepath=false", *args]
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env["GIT_OPTIONAL_LOCKS"] = "0"
        env["GIT_INDEX_FILE"] = str(self.index)   # this session's own index
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        res = subprocess.run(cmd, cwd=str(self.project), capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=timeout, env=env, **kwargs)
        if check and res.returncode != 0:
            raise RuntimeError(f"git {' '.join(args[:2])}: {res.stderr.strip()[:300]}")
        return res

    def ensure(self) -> None:
        if (self.git_dir / "HEAD").exists():
            return
        self.git_dir.mkdir(parents=True, exist_ok=True)
        self.hooks_dir.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q")
        for key, value in (("core.bare", "false"), ("core.untrackedCache", "true"),
                           ("feature.manyFiles", "true"), ("core.autocrlf", "false")):
            self._git("config", key, value)
        (self.git_dir / "info").mkdir(exist_ok=True)
        (self.git_dir / "info" / "exclude").write_text("\n".join(EXCLUDE) + "\n", encoding="utf-8")

    def _exclude_large(self) -> None:
        """Leave out untracked files bigger than MAX_FILE_BYTES (they would bloat every snapshot)."""
        res = self._git("ls-files", "--others", "--exclude-standard", "-z", check=False)
        big = []
        for rel in filter(None, res.stdout.split("\0")):
            try:
                if (self.project / rel).stat().st_size > MAX_FILE_BYTES:
                    big.append(rel)
            except OSError:
                continue
        if big:
            exclude = self.git_dir / "info" / "exclude"
            known = set(exclude.read_text(encoding="utf-8").splitlines())
            with open(exclude, "a", encoding="utf-8") as f:
                for rel in big:
                    line = "/" + rel.replace("\\", "/")
                    if line not in known:
                        f.write(line + "\n")
            self.skipped.extend(big)

    def snapshot(self, label: str) -> str:
        """Commit the folder's current state; returns the commit id."""
        self.ensure()
        self._exclude_large()
        self._git("add", "-A", "--ignore-errors", check=False)
        tree = self._git("write-tree").stdout.strip()
        if self._last is not None and self._last[0] == tree:
            return self._last[1]                                   # nothing changed: reuse the last one
        parent = self._last[1] if self._last else None
        args = ["commit-tree", tree, "-m", label[:200] or "snapshot"] + (["-p", parent] if parent else [])
        sha = self._git(*args).stdout.strip()
        self._git("update-ref", self.ref, sha)                     # keeps it (and its history) from being pruned
        self._last = (tree, sha)
        return sha

    # ------------------------------------------------------------------ other sessions in this folder

    def mark_active(self) -> None:
        try:
            folder = self.git_dir / "active"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / self.key).write_text(str(os.getpid()), encoding="utf-8")
        except OSError:
            return             # only used for a warning; snapshots work without it

    def mark_done(self) -> None:
        try:
            (self.git_dir / "active" / self.key).unlink(missing_ok=True)
        except OSError:
            return

    def others_active(self) -> int:
        """How many other sessions worked in this folder in the last few minutes."""
        folder = self.git_dir / "active"
        now = time.time()
        try:
            return sum(1 for p in folder.iterdir() if p.name != self.key
                       and now - p.stat().st_mtime < self.ACTIVE_SECONDS)
        except OSError:
            return 0

    def changes(self, current: str, target: str) -> list[tuple[str, str]]:
        """(status, path) of what restoring `target` over `current` will change: M, A (created), D (deleted)."""
        res = self._git("diff", "--name-status", "--no-renames", current, target, check=False)
        out = []
        for line in res.stdout.splitlines():
            status, _, path = line.partition("\t")
            if path:
                out.append(({"A": "A", "D": "D"}.get(status[:1], "M"), path))
        return out

    def restore(self, current: str, target: str) -> list[tuple[str, str]]:
        """Make the folder look like `target` (current = a snapshot of the folder right now)."""
        changes = self.changes(current, target)
        self._git("read-tree", target)
        self._git("checkout-index", "-a", "-f")
        for status, rel in changes:
            if status == "D":  # present now, absent in the target: remove (only files the snapshots track)
                path = self.project / rel
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                self._prune_empty_dirs(path.parent)
        return changes

    def _prune_empty_dirs(self, folder: Path) -> None:
        while folder != self.project and self.project in folder.parents:
            try:
                folder.rmdir()
            except OSError:
                return
            folder = folder.parent


class Rewind:
    """Checkpoints for every turn, and restoring code and/or conversation to any of them."""

    def __init__(self, home: Path, project: Path, session=None):
        self.session = session
        key = getattr(session, "id", "") or f"pid{os.getpid()}"
        self.shadow = ShadowRepo(home, project, key) if ShadowRepo.usable(project, home) else None
        self.files = Checkpoints()        # always kept: the fallback without git, and a cheap per-file log
        self.points: list[Point] = []
        self.epoch = 0
        self.disabled_reason = "" if self.shadow else "git is not available (or the folder is too broad)"
        self._pending: threading.Thread | None = None
        self._lock = threading.Lock()
        self.project = project.resolve()
        self._touched: set[int] = set()                          # turns in which a tool may have changed files
        self._after: dict[int, tuple[str, list[tuple[str, str]]]] = {}   # turn -> (end snapshot, its changes)

    # ------------------------------------------------------------------ recording

    @property
    def covers_commands(self) -> bool:
        return self.shadow is not None

    def load(self, events: list[dict]) -> None:
        """Checkpoints (and conversation epochs) recorded in a resumed session."""
        for ev in events:
            if ev.get("t") == "epoch":
                self.epoch = max(self.epoch, int(ev.get("epoch") or 0))
                continue
            fields = {k: ev.get(k) for k in ("turn", "prompt", "msg_index", "epoch", "sha")}
            fields["kind"] = ev.get("point") or "turn"
            fields["ts"] = float(ev.get("at") or ev.get("ts") or 0)
            fields["epoch"] = int(fields["epoch"] or 0)
            self.points.append(Point(**fields))
            self.epoch = max(self.epoch, fields["epoch"])

    def others_active(self) -> int:
        return self.shadow.others_active() if self.shadow is not None else 0

    def close(self) -> None:
        if self.shadow is not None:
            self.shadow.mark_done()

    def begin_turn(self, prompt: str, msg_index: int) -> Point:
        """Called before a turn. The snapshot runs in the background (the model is thinking anyway);
        tools that change files wait for it (wait_ready)."""
        point = Point(turn=len([p for p in self.points if p.kind == "turn"]) + 1, prompt=prompt[:300],
                      ts=time.time(), msg_index=msg_index, epoch=self.epoch)
        self.points.append(point)
        self.files.begin_turn(prompt)
        if self.shadow is not None:
            self.shadow.mark_active()
            self._pending = threading.Thread(target=self._snap, args=(point,), name="muyah-rewind", daemon=True)
            self._pending.start()
        else:
            self._log(point)
        return point

    def _snap(self, point: Point) -> None:
        started = time.monotonic()
        try:
            point.sha = self.shadow.snapshot(f"before turn {point.turn}: {point.prompt[:120]}")
        except Exception as e:  # never break a turn over a snapshot: fall back to per-file snapshots
            self.disabled_reason = f"snapshots failed: {e}"
            self.shadow = None
        if time.monotonic() - started > SLOW_SNAPSHOT_SECONDS and self.shadow is not None:
            self.disabled_reason = "this folder is too big to snapshot quickly; per-file snapshots are used instead"
            self.shadow = None
        self._log(point)

    def wait_ready(self, timeout: float = 120) -> None:
        """Before anything changes files: the turn's snapshot must be finished."""
        pending = self._pending
        if pending is not None and pending.is_alive():
            pending.join(timeout)

    def before_change(self) -> None:
        """A tool that may change files is about to run in the current turn."""
        turns = self.turns()
        if turns:
            self._touched.add(turns[-1].turn)
        self.wait_ready()

    def turn_changes(self) -> list[tuple[str, str]]:
        """(status, path) of what the last turn changed: M, A (created), D (deleted). With the shadow repo
        this includes changes made by commands; without it, only what Write/Edit changed."""
        turns = self.turns()
        if not turns:
            return []
        point = turns[-1]
        if point.turn not in self._touched:
            return []
        if point.turn in self._after:
            return self._after[point.turn][1]
        self.wait_ready()
        if self.shadow is not None and point.sha:
            try:
                after = self.shadow.snapshot(f"after turn {point.turn}")
                changes = self.shadow.changes(point.sha, after)
                self._after[point.turn] = (after, changes)
                return changes
            except Exception:  # fall back to what Write/Edit recorded
                pass
        recorded = self.files.turns[-1]["changes"] if self.files.turns else {}
        out = []
        for key, previous in recorded.items():
            rel = self._rel(Path(key))
            if previous is None:
                out.append(("A", rel))
            else:
                out.append(("M" if Path(key).exists() else "D", rel))
        return out

    def text_before_turn(self, rel: str) -> str | None:
        """A file's content from before the last turn (None if unknown)."""
        turns = self.turns()
        if not turns:
            return None
        point = turns[-1]
        if self.shadow is not None and point.sha:
            res = self.shadow._git("show", f"{point.sha}:{rel}", check=False)
            return res.stdout if res.returncode == 0 else None
        recorded = self.files.turns[-1]["changes"] if self.files.turns else {}
        for key, previous in recorded.items():
            if self._rel(Path(key)) == rel and previous is not None:
                return previous.decode("utf-8", errors="replace")
        return None

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project).as_posix()
        except ValueError:
            return path.as_posix()

    def record(self, path: Path, previous: bytes | None) -> None:
        self.files.record(path, previous)

    def new_epoch(self) -> None:
        """The conversation was cleared or compacted: earlier points can rewind code, not conversation."""
        self.epoch += 1
        if self.session is not None:
            self.session.log_event("epoch", epoch=self.epoch)

    def _log(self, point: Point) -> None:
        if self.session is not None:
            data = point.event()
            kind = data.pop("t")
            data.pop("ts", None)   # the session log stamps its own time
            data["at"] = point.ts
            data["point"] = data.pop("kind")  # "kind" is log_event's own argument name
            self.session.log_event(kind, **data)

    # ------------------------------------------------------------------ rewinding

    def turns(self) -> list[Point]:
        return [p for p in self.points if p.kind == "turn"]

    def last_safety(self) -> Point | None:
        return next((p for p in reversed(self.points) if p.kind == "safety"), None)

    def can_rewind_conversation(self, point: Point, agent=None) -> bool:
        """The conversation can go back to `point` if it has not been cleared/compacted since, and the
        point is still part of it (not in a stretch that an earlier rewind removed)."""
        if point.epoch != self.epoch:
            return False
        return agent is None or point.msg_index <= len(agent.messages)

    def preview(self, point: Point) -> tuple[str | None, list[tuple[str, str]]]:
        """Take the safety snapshot now and list what restoring `point`'s code would change."""
        self.wait_ready()
        if self.shadow is None or not point.sha:
            return None, []
        current = self.shadow.snapshot(f"safety before rewinding to turn {point.turn}")
        return current, self.shadow.changes(current, point.sha)

    def restore(self, point: Point, code: bool, conversation: bool, agent, safety_sha: str | None = None
                ) -> dict:
        """Rewind to `point`. Returns what happened (for the UI)."""
        with self._lock:
            result: dict = {"code": [], "conversation": False, "fallback": False}
            safety = Point(turn=0, prompt=f"before rewinding to turn {point.turn}", ts=time.time(),
                           msg_index=len(agent.messages), epoch=self.epoch, kind="safety",
                           messages=list(agent.messages[1:]))
            if code:
                if self.shadow is not None and point.sha:
                    if safety_sha is None:
                        safety_sha, _ = self.preview(point)
                    safety.sha = safety_sha
                    result["code"] = self.shadow.restore(safety_sha, point.sha)
                else:
                    result["fallback"] = True
                    result["code"] = self._fallback_code(point)
                if hasattr(agent, "ctx"):
                    agent.ctx.file_state.clear()  # files changed on disk: re-read before editing
            if conversation and self.can_rewind_conversation(point, agent):
                agent.messages = agent.messages[: point.msg_index]
                if agent.session is not None:
                    agent.session.log_replace(agent.messages[1:], "rewind")
                result["conversation"] = True
            self.points.append(safety)
            self._log(safety)
            # the points after `point` are history now; keep them listed (so a rewind can be redone)
            return result

    def undo_rewind(self, agent) -> dict | None:
        """Go back to how things were right before the last rewind."""
        safety = self.last_safety()
        if safety is None:
            return None
        result: dict = {"code": [], "conversation": False, "fallback": False}
        if self.shadow is not None and safety.sha:
            current = self.shadow.snapshot("safety before undoing a rewind")
            result["code"] = self.shadow.restore(current, safety.sha)
            agent.ctx.file_state.clear()
        if safety.messages:
            agent.messages = agent.messages[:1] + list(safety.messages)
            if agent.session is not None:
                agent.session.log_replace(agent.messages[1:], "undo rewind")
            result["conversation"] = True
        self.points.remove(safety)
        return result

    def _fallback_code(self, point: Point) -> list[tuple[str, str]]:
        """Without shadow git: undo per-file snapshots of every turn since `point` (Write/Edit only)."""
        index = self.turns().index(point) if point in self.turns() else len(self.files.turns)
        undone = []
        while len(self.files.turns) > index:
            res = self.files.undo()
            if res is None:
                break
            undone.extend(("M", line) for line in res[1])
        return undone
