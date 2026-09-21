"""Work in a separate git worktree: its own folder and branch, so parallel sessions never collide.

    muyah --worktree fix-login        # .muyah/worktrees/fix-login on branch muyah/fix-login
    muyah --worktree fix-login        # later: the same worktree again (continue with -c)

On creation:
  * files listed in .worktreeinclude (one glob per line, relative to the repo, e.g. `.env`, `config/*.local.json`)
    are copied in: untracked files such as secrets are otherwise missing from a fresh checkout,
  * .muyah/worktree-setup, if present, runs inside the new worktree (e.g. `npm install`).

On exit, a worktree without changes (nothing uncommitted, no new commits) is removed with `git worktree remove`,
which refuses to remove anything with changes. One with changes is kept, and you get the commands to
merge it or remove it yourself. MUYAH-CODE never deletes your work.

Sub-agents whose definition says `isolation: worktree` run in a worktree of their own the same way.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,60}$")
FOLDER = Path(".muyah") / "worktrees"


class WorktreeError(Exception):
    pass


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}
    res = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=300, **kwargs)
    if check and res.returncode != 0:
        raise WorktreeError(f"git {' '.join(args[:3])}: {(res.stderr or res.stdout).strip()[:400]}")
    return res


def repo_root(cwd: Path) -> Path:
    if not shutil.which("git"):
        raise WorktreeError("worktrees need git, which is not installed")
    res = _git(cwd, "rev-parse", "--show-toplevel", check=False)
    if res.returncode != 0:
        raise WorktreeError(f"{cwd} is not inside a git repository")
    return Path(res.stdout.strip()).resolve()


def main_root(path: Path) -> Path:
    """The main checkout of the repository a worktree belongs to."""
    common = Path(_git(path, "rev-parse", "--git-common-dir").stdout.strip())
    common = common if common.is_absolute() else (path / common)
    return common.resolve().parent


def branch_of(name: str) -> str:
    return f"muyah/{name}"


def create(cwd: Path, name: str) -> tuple[Path, bool]:
    """(worktree path, created now). An existing worktree with this name is reused."""
    if not NAME.match(name):
        raise WorktreeError(f"'{name}' is not a usable worktree name (letters, digits, . _ - only)")
    root = repo_root(cwd)
    path = root / FOLDER / name
    if (path / ".git").exists():
        return path, False
    _hide_folder(root)
    branch = branch_of(name)
    exists = _git(root, "rev-parse", "--verify", "-q", f"refs/heads/{branch}", check=False).returncode == 0
    if exists:
        _git(root, "worktree", "add", str(path), branch)
    else:
        _git(root, "worktree", "add", "-b", branch, str(path), "HEAD")
    base = _git(path, "rev-parse", "HEAD").stdout.strip()
    marker = _base_file(root, name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(base, encoding="utf-8")
    copy_includes(root, path)
    return path, True


def _git_dir(root: Path) -> Path:
    git_dir = Path(_git(root, "rev-parse", "--git-common-dir").stdout.strip())
    return git_dir if git_dir.is_absolute() else root / git_dir


def _base_file(root: Path, name: str) -> Path:
    """Where the commit a worktree started from is kept (inside .git, so the worktree itself stays clean)."""
    return _git_dir(root) / "muyah-worktrees" / name


def _hide_folder(root: Path) -> None:
    """.muyah/worktrees must never show up as untracked files in the main checkout."""
    exclude = _git_dir(root) / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    known = exclude.read_text(encoding="utf-8").splitlines() if exclude.exists() else []
    if "/.muyah/worktrees/" not in known:
        with open(exclude, "a", encoding="utf-8") as f:
            f.write(("\n" if known and known[-1] else "") + "/.muyah/worktrees/\n")


def copy_includes(root: Path, path: Path) -> list[str]:
    listing = root / ".worktreeinclude"
    if not listing.is_file():
        return []
    copied = []
    for raw in listing.read_text(encoding="utf-8").splitlines():
        pattern = raw.strip()
        if not pattern or pattern.startswith("#"):
            continue
        for src in root.glob(pattern):
            if not src.is_file() or FOLDER.as_posix() in src.relative_to(root).as_posix():
                continue
            rel = src.relative_to(root)
            dest = path / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(rel.as_posix())
    return copied


def setup_script(root: Path) -> Path | None:
    script = root / ".muyah" / "worktree-setup"
    return script if script.is_file() else None


def run_setup(root: Path, path: Path, shell, timeout: float = 900) -> tuple[int | None, str]:
    """Run .muyah/worktree-setup inside the new worktree with the Bash tool's shell."""
    from muyah_code.tools.shell import run_command

    script = setup_script(root)
    if script is None:
        return 0, ""
    code, out, timed_out = run_command(script.read_text(encoding="utf-8"), path, timeout, shell)
    return (None if timed_out else code), out


def changes(root: Path, path: Path, name: str) -> tuple[int, int]:
    """(uncommitted files, commits made since the worktree was created)."""
    dirty = len([ln for ln in _git(path, "status", "--porcelain").stdout.splitlines() if ln.strip()])
    marker = _base_file(root, name)
    if not marker.is_file():
        return dirty, 1          # unknown start: assume there is work to keep
    res = _git(path, "rev-list", "--count", f"{marker.read_text(encoding='utf-8').strip()}..HEAD", check=False)
    ahead = int(res.stdout.strip() or 0) if res.returncode == 0 else 1
    return dirty, ahead


def finish(path: Path, name: str) -> str:
    """At exit: remove the worktree if nothing would be lost, else keep it and say how to use or drop it."""
    root = main_root(path)
    dirty, ahead = changes(root, path, name)
    branch = branch_of(name)
    if not dirty and not ahead:
        _git(root, "worktree", "remove", str(path))       # git refuses if anything would be lost
        _git(root, "branch", "-d", branch, check=False)
        _base_file(root, name).unlink(missing_ok=True)
        return f"Worktree {name} had no changes and was removed."
    what = []
    if dirty:
        what.append(f"{dirty} uncommitted file{'s' if dirty != 1 else ''}")
    if ahead:
        what.append(f"{ahead} new commit{'s' if ahead != 1 else ''}")
    return (f"Worktree {name} kept ({', '.join(what)}): {path}\n"
            f"  continue:  muyah --worktree {name} -c\n"
            f"  merge:     git merge {branch}   (after committing in the worktree)\n"
            f"  remove:    git worktree remove --force \"{path}\" && git branch -D {branch}   (discards its changes)")
