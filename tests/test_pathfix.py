"""`muyah path`: make the launcher reachable from any folder. Never touches the real PATH or registry."""

import io
import os
from pathlib import Path

from rich.console import Console

from muyah_code import pathfix


def test_on_path_compares_folders_not_strings(tmp_path):
    folder = tmp_path / "Scripts"
    value = os.pathsep.join(["/somewhere/else", str(folder) + os.sep])
    assert pathfix.on_path(folder, value)
    assert not pathfix.on_path(tmp_path / "Other", value)


def test_posix_start_up_files_get_one_marked_line_once(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bashrc").write_text("alias ll='ls -l'", encoding="utf-8")
    folder = tmp_path / "venv" / "bin"
    first = pathfix._add_posix(folder, home)
    assert "Added" in first
    profile = (home / ".profile").read_text(encoding="utf-8")
    bashrc = (home / ".bashrc").read_text(encoding="utf-8")
    assert f'export PATH="{folder}:$PATH"' in profile and pathfix.MARKER in profile
    assert bashrc.startswith("alias ll='ls -l'\n") and str(folder) in bashrc
    assert "already" in pathfix._add_posix(folder, home)                  # twice changes nothing
    assert (home / ".profile").read_text(encoding="utf-8").count(str(folder)) == 1


def test_fix_adds_the_launcher_folder_and_says_to_open_a_new_terminal(tmp_path, monkeypatch):
    folder = tmp_path / "Scripts"
    folder.mkdir()
    (folder / pathfix.LAUNCHER).write_text("", encoding="utf-8")
    added = []
    monkeypatch.setattr(pathfix, "launcher_dirs", lambda: [tmp_path / "nothing", folder])
    monkeypatch.setattr(pathfix, "command_available", lambda: False)
    monkeypatch.setattr(pathfix, "add_to_path", lambda f, home=None: added.append(f) or f"Added {f} to your PATH.")
    monkeypatch.setenv("PATH", str(tmp_path / "elsewhere"))
    out = io.StringIO()
    assert pathfix.fix(Console(file=out, width=200)) == 0
    text = out.getvalue()
    assert added == [folder] and "in this window now" in text           # the one line that fixes this window
    assert ("$env:Path" in text) if os.name == "nt" else ("export PATH" in text)
    assert "python -m muyah_code path" in pathfix.startup_hint()


def test_nothing_to_do_when_muyah_already_works(tmp_path, monkeypatch):
    folder = tmp_path / "Scripts"
    folder.mkdir()
    (folder / pathfix.LAUNCHER).write_text("", encoding="utf-8")
    monkeypatch.setattr(pathfix, "launcher_dirs", lambda: [folder])
    monkeypatch.setattr(pathfix, "command_available", lambda: True)
    monkeypatch.setattr(pathfix, "add_to_path", lambda *a: (_ for _ in ()).throw(AssertionError("changed PATH")))
    monkeypatch.setenv("PATH", str(folder))
    out = io.StringIO()
    assert pathfix.fix(Console(file=out, width=200)) == 0 and "already works" in out.getvalue()
    assert pathfix.startup_hint() == ""


def test_installers_exist_and_end_with_the_path_step():
    root = Path(__file__).resolve().parent.parent
    ps1 = (root / "install.ps1").read_text(encoding="utf-8")
    sh = (root / "install.sh").read_text(encoding="utf-8")
    assert "muyah_code path" in ps1 and "muyah_code path" in sh
    assert "uv tool install" in ps1 and "pipx install" in ps1 and "uv tool install" in sh


def test_windows_copies_the_launcher_where_old_windows_already_look(tmp_path, monkeypatch):
    monkeypatch.setattr(pathfix.os, "name", "nt")
    folder = tmp_path / "Scripts"
    folder.mkdir()
    (folder / pathfix.LAUNCHER).write_bytes(b"launcher v2")
    old_bin, not_on_path = tmp_path / "local-bin", tmp_path / "elsewhere"
    old_bin.mkdir()
    not_on_path.mkdir()
    (old_bin / pathfix.LAUNCHER).write_bytes(b"launcher v1")          # from an earlier version
    path_value = os.pathsep.join([str(old_bin)])
    got = pathfix.install_shim(folder, path_value, candidates=(not_on_path, old_bin))
    assert got == old_bin and (old_bin / pathfix.LAUNCHER).read_bytes() == b"launcher v2"
    assert not (not_on_path / pathfix.LAUNCHER).exists()               # only where the window searches
    assert pathfix.install_shim(folder, "", candidates=(old_bin,)) is None
