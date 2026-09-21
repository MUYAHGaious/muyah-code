from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real ~/.muyah or the legacy colab-code config during tests."""
    home = tmp_path / "muyah_home"
    monkeypatch.setenv("MUYAH_HOME", str(home))
    for var in ("MUYAH_BASE_URL", "MUYAH_API_KEY", "MUYAH_MODEL", "MUYAH_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    import muyah_code.config as config

    monkeypatch.setattr(config, "LEGACY_CONFIG", tmp_path / "no-legacy.json")
    return home


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".muyah").mkdir()
    return root


@pytest.fixture
def ctx(project):
    from muyah_code.config import load_config
    from muyah_code.session import Checkpoints
    from muyah_code.tools.base import ToolContext

    cfg = load_config(cwd=project)
    c = ToolContext(cwd=project, project_root=project, config=cfg)
    c.services["checkpoints"] = Checkpoints()
    return c
