import pytest

from muyah_code.tools.base import ToolError
from muyah_code.tools.fs import EditTool, LSTool, ReadTool, WriteTool


def test_read_numbers_lines_and_records_state(ctx, project):
    (project / "a.py").write_text("one\ntwo\nthree\n")
    res = ReadTool().run({"file_path": "a.py"}, ctx)
    assert "     1\tone" in res.content and "     3\tthree" in res.content
    assert str((project / "a.py").resolve()) in ctx.file_state


def test_read_offset_limit(ctx, project):
    (project / "b.txt").write_text("\n".join(f"l{i}" for i in range(1, 101)))
    res = ReadTool().run({"file_path": "b.txt", "offset": 10, "limit": 5}, ctx)
    assert "    10\tl10" in res.content and "l15" not in res.content
    assert "Showing lines 10-14 of 100" in res.content


def test_read_missing_suggests_similar(ctx, project):
    (project / "config.py").write_text("x")
    with pytest.raises(ToolError, match="Did you mean: config.py"):
        ReadTool().run({"file_path": "confg.py"}, ctx)


def test_write_new_file_and_checkpoint(ctx, project):
    res = WriteTool().run({"file_path": "pkg/new.py", "content": "x = 1\n"}, ctx)
    assert (project / "pkg" / "new.py").read_text() == "x = 1\n"
    assert "Created" in res.content
    cp = ctx.service("checkpoints")
    label, restored = cp.undo()
    assert not (project / "pkg" / "new.py").exists()


def test_write_existing_requires_read(ctx, project):
    (project / "c.py").write_text("old")
    with pytest.raises(ToolError, match="must Read"):
        WriteTool().run({"file_path": "c.py", "content": "new"}, ctx)
    ReadTool().run({"file_path": "c.py"}, ctx)
    WriteTool().run({"file_path": "c.py", "content": "new"}, ctx)
    assert (project / "c.py").read_text() == "new"


def test_write_detects_external_modification(ctx, project):
    import os
    import time

    f = project / "d.py"
    f.write_text("v1")
    ReadTool().run({"file_path": "d.py"}, ctx)
    time.sleep(0.02)
    f.write_text("v2 by user")
    os.utime(f, ns=(time.time_ns(), time.time_ns() + 10_000_000))
    with pytest.raises(ToolError, match="modified on disk"):
        WriteTool().run({"file_path": "d.py", "content": "agent"}, ctx)


def test_edit_exact_unique(ctx, project):
    (project / "e.py").write_text("def a():\n    return 1\n\ndef b():\n    return 1\n")
    ReadTool().run({"file_path": "e.py"}, ctx)
    with pytest.raises(ToolError, match="matches 2 places"):
        EditTool().run({"file_path": "e.py", "old_string": "return 1", "new_string": "return 2"}, ctx)
    EditTool().run({"file_path": "e.py", "old_string": "def b():\n    return 1", "new_string": "def b():\n    return 2"},
                   ctx)
    assert (project / "e.py").read_text().endswith("def b():\n    return 2\n")


def test_edit_replace_all(ctx, project):
    (project / "f.py").write_text("x = 1\ny = x + x\n")
    ReadTool().run({"file_path": "f.py"}, ctx)
    EditTool().run({"file_path": "f.py", "old_string": "x", "new_string": "z", "replace_all": True}, ctx)
    assert (project / "f.py").read_text() == "z = 1\ny = z + z\n"


def test_edit_preserves_crlf(ctx, project):
    f = project / "g.txt"
    f.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    ReadTool().run({"file_path": "g.txt"}, ctx)
    EditTool().run({"file_path": "g.txt", "old_string": "beta\ngamma", "new_string": "BETA\ngamma"}, ctx)
    assert f.read_bytes() == b"alpha\r\nBETA\r\ngamma\r\n"


def test_edit_fuzzy_indentation_match(ctx, project):
    f = project / "h.py"
    f.write_text("class A:\n    def run(self):\n        return 1\n")
    ReadTool().run({"file_path": "h.py"}, ctx)
    # model forgot the class indentation level
    res = EditTool().run({"file_path": "h.py", "old_string": "def run(self):\n    return 1",
                          "new_string": "def run(self):\n    return 2"}, ctx)
    assert "ignoring whitespace" in res.content
    assert f.read_text() == "class A:\n    def run(self):\n        return 2\n"


def test_edit_not_found_gives_hint(ctx, project):
    (project / "i.py").write_text("def compute_total(items):\n    pass\n")
    ReadTool().run({"file_path": "i.py"}, ctx)
    with pytest.raises(ToolError, match="Closest lines"):
        EditTool().run({"file_path": "i.py", "old_string": "def compute_totals(items):", "new_string": "x"}, ctx)


def test_edit_preview_is_a_diff_and_does_not_write(ctx, project):
    f = project / "j.py"
    f.write_text("a = 1\n")
    ReadTool().run({"file_path": "j.py"}, ctx)
    diff = EditTool().preview({"file_path": "j.py", "old_string": "a = 1", "new_string": "a = 2"}, ctx)
    assert "-a = 1" in diff and "+a = 2" in diff
    assert f.read_text() == "a = 1\n"


def test_ls_skips_ignored_dirs(ctx, project):
    (project / "node_modules" / "x").mkdir(parents=True)
    (project / "src").mkdir()
    (project / "src" / "m.py").write_text("")
    out = LSTool().run({}, ctx).content
    assert "src/" in out and "m.py" in out and "node_modules" not in out
