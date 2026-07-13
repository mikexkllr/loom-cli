"""Undo snapshots and repo map / @mention expansion."""

import pytest

pytest.importorskip("pydantic")

from loom.core import repomap, undo


def test_snapshot_and_undo_roundtrip(tmp_path):
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("original")

    token = undo.current_turn_id.set("turn-1")
    try:
        undo.snapshot(tmp_path, "src/app.py")
        target.write_text("modified")
        undo.snapshot(tmp_path, "src/new_file.py")  # created this turn
        (tmp_path / "src" / "new_file.py").write_text("brand new")
    finally:
        undo.current_turn_id.reset(token)

    restored = undo.undo_last(tmp_path)
    assert sorted(restored) == ["src/app.py", "src/new_file.py"]
    assert target.read_text() == "original"
    assert not (tmp_path / "src" / "new_file.py").exists()  # creation undone
    assert undo.undo_last(tmp_path) == []  # nothing left


def test_first_snapshot_per_turn_wins(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("v1")
    token = undo.current_turn_id.set("turn-2")
    try:
        undo.snapshot(tmp_path, "f.txt")
        target.write_text("v2")
        undo.snapshot(tmp_path, "f.txt")  # later write in same turn — ignored
        target.write_text("v3")
    finally:
        undo.current_turn_id.reset(token)
    undo.undo_last(tmp_path)
    assert target.read_text() == "v1"


def test_no_turn_id_means_no_snapshot(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    undo.snapshot(tmp_path, "f.txt")
    assert undo.turns(tmp_path) == []


def test_repo_map_lists_top_level(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("")
    (tmp_path / "src" / "b.py").write_text("")
    (tmp_path / "README.md").write_text("")
    out = repomap.repo_map(tmp_path)
    assert "src/" in out and "README.md" in out


def test_expand_mentions_inlines_file(tmp_path):
    (tmp_path / "notes.txt").write_text("secret sauce")
    out = repomap.expand_mentions("look at @notes.txt please", tmp_path)
    assert "secret sauce" in out and "[Attached file: notes.txt]" in out


def test_expand_mentions_ignores_missing_and_escapes(tmp_path):
    text = "email me @alice and check @../outside.txt"
    assert repomap.expand_mentions(text, tmp_path) == text
