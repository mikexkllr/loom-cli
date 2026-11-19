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


# ------------------------------------------------------------ undo ordering


def _write_turn(root, turn_id, name, before, after):
    target = root / name
    target.write_text(before)
    token = undo.current_turn_id.set(turn_id)
    try:
        undo.snapshot(root, name)
        target.write_text(after)
    finally:
        undo.current_turn_id.reset(token)


def test_undo_rolls_back_the_latest_turn_past_turn_nine(tmp_path, monkeypatch):
    """Turn ids end in a counter and were sorted as text, where `…-t10` comes
    before `…-t2` — so from the tenth turn on, /undo restored turn nine."""
    clock = iter(range(1, 1000))
    monkeypatch.setattr(undo.time, "time", lambda: float(next(clock)))
    for n in range(1, 12):
        _write_turn(tmp_path, f"loom-20261119-101010-t{n}", f"f{n}.txt", f"before {n}", f"after {n}")

    assert undo.turns(tmp_path)[-1] == "loom-20261119-101010-t11"
    assert undo.undo_last(tmp_path) == ["f11.txt"]
    assert (tmp_path / "f11.txt").read_text() == "before 11"
    assert (tmp_path / "f9.txt").read_text() == "after 9"  # untouched
    assert undo.undo_last(tmp_path) == ["f10.txt"]
    assert undo.undo_last(tmp_path) == ["f9.txt"]


def test_parallel_snapshots_never_share_a_backup(tmp_path):
    """Parallel subagents snapshot from separate worker threads. Unlocked, two
    of them could read the same index, derive the same backup name, and leave
    /undo restoring one file's bytes into another."""
    import threading

    names = [f"file{i}.txt" for i in range(40)]
    for name in names:
        (tmp_path / name).write_text(f"original {name}")
    token = undo.current_turn_id.set("turn-parallel")
    try:
        start = threading.Barrier(len(names))

        def snap(name):
            start.wait()
            undo.snapshot(tmp_path, name)

        threads = [threading.Thread(target=snap, args=(n,)) for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        undo.current_turn_id.reset(token)
    for name in names:
        (tmp_path / name).write_text("clobbered")

    assert sorted(undo.undo_last(tmp_path)) == sorted(names)
    for name in names:
        assert (tmp_path / name).read_text() == f"original {name}"


def test_old_turns_are_pruned_and_the_newest_survive(tmp_path, monkeypatch):
    clock = iter(range(1, 10_000))
    monkeypatch.setattr(undo.time, "time", lambda: float(next(clock)))
    monkeypatch.setattr(undo, "KEEP_TURNS", 5)
    for n in range(1, 13):
        _write_turn(tmp_path, f"t{n}", "f.txt", f"v{n - 1}", f"v{n}")
    assert undo.turns(tmp_path) == ["t8", "t9", "t10", "t11", "t12"]
    assert undo.undo_last(tmp_path) == ["f.txt"]
    assert (tmp_path / "f.txt").read_text() == "v11"


def test_turns_without_a_recorded_start_still_order_by_age(tmp_path):
    """An index written before `created` existed falls back to its mtime."""
    import json
    import os

    root = tmp_path / ".loom" / "undo"
    for age, name in ((300, "zz-oldest"), (200, "mm-middle"), (100, "aa-newest")):
        (root / name).mkdir(parents=True)
        index = root / name / "index.json"
        index.write_text(json.dumps({"files": {}}))
        stamp = 1_800_000_000 - age
        os.utime(index, (stamp, stamp))
    assert undo.turns(tmp_path) == ["zz-oldest", "mm-middle", "aa-newest"]


def test_mentioned_files_lists_without_reading(tmp_path):
    (tmp_path / "a.py").write_text("print('a')")
    (tmp_path / "big.bin").write_bytes(b"x" * (repomap._MAX_MENTION_BYTES + 1))
    found = repomap.mentioned_files("see @a.py, @a.py again, @missing.py, @big.bin and @../escape", tmp_path)
    assert [rel for rel, _ in found] == ["a.py"]
