"""Session persistence: checkpointer selection and the /resume index."""

import pytest

pytest.importorskip("pydantic")

from loom.core import sessions as sessions_mod


def test_index_records_and_updates(tmp_path):
    sessions_mod.record(tmp_path, "t1", "fix the login bug in a very long descriptive sentence " * 5)
    sessions_mod.record(tmp_path, "t1", "follow-up")
    sessions_mod.record(tmp_path, "t2", "second session")
    rows = sessions_mod.load_index(tmp_path)
    assert [r["thread_id"] for r in rows] == ["t1", "t2"]
    assert rows[0]["turns"] == 2
    assert len(rows[0]["title"]) <= 80  # titles are truncated


def test_index_is_bounded(tmp_path):
    for i in range(60):
        sessions_mod.record(tmp_path, f"t{i}", f"task {i}")
    assert len(sessions_mod.load_index(tmp_path)) == 50


def test_sqlite_checkpointer_is_durable_when_available(tmp_path):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    checkpointer, durable = sessions_mod.make_checkpointer(tmp_path)
    assert durable is True
    assert (tmp_path / ".loom" / "sessions.db").exists()


def test_checkpointer_never_raises(tmp_path):
    # Whatever is (not) installed, make_checkpointer must return gracefully.
    checkpointer, durable = sessions_mod.make_checkpointer(tmp_path)
    assert isinstance(durable, bool)
