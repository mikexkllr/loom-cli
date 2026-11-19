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


def test_thread_ids_are_unique_within_one_second():
    """The id is the checkpointer's key. With a bare timestamp, `/clear` inside
    the second a session started handed back the same thread — history and all."""
    from loom.core import sessions

    ids = {sessions.new_thread_id() for _ in range(200)}
    assert len(ids) == 200
    import re

    # Still time-prefixed, so ids (and /resume's list) sort chronologically.
    assert all(re.fullmatch(r"loom-\d{8}-\d{6}-[0-9a-f]{6}", i) for i in ids)


def test_a_malformed_index_reads_as_no_sessions(tmp_path):
    from loom.core import sessions

    (tmp_path / ".loom").mkdir()
    (tmp_path / ".loom" / "sessions.json").write_text('{"not": "a list"}')
    assert sessions.load_index(tmp_path) == []
    sessions.record(tmp_path, "loom-x", "first message")  # and recording recovers it
    assert [row["thread_id"] for row in sessions.load_index(tmp_path)] == ["loom-x"]
