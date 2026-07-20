"""Approval modes (default / accept-edits / yolo) and loop mode."""

from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("rich")
pytest.importorskip("langchain_core")

from loom.core import settings as st
from loom.middleware import policy
from loom.middleware.policy import PolicyMiddleware
from loom.ui import slash
from loom.ui.repl import Session
from loom.ui.slash import parse_loop_args


def _session(tmp_path):
    return Session(st.load_settings(tmp_path), cwd=str(tmp_path))


# ----------------------------------------------------------------- modes


def test_cycle_approval_mode(tmp_path):
    s = _session(tmp_path)
    assert s.approval_mode == "default"
    assert s.cycle_approval_mode() == "accept-edits"
    assert s.accept_edits and not s.yolo
    assert s.cycle_approval_mode() == "plan"
    assert s.plan and not s.accept_edits and not s.yolo
    assert s.cycle_approval_mode() == "yolo"
    assert s.yolo and not s.plan and not s.accept_edits
    assert s.cycle_approval_mode() == "default"


def test_mode_command_sets_modes(tmp_path, capsys):
    s = _session(tmp_path)
    slash.dispatch(s, "/mode yolo")
    assert s.approval_mode == "yolo"
    slash.dispatch(s, "/mode accept-edits")
    assert s.approval_mode == "accept-edits"
    slash.dispatch(s, "/mode default")
    assert s.approval_mode == "default"
    slash.dispatch(s, "/mode")  # bare = cycle
    assert s.approval_mode == "accept-edits"
    slash.dispatch(s, "/mode nonsense")
    assert "unknown mode" in capsys.readouterr().out


def test_accept_edits_gates_only_file_writes(tmp_path):
    settings = st.Settings(permissions=st.Permissions(default_mode="ask"))
    mw = PolicyMiddleware(settings, cwd=str(tmp_path))

    def handler(_req):
        return SimpleNamespace(executed=True)

    def req(name, args):
        return SimpleNamespace(call={"name": name, "args": args, "id": "x"})

    policy.auto_approve.set(False)
    policy.auto_approve_edits.set(True)
    policy.confirm_callback.set(lambda n, i, r: False)  # user would decline
    try:
        edit = mw.wrap_tool_call(req("edit_file", {"path": "a", "old_string": "x", "new_string": "y"}), handler)
        assert getattr(edit, "executed", False) is True  # auto-approved
        shell = mw.wrap_tool_call(req("execute", {"command": "ls"}), handler)
        assert getattr(shell, "executed", False) is not True  # still asks → declined
    finally:
        policy.auto_approve_edits.set(False)


# ----------------------------------------------------------------- loop


def test_parse_loop_args_forms():
    assert parse_loop_args("fix the tests") == (10, "fix the tests", None)
    assert parse_loop_args("5 fix the tests") == (5, "fix the tests", None)
    assert parse_loop_args('3 fix it --until "pytest -q"') == (3, "fix it", "pytest -q")
    assert parse_loop_args('--until "pytest -q"') == (10, "", "pytest -q")
    assert parse_loop_args("999 x") == (100, "x", None)  # capped


def test_loop_stops_on_complete_token(tmp_path):
    s = _session(tmp_path)
    calls = []

    def fake_turn(text):
        calls.append(text)
        return "did some work" if len(calls) < 3 else "all done LOOP_COMPLETE"

    s.run_turn = fake_turn
    s.run_loop("do the thing", max_iters=10)
    assert len(calls) == 3
    assert "do the thing" in calls[0]
    assert "Continue the loop task" in calls[1]


def test_loop_until_command_feeds_failures_back(tmp_path):
    marker = tmp_path / "ok"
    s = _session(tmp_path)
    calls = []

    def fake_turn(text):
        calls.append(text)
        if len(calls) == 2:
            marker.write_text("done")  # second iteration "fixes" it
        return "working"

    s.run_turn = fake_turn
    s.run_loop("make it pass", max_iters=10, until=f"test -f {marker}")
    assert len(calls) == 2
    assert "still fails" in calls[1]  # check failure fed into iteration 2


def test_loop_respects_max_iters(tmp_path):
    s = _session(tmp_path)
    calls = []
    s.run_turn = lambda text: calls.append(text) or "never done"
    s.run_loop("endless", max_iters=4)
    assert len(calls) == 4


def test_loop_stops_when_interrupted(tmp_path):
    s = _session(tmp_path)
    calls = []

    def fake_turn(text):
        calls.append(text)
        s._interrupted = True
        return None

    s.run_turn = fake_turn
    s.run_loop("task", max_iters=10)
    assert len(calls) == 1


def test_run_turn_survives_model_connection_error(tmp_path, monkeypatch, capsys):
    """If both streaming and the synchronous model call fail, run_turn returns
    None and prints an error instead of crashing the REPL."""
    s = _session(tmp_path)

    class BrokenAgent:
        def stream(self, *a, **k):
            raise RuntimeError("Connection refused")

        def invoke(self, *a, **k):
            raise RuntimeError("API connection failed")

    s.bundle = SimpleNamespace(agent=BrokenAgent(), persistent=False, mode="normal", fallbacks={})
    result = s.run_turn("do something")
    assert result is None
    out = capsys.readouterr().out
    assert "model call failed" in out
