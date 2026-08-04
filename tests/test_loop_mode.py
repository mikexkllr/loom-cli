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


def test_decline_feedback_reaches_the_model(tmp_path):
    settings = st.Settings(permissions=st.Permissions(default_mode="ask"))
    mw = PolicyMiddleware(settings, cwd=str(tmp_path))

    def handler(_req):
        return SimpleNamespace(executed=True)

    req = SimpleNamespace(call={"name": "execute", "args": {"command": "ls"}, "id": "x"})
    policy.auto_approve.set(False)
    policy.confirm_callback.set(lambda n, i, r: (False, "run pytest instead"))
    try:
        result = mw.wrap_tool_call(req, handler)
        content = str(getattr(result, "content", result))
        assert "declined" in content
        assert "run pytest instead" in content
    finally:
        policy.confirm_callback.set(lambda n, i, r: False)


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


def test_tool_crash_is_reported(tmp_path, monkeypatch):
    """A tool that raises is the most common error a user sees, and until the
    policy wrapper caught it nothing reached the crash reporter: LangChain
    turns tool exceptions into an error ToolMessage inside the tool node, so
    they never reach the REPL's handler."""
    from loom.core import telemetry

    seen = []
    monkeypatch.setattr(telemetry, "report", lambda where, exc, **tags: seen.append((where, exc, tags)))

    settings = st.Settings(permissions=st.Permissions(default_mode="allow"))
    mw = PolicyMiddleware(settings, cwd=str(tmp_path))
    boom = RuntimeError("tool exploded")

    def handler(_req):
        raise boom

    with pytest.raises(RuntimeError):
        mw.wrap_tool_call(SimpleNamespace(call={"name": "grep", "args": {}, "id": "x"}), handler)
    assert seen == [("tool", boom, {"tool": "grep"})]


def test_tool_error_result_is_reported_without_its_body(tmp_path, monkeypatch):
    """Tools usually *return* their failure rather than raising. Report it —
    but send only the tool name and the exception class, never the message,
    which routinely quotes a path or a line of the user's file."""
    from loom.core import telemetry

    seen = []
    monkeypatch.setattr(
        telemetry, "capture_message", lambda msg, where="", **tags: seen.append((msg, where, tags))
    )

    policy._reported_tool_errors.clear()
    settings = st.Settings(permissions=st.Permissions(default_mode="allow"))
    mw = PolicyMiddleware(settings, cwd=str(tmp_path))
    result = SimpleNamespace(status="error", content="FileNotFoundError: /Users/someone/secret.py")
    mw.wrap_tool_call(SimpleNamespace(call={"name": "read_file", "args": {}, "id": "x"}), lambda _r: result)

    (msg, where, tags) = seen[0]
    assert msg == "tool failed: read_file" and where == "tool.result"
    assert tags == {"tool": "read_file", "error_type": "FileNotFoundError"}
    assert "secret.py" not in str(seen)

    # A coding agent misses on paths constantly; the repeats carry no new
    # information and would bury the real failures.
    mw.wrap_tool_call(SimpleNamespace(call={"name": "read_file", "args": {}, "id": "y"}), lambda _r: result)
    assert len(seen) == 1


def test_tool_success_is_not_reported(tmp_path, monkeypatch):
    from loom.core import telemetry

    seen = []
    monkeypatch.setattr(telemetry, "capture_message", lambda *a, **k: seen.append(a))
    policy._reported_tool_errors.clear()
    settings = st.Settings(permissions=st.Permissions(default_mode="allow"))
    mw = PolicyMiddleware(settings, cwd=str(tmp_path))
    mw.wrap_tool_call(
        SimpleNamespace(call={"name": "grep", "args": {}, "id": "x"}),
        lambda _r: SimpleNamespace(status="success", content="3 matches"),
    )
    assert seen == []
