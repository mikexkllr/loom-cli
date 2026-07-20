"""Plan mode: the read-only planning pass and the approve-&-execute gate."""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")
pytest.importorskip("rich")
pytest.importorskip("langchain_core")

from loom.core import settings as st
from loom.ui import slash
from loom.ui.repl import Session


def _session(tmp_path):
    return Session(st.load_settings(tmp_path), cwd=str(tmp_path))


# ----------------------------------------------------------------- mode wiring


def test_plan_command_toggles_and_rebuilds(tmp_path, capsys):
    s = _session(tmp_path)
    s.bundle = object()  # sentinel: rebuild() must drop it
    slash.dispatch(s, "/plan")
    assert s.plan is True
    assert s.bundle is None
    assert "approve" in capsys.readouterr().out
    slash.dispatch(s, "/plan")
    assert s.plan is False
    # explicit on/off args
    slash.dispatch(s, "/plan on")
    assert s.plan is True
    slash.dispatch(s, "/plan off")
    assert s.plan is False


def test_mode_command_accepts_plan(tmp_path, capsys):
    s = _session(tmp_path)
    slash.dispatch(s, "/mode plan")
    assert s.plan is True and not s.accept_edits and not s.yolo
    assert s.mode == "plan"
    assert "read-only planning" in capsys.readouterr().out
    slash.dispatch(s, "/mode default")
    assert s.plan is False


def test_plan_is_exclusive_with_other_modes(tmp_path):
    s = _session(tmp_path)
    s.set_mode("accept-edits")
    s.set_mode("plan")
    assert s.plan and not s.accept_edits
    slash.dispatch(s, "/yolo")  # yolo kicks the session out of plan mode
    assert s.yolo and not s.plan


def test_set_mode_rebuilds_only_on_plan_transitions(tmp_path):
    s = _session(tmp_path)
    s.bundle = sentinel = object()
    s.set_mode("accept-edits")  # no plan transition → bundle untouched
    assert s.bundle is sentinel
    s.set_mode("plan")
    assert s.bundle is None


# ----------------------------------------------------------------- approve & execute


def _plan_session(tmp_path, monkeypatch, choice: str):
    s = _session(tmp_path)
    s.set_mode("plan")
    s.bundle = object()  # pretend the plan-mode agent is built
    turns = []
    s.run_turn = lambda text: turns.append(text) or "ok"
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **k: choice)
    return s, turns


def test_approve_with_auto_accept_executes(tmp_path, monkeypatch):
    s, turns = _plan_session(tmp_path, monkeypatch, "1")
    s.offer_plan_execution()
    assert s.plan is False
    assert s.accept_edits is True
    assert s.bundle is None  # leaving plan mode recompiles the agent
    assert turns == [s.PLAN_EXECUTE_PROMPT]


def test_approve_with_manual_edits_executes(tmp_path, monkeypatch):
    s, turns = _plan_session(tmp_path, monkeypatch, "2")
    s.offer_plan_execution()
    assert s.plan is False
    assert s.accept_edits is False and s.yolo is False
    assert turns == [s.PLAN_EXECUTE_PROMPT]


def test_keep_planning_stays_in_plan_mode(tmp_path, monkeypatch):
    s, turns = _plan_session(tmp_path, monkeypatch, "3")
    s.offer_plan_execution()
    assert s.plan is True
    assert turns == []


def test_interrupt_at_gate_keeps_planning(tmp_path, monkeypatch):
    s, turns = _plan_session(tmp_path, monkeypatch, "1")

    def raise_interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("rich.prompt.Prompt.ask", raise_interrupt)
    s.offer_plan_execution()
    assert s.plan is True
    assert turns == []
