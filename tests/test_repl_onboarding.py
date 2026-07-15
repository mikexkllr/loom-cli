"""REPL first-run wiring: launches the setup wizard on a true first run,
falls back to the passive hint otherwise, and never crashes on cancel."""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")
pytest.importorskip("rich")

from loom.core import settings as st
from loom.ui import onboarding
from loom.ui import repl


def _session(tmp_path):
    return repl.Session(st.load_settings(tmp_path), cwd=str(tmp_path))


def test_runs_wizard_on_true_first_run(tmp_path, monkeypatch):
    monkeypatch.setattr(onboarding, "needs_onboarding", lambda root: True)
    calls = []
    monkeypatch.setattr(onboarding, "run", lambda console, **kw: calls.append(kw) or None)
    s = _session(tmp_path)
    reloaded = []
    monkeypatch.setattr(s, "reload_settings", lambda: reloaded.append("settings"))
    monkeypatch.setattr(s, "rebuild", lambda: reloaded.append("rebuild"))

    repl._maybe_run_onboarding(s)

    assert calls and calls[0]["root"] == s.cwd
    assert reloaded == ["settings", "rebuild"]


def test_falls_back_to_hint_when_not_first_run(tmp_path, monkeypatch):
    monkeypatch.setattr(onboarding, "needs_onboarding", lambda root: False)
    wizard_calls = []
    monkeypatch.setattr(onboarding, "run", lambda console, **kw: wizard_calls.append(1))
    hint_calls = []
    monkeypatch.setattr(repl, "_setup_hint", lambda session: hint_calls.append(1))
    s = _session(tmp_path)

    repl._maybe_run_onboarding(s)

    assert wizard_calls == []
    assert hint_calls == [1]


@pytest.mark.parametrize("exc", [KeyboardInterrupt, EOFError])
def test_cancel_does_not_crash_or_reload(tmp_path, monkeypatch, exc):
    monkeypatch.setattr(onboarding, "needs_onboarding", lambda root: True)

    def _cancel(console, **kw):
        raise exc

    monkeypatch.setattr(onboarding, "run", _cancel)
    s = _session(tmp_path)
    reloaded = []
    monkeypatch.setattr(s, "reload_settings", lambda: reloaded.append(1))
    monkeypatch.setattr(s, "rebuild", lambda: reloaded.append(1))

    repl._maybe_run_onboarding(s)  # must not raise

    assert reloaded == []  # cancelled before any reload
