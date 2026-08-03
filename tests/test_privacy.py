"""The privacy wizard step, the per-project gate, and /privacy's state display
— driven with scripted answers, the same harness as test_onboarding_interactive.
"""

import io

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("rich")

from rich.console import Console

from loom.core import telemetry as tel
from loom.ui import privacy as priv


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False)


class _Scripted:
    def __init__(self, answers: list):
        self.answers = list(answers)

    def __call__(self, *args, **kwargs):
        if not self.answers:
            raise EOFError("scripted answers exhausted")
        return self.answers.pop(0)


def _as_confirm(scripted):
    def _confirm(_console, _question, **_kwargs):
        return scripted()

    return _confirm


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(tel, "store_path", lambda: tmp_path / "telemetry.json")
    tel._reset_for_tests()
    yield tmp_path
    tel._reset_for_tests()


# --------------------------------------------------------------- prompt_mode


@pytest.mark.parametrize("answer,expected", [("1", "none"), ("2", "errors"), ("3", "full")])
def test_prompt_mode_maps_numbers_to_modes(monkeypatch, answer, expected):
    monkeypatch.setattr(priv, "Prompt", type("P", (), {"ask": staticmethod(lambda *a, **k: answer)}))
    assert priv.prompt_mode(_console()) == expected


def test_prompt_mode_defaults_to_the_current_mode(monkeypatch):
    seen = {}

    def _ask(*a, default=None, **k):
        seen["default"] = default
        return default

    monkeypatch.setattr(priv, "Prompt", type("P", (), {"ask": staticmethod(_ask)}))
    assert priv.prompt_mode(_console(), "errors") == "errors"
    assert seen["default"] == "2"


# ------------------------------------------------------------------------ run


def test_run_none_records_a_real_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(priv, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["1"]))}))
    consent = priv.run(_console(), tmp_path)
    assert consent.mode == "none"
    assert consent.decided is True
    assert tel.load().decided is True  # persisted — never asked again
    assert consent.projects == {}  # nothing to share, no project recorded


def test_run_errors_collects_a_dsn_and_shares_here(monkeypatch, tmp_path):
    monkeypatch.setattr(priv, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["2"]))}))
    monkeypatch.setattr(priv, "prompt_sentry_dsn", lambda console, current="": "https://k@o/1")
    consent = priv.run(_console(), tmp_path)
    assert consent.mode == "errors"
    assert consent.sentry_dsn == "https://k@o/1"
    # The directory setup ran in plainly consented — don't re-ask immediately.
    assert consent.projects[tel.project_key(tmp_path)]["share"] is True


def test_run_full_requires_a_second_explicit_yes(monkeypatch, tmp_path):
    """The mode-3 warning card ends in a confirm defaulting to no — declining
    falls back to errors (with a DSN) rather than silently upgrading."""
    monkeypatch.setattr(priv, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["3"]))}))
    monkeypatch.setattr(priv, "prompt_sentry_dsn", lambda console, current="": "https://k@o/1")
    monkeypatch.setattr(priv.render, "confirm", _as_confirm(_Scripted([False])))
    consent = priv.run(_console(), tmp_path)
    assert consent.mode == "errors"
    assert consent.langfuse_public_key == ""


def test_run_full_collects_and_stores_langfuse_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(priv, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["3"]))}))
    monkeypatch.setattr(priv, "prompt_sentry_dsn", lambda console, current="": "https://k@o/1")
    monkeypatch.setattr(priv.render, "confirm", _as_confirm(_Scripted([True])))
    monkeypatch.setattr(priv, "prompt_langfuse", lambda console, consent: ("pk", "sk", "https://cloud.langfuse.com"))
    consent = priv.run(_console(), tmp_path)
    assert consent.mode == "full"
    assert consent.langfuse_secret_key == "sk"
    assert tel.load().mode == "full"  # persisted


def test_run_full_without_keys_degrades(monkeypatch, tmp_path):
    monkeypatch.setattr(priv, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["3"]))}))
    monkeypatch.setattr(priv, "prompt_sentry_dsn", lambda console, current="": "https://k@o/1")
    monkeypatch.setattr(priv.render, "confirm", _as_confirm(_Scripted([True])))
    monkeypatch.setattr(priv, "prompt_langfuse", lambda console, consent: ("", "", "https://cloud.langfuse.com"))
    consent = priv.run(_console(), tmp_path)
    assert consent.mode == "errors"  # DSN present, so bug reports still work


# ------------------------------------------------------------- project gate


def test_ask_project_records_the_answer(monkeypatch, tmp_path):
    tel.save(tel.Consent(mode="errors", decided=True, sentry_dsn="x"))
    monkeypatch.setattr(priv.render, "confirm", _as_confirm(_Scripted([True])))
    assert priv.ask_project(_console(), tmp_path) is True
    assert tel.project_share(tmp_path) is True


def test_maybe_ask_project_is_silent_when_nothing_to_ask(monkeypatch, tmp_path):
    def _boom(*a, **k):
        raise AssertionError("asked a question with no consequence")

    monkeypatch.setattr(priv, "ask_project", _boom)
    # mode none: nothing shared anywhere, so no per-project question.
    tel.save(tel.Consent(mode="none", decided=True))
    priv.maybe_ask_project(_console(), tmp_path)
    # sharing mode but this project already answered.
    consent = tel.Consent(mode="errors", decided=True)
    consent = tel.record_project(tmp_path, False, consent)
    tel.save(consent)
    priv.maybe_ask_project(_console(), tmp_path)


def test_maybe_ask_project_asks_exactly_once(monkeypatch, tmp_path):
    tel.save(tel.Consent(mode="errors", decided=True, sentry_dsn="x"))
    monkeypatch.setattr(priv.render, "confirm", _as_confirm(_Scripted([True])))
    priv.maybe_ask_project(_console(), tmp_path)
    assert tel.project_share(tmp_path) is True
    # Second start in the same directory: no question.
    monkeypatch.setattr(priv, "ask_project", lambda *a, **k: pytest.fail("asked twice"))
    priv.maybe_ask_project(_console(), tmp_path)


# ------------------------------------------------------------------ set_mode


def test_set_mode_rejects_unknown_modes(tmp_path):
    assert priv.set_mode(_console(), "everything", tmp_path) is False
    assert tel.load().decided is False  # nothing recorded


def test_set_mode_persists_and_warns_about_missing_credentials(tmp_path, monkeypatch):
    console = _console()
    assert priv.set_mode(console, "errors", tmp_path) is True
    consent = tel.load()
    assert consent.mode == "errors" and consent.decided is True
    assert "SENTRY_DSN" in console.file.getvalue()  # warned it can't work yet


def test_set_mode_none_needs_no_credentials(tmp_path):
    console = _console()
    assert priv.set_mode(console, "none", tmp_path) is True
    assert "still needs" not in console.file.getvalue()


# ---------------------------------------------------------------- doctor_row


def test_doctor_row_none_mode(tmp_path):
    ok, label, detail = priv.doctor_row(tmp_path)
    assert ok is True and "nothing leaves" in detail


def test_doctor_row_warns_when_mode_cannot_work(tmp_path, monkeypatch):
    for key in ("SENTRY_DSN",):
        monkeypatch.delenv(key, raising=False)
    tel.save(tel.Consent(mode="errors", decided=True))
    ok, _, detail = priv.doctor_row(tmp_path)
    assert ok is None and "SENTRY_DSN" in detail


def test_doctor_row_reports_project_gate(tmp_path):
    consent = tel.Consent(mode="errors", decided=True, sentry_dsn="https://k@o/1")
    tel.save(consent)
    ok, _, detail = priv.doctor_row(tmp_path)
    assert ok is True and "not asked yet" in detail
    tel.save(tel.record_project(tmp_path, True))
    ok, _, detail = priv.doctor_row(tmp_path)
    assert ok is True and "sharing" in detail
