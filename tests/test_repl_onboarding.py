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
    s = _session(tmp_path)
    monkeypatch.setattr(onboarding, "run", lambda console, **kw: calls.append(kw) or s.settings)
    monkeypatch.setattr(onboarding, "maybe_setup_playwright", lambda console, settings: None)
    # "run setup now?" — yes. Piped stdin answers no (render.confirm declines
    # on EOF), so the offer has to be scripted like any other prompt.
    monkeypatch.setattr(repl.render, "confirm", lambda *a, **k: True)
    reloaded = []
    monkeypatch.setattr(s, "reload_settings", lambda: reloaded.append("settings"))
    monkeypatch.setattr(s, "rebuild", lambda: reloaded.append("rebuild"))

    repl._maybe_run_onboarding(s)

    assert calls and calls[0]["root"] == s.cwd
    assert reloaded == ["settings", "rebuild"]


def test_declining_the_offer_runs_nothing(tmp_path, monkeypatch):
    """Saying no to setup must not record an answer either way — the offer
    simply comes back on the next start."""
    monkeypatch.setattr(onboarding, "needs_onboarding", lambda root: True)
    wizard_calls = []
    monkeypatch.setattr(onboarding, "run", lambda console, **kw: wizard_calls.append(1))
    monkeypatch.setattr(repl.render, "confirm", lambda *a, **k: False)
    hint_calls = []
    monkeypatch.setattr(repl, "_setup_hint", lambda session: hint_calls.append(1))
    s = _session(tmp_path)

    repl._maybe_run_onboarding(s)

    assert wizard_calls == []
    assert hint_calls == [1]


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
    monkeypatch.setattr(repl.render, "confirm", lambda *a, **k: True)  # accept the offer, then cancel inside
    s = _session(tmp_path)
    reloaded = []
    monkeypatch.setattr(s, "reload_settings", lambda: reloaded.append(1))
    monkeypatch.setattr(s, "rebuild", lambda: reloaded.append(1))

    repl._maybe_run_onboarding(s)  # must not raise

    assert reloaded == []  # cancelled before any reload


# ------------------------------------------------------------ privacy gates


def test_non_first_run_with_consent_undecided_asks_privacy(tmp_path, monkeypatch):
    """An install that predates privacy modes has settings.json, so the wizard
    never fires — but it has never answered the question and must be asked."""
    from loom.ui import privacy as privacy_mod

    monkeypatch.setattr(onboarding, "needs_onboarding", lambda root: False)
    monkeypatch.setattr(repl, "_setup_hint", lambda session: None)
    calls = []
    monkeypatch.setattr(privacy_mod, "run", lambda console, root=".": calls.append(root))
    s = _session(tmp_path)

    repl._maybe_run_onboarding(s)

    assert calls == [s.cwd]


def test_non_first_run_with_consent_decided_asks_about_the_project(tmp_path, monkeypatch):
    from loom.core import telemetry as tel
    from loom.ui import privacy as privacy_mod

    monkeypatch.setattr(onboarding, "needs_onboarding", lambda root: False)
    monkeypatch.setattr(repl, "_setup_hint", lambda session: None)
    consent = tel.Consent(mode="errors", decided=True, sentry_dsn="https://x@y/1")
    tel.save(consent)
    project_calls = []
    monkeypatch.setattr(privacy_mod, "maybe_ask_project", lambda console, root=".": project_calls.append(root))
    run_calls = []
    monkeypatch.setattr(privacy_mod, "run", lambda console, root=".": run_calls.append(root))
    s = _session(tmp_path)

    repl._maybe_run_onboarding(s)

    assert run_calls == []
    assert project_calls == [s.cwd]


def test_privacy_prompt_failure_never_blocks_startup(tmp_path, monkeypatch):
    from loom.ui import privacy as privacy_mod

    monkeypatch.setattr(onboarding, "needs_onboarding", lambda root: False)
    monkeypatch.setattr(repl, "_setup_hint", lambda session: None)

    def _boom(console, root="."):
        raise RuntimeError("telemetry store on fire")

    monkeypatch.setattr(privacy_mod, "run", _boom)
    s = _session(tmp_path)

    repl._maybe_run_onboarding(s)  # must not raise


# --------------------------------------------------------------- setup hint


def _hint_output(tmp_path, monkeypatch, models: dict, env: dict | None = None) -> str:
    """Capture what the startup hint prints for a given config + settings env."""
    import json

    from loom.core import ollama as ollama_mod

    (tmp_path / ".loom").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".loom" / "settings.json").write_text(
        json.dumps({"models": models, "env": env or {}})
    )
    monkeypatch.setattr(
        ollama_mod, "status", lambda cfg: ollama_mod.OllamaStatus(False, False, [], cfg.ollama_endpoint)
    )
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "OPENCODE_GO_API_KEY", "OPENCODE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    s = _session(tmp_path)
    with s.console.capture() as cap:
        repl._setup_hint(s)
    return cap.get()


_ALL_GO = {
    "orchestrator": "go:glm-5",
    "advisor": "go:glm-5",
    "escalation_model": "go:glm-5",
    "cloud_fallback": "go:glm-5",
    "subagents": dict.fromkeys(
        ("explorer", "editor", "bash", "searcher", "general-purpose", "tester", "reviewer"), "go:glm-5"
    ),
}


def test_hint_sees_a_key_the_wizard_wrote_to_settings(tmp_path, monkeypatch):
    """The wizard stores keys in settings.json's env block, not the shell.
    Checking os.environ alone told a working setup its tasks would fail."""
    out = _hint_output(tmp_path, monkeypatch, _ALL_GO, {"OPENCODE_GO_API_KEY": "sk-test"})
    assert out.strip() == "", out


def test_hint_names_the_provider_actually_configured(tmp_path, monkeypatch):
    # No key anywhere: it must ask for the one this config needs, not for
    # ANTHROPIC_API_KEY, which no role here would ever use.
    out = _hint_output(tmp_path, monkeypatch, _ALL_GO)
    assert "OPENCODE_GO_API_KEY" in out
    assert "ANTHROPIC_API_KEY" not in out


def test_hint_accepts_the_shared_opencode_key(tmp_path, monkeypatch):
    # One OPENCODE_API_KEY covers both Zen and Go.
    out = _hint_output(tmp_path, monkeypatch, _ALL_GO, {"OPENCODE_API_KEY": "sk-shared"})
    assert out.strip() == "", out
