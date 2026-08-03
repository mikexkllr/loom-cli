"""Model preflight: does the configured model actually answer?

The bug that motivated this: a valid OpenCode Go key with an orchestrator set
to a model the account may not call in this region. Loom reported it as a
credentials problem, so the obvious fix — re-paste the key — could never work.
"is the key present", "is the key accepted", and "is this model callable for
you" are three different questions, and these tests keep them apart.
"""

from __future__ import annotations

import concurrent.futures

import pytest

from loom.core import config as cfg
from loom.core import preflight


class _Boom(Exception):
    """Stands in for a provider SDK error, which is what actually reaches us."""

    def __init__(self, message: str, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _config() -> cfg.LoomConfig:
    return cfg.LoomConfig()


def _fake_cloud(monkeypatch, *, raises=None, reply="ok"):
    class _Model:
        def invoke(self, _prompt):
            if raises is not None:
                raise raises
            return type("R", (), {"content": reply})()

    monkeypatch.setattr(preflight, "_status_code", preflight._status_code)
    monkeypatch.setattr(
        "loom.core.model_router.build_model", lambda *a, **k: _Model(), raising=False
    )


# ------------------------------------------------------------------- cloud


def test_401_is_a_credentials_problem(monkeypatch):
    _fake_cloud(monkeypatch, raises=_Boom("Unauthorized", status_code=401))
    check = preflight.check_model("anthropic:claude-sonnet-5", _config())
    assert not check.ok
    assert check.state == "auth"
    assert "rejected" in check.hint


def test_403_is_not_a_credentials_problem(monkeypatch):
    """The regression this whole module exists for. A 403 means the key was
    accepted and *this model* was refused — telling someone to re-enter their
    key sends them somewhere the fix cannot be."""
    body = {
        "error": {
            "type": "RegionError",
            "message": "The latest version of this model is only available hosted in China "
            "and requires explicit opt in",
        }
    }
    _fake_cloud(monkeypatch, raises=_Boom("Error code: 403", status_code=403, body=body))
    check = preflight.check_model("go:deepseek-v4-flash", _config())
    assert not check.ok
    assert check.state == "forbidden"
    # The provider's own words survive — they carry the actual reason.
    assert "only available hosted in China" in check.detail
    # And the hint must not send them back to the key.
    assert "key works" in check.hint
    assert "re-run" not in check.hint


def test_404_reads_as_a_wrong_model_id(monkeypatch):
    _fake_cloud(monkeypatch, raises=_Boom("model not found", status_code=404))
    check = preflight.check_model("go:typo-v9", _config())
    assert check.state == "missing"
    assert "model id" in check.hint


def test_rate_limited_counts_as_working(monkeypatch):
    """429 answers the question we actually asked: the key and model are valid.
    Failing setup over a transient limit would be wrong."""
    _fake_cloud(monkeypatch, raises=_Boom("slow down", status_code=429))
    check = preflight.check_model("anthropic:claude-sonnet-5", _config())
    assert check.ok
    assert "valid" in check.detail


def test_a_successful_probe_reports_what_came_back(monkeypatch):
    _fake_cloud(monkeypatch, reply="ok")
    check = preflight.check_model("anthropic:claude-sonnet-5", _config())
    assert check.ok and check.state == "ok" and "ok" in check.detail


def test_a_hung_provider_does_not_hang_setup(monkeypatch):
    import time

    class _Slow:
        def invoke(self, _prompt):
            time.sleep(30)

    monkeypatch.setattr("loom.core.model_router.build_model", lambda *a, **k: _Slow(), raising=False)
    check = preflight.check_model("anthropic:claude-sonnet-5", _config(), timeout=0.2)
    assert not check.ok and check.state == "timeout"


def test_a_missing_env_var_is_reported_without_a_network_call(monkeypatch):
    def _explode(*a, **k):
        raise RuntimeError("OPENCODE_GO_API_KEY is not set — required for the 'opencode_go' provider")

    monkeypatch.setattr("loom.core.model_router.build_model", _explode, raising=False)
    check = preflight.check_model("go:glm-5.2", _config())
    assert check.state == "auth" and "not set" in check.detail


# ------------------------------------------------------------------- local


def test_local_model_reports_the_daemon_being_down(monkeypatch):
    monkeypatch.setattr(
        preflight.ollama_mod,
        "status",
        lambda c: type("S", (), {"running": False, "installed": True, "models": [], "endpoint": "http://x"})(),
    )
    check = preflight.check_model("ollama/qwen3.5:9b", _config())
    assert not check.ok and check.state == "offline"
    assert "not reachable" in check.detail


def test_local_model_reports_a_tag_that_is_not_pulled(monkeypatch):
    monkeypatch.setattr(
        preflight.ollama_mod,
        "status",
        lambda c: type("S", (), {"running": True, "installed": True, "models": ["other:1b"], "endpoint": "e"})(),
    )
    check = preflight.check_model("ollama/qwen3.5:9b", _config())
    assert check.state == "not-pulled"
    assert "loom models pull qwen3.5:9b" in check.hint


def test_local_model_that_is_present_passes_without_inference(monkeypatch):
    """Waking a cold local model can take a minute and proves nothing the tag
    list hasn't already answered."""
    monkeypatch.setattr(
        preflight.ollama_mod,
        "status",
        lambda c: type("S", (), {"running": True, "installed": True, "models": ["qwen3.5:9b"], "endpoint": "e"})(),
    )
    monkeypatch.setattr(
        "loom.core.model_router.build_model",
        lambda *a, **k: pytest.fail("a served local model must not be probed by inference"),
        raising=False,
    )
    assert preflight.check_model("ollama/qwen3.5:9b", _config()).ok


# -------------------------------------------------------------------- plan


def test_roles_sharing_a_model_are_probed_once(monkeypatch):
    """Probes are real billed calls — seven subagents on one tag must not mean
    seven round-trips."""
    calls: list[str] = []

    class _Model:
        def __init__(self, name):
            self.name = name

        def invoke(self, _prompt):
            calls.append(self.name)
            return type("R", (), {"content": "ok"})()

    monkeypatch.setattr(
        "loom.core.model_router.build_model", lambda m, c: _Model(m), raising=False
    )
    plan = {
        "orchestrator": "anthropic:claude-sonnet-5",
        "escalation": "anthropic:claude-sonnet-5",
        "reviewer": "anthropic:claude-sonnet-5",
        "advisor": "anthropic:claude-opus-4-8",
    }
    results = list(preflight.check_plan(plan, _config()))
    assert len(calls) == 2, calls
    roles = {c.model: set(r) for r, c in results}
    assert roles["anthropic:claude-sonnet-5"] == {"orchestrator", "escalation", "reviewer"}


def test_provider_message_is_extracted_from_a_string_only_error(monkeypatch):
    """Some wrappers flatten the body into the exception text; the provider's
    sentence is still the useful part."""
    exc = _Boom("Error code: 403 - {'type': 'error', 'error': {'message': 'not opted in'}}")
    assert preflight._provider_message(exc) == "not opted in"
    assert preflight._status_code(exc) == 403


# ------------------------------------------------------------------ wizard


def test_the_wizard_can_skip_probing(monkeypatch, tmp_path):
    """`verify=False` must make no model calls at all — it is what keeps the
    test suite and unattended runs offline."""
    from loom.ui import onboarding as ob

    monkeypatch.setattr(
        "loom.core.model_router.build_model",
        lambda *a, **k: pytest.fail("verify=False still called a model"),
        raising=False,
    )
    monkeypatch.setattr(ob, "_verify_and_repair", lambda *a, **k: pytest.fail("probed anyway"))
    answers = iter(["quick", "1", "user"])
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(lambda *a, **k: next(answers))}))
    monkeypatch.setattr(ob.render, "confirm", lambda *a, **k: False)
    monkeypatch.setattr(ob.settings_mod, "USER_SETTINGS_PATH", tmp_path / "settings.json")
    from loom.ui.theme import make_console
    from loom.core.settings import UISettings

    ob.run(make_console(UISettings()), root=str(tmp_path), verify=False, privacy=False)


def test_timeout_error_is_caught_not_propagated():
    """concurrent.futures.TimeoutError is not a subclass of the SDK errors we
    classify, so it needs its own arm — this pins that it has one."""
    assert issubclass(concurrent.futures.TimeoutError, Exception)
    # Reasoning models think for 10-20s before their first token on a probe as
    # small as "say ok", so the bound has to clear that with room to spare or
    # healthy models get reported dead.
    assert preflight.TIMEOUT_SECONDS >= 45
