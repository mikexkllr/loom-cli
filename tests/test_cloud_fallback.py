"""Resolving local roles Ollama can't serve — another local model before the cloud."""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langchain_core")

from loom.core import ollama
from loom.core.config import LoomConfig
from loom.core.local_pool import LocalPool, build_pool, plan_local_roles, substitute
from loom.core.orchestrator import _require_ollama, apply_cloud_fallback

WINDOWS = {
    "ollama/qwen3:4b": 32768,
    "ollama/deepseek-coder:14b": 65536,
}


def _config(**kw):
    defaults = dict(
        orchestrator="claude-sonnet-4-6",
        subagents={
            "explorer": "ollama/qwen3:4b",
            "editor": "ollama/deepseek-coder:14b",
            "reviewer": "claude-haiku-4-5",
        },
        cloud_fallback="claude-haiku-4-5",
        context_windows=dict(WINDOWS),
    )
    defaults.update(kw)
    return LoomConfig(**defaults)


def _status(running: bool, models: list[str]):
    return ollama.OllamaStatus(installed=True, running=running, models=models, endpoint="http://x")


def _pool(running: bool, models: list[str]) -> LocalPool:
    return build_pool(_config(), _status(running, models))


# ----- cloud is the last resort -------------------------------------------


def test_daemon_down_reroutes_all_local_roles(monkeypatch):
    monkeypatch.setattr(ollama, "status", lambda cfg: _status(False, []))
    plan = apply_cloud_fallback(_config())
    assert set(plan.cloud) == {"explorer", "editor"}
    assert plan.substituted == {}  # nothing local to stand in
    assert plan.config.subagents["explorer"] == "claude-haiku-4-5"
    assert plan.config.subagents["editor"] == "claude-haiku-4-5"
    assert plan.config.subagents["reviewer"] == "claude-haiku-4-5"  # untouched (was cloud)


def test_missing_model_prefers_a_served_local_model_over_the_cloud(monkeypatch):
    """The regression this whole path exists for: one tag not pulled used to
    put a role on a billed cloud model while another local model sat idle."""
    monkeypatch.setattr(ollama, "status", lambda cfg: _status(True, ["qwen3:4b"]))
    plan = apply_cloud_fallback(_config())
    assert plan.cloud == {}  # nothing billed
    assert plan.substituted == {"editor": "ollama/deepseek-coder:14b"}
    assert plan.config.subagents["explorer"] == "ollama/qwen3:4b"  # served — kept as configured
    assert plan.config.subagents["editor"] == "ollama/qwen3:4b"  # stood up locally


def test_all_served_is_a_noop(monkeypatch):
    monkeypatch.setattr(
        ollama, "status", lambda cfg: _status(True, ["qwen3:4b", "deepseek-coder:14b"])
    )
    original = _config()
    plan = apply_cloud_fallback(original)
    assert not plan.changed
    assert plan.config is original


def test_no_local_roles_never_touches_network(monkeypatch):
    def boom(cfg):
        raise AssertionError("ollama.status should not be called")

    monkeypatch.setattr(ollama, "status", boom)
    original = _config(subagents={"reviewer": "claude-haiku-4-5"})
    plan = apply_cloud_fallback(original)
    assert not plan.changed and plan.config is original


def test_local_orchestrator_reroutes_to_cloud_when_nothing_local(monkeypatch):
    monkeypatch.setattr(ollama, "status", lambda cfg: _status(False, []))
    plan = apply_cloud_fallback(_config(orchestrator="ollama/qwen3:14b"))
    assert "orchestrator" in plan.cloud
    assert plan.config.orchestrator == "claude-haiku-4-5"


def test_local_orchestrator_prefers_a_served_local_model(monkeypatch):
    monkeypatch.setattr(ollama, "status", lambda cfg: _status(True, ["qwen3:4b"]))
    plan = apply_cloud_fallback(_config(orchestrator="ollama/qwen3:14b"))
    assert plan.cloud == {}
    assert plan.substituted["orchestrator"] == "ollama/qwen3:14b"
    assert plan.config.orchestrator == "ollama/qwen3:4b"


def test_latest_tag_counts_as_served(monkeypatch):
    monkeypatch.setattr(
        ollama, "status", lambda cfg: _status(True, ["qwen3:4b:latest", "deepseek-coder:14b"])
    )
    # "qwen3:4b" resolves to name "qwen3:4b"; ":latest"-suffixed install matches.
    assert not apply_cloud_fallback(_config()).changed


def test_local_only_mode_substitutes_but_never_reaches_for_the_cloud():
    pool = _pool(True, ["qwen3:4b"])
    plan = plan_local_roles(_config(), pool, allow_cloud=False)
    assert plan.cloud == {}
    assert plan.config.subagents["editor"] == "ollama/qwen3:4b"


def test_local_only_mode_leaves_a_role_alone_when_nothing_local_is_served():
    plan = plan_local_roles(_config(), _pool(True, []), allow_cloud=False)
    assert plan.cloud == {} and plan.substituted == {}
    assert plan.config.subagents["editor"] == "ollama/deepseek-coder:14b"


# ----- picking the stand-in ------------------------------------------------


def test_substitute_picks_smallest_model_that_is_roomy_enough():
    config = _config(
        subagents={
            "a": "ollama/small",
            "b": "ollama/mid",
            "c": "ollama/big",
            "editor": "ollama/deepseek-coder:14b",  # 65536, missing
        },
        context_windows={
            "ollama/small": 8192,
            "ollama/mid": 65536,
            "ollama/big": 131072,
            **WINDOWS,
        },
    )
    pool = build_pool(config, _status(True, ["small", "mid", "big"]))
    # mid matches the missing model's 65536 exactly — no reason to pay for big.
    assert substitute("ollama/deepseek-coder:14b", config, pool) == "ollama/mid"


def test_substitute_falls_back_to_the_largest_when_none_are_roomy_enough():
    config = _config(
        subagents={"a": "ollama/small", "b": "ollama/mid", "editor": "ollama/deepseek-coder:14b"},
        context_windows={"ollama/small": 8192, "ollama/mid": 16384, **WINDOWS},
    )
    pool = build_pool(config, _status(True, ["small", "mid"]))
    assert substitute("ollama/deepseek-coder:14b", config, pool) == "ollama/mid"


def test_substitute_returns_none_when_nothing_is_served():
    assert substitute("ollama/qwen3:4b", _config(), _pool(False, [])) is None


# ----- _require_ollama ------------------------------------------------------


def test_require_ollama_raises_clear_error(monkeypatch):
    monkeypatch.setattr(ollama, "status", lambda cfg: _status(False, []))
    with pytest.raises(RuntimeError, match="local-only mode needs local models"):
        _require_ollama(_config(), "local-only")


def test_require_ollama_ok_when_running(monkeypatch):
    monkeypatch.setattr(ollama, "status", lambda cfg: _status(True, []))
    _require_ollama(_config(), "airgap")  # no raise


def test_require_ollama_skipped_for_all_cloud_config(monkeypatch):
    def boom(cfg):
        raise AssertionError("should not be called")

    monkeypatch.setattr(ollama, "status", boom)
    _require_ollama(
        _config(orchestrator="claude-sonnet-4-6", subagents={"reviewer": "claude-haiku-4-5"}),
        "airgap",
    )


def test_a_local_advisor_is_planned_too_because_the_reviewer_inherits_it():
    config = _config(
        advisor="ollama/deepseek-coder:14b",  # not pulled
        subagents={"explorer": "ollama/qwen3:4b"},
    )
    plan = plan_local_roles(config, build_pool(config, _status(True, ["qwen3:4b"])))
    assert plan.substituted == {"advisor": "ollama/deepseek-coder:14b"}
    assert plan.config.advisor == "ollama/qwen3:4b"


def test_a_cloud_advisor_is_left_alone():
    config = _config(advisor="claude-opus-4-8")
    plan = plan_local_roles(config, build_pool(config, _status(False, [])))
    assert "advisor" not in plan.substituted and "advisor" not in plan.cloud
    assert plan.config.advisor == "claude-opus-4-8"
