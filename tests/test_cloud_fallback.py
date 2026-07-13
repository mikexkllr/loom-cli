"""Cloud fallback when Ollama can't serve the configured local models."""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langchain_core")

from loom.core import ollama
from loom.core.config import LoomConfig
from loom.core.orchestrator import _require_ollama, apply_cloud_fallback


def _config(**kw):
    defaults = dict(
        orchestrator="claude-sonnet-4-6",
        subagents={
            "explorer": "ollama/qwen3:4b",
            "editor": "ollama/deepseek-coder:14b",
            "reviewer": "claude-haiku-4-5",
        },
        cloud_fallback="claude-haiku-4-5",
    )
    defaults.update(kw)
    return LoomConfig(**defaults)


def _status(running: bool, models: list[str]):
    return ollama.OllamaStatus(installed=True, running=running, models=models, endpoint="http://x")


def test_daemon_down_reroutes_all_local_roles(monkeypatch):
    monkeypatch.setattr(ollama, "status", lambda cfg: _status(False, []))
    config, fallbacks = apply_cloud_fallback(_config())
    assert set(fallbacks) == {"explorer", "editor"}
    assert config.subagents["explorer"] == "claude-haiku-4-5"
    assert config.subagents["editor"] == "claude-haiku-4-5"
    assert config.subagents["reviewer"] == "claude-haiku-4-5"  # untouched (was cloud)


def test_partial_missing_only_reroutes_missing(monkeypatch):
    monkeypatch.setattr(ollama, "status", lambda cfg: _status(True, ["qwen3:4b"]))
    config, fallbacks = apply_cloud_fallback(_config())
    assert set(fallbacks) == {"editor"}
    assert config.subagents["explorer"] == "ollama/qwen3:4b"  # served — kept local
    assert config.subagents["editor"] == "claude-haiku-4-5"


def test_all_served_is_a_noop(monkeypatch):
    monkeypatch.setattr(ollama, "status", lambda cfg: _status(True, ["qwen3:4b", "deepseek-coder:14b"]))
    original = _config()
    config, fallbacks = apply_cloud_fallback(original)
    assert fallbacks == {}
    assert config is original


def test_no_local_roles_never_touches_network(monkeypatch):
    def boom(cfg):
        raise AssertionError("ollama.status should not be called")

    monkeypatch.setattr(ollama, "status", boom)
    original = _config(subagents={"reviewer": "claude-haiku-4-5"})
    config, fallbacks = apply_cloud_fallback(original)
    assert fallbacks == {} and config is original


def test_local_orchestrator_reroutes_too(monkeypatch):
    monkeypatch.setattr(ollama, "status", lambda cfg: _status(False, []))
    config, fallbacks = apply_cloud_fallback(_config(orchestrator="ollama/qwen3:14b"))
    assert "orchestrator" in fallbacks
    assert config.orchestrator == "claude-haiku-4-5"


def test_latest_tag_counts_as_served(monkeypatch):
    monkeypatch.setattr(ollama, "status", lambda cfg: _status(True, ["qwen3:4b:latest", "deepseek-coder:14b"]))
    # "qwen3:4b" resolves to name "qwen3:4b"; ":latest"-suffixed install matches.
    config, fallbacks = apply_cloud_fallback(_config())
    assert fallbacks == {}


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
    _require_ollama(_config(orchestrator="claude-sonnet-4-6", subagents={"reviewer": "claude-haiku-4-5"}), "airgap")
