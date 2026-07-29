"""Keeping work on the local daemon: context detection + the escalation ladder."""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langchain_core")

from loom.core import ollama
from loom.core.config import LoomConfig
from loom.core.local_pool import (
    build_pool,
    detect_context_windows,
    escalation_ladder,
    local_escalation,
)
from loom.middleware.prompt_size_guard import PromptSizeGuard


def _config(**kw):
    defaults = dict(
        orchestrator="claude-sonnet-4-6",
        subagents={"explorer": "ollama/small", "editor": "ollama/big"},
        escalation_model="claude-sonnet-4-6",
        escalation_threshold=0.85,
        context_windows={"ollama/small": 10_000, "ollama/big": 100_000},
    )
    defaults.update(kw)
    return LoomConfig(**defaults)


def _pool(config, models):
    return build_pool(
        config,
        ollama.OllamaStatus(installed=True, running=True, models=models, endpoint="http://x"),
    )


# ----- context-window detection --------------------------------------------


def test_detect_fills_in_windows_for_models_with_no_configured_entry(monkeypatch):
    monkeypatch.setattr(ollama, "context_length", lambda tag, endpoint: 40_000)
    config = _config(context_windows={}, max_local_context=131_072)
    detected = detect_context_windows(config, _pool(config, ["small", "big"]))
    # Without detection both would silently fall back to the 32K default and
    # escalate to the cloud on prompts they can actually hold.
    assert detected.context_windows == {"ollama/small": 40_000, "ollama/big": 40_000}


def test_detect_caps_at_max_local_context(monkeypatch):
    monkeypatch.setattr(ollama, "context_length", lambda tag, endpoint: 262_144)
    config = _config(context_windows={}, max_local_context=65_536)
    detected = detect_context_windows(config, _pool(config, ["small", "big"]))
    assert detected.context_windows["ollama/small"] == 65_536


def test_detect_caps_at_the_hardware_budget_when_unset(monkeypatch):
    """An unset max_local_context is sized from this machine's GPU memory, not
    a fixed number — a 256K-context model on a small box gets a cache it can
    actually allocate."""
    from loom.core import recommendations

    monkeypatch.setattr(ollama, "context_length", lambda tag, endpoint: 262_144)
    monkeypatch.setattr(recommendations, "auto_context_budget", lambda: 32_768)
    config = _config(context_windows={}, max_local_context=None)
    detected = detect_context_windows(config, _pool(config, ["small", "big"]))
    assert detected.context_windows["ollama/small"] == 32_768


def test_configured_windows_win_over_detection(monkeypatch):
    def boom(tag, endpoint):
        raise AssertionError("configured models should not be probed")

    monkeypatch.setattr(ollama, "context_length", boom)
    config = _config()
    assert detect_context_windows(config, _pool(config, ["small", "big"])) is config


def test_unserved_and_undetectable_models_are_left_alone(monkeypatch):
    monkeypatch.setattr(ollama, "context_length", lambda tag, endpoint: None)
    config = _config(context_windows={})
    assert detect_context_windows(config, _pool(config, ["small"])) is config
    # Nothing served at all → no probing, no change.
    assert detect_context_windows(config, _pool(config, [])) is config


# ----- the ladder -----------------------------------------------------------


def test_ladder_lists_served_local_models_smallest_window_first():
    config = _config()
    assert escalation_ladder(config, _pool(config, ["big", "small"])) == (
        ("ollama/small", 10_000),
        ("ollama/big", 100_000),
    )


def test_ladder_excludes_models_the_daemon_does_not_serve():
    config = _config()
    assert escalation_ladder(config, _pool(config, ["small"])) == (("ollama/small", 10_000),)


def test_local_escalation_picks_the_smallest_model_that_still_fits():
    config = _config(
        context_windows={"ollama/small": 10_000, "ollama/mid": 50_000, "ollama/big": 100_000},
        subagents={"a": "ollama/small", "b": "ollama/mid", "c": "ollama/big"},
    )
    ladder = escalation_ladder(config, _pool(config, ["small", "mid", "big"]))
    # 20k overflows small (10k) but sits well inside mid's 85% of 50k.
    assert local_escalation(20_000, "ollama/small", config, ladder) == "ollama/mid"


def test_local_escalation_returns_none_when_no_local_model_has_headroom():
    config = _config()
    ladder = escalation_ladder(config, _pool(config, ["small", "big"]))
    assert local_escalation(200_000, "ollama/small", config, ladder) is None


def test_local_escalation_never_picks_the_model_it_is_escalating_from():
    config = _config()
    ladder = escalation_ladder(config, _pool(config, ["small", "big"]))
    # 9k fits small's own 85% bar, but small is the one overflowing — skip it.
    assert local_escalation(9_000, "ollama/small", config, ladder) == "ollama/big"


# ----- the guard climbs the ladder before spending -------------------------


class _Request:
    """Minimal stand-in for LangChain's ModelRequest."""

    def __init__(self, text: str) -> None:
        self.messages = [type("M", (), {"content": text})()]
        self.system_prompt = ""
        self.model = None

    def override(self, model):
        self.model = model
        return self


def _guard(config, ladder, monkeypatch, built):
    guard = PromptSizeGuard("ollama/small", config, ladder)
    monkeypatch.setattr(
        "loom.middleware.prompt_size_guard.build_model",
        lambda model, cfg: built.append(model) or f"<{model}>",
    )
    return guard


def test_guard_escalates_to_a_local_model_and_stays_off_the_cloud(monkeypatch):
    config = _config()
    ladder = escalation_ladder(config, _pool(config, ["small", "big"]))
    built: list[str] = []
    guard = _guard(config, ladder, monkeypatch, built)

    # ~20k tokens: past small's 8.5k bar, comfortably inside big.
    request = guard._maybe_escalate(_Request("x" * 80_000))

    assert built == ["ollama/big"]
    assert request.model == "<ollama/big>"
    assert guard.local_escalation_count == 1
    assert guard.escalation_count == 0  # nothing billed


def test_guard_falls_through_to_the_cloud_when_no_local_model_fits(monkeypatch):
    config = _config()
    ladder = escalation_ladder(config, _pool(config, ["small", "big"]))
    built: list[str] = []
    guard = _guard(config, ladder, monkeypatch, built)

    request = guard._maybe_escalate(_Request("x" * 2_000_000))  # ~500k tokens

    assert built == ["claude-sonnet-4-6"]
    assert request.model == "<claude-sonnet-4-6>"
    assert guard.escalation_count == 1
    assert guard.local_escalation_count == 0


def test_guard_with_no_ladder_behaves_as_before(monkeypatch):
    config = _config()
    built: list[str] = []
    guard = _guard(config, (), monkeypatch, built)

    guard._maybe_escalate(_Request("x" * 80_000))

    assert built == ["claude-sonnet-4-6"]
    assert guard.escalation_count == 1


def test_guard_leaves_prompts_that_fit_untouched(monkeypatch):
    config = _config()
    ladder = escalation_ladder(config, _pool(config, ["small", "big"]))
    built: list[str] = []
    guard = _guard(config, ladder, monkeypatch, built)

    request = _Request("x" * 400)  # ~100 tokens
    assert guard._maybe_escalate(request) is request
    assert built == []
    assert guard.escalation_count == 0 and guard.local_escalation_count == 0


def test_guard_falls_back_to_cloud_when_the_local_model_cannot_be_built(monkeypatch):
    config = _config()
    ladder = escalation_ladder(config, _pool(config, ["small", "big"]))
    built: list[str] = []

    def _build(model, cfg):
        built.append(model)
        if config.is_local(model):
            raise RuntimeError("ollama went away mid-run")
        return f"<{model}>"

    guard = PromptSizeGuard("ollama/small", config, ladder)
    monkeypatch.setattr("loom.middleware.prompt_size_guard.build_model", _build)

    request = guard._maybe_escalate(_Request("x" * 80_000))

    assert built == ["ollama/big", "claude-sonnet-4-6"]
    assert request.model == "<claude-sonnet-4-6>"
    assert guard.local_escalation_count == 0 and guard.escalation_count == 1


# ----- end to end through build_orchestrator -------------------------------


def _settings(**kw):
    from loom.core.settings import Settings

    defaults = dict(
        orchestrator="claude-sonnet-4-6",
        subagents={
            "explorer": "ollama/qwen3:4b",
            "editor": "ollama/qwen3:27b",
            "bash": "ollama/qwen3:4b",
            "searcher": "ollama/qwen3:4b",
            "reviewer": "claude-haiku-4-5",
            "general-purpose": "ollama/qwen3:4b",
            "tester": "ollama/qwen3:4b",
        },
        cloud_fallback="claude-haiku-4-5",
        context_windows={"ollama/qwen3:4b": 32_768, "ollama/qwen3:27b": 65_536},
    )
    defaults.update(kw)
    return Settings(models=LoomConfig(**defaults))


def _stub_daemon(monkeypatch, models):
    monkeypatch.setattr(
        ollama,
        "status",
        lambda cfg: ollama.OllamaStatus(True, True, list(models), "http://x"),
    )


def test_build_keeps_an_unpulled_role_local_instead_of_billing_it(monkeypatch):
    pytest.importorskip("deepagents")
    from loom.core.orchestrator import build_orchestrator

    _stub_daemon(monkeypatch, ["qwen3:4b"])  # the 27b editor model isn't pulled
    bundle = build_orchestrator(_settings())

    assert bundle.fallbacks == {}  # nothing was sent to the cloud
    assert bundle.substitutions == {"editor": "ollama/qwen3:27b"}
    assert bundle.active_config.subagents["editor"] == "ollama/qwen3:4b"


def test_build_falls_back_to_cloud_only_when_the_daemon_is_down(monkeypatch):
    pytest.importorskip("deepagents")
    from loom.core.orchestrator import build_orchestrator

    monkeypatch.setattr(
        ollama, "status", lambda cfg: ollama.OllamaStatus(True, False, [], "http://x")
    )
    bundle = build_orchestrator(_settings())

    assert bundle.substitutions == {}
    assert "editor" in bundle.fallbacks and "explorer" in bundle.fallbacks
    assert bundle.active_config.subagents["editor"] == "claude-haiku-4-5"


def test_build_hands_every_local_subagent_the_escalation_ladder(monkeypatch):
    pytest.importorskip("deepagents")
    from loom.core.orchestrator import build_orchestrator

    _stub_daemon(monkeypatch, ["qwen3:4b", "qwen3:27b"])
    bundle = build_orchestrator(_settings())

    assert bundle.guards, "local subagents should carry prompt-size guards"
    for guard in bundle.guards:
        assert guard.ladder == (("ollama/qwen3:4b", 32_768), ("ollama/qwen3:27b", 65_536))
