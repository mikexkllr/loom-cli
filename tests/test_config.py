"""Config loading, validation, and `config set` round-trips."""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")

from loom.core import config as cfg


def test_default_config_loads_and_validates():
    c = cfg.load_config(path=cfg.DEFAULT_CONFIG_PATH)
    assert c.orchestrator
    assert "explorer" in c.subagents
    assert c.advisor
    assert 0 < c.compaction_threshold <= 1


def test_is_local_detection():
    c = cfg.load_config(path=cfg.DEFAULT_CONFIG_PATH)
    assert c.is_local("ollama/qwen3:4b")
    assert c.is_local("ollama:llama3.2:3b")
    assert not c.is_local("claude-sonnet-4-6")
    assert not c.is_local("gpt-4o")


def test_invalid_advisor_threshold_rejected():
    with pytest.raises(Exception):
        cfg.LoomConfig(advisor_threshold="sometimes")


def test_invalid_fraction_rejected():
    with pytest.raises(Exception):
        cfg.LoomConfig(compaction_threshold=1.5)


def test_set_value_nested_and_persisted(tmp_path):
    target = tmp_path / "config.yaml"
    cfg.save_config(cfg.load_config(path=cfg.DEFAULT_CONFIG_PATH), path=target)

    cfg.set_value("orchestrator", "gpt-4o", path=target)
    cfg.set_value("subagents.editor", "ollama/qwen3:14b", path=target)

    reloaded = cfg.load_config(path=target)
    assert reloaded.orchestrator == "gpt-4o"
    assert reloaded.subagents["editor"] == "ollama/qwen3:14b"


def test_set_value_coerces_scalars(tmp_path):
    target = tmp_path / "config.yaml"
    cfg.save_config(cfg.load_config(path=cfg.DEFAULT_CONFIG_PATH), path=target)
    cfg.set_value("worktree_isolation", "false", path=target)
    cfg.set_value("max_nesting_depth", "3", path=target)
    reloaded = cfg.load_config(path=target)
    assert reloaded.worktree_isolation is False
    assert reloaded.max_nesting_depth == 3
