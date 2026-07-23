"""Layered settings loading + set_value round-trips."""

import json

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")

from loom.core import settings as st


def test_defaults_load_and_embed_models():
    s = st.load_settings()
    assert s.models.orchestrator  # from config.yaml defaults
    assert s.permissions.default_mode in {"allow", "ask", "deny"}
    assert "read_file" in s.permissions.allow
    assert s.ui.prompt_symbol


def test_project_layer_overrides_user(tmp_path, monkeypatch):
    # Point the loader at a temp project with a .loom/settings.json.
    proj = tmp_path / "proj"
    (proj / ".loom").mkdir(parents=True)
    (proj / ".loom" / "settings.json").write_text(
        json.dumps({"ui": {"theme": "light"}, "permissions": {"default_mode": "allow"}})
    )
    s = st.load_settings(root=proj)
    assert s.ui.theme == "light"
    assert s.permissions.default_mode == "allow"


def test_local_layer_beats_project(tmp_path):
    proj = tmp_path / "proj"
    (proj / ".loom").mkdir(parents=True)
    (proj / ".loom" / "settings.json").write_text(json.dumps({"ui": {"theme": "light"}}))
    (proj / ".loom" / "settings.local.json").write_text(json.dumps({"ui": {"theme": "mono"}}))
    s = st.load_settings(root=proj)
    assert s.ui.theme == "mono"


def test_invalid_theme_rejected(tmp_path):
    proj = tmp_path / "proj"
    (proj / ".loom").mkdir(parents=True)
    (proj / ".loom" / "settings.json").write_text(json.dumps({"ui": {"theme": "neon"}}))
    with pytest.raises(Exception):
        st.load_settings(root=proj)


def test_apply_env_does_not_overwrite(monkeypatch):
    s = st.load_settings()
    s.env = {"LOOM_TEST_VAR": "from_settings"}
    monkeypatch.setenv("LOOM_TEST_VAR", "from_shell")
    s.apply_env()
    import os

    assert os.environ["LOOM_TEST_VAR"] == "from_shell"  # setdefault, not overwrite


def test_set_value_env_stays_string(tmp_path, monkeypatch):
    # env.* values must stay str (they become process env vars) — numeric-
    # looking values like "1"/"0" must not be coerced to int/bool.
    monkeypatch.setattr(st, "USER_SETTINGS_PATH", tmp_path / "settings.json")
    st.set_value("env.LOOM_USE_BEDROCK", "1")
    s = st.set_value("env.ANTHROPIC_BEDROCK_BASE_URL", "https://example.com")
    assert s.env["LOOM_USE_BEDROCK"] == "1"
    assert s.env["ANTHROPIC_BEDROCK_BASE_URL"] == "https://example.com"


def test_apply_env_translates_legacy_bedrock_flag(monkeypatch):
    # Loom's Bedrock opt-in once reused Claude Code's env var name; old env
    # blocks are translated to LOOM_USE_BEDROCK and the Claude Code name is
    # never exported (it would reconfigure Claude Code in subshells).
    import os

    monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)
    monkeypatch.delenv("LOOM_USE_BEDROCK", raising=False)
    s = st.load_settings()
    s.env = {"CLAUDE_CODE_USE_BEDROCK": "1"}
    s.apply_env()
    try:
        assert os.environ.get("LOOM_USE_BEDROCK") == "1"
        assert "CLAUDE_CODE_USE_BEDROCK" not in os.environ
    finally:
        os.environ.pop("LOOM_USE_BEDROCK", None)


def test_project_settings_json_can_set_env(tmp_path):
    # Project-level .loom/settings.json can also carry env vars, overriding
    # (deep-merging on top of) the user/home layer.
    proj = tmp_path / "proj"
    (proj / ".loom").mkdir(parents=True)
    (proj / ".loom" / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BEDROCK_BASE_URL": "https://project.example.com"}})
    )
    s = st.load_settings(root=proj)
    assert s.env["ANTHROPIC_BEDROCK_BASE_URL"] == "https://project.example.com"


def test_set_model_value_overrides_prior_setup(tmp_path, monkeypatch):
    # Regression: /model must win over a models block a prior /setup wrote.
    # set_value("models.*") used to write config.yaml, but settings.json
    # deep-merges over config.yaml, so the change persisted-but-never-applied.
    us = tmp_path / "settings.json"
    monkeypatch.setattr(st, "USER_SETTINGS_PATH", us)
    # A prior /setup pinned the orchestrator in the winning layer.
    us.write_text(json.dumps({"models": {"orchestrator": "claude-sonnet-5"}}))

    st.set_value("models.orchestrator", "ollama/qwen3.6:27b")

    # Effective on a fresh load (== restart) and landed in settings.json.
    assert st.load_settings().models.orchestrator == "ollama/qwen3.6:27b"
    assert json.loads(us.read_text())["models"]["orchestrator"] == "ollama/qwen3.6:27b"


def test_set_model_value_nested_subagent_key(tmp_path, monkeypatch):
    # Nested keys (models.subagents.<role>) must merge, not clobber siblings.
    us = tmp_path / "settings.json"
    monkeypatch.setattr(st, "USER_SETTINGS_PATH", us)
    st.set_value("models.subagents.editor", "ollama/qwen3.5:9b")
    st.set_value("models.subagents.tester", "ollama/qwen3.5:4b")
    s = st.load_settings()
    assert s.models.subagents["editor"] == "ollama/qwen3.5:9b"
    assert s.models.subagents["tester"] == "ollama/qwen3.5:4b"


def test_set_model_value_invalid_leaves_file_intact(tmp_path, monkeypatch):
    # A value the schema rejects must not corrupt settings.json.
    us = tmp_path / "settings.json"
    monkeypatch.setattr(st, "USER_SETTINGS_PATH", us)
    st.set_value("models.orchestrator", "ollama/qwen3.6:27b")
    before = us.read_text()
    with pytest.raises(Exception):
        st.set_value("models.compaction_threshold", "5.0")  # must be in (0, 1]
    assert us.read_text() == before
