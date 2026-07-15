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
    st.set_value("env.CLAUDE_CODE_USE_BEDROCK", "1")
    s = st.set_value("env.ANTHROPIC_BEDROCK_BASE_URL", "https://example.com")
    assert s.env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert s.env["ANTHROPIC_BEDROCK_BASE_URL"] == "https://example.com"


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
