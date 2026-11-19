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


# ------------------------------------------------- set_value writes one key


def test_set_value_writes_only_the_key_it_was_given(tmp_path, monkeypatch):
    """`/theme light` inside a project used to dump the *merged* settings into
    the user file — the project's permissions, hooks and env included, plus
    whatever its git-ignored settings.local.json held — where they then applied
    to every other project on the machine."""
    us = tmp_path / "home" / "settings.json"
    monkeypatch.setattr(st, "USER_SETTINGS_PATH", us)
    proj = tmp_path / "proj"
    (proj / ".loom").mkdir(parents=True)
    (proj / ".loom" / "settings.json").write_text(
        json.dumps(
            {
                "permissions": {"default_mode": "allow", "allow": ["execute"]},
                "hooks": {"pre_tool_use": [{"matcher": "*", "command": "echo project-hook"}]},
            }
        )
    )
    (proj / ".loom" / "settings.local.json").write_text(json.dumps({"env": {"PROJECT_SECRET": "s3cret"}}))

    merged = st.set_value("ui.theme", "loom-light", proj)

    # The project still sees its own layers …
    assert merged.ui.theme == "loom-light"
    assert merged.permissions.default_mode == "allow"
    # … and the user file holds the one key, nothing lifted from the project
    # and nothing frozen out of the packaged defaults.
    assert json.loads(us.read_text()) == {"ui": {"theme": "loom-light"}}
    elsewhere = st.load_settings(tmp_path / "another-project")
    assert elsewhere.permissions.default_mode == "ask"
    assert elsewhere.hooks.pre_tool_use == []
    assert "PROJECT_SECRET" not in elsewhere.env


def test_set_value_keeps_what_the_user_file_already_had(tmp_path, monkeypatch):
    us = tmp_path / "settings.json"
    monkeypatch.setattr(st, "USER_SETTINGS_PATH", us)
    us.write_text(json.dumps({"env": {"MY_KEY": "abc"}, "ui": {"compact": True}}))
    st.set_value("ui.theme", "mono")
    assert json.loads(us.read_text()) == {"env": {"MY_KEY": "abc"}, "ui": {"compact": True, "theme": "mono"}}


@pytest.mark.parametrize("key", ["ui.them", "bogus.key", "models.orchestator", "permissions.mode"])
def test_set_value_rejects_a_key_that_names_no_setting(tmp_path, monkeypatch, key):
    """Unknown keys are ignored on load, so writing one reported success and
    changed nothing."""
    us = tmp_path / "settings.json"
    monkeypatch.setattr(st, "USER_SETTINGS_PATH", us)
    with pytest.raises(ValueError, match="unknown key"):
        st.set_value(key, "x")
    assert not us.exists()


def test_set_value_accepts_any_key_under_a_free_form_map(tmp_path, monkeypatch):
    us = tmp_path / "settings.json"
    monkeypatch.setattr(st, "USER_SETTINGS_PATH", us)
    st.set_value("env.ANYTHING_AT_ALL", "1")
    st.set_value("mcp_servers.graphify.enabled", "true")
    s = st.set_value("models.subagents.editor", "ollama/qwen3.5:9b")
    assert s.env["ANYTHING_AT_ALL"] == "1"
    assert s.mcp_servers["graphify"].enabled is True
    assert s.mcp_servers["graphify"].command == "graphify"  # merged with the packaged entry
    assert s.models.subagents["editor"] == "ollama/qwen3.5:9b"


def test_set_value_invalid_value_leaves_the_file_alone(tmp_path, monkeypatch):
    us = tmp_path / "settings.json"
    monkeypatch.setattr(st, "USER_SETTINGS_PATH", us)
    st.set_value("ui.theme", "mono")
    before = us.read_text()
    with pytest.raises(Exception):
        st.set_value("ui.theme", "neon")
    with pytest.raises(Exception):
        st.set_value("permissions.default_mode", "sometimes")
    assert us.read_text() == before


# ------------------------------------------------------ apply_env + rotation


def test_apply_env_replaces_a_value_loom_injected_itself(monkeypatch):
    """A key rotated in /setup was written to settings.json and then ignored
    until restart: the old value was already in os.environ, and `setdefault`
    would not touch it — so the wizard verified with the key just replaced."""
    import os

    from loom.core import model_router

    monkeypatch.delenv("LOOM_TEST_ROTATED", raising=False)
    monkeypatch.setattr(st, "_INJECTED_ENV", {})
    cleared = []
    monkeypatch.setattr(model_router, "clear_model_cache", lambda: cleared.append(True))

    s = st.load_settings()
    s.env = {"LOOM_TEST_ROTATED": "old-key"}
    s.apply_env()
    assert os.environ["LOOM_TEST_ROTATED"] == "old-key"

    s.env = {"LOOM_TEST_ROTATED": "new-key"}
    s.apply_env()
    try:
        assert os.environ["LOOM_TEST_ROTATED"] == "new-key"
        # Models built with the old key must not be handed out again.
        assert len(cleared) == 2
        s.apply_env()  # nothing changed: no rebuild
        assert len(cleared) == 2
    finally:
        os.environ.pop("LOOM_TEST_ROTATED", None)


def test_apply_env_still_never_overrides_the_shell_after_a_rotation(monkeypatch):
    import os

    monkeypatch.setattr(st, "_INJECTED_ENV", {})
    monkeypatch.delenv("LOOM_TEST_SHELL", raising=False)
    s = st.load_settings()
    s.env = {"LOOM_TEST_SHELL": "from-settings"}
    s.apply_env()
    # The user exports their own value afterwards: that is theirs now.
    monkeypatch.setenv("LOOM_TEST_SHELL", "from-shell")
    s.env = {"LOOM_TEST_SHELL": "from-settings-v2"}
    s.apply_env()
    assert os.environ["LOOM_TEST_SHELL"] == "from-shell"
