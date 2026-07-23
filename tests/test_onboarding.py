"""Onboarding wizard: pure logic (settings I/O, plan-building) — no terminal
interaction needed. See tests/test_providers.py for the provider catalog and
tests/test_recommendations.py for hardware detection."""

import json

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")

from loom.core import providers as prov
from loom.core import recommendations as rec
from loom.core import settings as settings_mod
from loom.ui import onboarding as ob

HW = rec.Hardware(os_name="Darwin", ram_gb=32, gpu_vendor="apple", vram_gb=32)


@pytest.fixture(autouse=True)
def _isolated_user_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod, "USER_SETTINGS_PATH", tmp_path / "user-settings.json")
    return tmp_path


def test_default_role_plan_local_only():
    plan = ob.default_role_plan(HW, "qwen3:14b", None)
    assert set(plan) == set(ob.ALL_ROLES)
    assert all(v == "ollama/qwen3:14b" for v in plan.values())


def test_default_role_plan_mixes_local_and_cloud():
    plan = ob.default_role_plan(HW, "qwen2.5-coder:32b", prov.get("anthropic"))
    for role in ob._DEFAULT_LOCAL_ROLES:
        assert plan[role] == "ollama/qwen2.5-coder:32b"
    assert plan["orchestrator"] == "anthropic:claude-sonnet-5"
    assert plan["advisor"] == "anthropic:claude-opus-4-8"  # flagship tier
    assert plan["reviewer"] == "anthropic:claude-haiku-4-5"  # light tier
    assert plan["escalation"] == "anthropic:claude-sonnet-5"  # main tier


def test_default_role_plan_tier_models_overrides_provider_defaults():
    tier_models = {"main": "claude-haiku-4-5", "light": "claude-sonnet-5"}
    plan = ob.default_role_plan(HW, "qwen2.5-coder:32b", prov.get("anthropic"), tier_models)
    assert plan["orchestrator"] == "anthropic:claude-haiku-4-5"  # main tier, overridden
    assert plan["escalation"] == "anthropic:claude-haiku-4-5"  # main tier, overridden
    assert plan["reviewer"] == "anthropic:claude-sonnet-5"  # light tier, overridden
    assert plan["advisor"] == "anthropic:claude-opus-4-8"  # flagship tier, untouched


def test_default_role_plan_tier_models_partial_falls_back_per_tier():
    """A tier missing from tier_models still gets the provider's own default
    for that tier, not a crash or an empty string."""
    plan = ob.default_role_plan(HW, "qwen2.5-coder:32b", prov.get("anthropic"), {"main": "claude-sonnet-4-6"})
    assert plan["orchestrator"] == "anthropic:claude-sonnet-4-6"  # main — overridden
    assert plan["advisor"] == "anthropic:claude-opus-4-8"  # flagship — provider default, no override given
    assert plan["reviewer"] == "anthropic:claude-haiku-4-5"  # light — provider default, no override given


def test_apply_plan_user_scope_writes_and_reloads(tmp_path):
    plan = {"orchestrator": "anthropic:claude-sonnet-4-6", "editor": "ollama/qwen3:14b"}
    settings = ob.apply_plan(tmp_path, "user", plan, {"ANTHROPIC_API_KEY": "test-key"})
    assert settings.models.orchestrator == "anthropic:claude-sonnet-4-6"
    assert settings.models.subagents["editor"] == "ollama/qwen3:14b"
    assert settings.env["ANTHROPIC_API_KEY"] == "test-key"
    assert settings_mod.USER_SETTINGS_PATH.exists()


def test_apply_plan_project_scope_writes_project_file(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    plan = {"orchestrator": "openai:gpt-5.2"}
    settings = ob.apply_plan(proj, "project", plan, {})
    project_file = settings_mod.project_settings_paths(proj)[0]
    assert project_file.exists()
    assert settings.models.orchestrator == "openai:gpt-5.2"
    # User-level file must stay untouched.
    assert not settings_mod.USER_SETTINGS_PATH.exists()


def test_apply_plan_preserves_existing_unrelated_settings(tmp_path):
    target = settings_mod.USER_SETTINGS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"ui": {"theme": "light"}, "models": {"advisor": "claude-opus-4-8"}}))

    settings = ob.apply_plan(tmp_path, "user", {"orchestrator": "anthropic:claude-sonnet-4-6"}, {})
    assert settings.ui.theme == "light"
    assert settings.models.orchestrator == "anthropic:claude-sonnet-4-6"
    assert settings.models.advisor == "claude-opus-4-8"  # untouched by this call


def test_apply_plan_merges_subagents_without_dropping_others(tmp_path):
    ob.apply_plan(tmp_path, "user", {"editor": "ollama/deepseek-coder:14b"}, {})
    settings = ob.apply_plan(tmp_path, "user", {"tester": "ollama/qwen3:14b"}, {})
    assert settings.models.subagents["editor"] == "ollama/deepseek-coder:14b"
    assert settings.models.subagents["tester"] == "ollama/qwen3:14b"


def test_apply_plan_rejects_invalid_scope(tmp_path):
    with pytest.raises(ValueError):
        ob.apply_plan(tmp_path, "nowhere", {}, {})


def test_apply_plan_rejects_invalid_model_value(tmp_path):
    # A non-string value for a str field must fail validation before the
    # file is ever written, so a bad wizard answer can't corrupt settings.json.
    with pytest.raises(Exception):
        ob.apply_plan(tmp_path, "user", {"orchestrator": ["not", "a", "string"]}, {})
    assert not settings_mod.USER_SETTINGS_PATH.exists()


def test_missing_credentials_empty_when_env_present():
    p = prov.get("anthropic")
    assert ob.missing_credentials(p, {"ANTHROPIC_API_KEY": "x"}) == []


def test_missing_credentials_flags_unset_required_vars():
    p = prov.get("anthropic")
    missing = ob.missing_credentials(p, {})
    assert [v.key for v in missing] == ["ANTHROPIC_API_KEY"]


def test_missing_credentials_ignores_optional_vars():
    p = prov.get("anthropic_bedrock")
    missing = ob.missing_credentials(p, {"AWS_BEARER_TOKEN_BEDROCK": "x", "LOOM_USE_BEDROCK": "1"})
    assert missing == []  # ANTHROPIC_BEDROCK_BASE_URL is optional


def test_missing_credentials_reads_real_environ(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
    p = prov.get("anthropic")
    assert ob.missing_credentials(p, {}) == []


def test_needs_onboarding_true_when_no_settings_anywhere(tmp_path):
    assert ob.needs_onboarding(tmp_path) is True


def test_needs_onboarding_false_after_user_settings_written(tmp_path):
    ob.apply_plan(tmp_path, "user", {"orchestrator": "anthropic:claude-sonnet-4-6"}, {})
    assert ob.needs_onboarding(tmp_path) is False


def test_needs_onboarding_false_after_project_settings_written(tmp_path):
    assert ob.needs_onboarding(tmp_path) is True
    ob.apply_plan(tmp_path, "project", {"orchestrator": "anthropic:claude-sonnet-4-6"}, {})
    assert ob.needs_onboarding(tmp_path) is False
