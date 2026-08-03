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
    assert plan["escalation"] == "anthropic:claude-sonnet-5"  # main tier


def test_quick_setup_sends_only_the_three_whole_task_roles_to_the_cloud():
    """Loom's argument is that everything touching raw file content, shell
    output or test logs stays local — only the roles that reason across the
    whole task are worth paying for. `reviewer` reads diffs, so it is local."""
    assert set(ob._DEFAULT_CLOUD_ROLES) == {"orchestrator", "advisor", "escalation"}
    plan = ob.default_role_plan(HW, "qwen2.5-coder:32b", prov.get("anthropic"))
    for role in ("reviewer", "explorer", "editor", "bash", "searcher", "general-purpose", "tester"):
        assert plan[role] == "ollama/qwen2.5-coder:32b", f"{role} should be local in quick setup"
    assert set(ob._DEFAULT_LOCAL_ROLES) | set(ob._DEFAULT_CLOUD_ROLES) == set(ob.ALL_ROLES)


def test_default_role_plan_tier_models_overrides_provider_defaults():
    tier_models = {"main": "claude-haiku-4-5", "light": "claude-sonnet-5"}
    plan = ob.default_role_plan(HW, "qwen2.5-coder:32b", prov.get("anthropic"), tier_models)
    assert plan["orchestrator"] == "anthropic:claude-haiku-4-5"  # main tier, overridden
    assert plan["escalation"] == "anthropic:claude-haiku-4-5"  # main tier, overridden
    assert plan["advisor"] == "anthropic:claude-opus-4-8"  # flagship tier, untouched


def test_default_role_plan_tier_models_partial_falls_back_per_tier():
    """A tier missing from tier_models still gets the provider's own default
    for that tier, not a crash or an empty string."""
    plan = ob.default_role_plan(HW, "qwen2.5-coder:32b", prov.get("anthropic"), {"main": "claude-sonnet-4-6"})
    assert plan["orchestrator"] == "anthropic:claude-sonnet-4-6"  # main — overridden
    assert plan["advisor"] == "anthropic:claude-opus-4-8"  # flagship — provider default, no override given


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


# ---------------------------------------------------------------------------
# a save that another layer overrides
# ---------------------------------------------------------------------------


def _pin_project(root, models):
    import json

    target = settings_mod.project_settings_paths(root)[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"models": models}))


def test_a_user_save_shadowed_by_the_project_is_detected(tmp_path):
    """The report: quick setup saved go:glm-5 to the user layer, said "saved —
    reload complete", and the next turn ran on the project layer's Anthropic
    models. Writing has to be checked against what actually resolves."""
    _pin_project(tmp_path, {"orchestrator": "anthropic:claude-sonnet-5"})
    plan = {"orchestrator": "go:glm-5", "editor": "ollama/qwen3.5:2b"}
    settings = ob.apply_plan(tmp_path, "user", plan, {})

    assert settings.models.orchestrator == "anthropic:claude-sonnet-5"  # the write did nothing
    shadowed = ob.shadowed_roles(plan, settings)
    assert "orchestrator" in shadowed
    assert shadowed["orchestrator"] == ("go:glm-5", "anthropic:claude-sonnet-5")
    # A role the project doesn't pin still lands normally.
    assert "editor" not in shadowed


def test_saving_to_the_winning_layer_clears_the_shadow(tmp_path):
    _pin_project(tmp_path, {"orchestrator": "anthropic:claude-sonnet-5"})
    plan = {"orchestrator": "go:glm-5"}
    settings = ob.apply_plan(tmp_path, "project", plan, {})
    assert settings.models.orchestrator == "go:glm-5"
    assert ob.shadowed_roles(plan, settings) == {}


def test_nothing_is_reported_when_the_save_takes_effect(tmp_path):
    plan = {"orchestrator": "go:glm-5", "editor": "ollama/qwen3.5:2b"}
    settings = ob.apply_plan(tmp_path, "user", plan, {})
    assert ob.shadowed_roles(plan, settings) == {}


def test_effective_model_covers_every_role_shape(tmp_path):
    """orchestrator/advisor are attributes, escalation is renamed, subagents
    live in a dict — a role read the wrong way would look permanently shadowed."""
    plan = {role: "ollama/qwen3.5:2b" for role in ob.ALL_ROLES}
    settings = ob.apply_plan(tmp_path, "user", plan, {})
    for role in ob.ALL_ROLES:
        assert ob.effective_model(role, settings) == "ollama/qwen3.5:2b", role
    assert ob.shadowed_roles(plan, settings) == {}


def test_quick_setup_points_the_fallback_at_the_chosen_provider():
    """When Ollama is down every local role runs on cloud_fallback. Leaving it
    on whatever a previous config named meant one dead daemon produced
    "OPENCODE_ZEN_API_KEY is not set" for someone who only ever set up Go."""
    plan = ob.default_role_plan(HW, "qwen3.5:2b", prov.get("anthropic"))
    assert plan[ob.FALLBACK_KEY].startswith("anthropic:")
    # And it is the cheap tier — a fallback is a stopgap, not the main model.
    assert plan[ob.FALLBACK_KEY] == "anthropic:claude-haiku-4-5"


def test_local_only_setup_leaves_the_fallback_alone():
    """No cloud provider was chosen, so there is nothing to point it at."""
    plan = ob.default_role_plan(HW, "qwen3.5:2b", None)
    assert ob.FALLBACK_KEY not in plan


def test_the_fallback_is_written_as_a_top_level_key(tmp_path):
    plan = {"orchestrator": "go:glm-5", ob.FALLBACK_KEY: "go:glm-5-air"}
    settings = ob.apply_plan(tmp_path, "user", plan, {})
    assert settings.models.cloud_fallback == "go:glm-5-air"
    # ...and not mistaken for a subagent named "cloud_fallback".
    assert ob.FALLBACK_KEY not in settings.models.subagents
    assert ob.shadowed_roles(plan, settings) == {}
