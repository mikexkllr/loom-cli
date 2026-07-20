"""Provider catalog: every entry must resolve to a provider model_router knows."""

import pytest

pytest.importorskip("pydantic")

from loom.core import model_router as mr
from loom.core import providers as prov


def test_every_provider_resolves_to_a_known_model_router_provider():
    for p in prov.PROVIDERS:
        model_str = p.model_string("some-model-id")
        rm = mr.resolve(model_str)
        assert rm.is_local == (p.kind == "local")


def test_get_roundtrips_by_id():
    for p in prov.PROVIDERS:
        assert prov.get(p.id) is p


def test_get_unknown_id_raises():
    with pytest.raises(KeyError):
        prov.get("does-not-exist")


def test_cloud_providers_excludes_ollama():
    ids = {p.id for p in prov.cloud_providers()}
    assert "ollama" not in ids
    assert "anthropic" in ids


def test_model_string_local_uses_ollama_prefix():
    ollama = prov.get("ollama")
    assert ollama.model_string("qwen3:14b") == "ollama/qwen3:14b"


@pytest.mark.parametrize("provider_id", [p.id for p in prov.PROVIDERS if p.kind == "cloud"])
def test_cloud_provider_has_at_least_one_env_var_or_is_ambient_auth(provider_id):
    p = prov.get(provider_id)
    # Vertex AI uses ADC (no API key env var) by design; everything else
    # needs at least one credential-ish env var.
    if provider_id == "google_vertexai":
        assert p.env_vars  # still has project/region vars
    else:
        assert any(v.secret for v in p.env_vars)


@pytest.mark.parametrize("provider_id", [p.id for p in prov.PROVIDERS if p.kind == "cloud"])
def test_cloud_provider_has_a_main_model(provider_id):
    assert prov.get(provider_id).main_model


def test_model_for_tier_falls_back_to_main_when_tier_unset():
    p = prov.get("google_ai_studio")  # has no light_model
    assert p.light_model == ""
    assert p.model_for_tier("light") == p.main_model


def test_model_for_tier_returns_specific_tier_when_set():
    p = prov.get("anthropic")
    assert p.model_for_tier("main") == "claude-sonnet-5"
    assert p.model_for_tier("flagship") == "claude-opus-4-8"
    assert p.model_for_tier("light") == "claude-haiku-4-5"


def test_example_models_deduplicates_and_skips_blank():
    p = prov.get("opencode_go")  # main == light in the catalog
    assert len(p.example_models) == len(set(p.example_models))
    assert "" not in p.example_models
