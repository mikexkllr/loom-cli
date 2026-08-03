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


# ------------------------------------------------- optional-extra providers


def test_providers_behind_an_extra_report_unavailable_when_it_is_absent():
    """A frozen binary bundles none of the extras, so `is_available` is what
    keeps the wizard from offering a provider that cannot possibly run."""
    import importlib.util

    for p in prov.PROVIDERS:
        if not p.pip_extra:
            assert prov.is_available(p.id), f"{p.id} is a base provider"
            continue
        module = prov._EXTRA_MODULE.get(p.pip_extra)
        installed = module is None or importlib.util.find_spec(module) is not None
        assert prov.is_available(p.id) is installed


def test_every_extra_has_an_import_name_to_probe():
    # Without a mapping, is_available() silently returns True and the provider
    # fails at first use instead of in the picker.
    for p in prov.PROVIDERS:
        if p.pip_extra:
            assert p.pip_extra in prov._EXTRA_MODULE, p.pip_extra


def test_install_hint_never_tells_a_binary_user_to_run_uv_sync(monkeypatch):
    """`uv sync` is impossible in a frozen bundle — no project, no uv, no
    site-packages. Saying it anyway sends the user in a circle."""
    from loom.core import update

    monkeypatch.setattr(update, "is_frozen", lambda: True)
    frozen = prov.extra_install_hint("bedrock")
    assert "standalone binary" in frozen
    assert "source install" in frozen

    monkeypatch.setattr(update, "is_frozen", lambda: False)
    source = prov.extra_install_hint("bedrock")
    assert "uv sync --extra bedrock" in source
    assert frozen != source
