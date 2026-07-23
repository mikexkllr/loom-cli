"""Live cloud model catalogs: httpx calls are mocked — see test_ollama.py for
the same monkeypatch-httpx convention used elsewhere in this repo."""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("httpx")

import httpx

from loom.core import model_catalog as catalog
from loom.core import providers as prov


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=self)

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# can_list / needs_no_credential
# ---------------------------------------------------------------------------


def test_can_list_covers_rest_listable_providers():
    for pid in ("anthropic", "openai", "google_ai_studio", "opencode_zen", "opencode_go", "openai_compatible"):
        assert catalog.can_list(prov.get(pid))


def test_can_list_excludes_sdk_only_providers():
    assert not catalog.can_list(prov.get("anthropic_bedrock"))
    assert not catalog.can_list(prov.get("google_vertexai"))


def test_needs_no_credential_only_for_opencode_gateways():
    assert catalog.needs_no_credential(prov.get("opencode_zen"))
    assert catalog.needs_no_credential(prov.get("opencode_go"))
    assert not catalog.needs_no_credential(prov.get("anthropic"))


# ---------------------------------------------------------------------------
# list_models — per-provider branches
# ---------------------------------------------------------------------------


def test_list_models_openai_parses_data_ids(monkeypatch):
    monkeypatch.setattr(
        catalog.httpx, "get", lambda url, **k: _FakeResponse({"data": [{"id": "gpt-5.6-terra"}, {"id": "gpt-5.6-sol"}]})
    )
    assert catalog.list_models(prov.get("openai"), {"OPENAI_API_KEY": "x"}) == ["gpt-5.6-sol", "gpt-5.6-terra"]


def test_list_models_openai_without_key_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not call httpx without a key")

    monkeypatch.setattr(catalog.httpx, "get", boom)
    assert catalog.list_models(prov.get("openai"), {}) == []


def test_list_models_anthropic_uses_x_api_key_header(monkeypatch):
    seen = {}

    def fake_get(url, headers=None, **k):
        seen["url"], seen["headers"] = url, headers
        return _FakeResponse({"data": [{"id": "claude-sonnet-5"}]})

    monkeypatch.setattr(catalog.httpx, "get", fake_get)
    assert catalog.list_models(prov.get("anthropic"), {"ANTHROPIC_API_KEY": "sk-ant-x"}) == ["claude-sonnet-5"]
    assert seen["url"] == "https://api.anthropic.com/v1/models"
    assert seen["headers"]["x-api-key"] == "sk-ant-x"
    assert seen["headers"]["anthropic-version"]


def test_list_models_google_ai_studio_filters_to_generate_content(monkeypatch):
    payload = {
        "models": [
            {"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]},
        ]
    }
    monkeypatch.setattr(catalog.httpx, "get", lambda url, **k: _FakeResponse(payload))
    assert catalog.list_models(prov.get("google_ai_studio"), {"GOOGLE_API_KEY": "x"}) == ["gemini-3.5-flash"]


def test_list_models_opencode_zen_filters_unsupported_families(monkeypatch):
    payload = {
        "data": [
            {"id": "gpt-5.5"},
            {"id": "claude-sonnet-5"},
            {"id": "minimax-m3"},
            {"id": "qwen3.6-plus"},
        ]
    }
    monkeypatch.setattr(catalog.httpx, "get", lambda url, **k: _FakeResponse(payload))
    models = catalog.list_models(prov.get("opencode_zen"), {})
    assert models == ["claude-sonnet-5", "gpt-5.5"]
    assert "minimax-m3" not in models
    assert "qwen3.6-plus" not in models


def test_list_models_opencode_zen_works_without_credentials(monkeypatch):
    """Zen's /models listing is public — no API key required to browse it."""
    monkeypatch.setattr(catalog.httpx, "get", lambda url, **k: _FakeResponse({"data": [{"id": "gpt-5.5"}]}))
    assert catalog.list_models(prov.get("opencode_zen"), {}) == ["gpt-5.5"]


def test_list_models_openai_compatible_needs_base_url():
    assert catalog.list_models(prov.get("openai_compatible"), {}) == []


def test_list_models_bedrock_and_vertexai_always_empty():
    assert catalog.list_models(prov.get("anthropic_bedrock"), {"AWS_BEARER_TOKEN_BEDROCK": "x"}) == []
    assert catalog.list_models(prov.get("google_vertexai"), {"GOOGLE_CLOUD_PROJECT": "p"}) == []


def test_list_models_swallows_http_errors(monkeypatch):
    def raise_connect(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(catalog.httpx, "get", raise_connect)
    assert catalog.list_models(prov.get("openai"), {"OPENAI_API_KEY": "x"}) == []


def test_list_models_swallows_bad_status(monkeypatch):
    monkeypatch.setattr(catalog.httpx, "get", lambda url, **k: _FakeResponse({}, status_code=401))
    assert catalog.list_models(prov.get("openai"), {"OPENAI_API_KEY": "bad"}) == []


def test_list_models_swallows_malformed_top_level_json(monkeypatch):
    """A server returning a bare JSON array instead of {"data": [...]} must
    not crash the picker — list_models() never raises."""
    monkeypatch.setattr(catalog.httpx, "get", lambda url, **k: _FakeResponse(["not", "a", "dict"]))
    assert catalog.list_models(prov.get("openai"), {"OPENAI_API_KEY": "x"}) == []


# ---------------------------------------------------------------------------
# available_models — the fallback wrapper callers actually use
# ---------------------------------------------------------------------------


def test_available_models_falls_back_to_examples_without_credentials():
    models, is_live = catalog.available_models(prov.get("anthropic"), {})
    assert is_live is False
    assert models == list(prov.get("anthropic").example_models)


def test_available_models_uses_live_catalog_when_reachable(monkeypatch):
    monkeypatch.setattr(
        catalog.httpx, "get", lambda url, **k: _FakeResponse({"data": [{"id": "claude-opus-4-8"}]})
    )
    models, is_live = catalog.available_models(prov.get("anthropic"), {"ANTHROPIC_API_KEY": "x"})
    assert is_live is True
    assert models == ["claude-opus-4-8"]


def test_available_models_falls_back_on_empty_live_result(monkeypatch):
    monkeypatch.setattr(catalog.httpx, "get", lambda url, **k: _FakeResponse({"data": []}))
    models, is_live = catalog.available_models(prov.get("anthropic"), {"ANTHROPIC_API_KEY": "x"})
    assert is_live is False
    assert models == list(prov.get("anthropic").example_models)


def test_available_models_bedrock_always_uses_examples():
    models, is_live = catalog.available_models(prov.get("anthropic_bedrock"), {"AWS_BEARER_TOKEN_BEDROCK": "x"})
    assert is_live is False
    assert models == list(prov.get("anthropic_bedrock").example_models)


def test_available_models_zen_is_live_even_with_no_env(monkeypatch):
    monkeypatch.setattr(catalog.httpx, "get", lambda url, **k: _FakeResponse({"data": [{"id": "gpt-5.5"}]}))
    models, is_live = catalog.available_models(prov.get("opencode_zen"), {})
    assert is_live is True
    assert models == ["gpt-5.5"]


def test_available_models_custom_endpoint_attempts_live_without_optional_api_key(monkeypatch):
    """LOOM_CUSTOM_API_KEY is optional (self-hosted no-auth servers) — a
    fully-configured base URL alone must be enough to try a live fetch."""
    monkeypatch.setattr(catalog.httpx, "get", lambda url, **k: _FakeResponse({"data": [{"id": "local-model"}]}))
    models, is_live = catalog.available_models(
        prov.get("openai_compatible"), {"LOOM_CUSTOM_BASE_URL": "http://localhost:8000/v1"}
    )
    assert is_live is True
    assert models == ["local-model"]


def test_available_models_custom_endpoint_without_base_url_uses_examples():
    models, is_live = catalog.available_models(prov.get("openai_compatible"), {})
    assert is_live is False
    assert models == list(prov.get("openai_compatible").example_models)
