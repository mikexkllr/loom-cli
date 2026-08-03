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


def test_can_list_excludes_only_providers_whose_sdk_is_absent():
    # Bedrock is listable: boto3 signs the control-plane calls and is bundled
    # into the binary. Vertex AI is not — its SDK is 331MB and deliberately
    # left out, so there is nothing to list with.
    assert catalog.can_list(prov.get("anthropic_bedrock"))
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


# ------------------------------------------------------------------ bedrock
#
# boto3 is stubbed: these pin the shape Loom asks for and how it degrades, not
# AWS's behaviour. The call has not been run against a live AWS account.


class _FakeBedrockClient:
    def __init__(self, profiles=None, models=None, fail_profiles=False, fail_models=False):
        self._profiles = profiles or []
        self._models = models or []
        self._fail_profiles = fail_profiles
        self._fail_models = fail_models
        self.list_kwargs = None

    def get_paginator(self, name):
        assert name == "list_inference_profiles"
        client = self

        class _Paginator:
            def paginate(self):
                if client._fail_profiles:
                    raise RuntimeError("AccessDeniedException")
                return [{"inferenceProfileSummaries": client._profiles}]

        return _Paginator()

    def list_foundation_models(self, **kwargs):
        if self._fail_models:
            raise RuntimeError("AccessDeniedException")
        self.list_kwargs = kwargs
        return {"modelSummaries": self._models}


def _stub_boto3(monkeypatch, client):
    import sys
    import types

    boto3 = types.ModuleType("boto3")
    boto3.client = lambda *a, **k: client
    botocore = types.ModuleType("botocore")
    config_mod = types.ModuleType("botocore.config")
    config_mod.Config = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", config_mod)


def _bedrock_ids(monkeypatch, client, env=None):
    _stub_boto3(monkeypatch, client)
    for key in ("ANTHROPIC_BEDROCK_BASE_URL", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(key, raising=False)
    return catalog.list_models(prov.get("anthropic_bedrock"), env or {})


def test_bedrock_merges_inference_profiles_and_foundation_models(monkeypatch):
    client = _FakeBedrockClient(
        profiles=[
            {"inferenceProfileId": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
            {"inferenceProfileId": "eu.amazon.nova-pro-v1:0"},  # not Anthropic
        ],
        models=[
            {"modelId": "anthropic.claude-3-5-haiku-20241022-v1:0", "inferenceTypesSupported": ["ON_DEMAND"]},
        ],
    )
    ids = _bedrock_ids(monkeypatch, client)
    assert "us.anthropic.claude-sonnet-4-5-20250929-v1:0" in ids
    assert "anthropic.claude-3-5-haiku-20241022-v1:0" in ids
    assert "eu.amazon.nova-pro-v1:0" not in ids
    assert client.list_kwargs == {"byProvider": "anthropic", "byOutputModality": "TEXT"}


def test_bedrock_skips_models_without_on_demand_throughput(monkeypatch):
    # A provisioned-only id is listed by AWS but fails at invoke time with
    # "on-demand throughput isn't supported" — offering it is a trap.
    client = _FakeBedrockClient(
        models=[
            {"modelId": "anthropic.claude-provisioned-v1:0", "inferenceTypesSupported": ["PROVISIONED"]},
            {"modelId": "anthropic.claude-ok-v1:0", "inferenceTypesSupported": ["ON_DEMAND"]},
        ]
    )
    assert _bedrock_ids(monkeypatch, client) == ["anthropic.claude-ok-v1:0"]


def test_bedrock_survives_losing_either_half(monkeypatch):
    # Listing profiles and listing foundation models are separate IAM actions;
    # being denied one must not cost the other.
    only_models = _FakeBedrockClient(
        models=[{"modelId": "anthropic.claude-ok-v1:0", "inferenceTypesSupported": ["ON_DEMAND"]}],
        fail_profiles=True,
    )
    assert _bedrock_ids(monkeypatch, only_models) == ["anthropic.claude-ok-v1:0"]

    only_profiles = _FakeBedrockClient(
        profiles=[{"inferenceProfileId": "us.anthropic.claude-sonnet-4-5-v1:0"}], fail_models=True
    )
    assert _bedrock_ids(monkeypatch, only_profiles) == ["us.anthropic.claude-sonnet-4-5-v1:0"]


def test_bedrock_does_not_query_aws_when_pointed_at_a_proxy(monkeypatch):
    """ANTHROPIC_BEDROCK_BASE_URL means a corporate gateway. Asking real AWS
    what it hosts would answer a question nobody asked."""
    client = _FakeBedrockClient(models=[{"modelId": "x", "inferenceTypesSupported": ["ON_DEMAND"]}])
    _stub_boto3(monkeypatch, client)
    ids = catalog.list_models(
        prov.get("anthropic_bedrock"), {"ANTHROPIC_BEDROCK_BASE_URL": "https://proxy.corp/v1"}
    )
    assert ids == []


def test_bedrock_falls_back_to_examples_without_boto3(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "boto3", None)  # import raises
    models, is_live = catalog.available_models(prov.get("anthropic_bedrock"), {})
    assert is_live is False
    assert models == list(prov.get("anthropic_bedrock").example_models)
