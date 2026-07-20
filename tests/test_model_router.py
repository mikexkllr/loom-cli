"""Model-string resolution and escalation logic (no network / no model build)."""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")

from loom.core import config as cfg
from loom.core import model_router as mr


@pytest.mark.parametrize(
    "raw,provider,name",
    [
        ("ollama/qwen3:4b", "ollama", "qwen3:4b"),
        ("ollama:llama3.2:3b", "ollama", "llama3.2:3b"),
        ("claude-sonnet-4-6", "anthropic", "claude-sonnet-4-6"),
        ("anthropic:claude-haiku-4-5", "anthropic", "claude-haiku-4-5"),
        ("gpt-4o", "openai", "gpt-4o"),
        ("openai:gpt-4o-mini", "openai", "gpt-4o-mini"),
        ("o3", "openai", "o3"),
        ("gemini-2.5-pro", "google_genai", "gemini-2.5-pro"),
        ("vertexai:gemini-2.5-pro", "google_vertexai", "gemini-2.5-pro"),
        ("vertex:gemini-2.5-pro", "google_vertexai", "gemini-2.5-pro"),
        ("zen:glm-5.2", "opencode_zen", "glm-5.2"),
        ("opencode-zen:glm-5.2", "opencode_zen", "glm-5.2"),
        ("go:deepseek-v4-flash", "opencode_go", "deepseek-v4-flash"),
        ("opencode-go:deepseek-v4-flash", "opencode_go", "deepseek-v4-flash"),
        ("custom:my-self-hosted-model", "custom", "my-self-hosted-model"),
    ],
)
def test_resolve(raw, provider, name):
    rm = mr.resolve(raw)
    assert rm.provider == provider
    assert rm.name == name
    assert rm.is_local == (provider == "ollama")


def test_estimate_tokens_monotonic():
    assert mr.estimate_tokens("x" * 4) <= mr.estimate_tokens("x" * 400)
    assert mr.estimate_tokens("") >= 1


def test_should_escalate_local_over_threshold():
    c = cfg.load_config(path=cfg.DEFAULT_CONFIG_PATH)
    model = "ollama/qwen3:4b"
    window = c.context_window_for(model)
    over = int(window * c.escalation_threshold) + 10
    under = int(window * c.escalation_threshold) - 10
    assert mr.should_escalate(over, model, c) is True
    assert mr.should_escalate(under, model, c) is False


def test_cloud_models_never_escalate():
    c = cfg.load_config(path=cfg.DEFAULT_CONFIG_PATH)
    assert mr.should_escalate(10**9, "claude-sonnet-4-6", c) is False


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, False),
        ({"LOOM_USE_BEDROCK": "1"}, True),
        ({"LOOM_USE_BEDROCK": "true"}, True),
        ({"LOOM_USE_BEDROCK": "0"}, False),
        # Claude Code's own flag must NOT flip Loom's routing.
        ({"CLAUDE_CODE_USE_BEDROCK": "1"}, False),
        ({"ANTHROPIC_BEDROCK_BASE_URL": "https://example.com"}, True),
    ],
)
def test_use_bedrock_flag(monkeypatch, env, expected):
    monkeypatch.delenv("LOOM_USE_BEDROCK", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)
    monkeypatch.delenv("ANTHROPIC_BEDROCK_BASE_URL", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert mr._use_bedrock() is expected


def test_anthropic_routes_through_bedrock_when_flagged(monkeypatch):
    pytest.importorskip("langchain_aws")
    monkeypatch.setenv("LOOM_USE_BEDROCK", "1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
    monkeypatch.setenv("ANTHROPIC_BEDROCK_BASE_URL", "https://example.com")
    mr._build_cached.cache_clear()
    try:
        from langchain_aws import ChatAnthropicBedrock

        model = mr._build_cached("anthropic", "claude-sonnet-4-6", "", 0)
        assert isinstance(model, ChatAnthropicBedrock)
    finally:
        mr._build_cached.cache_clear()


@pytest.mark.parametrize(
    "provider,api_key_env,default_base_url",
    [
        ("opencode_zen", "OPENCODE_ZEN_API_KEY", "https://opencode.ai/zen/v1"),
        ("opencode_go", "OPENCODE_GO_API_KEY", "https://opencode.ai/zen/go/v1"),
    ],
)
def test_opencode_presets_build_chat_openai(monkeypatch, provider, api_key_env, default_base_url):
    pytest.importorskip("langchain_openai")
    from langchain_openai import ChatOpenAI

    monkeypatch.setenv(api_key_env, "test-key")
    mr._build_cached.cache_clear()
    try:
        model = mr._build_cached(provider, "some-model", "", 0)
        assert isinstance(model, ChatOpenAI)
        assert model.openai_api_base == default_base_url
    finally:
        mr._build_cached.cache_clear()


def test_opencode_zen_falls_back_to_shared_api_key(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_API_KEY", "shared-key")
    mr._build_cached.cache_clear()
    try:
        model = mr._build_cached("opencode_zen", "glm-5.2", "", 0)
        assert model.openai_api_key.get_secret_value() == "shared-key"
    finally:
        mr._build_cached.cache_clear()


def test_custom_provider_requires_base_url(monkeypatch):
    monkeypatch.delenv("LOOM_CUSTOM_BASE_URL", raising=False)
    mr._build_cached.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="LOOM_CUSTOM_BASE_URL"):
            mr._build_cached("custom", "my-model", "", 0)
    finally:
        mr._build_cached.cache_clear()


def test_custom_provider_requires_api_key(monkeypatch):
    monkeypatch.setenv("LOOM_CUSTOM_BASE_URL", "https://example.com/v1")
    monkeypatch.delenv("LOOM_CUSTOM_API_KEY", raising=False)
    mr._build_cached.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="LOOM_CUSTOM_API_KEY"):
            mr._build_cached("custom", "my-model", "", 0)
    finally:
        mr._build_cached.cache_clear()


def test_custom_provider_builds_chat_openai(monkeypatch):
    pytest.importorskip("langchain_openai")
    from langchain_openai import ChatOpenAI

    monkeypatch.setenv("LOOM_CUSTOM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LOOM_CUSTOM_API_KEY", "test-key")
    mr._build_cached.cache_clear()
    try:
        model = mr._build_cached("custom", "my-model", "", 0)
        assert isinstance(model, ChatOpenAI)
        assert model.openai_api_base == "https://example.com/v1"
    finally:
        mr._build_cached.cache_clear()
