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


@pytest.mark.parametrize("provider,api_key_env", [("opencode_zen", "OPENCODE_ZEN_API_KEY"), ("opencode_go", "OPENCODE_GO_API_KEY")])
def test_opencode_gateways_get_a_session_id_and_user_agent(monkeypatch, provider, api_key_env):
    """Go answers 400 MissingSessionID without x-opencode-session (Oct 2026), and
    both gateways ask clients to name themselves rather than send the SDK's
    generic User-Agent. One id per process, shared by every role and turn."""
    pytest.importorskip("langchain_openai")
    from loom import __version__

    monkeypatch.setenv(api_key_env, "test-key")
    mr._build_cached.cache_clear()
    try:
        a = mr._build_cached(provider, "glm-5.3", "", 0)
        b = mr._build_cached(provider, "kimi-k2.7-code", "", 0)
        assert a.default_headers["x-opencode-session"].startswith("loom-")
        assert a.default_headers["x-opencode-session"] == b.default_headers["x-opencode-session"]
        assert a.default_headers["User-Agent"] == f"loom/{__version__}"
    finally:
        mr._build_cached.cache_clear()


def test_custom_endpoints_get_no_opencode_headers(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("LOOM_CUSTOM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LOOM_CUSTOM_API_KEY", "k")
    mr._build_cached.cache_clear()
    try:
        model = mr._build_cached("custom", "my-model", "", 0)
        assert "x-opencode-session" not in (model.default_headers or {})
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


def test_custom_provider_builds_without_an_api_key(monkeypatch):
    # The provider registry marks LOOM_CUSTOM_API_KEY optional — LM Studio,
    # vLLM and llama.cpp serve with no auth — and the wizard says so. The
    # router used to refuse to build such an endpoint anyway, so a keyless
    # setup passed the wizard and then failed on the first call.
    pytest.importorskip("langchain_openai")
    from loom.core import providers

    key = next(v for v in providers.get("openai_compatible").env_vars if v.key == "LOOM_CUSTOM_API_KEY")
    assert key.required is False  # the claim this test holds the router to

    monkeypatch.setenv("LOOM_CUSTOM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.delenv("LOOM_CUSTOM_API_KEY", raising=False)
    mr._build_cached.cache_clear()
    try:
        model = mr._build_cached("custom", "my-model", "", 0)
        assert str(model.openai_api_base).rstrip("/") == "http://localhost:1234/v1"
        assert model.openai_api_key.get_secret_value()  # a placeholder, never empty
    finally:
        mr._build_cached.cache_clear()


def test_gateway_providers_still_require_a_key(monkeypatch):
    # Only the self-hosted route is keyless; a gateway without its key must
    # still fail at build time with the name of the variable to set.
    for name in ("OPENCODE_ZEN_API_KEY", "OPENCODE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    mr._build_cached.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="OPENCODE_ZEN_API_KEY"):
            mr._build_cached("opencode_zen", "glm-5.2", "", 0)
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


def test_opencode_go_falls_back_to_the_same_shared_api_key(monkeypatch):
    """One OpenCode key covers both gateways — verified live: a Zen key answers
    on the Go endpoint. Users with both should not need to set two variables."""
    pytest.importorskip("langchain_openai")
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_API_KEY", "shared-key")
    mr._build_cached.cache_clear()
    try:
        model = mr._build_cached("opencode_go", "glm-5.2", "", 0)
        assert model.openai_api_key.get_secret_value() == "shared-key"
    finally:
        mr._build_cached.cache_clear()


def test_a_dedicated_key_still_beats_the_shared_one(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENCODE_API_KEY", "shared-key")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-key")
    mr._build_cached.cache_clear()
    try:
        assert mr._build_cached("opencode_go", "glm-5.2", "", 0
                                ).openai_api_key.get_secret_value() == "go-key"
    finally:
        mr._build_cached.cache_clear()


def test_both_opencode_gateways_coexist_in_one_session(monkeypatch):
    """The reason these endpoints are built directly instead of through
    init_chat_model: each reads its own base_url/api_key rather than a shared
    OPENAI_*, so a fleet can mix a Go orchestrator with Zen subagents."""
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-key")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated")
    mr._build_cached.cache_clear()
    try:
        zen = mr._build_cached("opencode_zen", "deepseek-v4-flash-free", "", 0)
        go = mr._build_cached("opencode_go", "glm-5.2", "", 0)
        assert zen.openai_api_base != go.openai_api_base
        assert zen.openai_api_key.get_secret_value() == "zen-key"
        assert go.openai_api_key.get_secret_value() == "go-key"
    finally:
        mr._build_cached.cache_clear()


def test_a_self_hosted_mirror_overrides_the_default_base_url(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-key")
    monkeypatch.setenv("OPENCODE_GO_BASE_URL", "http://localhost:9000/v1")
    mr._build_cached.cache_clear()
    try:
        model = mr._build_cached("opencode_go", "glm-5.2", "", 0)
        assert model.openai_api_base == "http://localhost:9000/v1"
    finally:
        mr._build_cached.cache_clear()


def test_is_local_agrees_with_resolve_whatever_the_case():
    """`Ollama/qwen3:4b` resolved to the Ollama provider and read as cloud to
    everything asking the config — billed, and without a prompt-size guard."""
    from loom.core.config import LoomConfig

    config = LoomConfig()
    for model in ("ollama/qwen3:4b", "Ollama/qwen3:4b", "OLLAMA:qwen3", " ollama/x "):
        assert config.is_local(model) and mr.resolve(model).is_local, model
    for model in ("claude-sonnet-5", "zen/ollama-ish", "go:glm-5.2", ""):
        assert not config.is_local(model), model


def test_an_empty_model_string_is_a_clear_error_not_a_provider_guess():
    from loom.core.config import LoomConfig

    with pytest.raises(ValueError, match="no model configured"):
        mr.build_model("", LoomConfig())
    with pytest.raises(ValueError, match="no model configured"):
        mr.build_model("   ", LoomConfig())


def test_clear_model_cache_forgets_built_models():
    pytest.importorskip("langchain_ollama")
    from loom.core.config import LoomConfig

    mr.clear_model_cache()
    first = mr.build_model("ollama/qwen3:4b", LoomConfig())
    assert mr.build_model("ollama/qwen3:4b", LoomConfig()) is first
    mr.clear_model_cache()
    assert mr.build_model("ollama/qwen3:4b", LoomConfig()) is not first
    mr.clear_model_cache()
