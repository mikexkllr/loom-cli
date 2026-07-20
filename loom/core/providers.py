"""Catalog of model providers Loom knows how to talk to.

This is the single source of truth the onboarding wizard and ``/model``
picker use to list providers, ask for the right credentials, and build a
config model string that :mod:`loom.core.model_router` understands. Adding a
new provider means adding one :class:`ProviderInfo` here plus (if it isn't a
plain OpenAI-compatible endpoint) a branch in ``model_router._build_cached``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvVar:
    """One environment variable a provider needs, surfaced by the wizard."""

    key: str
    label: str
    secret: bool = True
    required: bool = True
    default: str = ""


@dataclass(frozen=True)
class ProviderInfo:
    """Static description of a model provider for the onboarding UI.

    ``main_model``/``flagship_model``/``light_model`` let the wizard
    auto-suggest a sensible model per role in "quick setup": main for the
    orchestrator/escalation, flagship (the provider's strongest model) for
    the advisor, light (cheap/fast) for the reviewer subagent. Cloud
    providers should set at least ``main_model``; ``flagship_model``/
    ``light_model`` default to it when left blank.
    """

    id: str
    label: str
    kind: str  # "local" | "cloud"
    prefix: str  # model-string prefix understood by model_router.resolve()
    env_vars: tuple[EnvVar, ...] = field(default_factory=tuple)
    main_model: str = ""
    flagship_model: str = ""
    light_model: str = ""
    docs_url: str = ""
    notes: str = ""
    pip_extra: str | None = None  # `pip install -e ".[<pip_extra>]"` if not a base dep

    def model_string(self, model_id: str) -> str:
        """Build a config model string ``model_router.resolve()`` understands."""
        if self.kind == "local":
            return f"ollama/{model_id}"
        return f"{self.prefix}:{model_id}"

    def model_for_tier(self, tier: str) -> str:
        """``tier`` is "main" | "flagship" | "light"; unset tiers fall back to main."""
        return {"main": self.main_model, "flagship": self.flagship_model, "light": self.light_model}.get(
            tier, self.main_model
        ) or self.main_model

    @property
    def example_models(self) -> tuple[str, ...]:
        """De-duplicated (main, flagship, light) for display purposes."""
        seen: list[str] = []
        for m in (self.main_model, self.flagship_model, self.light_model):
            if m and m not in seen:
                seen.append(m)
        return tuple(seen)


PROVIDERS: tuple[ProviderInfo, ...] = (
    ProviderInfo(
        id="ollama",
        label="Local (Ollama)",
        kind="local",
        prefix="ollama",
        main_model="qwen3.5:9b",
        docs_url="https://ollama.com",
        notes="Free, private, runs on your machine. Metal on macOS, CUDA/ROCm on Linux/Windows.",
    ),
    ProviderInfo(
        id="anthropic",
        label="Anthropic (Claude)",
        kind="cloud",
        prefix="anthropic",
        env_vars=(EnvVar("ANTHROPIC_API_KEY", "Anthropic API key"),),
        main_model="claude-sonnet-5",
        flagship_model="claude-opus-4-8",
        light_model="claude-haiku-4-5",
        docs_url="https://console.anthropic.com/settings/keys",
    ),
    ProviderInfo(
        id="anthropic_bedrock",
        label="Anthropic via AWS Bedrock",
        kind="cloud",
        prefix="anthropic",
        env_vars=(
            EnvVar("CLAUDE_CODE_USE_BEDROCK", "Enable Bedrock routing", secret=False, default="1"),
            EnvVar("AWS_BEARER_TOKEN_BEDROCK", "Bedrock API key / bearer token"),
            EnvVar("ANTHROPIC_BEDROCK_BASE_URL", "Bedrock base URL (blank = real AWS)", secret=False, required=False),
        ),
        main_model="claude-sonnet-5",
        flagship_model="claude-opus-4-8",
        docs_url="https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html",
        notes="Also works with real AWS credentials (AWS_ACCESS_KEY_ID/SECRET) instead of a bearer token.",
        pip_extra="bedrock",
    ),
    ProviderInfo(
        id="openai",
        label="OpenAI",
        kind="cloud",
        prefix="openai",
        env_vars=(EnvVar("OPENAI_API_KEY", "OpenAI API key"),),
        main_model="gpt-5.6-terra",
        flagship_model="gpt-5.6-sol",
        light_model="gpt-5.6-luna",
        docs_url="https://platform.openai.com/api-keys",
    ),
    ProviderInfo(
        id="openai_compatible",
        label="OpenAI-compatible API (custom endpoint)",
        kind="cloud",
        prefix="custom",
        env_vars=(
            EnvVar("LOOM_CUSTOM_BASE_URL", "Base URL (e.g. http://localhost:8000/v1)", secret=False),
            EnvVar("LOOM_CUSTOM_API_KEY", "API key", required=False),
        ),
        main_model="your-model-id",
        notes="Any server speaking the OpenAI Chat Completions API: vLLM, LM Studio, Together, Groq, etc.",
    ),
    ProviderInfo(
        id="opencode_zen",
        label="OpenCode Zen",
        kind="cloud",
        prefix="zen",
        env_vars=(EnvVar("OPENCODE_ZEN_API_KEY", "OpenCode Zen API key"),),
        main_model="gpt-5.5",
        flagship_model="gpt-5.5-pro",
        light_model="big-pickle",
        docs_url="https://opencode.ai/docs/zen/",
        notes="Curated pay-per-use model gateway; several models are free. Only OpenAI-shaped models are "
        "supported so far — MiniMax/Qwen-style Anthropic-shaped Zen models aren't wired up yet.",
    ),
    ProviderInfo(
        id="opencode_go",
        label="OpenCode Go",
        kind="cloud",
        prefix="go",
        env_vars=(EnvVar("OPENCODE_GO_API_KEY", "OpenCode Go API key"),),
        main_model="deepseek-v4-flash",
        flagship_model="glm-5.2",
        light_model="deepseek-v4-flash",
        docs_url="https://opencode.ai/docs/go/",
        notes="$5 first month / $10 mo subscription, flat usage limits. GLM/Kimi/DeepSeek/MiMo models route "
        "here; MiniMax/Qwen are Anthropic-shaped and aren't wired up yet.",
    ),
    ProviderInfo(
        id="google_ai_studio",
        label="Google AI Studio (Gemini API)",
        kind="cloud",
        prefix="google",
        env_vars=(EnvVar("GOOGLE_API_KEY", "Google AI Studio API key"),),
        main_model="gemini-3.5-flash",
        flagship_model="gemini-3.1-pro-preview",
        docs_url="https://aistudio.google.com/apikey",
        notes="Personal/dev API key, no GCP project needed.",
    ),
    ProviderInfo(
        id="google_vertexai",
        label="Google Vertex AI",
        kind="cloud",
        prefix="vertexai",
        env_vars=(
            EnvVar("GOOGLE_CLOUD_PROJECT", "GCP project id", secret=False),
            EnvVar("GOOGLE_CLOUD_LOCATION", "GCP region", secret=False, default="us-central1"),
        ),
        main_model="gemini-3.5-flash",
        flagship_model="gemini-3.1-pro-preview",
        docs_url="https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal",
        notes="Uses Application Default Credentials (`gcloud auth application-default login`) — no API key.",
        pip_extra="vertexai",
    ),
)

_BY_ID = {p.id: p for p in PROVIDERS}


def get(provider_id: str) -> ProviderInfo:
    return _BY_ID[provider_id]


def cloud_providers() -> tuple[ProviderInfo, ...]:
    return tuple(p for p in PROVIDERS if p.kind == "cloud")
