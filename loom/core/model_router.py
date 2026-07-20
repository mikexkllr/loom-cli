"""Unified model interface + escalation logic (build step 1).

Loom mixes local Ollama models and several cloud providers behind one resolver.
Given a config model string (``ollama/qwen3:4b``, ``claude-sonnet-4-6``,
``gpt-4o``, ``gemini-2.5-pro`` ...) :func:`build_model` returns a ready-to-use
LangChain ``BaseChatModel``.

Local models go through ``langchain-ollama`` (Metal on macOS, CUDA on
Linux/Windows — no per-platform code needed). Cloud models go through
``init_chat_model`` with an explicit provider so we never depend on fuzzy
name inference.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loom.core.config import LoomConfig

if TYPE_CHECKING:  # avoid importing heavy deps at module load
    from langchain_core.language_models import BaseChatModel


# ----------------------------------------------------------------------------
# Provider resolution
# ----------------------------------------------------------------------------

# Prefix/pattern -> internal provider id. Providers recognized natively by
# ``init_chat_model`` (anthropic/openai/google_genai/google_vertexai) use its
# id directly; the rest (opencode_zen/opencode_go/custom) are OpenAI-compatible
# endpoints Loom builds directly in ``_build_cached``.
_CLOUD_PROVIDERS = {
    "claude": "anthropic",
    "anthropic": "anthropic",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "openai": "openai",
    "gemini": "google_genai",
    "google": "google_genai",
    "vertexai": "google_vertexai",
    "vertex": "google_vertexai",
    "zen": "opencode_zen",
    "opencode-zen": "opencode_zen",
    "go": "opencode_go",
    "opencode-go": "opencode_go",
    "custom": "custom",
}


@dataclass(frozen=True)
class ResolvedModel:
    """A model string normalized into provider + bare model name."""

    raw: str
    provider: str  # "ollama" | "anthropic" | "openai" | "google_genai"
    name: str  # bare model id, e.g. "qwen3:4b" or "claude-sonnet-4-6"

    @property
    def is_local(self) -> bool:
        return self.provider == "ollama"


def resolve(model: str) -> ResolvedModel:
    """Normalize a config model string into a :class:`ResolvedModel`.

    Accepts both ``provider/model`` (LiteLLM) and ``provider:model`` (LangChain)
    separators, and provider-less cloud names (mapped by prefix).
    """
    raw = model.strip()

    # Explicit provider via "/" or ":" separator.
    for sep in ("/", ":"):
        if sep in raw:
            head, tail = raw.split(sep, 1)
            head_l = head.lower()
            if head_l == "ollama":
                return ResolvedModel(raw, "ollama", tail)
            if head_l in _CLOUD_PROVIDERS:
                return ResolvedModel(raw, _CLOUD_PROVIDERS[head_l], tail)
            # ":" inside an ollama tag (e.g. "qwen3:4b") is a model tag, not a
            # provider separator — fall through to prefix inference below.
            if sep == ":":
                break

    # Provider-less cloud name: infer from leading token.
    prefix = raw.split("-", 1)[0].lower()
    provider = _CLOUD_PROVIDERS.get(prefix)
    if provider is None:
        # Unknown bare name — assume Anthropic-style cloud; init_chat_model will
        # raise a clear error if that's wrong.
        provider = "anthropic"
    return ResolvedModel(raw, provider, raw)


# ----------------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------------


def _use_bedrock() -> bool:
    """True if Claude models should route through AWS Bedrock (or a Bedrock-
    compatible corporate proxy) instead of the direct Anthropic API.

    ``LOOM_USE_BEDROCK=1`` or an explicit ``ANTHROPIC_BEDROCK_BASE_URL`` opts
    in. Set via ``settings.json``'s ``env`` block (see ``loom settings set
    env.LOOM_USE_BEDROCK 1``). Deliberately NOT Claude Code's
    ``CLAUDE_CODE_USE_BEDROCK`` — Loom settings must not reconfigure a Claude
    Code running in the same shell, and vice versa (legacy env blocks are
    translated in ``Settings.apply_env``).
    """
    flag = os.environ.get("LOOM_USE_BEDROCK", "").strip().lower()
    return flag in {"1", "true", "yes"} or bool(os.environ.get("ANTHROPIC_BEDROCK_BASE_URL"))


# OpenAI-compatible endpoints Loom builds directly (rather than via
# init_chat_model) so each one reads its *own* base_url/api_key env vars and
# several can coexist in one session without clobbering a shared OPENAI_*.
# base_url_env is checked first so a self-hosted Zen/Go mirror can override it.
_OPENAI_COMPATIBLE: dict[str, dict[str, str]] = {
    "opencode_zen": {
        "base_url": "https://opencode.ai/zen/v1",
        "base_url_env": "OPENCODE_ZEN_BASE_URL",
        "api_key_env": "OPENCODE_ZEN_API_KEY",
        "api_key_env_fallback": "OPENCODE_API_KEY",
    },
    "opencode_go": {
        "base_url": "https://opencode.ai/zen/go/v1",
        "base_url_env": "OPENCODE_GO_BASE_URL",
        "api_key_env": "OPENCODE_GO_API_KEY",
        "api_key_env_fallback": "OPENCODE_API_KEY",
    },
    "custom": {
        "base_url": "",
        "base_url_env": "LOOM_CUSTOM_BASE_URL",
        "api_key_env": "LOOM_CUSTOM_API_KEY",
        "api_key_env_fallback": "",
    },
}


def _build_openai_compatible(provider: str, name: str) -> "BaseChatModel":
    from langchain_openai import ChatOpenAI

    spec = _OPENAI_COMPATIBLE[provider]
    base_url = os.environ.get(spec["base_url_env"]) or spec["base_url"]
    api_key = os.environ.get(spec["api_key_env"]) or (
        os.environ.get(spec["api_key_env_fallback"]) if spec["api_key_env_fallback"] else None
    )
    if not base_url:
        raise RuntimeError(f"{spec['base_url_env']} is not set — required for the {provider!r} provider.")
    if not api_key:
        raise RuntimeError(
            f"{spec['api_key_env']} is not set — required for the {provider!r} provider "
            "(set it via settings.json's env block, e.g. `loom settings set "
            f"env.{spec['api_key_env']} <key>`)."
        )
    return ChatOpenAI(model=name, base_url=base_url, api_key=api_key)


@functools.lru_cache(maxsize=64)
def _build_cached(provider: str, name: str, ollama_endpoint: str, num_ctx: int) -> "BaseChatModel":
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=name,
            base_url=ollama_endpoint,
            num_ctx=num_ctx,
            # Deterministic-ish defaults; subagents can override per-call.
            temperature=0.0,
        )

    if provider in _OPENAI_COMPATIBLE:
        return _build_openai_compatible(provider, name)

    if provider == "anthropic" and _use_bedrock():
        try:
            from langchain_aws import ChatAnthropicBedrock
        except ImportError as exc:
            raise ImportError(
                "LOOM_USE_BEDROCK / ANTHROPIC_BEDROCK_BASE_URL is set, but "
                "langchain-aws isn't installed. Run `pip install -e '.[bedrock]'` "
                "(or `pip install langchain-aws`)."
            ) from exc

        # ChatAnthropicBedrock wraps anthropic's AnthropicBedrock client, which
        # reads AWS_BEARER_TOKEN_BEDROCK / ANTHROPIC_BEDROCK_BASE_URL (or real
        # AWS credentials) straight from the environment — nothing else to pass.
        return ChatAnthropicBedrock(model=name)

    if provider == "google_vertexai":
        try:
            from langchain.chat_models import init_chat_model

            return init_chat_model(name, model_provider=provider)
        except ImportError as exc:
            raise ImportError(
                "Google Vertex AI needs langchain-google-vertexai. Run "
                "`pip install -e '.[vertexai]'` (or `pip install langchain-google-vertexai`)."
            ) from exc

    from langchain.chat_models import init_chat_model

    return init_chat_model(name, model_provider=provider)


def build_model(model: str, config: LoomConfig) -> "BaseChatModel":
    """Return a LangChain chat model for the given config model string."""
    rm = resolve(model)
    num_ctx = config.context_window_for(rm.raw) if rm.is_local else 0
    return _build_cached(rm.provider, rm.name, config.ollama_endpoint, num_ctx)


# ----------------------------------------------------------------------------
# Token estimation + escalation
# ----------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token).

    Used by the prompt-size guard to decide on escalation without paying for a
    real tokenizer round-trip on every model call.
    """
    return max(1, len(text) // 4)


def should_escalate(prompt_tokens: int, model: str, config: LoomConfig) -> bool:
    """True if a local model's prompt is too large and we should use the cloud."""
    if not config.is_local(model):
        return False
    window = config.context_window_for(model)
    return prompt_tokens >= window * config.escalation_threshold


def escalation_target(config: LoomConfig) -> str:
    return config.escalation_model
