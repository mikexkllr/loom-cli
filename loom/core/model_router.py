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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loom.core.config import LoomConfig

if TYPE_CHECKING:  # avoid importing heavy deps at module load
    from langchain_core.language_models import BaseChatModel


# ----------------------------------------------------------------------------
# Provider resolution
# ----------------------------------------------------------------------------

# Prefix/pattern -> LangChain provider id used by init_chat_model.
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
