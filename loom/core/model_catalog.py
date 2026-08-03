"""Best-effort live model catalogs for cloud providers.

Some providers expose a plain REST "list models" endpoint that only needs an
API key (or nothing at all) — those get queried directly with ``httpx``.
Bedrock needs SigV4-signed requests, so it goes through boto3 (bundled in the
binary for exactly this). Vertex AI needs OAuth via Application Default
Credentials and its SDK is not bundled, so it is still not attempted.
Everywhere dynamic listing isn't possible, or the request fails/times
out/lacks credentials, :func:`available_models` falls back to
:attr:`~loom.core.providers.ProviderInfo.example_models` — the picker never
comes up empty.
"""

from __future__ import annotations

import os

import httpx

from loom.core.providers import ProviderInfo

TIMEOUT = 4.0

# Providers Loom can enumerate. Most are a plain, key-only (or public) REST
# endpoint; anthropic_bedrock is the exception and goes through boto3.
_LISTABLE = {
    "anthropic",
    "openai",
    "google_ai_studio",
    "opencode_zen",
    "opencode_go",
    "openai_compatible",
    "anthropic_bedrock",
}

# OpenCode's Zen/Go gateways list every model they route to, but a few
# families are only reachable there via a non-OpenAI wire shape (Anthropic
# Messages API, not chat/completions) that Loom's ChatOpenAI-based client
# can't speak yet — see the opencode_zen/opencode_go notes in providers.py.
# Filtered out here so the picker never offers a model that will fail.
_UNSUPPORTED_PREFIXES: dict[str, tuple[str, ...]] = {
    "opencode_zen": ("minimax", "qwen"),
    "opencode_go": ("minimax", "qwen"),
}


def can_list(provider: ProviderInfo) -> bool:
    return provider.id in _LISTABLE


def needs_no_credential(provider: ProviderInfo) -> bool:
    """True for providers whose listing endpoint is public — worth querying
    opportunistically even before the user has entered an API key."""
    return provider.id in {"opencode_zen", "opencode_go"}


def _has_credentials(provider: ProviderInfo, env: dict[str, str]) -> bool:
    """True if every *required* env var for ``provider`` is available.

    Deliberately keyed on "required", not "secret": the custom OpenAI-
    compatible provider's only required var is its (non-secret) base URL —
    the API key is optional, since plenty of self-hosted servers (local
    vLLM, LM Studio) need no auth at all. Gating on secrecy would mean a
    fully-reachable no-auth endpoint never gets a live listing attempt.
    """
    return all(env.get(v.key) or os.environ.get(v.key) for v in provider.env_vars if v.required)


def _filter_unsupported(provider_id: str, ids: list[str]) -> list[str]:
    prefixes = _UNSUPPORTED_PREFIXES.get(provider_id, ())
    if not prefixes:
        return ids
    return [i for i in ids if not i.lower().startswith(prefixes)]


def list_models(provider: ProviderInfo, env: dict[str, str]) -> list[str]:
    """Live model ids for ``provider``, or ``[]`` if unsupported, lacking
    credentials, or the request fails for any reason. Never raises."""

    def get(key: str) -> str:
        return env.get(key) or os.environ.get(key, "")

    try:
        if provider.id == "anthropic":
            return _anthropic(get("ANTHROPIC_API_KEY"))
        if provider.id == "openai":
            api_key = get("OPENAI_API_KEY")
            if not api_key:
                return []
            return _openai_compatible("https://api.openai.com/v1", api_key)
        if provider.id == "google_ai_studio":
            return _google_ai_studio(get("GOOGLE_API_KEY"))
        if provider.id == "opencode_zen":
            ids = _openai_compatible(
                get("OPENCODE_ZEN_BASE_URL") or "https://opencode.ai/zen/v1",
                get("OPENCODE_ZEN_API_KEY") or get("OPENCODE_API_KEY"),
            )
            return _filter_unsupported(provider.id, ids)
        if provider.id == "opencode_go":
            ids = _openai_compatible(
                get("OPENCODE_GO_BASE_URL") or "https://opencode.ai/zen/go/v1",
                get("OPENCODE_GO_API_KEY") or get("OPENCODE_API_KEY"),
            )
            return _filter_unsupported(provider.id, ids)
        if provider.id == "openai_compatible":
            base = get("LOOM_CUSTOM_BASE_URL")
            if not base:
                return []
            return _openai_compatible(base, get("LOOM_CUSTOM_API_KEY"))
        if provider.id == "anthropic_bedrock":
            return _bedrock(get)
    except (httpx.HTTPError, KeyError, ValueError, TypeError, AttributeError):
        pass
    return []


def available_models(provider: ProviderInfo, env: dict[str, str]) -> tuple[list[str], bool]:
    """``(models, is_live)`` for ``provider``.

    Attempts a live catalog fetch when one's possible and worth trying (a
    public endpoint, or credentials already known); otherwise — or if the
    fetch comes back empty — falls back to the provider's hardcoded
    ``example_models`` so callers always get something to show.
    """
    if can_list(provider) and (
        needs_no_credential(provider)
        or _has_credentials(provider, env)
        # Bedrock's credentials may be an AWS profile, an instance role or
        # SSO — boto3 resolves those from places Loom never sees, so gating on
        # env vars alone would skip a listing that would have worked.
        or provider.id == "anthropic_bedrock"
    ):
        live = list_models(provider, env)
        if live:
            return live, True
    return list(provider.example_models), False


def _openai_compatible(base_url: str, api_key: str) -> list[str]:
    if not base_url:
        return []
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    return sorted({m["id"] for m in resp.json().get("data", []) if m.get("id")})


def _anthropic(api_key: str) -> list[str]:
    if not api_key:
        return []
    resp = httpx.get(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return sorted({m["id"] for m in resp.json().get("data", []) if m.get("id")})


def _google_ai_studio(api_key: str) -> list[str]:
    if not api_key:
        return []
    resp = httpx.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key, "pageSize": 1000},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    out: set[str] = set()
    for m in resp.json().get("models", []):
        name = m.get("name", "")
        if name.startswith("models/") and "generateContent" in m.get("supportedGenerationMethods", []):
            out.add(name[len("models/") :])
    return sorted(out)


def _bedrock(get) -> list[str]:
    """Claude model ids callable on this AWS account, newest-looking first.

    Bedrock has no single "list models" URL you can curl: the ids live on the
    *control plane* (``bedrock``), not the runtime (``bedrock-runtime``, the
    one that serves Converse), and the requests are SigV4-signed. boto3 is
    bundled for exactly this, so it does the signing.

    Two calls, because either alone gives an incomplete picture:

    * ``list_inference_profiles`` — the cross-region ids (``us.anthropic.…``).
      Most current Claude models are offered *only* this way, and passing the
      bare foundation-model id instead fails at invoke time with "on-demand
      throughput isn't supported".
    * ``list_foundation_models`` — everything else, filtered to Anthropic text
      models that are actually on-demand invokable.

    A corporate Bedrock proxy is not queried: ``ANTHROPIC_BEDROCK_BASE_URL``
    points at someone else's gateway whose catalog shape Loom cannot assume,
    so the picker falls back to the example models there.
    """
    if get("ANTHROPIC_BEDROCK_BASE_URL"):
        return []
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        return []

    region = get("AWS_REGION") or get("AWS_DEFAULT_REGION") or "us-east-1"
    timeout = Config(connect_timeout=TIMEOUT, read_timeout=TIMEOUT, retries={"max_attempts": 1})
    try:
        client = boto3.client("bedrock", region_name=region, config=timeout)
    except Exception:
        return []

    ids: set[str] = set()
    try:
        for page in client.get_paginator("list_inference_profiles").paginate():
            for profile in page.get("inferenceProfileSummaries", []):
                pid = profile.get("inferenceProfileId", "")
                if "anthropic" in pid.lower():
                    ids.add(pid)
    except Exception:
        pass  # permission to list profiles is separate; foundation models may still work
    try:
        resp = client.list_foundation_models(byProvider="anthropic", byOutputModality="TEXT")
        for model in resp.get("modelSummaries", []):
            mid = model.get("modelId", "")
            if mid and "ON_DEMAND" in (model.get("inferenceTypesSupported") or []):
                ids.add(mid)
    except Exception:
        pass
    return sorted(ids, reverse=True)
