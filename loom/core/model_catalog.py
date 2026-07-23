"""Best-effort live model catalogs for cloud providers.

Some providers expose a plain REST "list models" endpoint that only needs an
API key (or nothing at all) — those get queried directly with ``httpx``.
Bedrock (SigV4-signed requests) and Vertex AI (OAuth via Application Default
Credentials) need their own SDK's credential machinery to list anything, so
Loom doesn't attempt it here. Everywhere dynamic listing isn't possible, or
the request fails/times out/lacks credentials, :func:`available_models` falls
back to :attr:`~loom.core.providers.ProviderInfo.example_models` — the picker
never comes up empty.
"""

from __future__ import annotations

import os

import httpx

from loom.core.providers import ProviderInfo

TIMEOUT = 4.0

# Providers with a plain, key-only (or public) REST listing endpoint.
_LISTABLE = {"anthropic", "openai", "google_ai_studio", "opencode_zen", "opencode_go", "openai_compatible"}

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
    if can_list(provider) and (needs_no_credential(provider) or _has_credentials(provider, env)):
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
