"""What the local Ollama daemon can actually do this session — and how to keep
work on it.

Loom's cost/privacy pitch only pays off when local models handle the work they
are capable of. Three places used to leak to the cloud even with a healthy
Ollama:

* a role whose *specific* tag wasn't pulled went to ``cloud_fallback``, even
  when another perfectly good local model was sitting there served;
* an oversized prompt on a small model escalated straight to
  ``escalation_model``, skipping the larger local model right next to it;
* a local model with no ``context_windows`` entry was assumed to hold 32K, so
  it escalated on prompts it could have held — most local coders ship far
  bigger windows than that.

This module answers those three with one probe of the daemon: :func:`build_pool`
asks Ollama once what's installed, and everything else here is pure logic over
that snapshot. Cloud stays the last resort, not the first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loom.core.config import LoomConfig
from loom.core.model_router import resolve

if TYPE_CHECKING:
    from loom.core.ollama import OllamaStatus


@dataclass(frozen=True)
class LocalPool:
    """Snapshot of the local models available this session.

    Built once per orchestrator assembly so a single ``/api/tags`` round-trip
    serves the fallback planner, the context-window detector, and the
    escalation ladder.
    """

    running: bool
    served: frozenset[str]  # installed Ollama tags, as reported by the daemon
    endpoint: str

    def serves(self, model_string: str) -> bool:
        """True if ``model_string`` is a local model this daemon can serve now."""
        from loom.core import ollama

        if not self.running:
            return False
        rm = resolve(model_string)
        return rm.is_local and ollama.is_served(rm.name, self.served)

    @classmethod
    def offline(cls, endpoint: str = "") -> "LocalPool":
        return cls(running=False, served=frozenset(), endpoint=endpoint)


def build_pool(config: LoomConfig, status: "OllamaStatus | None" = None) -> LocalPool:
    """Probe the daemon once and snapshot what it serves.

    Never raises: an unreachable daemon is an empty pool, which every consumer
    here already treats as "nothing local is available".
    """
    from loom.core import ollama

    if status is None:
        try:
            status = ollama.status(config)
        except Exception:
            return LocalPool.offline(config.ollama_endpoint)
    return LocalPool(
        running=status.running,
        served=frozenset(status.models),
        endpoint=status.endpoint,
    )


# ----------------------------------------------------------------------------
# Context windows: ask the model, don't guess
# ----------------------------------------------------------------------------


def detect_context_windows(config: LoomConfig, pool: LocalPool) -> LoomConfig:
    """Fill in ``context_windows`` for served local models that have no entry.

    Without this, an unlisted local model falls back to
    ``context_window_for``'s conservative 32K default — which both shrinks the
    ``num_ctx`` Loom asks Ollama for and makes the prompt-size guard escalate to
    the cloud on prompts the model could have held. Ollama reports the real
    trained context length via ``/api/show``, so use it.

    Configured entries always win (an explicit window is a deliberate
    memory-budget decision), and detected values are capped so a 256K-context
    model doesn't get a KV cache this machine can't allocate — at
    ``max_local_context`` when it's set, otherwise at whatever this box's
    GPU/unified memory can carry (see
    :func:`loom.core.recommendations.context_budget`).
    """
    from loom.core import ollama
    from loom.core.recommendations import auto_context_budget

    missing = [
        m
        for m in dict.fromkeys(config.all_models().values())
        if config.is_local(m) and m not in config.context_windows and pool.serves(m)
    ]
    if not missing:
        return config

    ceiling = config.max_local_context or auto_context_budget()
    detected: dict[str, int] = {}
    for model in missing:
        length = ollama.context_length(resolve(model).name, pool.endpoint)
        if length:
            detected[model] = min(length, ceiling)
    if not detected:
        return config
    return config.model_copy(update={"context_windows": {**config.context_windows, **detected}})


# ----------------------------------------------------------------------------
# Role substitution: another local model beats any cloud model
# ----------------------------------------------------------------------------


def local_candidates(config: LoomConfig, pool: LocalPool) -> list[str]:
    """Distinct config-declared local models the daemon serves, small window first.

    Deliberately limited to models the config already names: substituting an
    arbitrary tag that happens to be installed (someone's embedding model, a
    vision model) would be unpredictable, and Loom would have no idea what it
    is good at.
    """
    seen = [m for m in dict.fromkeys(config.all_models().values()) if pool.serves(m)]
    return sorted(seen, key=lambda m: config.context_window_for(m))


def substitute(missing: str, config: LoomConfig, pool: LocalPool) -> str | None:
    """Pick the served local model that best stands in for ``missing``.

    Context window is the capability proxy Loom already tracks per model, so:
    prefer the *smallest* served model that is at least as roomy as the one
    that's missing (cheapest adequate replacement); if nothing reaches that
    bar, take the largest available rather than giving up on local. Returns
    None only when the daemon serves nothing usable.
    """
    candidates = [m for m in local_candidates(config, pool) if m != missing]
    if not candidates:
        return None
    want = config.context_window_for(missing)
    for model in candidates:  # ascending window
        if config.context_window_for(model) >= want:
            return model
    return candidates[-1]


@dataclass
class RolePlan:
    """Outcome of resolving every configured role against the live daemon."""

    config: LoomConfig
    # role -> the local model it was configured with, replaced by another
    # *local* model that is actually served. No cloud call, no cost.
    substituted: dict[str, str]
    # role -> the local model it was configured with, now running on the
    # billed cloud fallback because nothing local could serve it.
    cloud: dict[str, str]

    @property
    def changed(self) -> bool:
        return bool(self.substituted or self.cloud)


def plan_local_roles(
    config: LoomConfig,
    pool: LocalPool,
    *,
    allow_cloud: bool = True,
) -> RolePlan:
    """Resolve local roles the daemon can't serve, preferring local over cloud.

    For each role (the subagents plus a local orchestrator or advisor) whose
    model isn't served: swap in another served local model if there is one, and
    only otherwise fall back to ``config.cloud_fallback``. With ``allow_cloud=False``
    (local-only / airgap, where a cloud call is not on the table) an
    unsubstitutable role is left as configured — the caller has already
    verified the daemon is up, and a clear failure on that one role beats
    silently breaking the mode's guarantee.
    """
    roles: dict[str, str] = {
        role: model for role, model in config.subagents.items() if config.is_local(model)
    }
    # orchestrator and advisor live in their own config fields rather than the
    # subagents map, but they need the same treatment — the advisor doubly so,
    # since an unassigned reviewer role inherits its model.
    top_level = {
        role: model
        for role, model in (("orchestrator", config.orchestrator), ("advisor", config.advisor))
        if config.is_local(model)
    }
    if not roles and not top_level:
        return RolePlan(config, {}, {})

    substituted: dict[str, str] = {}
    cloud: dict[str, str] = {}
    subagents = dict(config.subagents)
    update: dict[str, object] = {}

    def _resolve(role: str, model: str) -> str | None:
        """Replacement for ``role``'s model, or None to leave it as configured."""
        if pool.serves(model):
            return None
        replacement = substitute(model, config, pool)
        if replacement is None:
            replacement = config.cloud_fallback if allow_cloud else None
        if replacement is None:
            return None
        (substituted if config.is_local(replacement) else cloud)[role] = model
        return replacement

    for role, model in roles.items():
        replacement = _resolve(role, model)
        if replacement is not None:
            subagents[role] = replacement
    if subagents != config.subagents:
        update["subagents"] = subagents

    for role, model in top_level.items():
        replacement = _resolve(role, model)
        if replacement is not None:
            update[role] = replacement

    if not substituted and not cloud:
        return RolePlan(config, {}, {})
    return RolePlan(config.model_copy(update=update), substituted, cloud)


# ----------------------------------------------------------------------------
# Escalation ladder: climb locally before reaching for the cloud
# ----------------------------------------------------------------------------


def escalation_ladder(config: LoomConfig, pool: LocalPool) -> tuple[tuple[str, int], ...]:
    """Served local models as ``(model_string, context_window)``, smallest first.

    Handed to each local subagent's :class:`~loom.middleware.prompt_size_guard.
    PromptSizeGuard` so an oversized prompt can climb to a roomier *local*
    model before it costs anything.
    """
    return tuple((m, config.context_window_for(m)) for m in local_candidates(config, pool))


def local_escalation(
    prompt_tokens: int,
    current: str,
    config: LoomConfig,
    ladder: tuple[tuple[str, int], ...],
) -> str | None:
    """Smallest local model on ``ladder`` that still fits ``prompt_tokens``.

    "Fits" uses the same ``escalation_threshold`` the guard escalates on, so a
    model is only chosen if the prompt would not immediately re-escalate off
    it. None means no local model has the headroom — that's when the cloud
    escalation model earns its call.
    """
    for model, window in ladder:  # ascending window
        if model == current:
            continue
        if prompt_tokens < window * config.escalation_threshold:
            return model
    return None
