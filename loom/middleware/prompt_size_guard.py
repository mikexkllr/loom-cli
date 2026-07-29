"""Prompt-size guard: auto-escalate a single model call when a local model's
prompt approaches its context window (build step: middleware).

Escalation is a two-rung ladder — a roomier *local* model first, the cloud only
when nothing local has the headroom. Overflowing a 4B model's window is a
capacity problem, not a difficulty one, so paying for a cloud call while a
larger Ollama model sits served and idle is pure waste.

Implemented as a LangChain ``AgentMiddleware`` using the ``wrap_model_call``
hook, which lets us inspect the outgoing request and, if needed, swap the model
for just that call — without failing the subagent or losing its transcript.

The hook signature follows LangChain v1's middleware protocol. We keep the
implementation defensive: if the request object doesn't expose what we expect,
we pass through untouched rather than break the run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from loom.core.config import LoomConfig
from loom.core.local_pool import local_escalation
from loom.core.model_router import build_model, estimate_tokens, should_escalate

try:  # LangChain v1 middleware base
    from langchain.agents.middleware import AgentMiddleware
except Exception:  # pragma: no cover - allows import without langchain installed

    class AgentMiddleware:  # type: ignore[no-redef]
        """Fallback shim so the module imports without langchain present."""


if TYPE_CHECKING:
    from langchain.agents.middleware import ModelRequest, ModelResponse


class PromptSizeGuard(AgentMiddleware):
    """Escalate an oversized local-model call — locally first, cloud last.

    A 4B explorer overflowing its window does not mean the work needs a cloud
    model; it usually means it needs a *roomier* model, and on a hybrid box
    there is often one already loaded. So the guard climbs the local ladder
    first and only spends a cloud call when nothing local has the headroom.

    Parameters
    ----------
    local_model:
        The config model string this subagent normally runs on (e.g.
        ``ollama/qwen3:4b``). Only local models are ever escalated.
    config:
        The active Loom config — supplies window sizes, threshold, and the
        escalation target model.
    ladder:
        ``(model_string, context_window)`` for every local model the daemon
        actually serves, smallest window first (see
        :func:`loom.core.local_pool.escalation_ladder`). Empty means no local
        headroom is known, which restores the plain local -> cloud behavior.
    """

    def __init__(
        self,
        local_model: str,
        config: LoomConfig,
        ladder: tuple[tuple[str, int], ...] = (),
    ) -> None:
        super().__init__()
        self.local_model = local_model
        self.config = config
        self.ladder = ladder
        self._escalations = 0
        self._local_escalations = 0

    # LangChain v1 hook: wrap a single model invocation (sync + async).
    def wrap_model_call(
        self,
        request: "ModelRequest",
        handler: Callable[["ModelRequest"], "ModelResponse"],
    ) -> "ModelResponse":
        return handler(self._maybe_escalate(request))

    async def awrap_model_call(
        self,
        request: "ModelRequest",
        handler: Callable[["ModelRequest"], "ModelResponse"],
    ) -> "ModelResponse":
        return await handler(self._maybe_escalate(request))

    def _maybe_escalate(self, request: "ModelRequest") -> "ModelRequest":
        if not self.config.is_local(self.local_model):
            return request
        prompt_tokens = self._estimate_request_tokens(request)
        if not should_escalate(prompt_tokens, self.local_model, self.config):
            return request

        # Rung 1: a bigger local model that still fits the prompt. Free, private,
        # and no context leaves the machine.
        target = local_escalation(prompt_tokens, self.local_model, self.config, self.ladder)
        if target is not None:
            swapped = self._swap(request, target)
            if swapped is not None:
                self._local_escalations += 1
                return swapped

        # Rung 2: the cloud. Only once no local model has the headroom.
        self._escalations += 1
        swapped = self._swap(request, self.config.escalation_model)
        # Can't build the cloud model (e.g. missing API key) → let the local
        # model try rather than failing the whole call.
        return swapped if swapped is not None else request

    def _swap(self, request: "ModelRequest", model_string: str) -> "ModelRequest | None":
        """Rebind ``request`` to ``model_string``, or None if it can't be built."""
        try:
            return self._with_model(request, build_model(model_string, self.config))
        except Exception:
            return None

    # ----- helpers (defensive against API drift) -----

    @staticmethod
    def _estimate_request_tokens(request: Any) -> int:
        messages = getattr(request, "messages", None) or []
        system = getattr(request, "system_prompt", "") or ""
        total = estimate_tokens(str(system))
        for msg in messages:
            content = getattr(msg, "content", msg)
            total += estimate_tokens(str(content))
        return total

    @staticmethod
    def _with_model(request: Any, model: Any) -> Any:
        # v1 ModelRequest exposes .override(...); fall back to attribute set.
        override = getattr(request, "override", None)
        if callable(override):
            return override(model=model)
        try:
            request.model = model
        except Exception:
            pass
        return request

    @property
    def escalation_count(self) -> int:
        """Calls that had to leave the machine for the cloud escalation model."""
        return self._escalations

    @property
    def local_escalation_count(self) -> int:
        """Calls kept local by climbing to a roomier Ollama model instead."""
        return self._local_escalations
