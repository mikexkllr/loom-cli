"""Prompt-size guard: auto-escalate a single model call to the cloud when a
local model's prompt approaches its context window (build step: middleware).

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
from loom.core.model_router import build_model, estimate_tokens, should_escalate

try:  # LangChain v1 middleware base
    from langchain.agents.middleware import AgentMiddleware
except Exception:  # pragma: no cover - allows import without langchain installed
    class AgentMiddleware:  # type: ignore[no-redef]
        """Fallback shim so the module imports without langchain present."""

if TYPE_CHECKING:
    from langchain.agents.middleware import ModelRequest, ModelResponse


class PromptSizeGuard(AgentMiddleware):
    """Escalate oversized local-model calls to a cloud fallback.

    Parameters
    ----------
    local_model:
        The config model string this subagent normally runs on (e.g.
        ``ollama/qwen3:4b``). Only local models are ever escalated.
    config:
        The active Loom config — supplies window sizes, threshold, and the
        escalation target model.
    """

    def __init__(self, local_model: str, config: LoomConfig) -> None:
        super().__init__()
        self.local_model = local_model
        self.config = config
        self._escalations = 0

    # LangChain v1 hook: wrap a single model invocation.
    def wrap_model_call(
        self,
        request: "ModelRequest",
        handler: Callable[["ModelRequest"], "ModelResponse"],
    ) -> "ModelResponse":
        if not self.config.is_local(self.local_model):
            return handler(request)

        prompt_tokens = self._estimate_request_tokens(request)
        if should_escalate(prompt_tokens, self.local_model, self.config):
            self._escalations += 1
            target = self.config.escalation_model
            try:
                escalated = build_model(target, self.config)
                request = self._with_model(request, escalated)
            except Exception:
                # If we can't build the cloud model (e.g. missing API key),
                # fall back to the local model and let it try.
                pass
        return handler(request)

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
        return self._escalations
