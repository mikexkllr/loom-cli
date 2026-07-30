"""Delegation guard: cap the orchestrator's own file reads per turn.

Loom's whole premise is that the orchestrator plans and routes while subagents
absorb the noisy work. Its search tools are already gone (see
``_orchestrator_excluded_tools``), but ``read_file`` has to stay: confirming the
one path a subagent named, or re-reading a change it reported, is legitimately
the orchestrator's job.

The failure mode that leaves is drift. Given ``read_file`` and a repo, a strong
cloud model will happily read fifteen files "just to be sure" — which is exactly
the context pollution subagents exist to prevent, at cloud prices. Prompt
wording does not hold here; the model reads the rule, agrees with it, and then
reads the fifteen files anyway.

So the budget is structural. After ``budget`` reads in the current user turn,
``read_file`` is removed from the request and the orchestrator has no option but
to delegate. The system prompt states the budget up front, so the tool
disappearing is expected rather than confusing — and because the note lives in
the stable prompt prefix rather than being appended mid-turn, this never
invalidates the provider's prompt cache.
"""

from __future__ import annotations

from typing import Any, Callable

try:
    from langchain.agents.middleware import AgentMiddleware
except Exception:  # pragma: no cover - allows import without langchain

    class AgentMiddleware:  # type: ignore[no-redef]
        pass


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None
    return getattr(tool, "name", None)


class DelegationGuard(AgentMiddleware):
    """Remove ``tool`` from the request once it has been used ``budget`` times
    in the current user turn.

    Parameters
    ----------
    budget:
        How many direct calls the orchestrator gets per turn. ``0`` removes the
        tool outright; a negative value disables the guard.
    tool:
        The tool to meter. ``read_file`` in practice.
    """

    def __init__(self, budget: int, *, tool: str = "read_file") -> None:
        super().__init__()
        self.budget = budget
        self.tool = tool
        self._blocked = 0

    # ----- LangChain v1 hooks -----

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._maybe_revoke(request))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return await handler(self._maybe_revoke(request))

    # ----- implementation -----

    def _maybe_revoke(self, request: Any) -> Any:
        if self.budget < 0:
            return request
        if self.spent(getattr(request, "messages", None) or []) < self.budget:
            return request
        tools = getattr(request, "tools", None)
        if not tools or not any(_tool_name(t) == self.tool for t in tools):
            return request  # already gone; nothing to revoke
        self._blocked += 1
        override = getattr(request, "override", None)
        if not callable(override):  # pragma: no cover - API drift
            return request
        return override(tools=[t for t in tools if _tool_name(t) != self.tool])

    def spent(self, messages: list[Any]) -> int:
        """Calls to ``self.tool`` made since the last user message.

        The budget resets every turn: a fresh request from the user is a fresh
        reason to look at a file. Counting back to the most recent
        ``HumanMessage`` gives that for free and survives compaction, which
        rewrites older history but never drops the current turn.
        """
        used = 0
        for msg in reversed(messages):
            if _is_human(msg):
                break
            for call in getattr(msg, "tool_calls", None) or []:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name == self.tool:
                    used += 1
        return used

    @property
    def blocked_count(self) -> int:
        """Model calls that were made with the tool revoked — i.e. how often the
        orchestrator hit its read budget and had to delegate instead."""
        return self._blocked


def _is_human(msg: Any) -> bool:
    if getattr(msg, "type", None) == "human":
        return True
    # Tuples/dicts appear when a caller seeds state by hand rather than with
    # message objects (Loom's non-persistent path does exactly that).
    if isinstance(msg, tuple) and msg and msg[0] in ("user", "human"):
        return True
    if isinstance(msg, dict) and msg.get("role") in ("user", "human"):
        return True
    return False
