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

Withdrawing the tool from the *offered* set is not by itself enforcement, which
is the trap the first version of this guard fell into. ``ToolNode`` still holds
every tool the agent was built with, so a model that emits ``read_file`` anyway —
and models do, having just watched four such calls succeed in the visible
history — got its file. Trusting a model not to call a tool that vanished from
its schema is the same bet as trusting it to follow the prompt. So the budget is
enforced twice: withdrawn at the model call, and refused at the tool call.
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


def _call_field(call: Any, field: str) -> Any:
    return call.get(field) if isinstance(call, dict) else getattr(call, field, None)


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
        self._refused = 0

    # ----- LangChain v1 hooks -----

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._maybe_revoke(request))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return await handler(self._maybe_revoke(request))

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        refusal = self._maybe_refuse(request)
        return refusal if refusal is not None else handler(request)

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        refusal = self._maybe_refuse(request)
        return refusal if refusal is not None else await handler(request)

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

    def _maybe_refuse(self, request: Any) -> Any:
        """A ``ToolMessage`` refusing an over-budget call, or None to let it run.

        Counting stops at *this* call rather than totalling the turn, so a batch
        of parallel reads degrades in order — the ones inside the budget run and
        only the surplus is refused. Totalling would reject the whole batch,
        including the first call, whenever the batch straddled the limit.
        """
        if self.budget < 0:
            return None
        call = getattr(request, "tool_call", None) or getattr(request, "call", None)
        if call is None or _call_field(call, "name") != self.tool:
            return None
        messages = _messages_of(getattr(request, "state", None))
        if self.spent(messages, before_id=_call_field(call, "id")) < self.budget:
            return None
        self._refused += 1
        return self._refusal(_call_field(call, "id"))

    def _refusal(self, tool_call_id: Any) -> Any:
        # Only `explorer` is named: it survives every run mode, and pointing the
        # model at a role that mode-filtering removed would be its own dead end.
        message = (
            f"[read budget] {self.tool} is spent for this turn "
            f"({self.budget} direct call{'s' if self.budget != 1 else ''} allowed). "
            "Delegate the reading: call `task` with the subagent that fits — "
            "`explorer` to find and read a file — and say exactly what you need "
            "back. A subagent's reading does not count against this budget."
        )
        try:
            from langchain_core.messages import ToolMessage

            return ToolMessage(content=message, tool_call_id=str(tool_call_id or "read-budget"))
        except Exception:  # pragma: no cover - langchain always present in practice
            return message

    def spent(self, messages: list[Any], *, before_id: Any = None) -> int:
        """Calls to ``self.tool`` made since the last user message.

        The budget resets every turn: a fresh request from the user is a fresh
        reason to look at a file. Counting from the most recent ``HumanMessage``
        gives that for free and survives compaction, which rewrites older history
        but never drops the current turn.

        ``before_id`` stops the count at a specific tool call, which is what the
        tool-call hook needs: by then the model's message is already in state, so
        a plain total would include the very call being judged.
        """
        used = 0
        for msg in _current_turn(messages):
            for call in getattr(msg, "tool_calls", None) or []:
                if before_id is not None and _call_field(call, "id") == before_id:
                    return used
                if _call_field(call, "name") == self.tool:
                    used += 1
        return used

    @property
    def blocked_count(self) -> int:
        """Model calls that were made with the tool revoked — i.e. how often the
        orchestrator hit its read budget and had to delegate instead."""
        return self._blocked

    @property
    def refused_count(self) -> int:
        """Over-budget calls the model made anyway, and that were rejected. A
        non-zero count means the model tried to read past a tool it could no
        longer see — the reason withdrawing it is not enough on its own."""
        return self._refused


def _current_turn(messages: list[Any]) -> list[Any]:
    """Everything after the most recent user message."""
    start = 0
    for i, msg in enumerate(messages):
        if _is_human(msg):
            start = i + 1
    return list(messages[start:])


def _messages_of(state: Any) -> list[Any]:
    """``messages`` out of whatever shape the graph state arrives in."""
    if state is None:
        return []
    if isinstance(state, dict):
        messages = state.get("messages")
    else:
        messages = getattr(state, "messages", None)
    return list(messages) if messages else []


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
