"""Delegation reminder: every direct read nudges the orchestrator toward `explorer`.

Loom's whole premise is that the orchestrator plans and routes while subagents
absorb the noisy work. Its search tools are already gone (see
``_orchestrator_excluded_tools``) — browsing a repo is structurally impossible
from the orchestrator's seat. ``read_file`` has to stay: confirming the one path
a subagent named, or re-reading a change it reported, is legitimately the
orchestrator's job.

What that leaves is drift: a strong cloud model holding ``read_file`` drifts
from "confirm this line" into "let me look around", one file at a time, at cloud
prices. Loom used to answer that with a hard per-turn budget that withdrew the
tool. This replaces the cap with a reminder: every ``read_file`` result the
orchestrator gets back ends with one line saying that anything larger than a
targeted check belongs to ``explorer``. The tool never disappears, so a
legitimate run of confirmations is never cut off; the nudge arrives exactly when
the model is reading, which is when the drift happens.

The reminder rides on the tool result rather than the system prompt on purpose.
Tool results are appended at the end of the conversation, so they never touch the
cached prompt prefix; editing or re-appending the system prompt mid-turn would
invalidate the provider's prompt cache on every call.
"""

from __future__ import annotations

from typing import Any, Callable

try:
    from langchain.agents.middleware import AgentMiddleware
except Exception:  # pragma: no cover - allows import without langchain

    class AgentMiddleware:  # type: ignore[no-redef]
        pass


REMINDER = (
    "[loom] Reminder: read_file is for confirming one specific spot. For anything "
    "larger (mapping how something works, tracing a flow, reading several files) "
    "spawn `explorer` with `task` and work from its summary."
)


def _call_field(call: Any, field: str) -> Any:
    return call.get(field) if isinstance(call, dict) else getattr(call, field, None)


class DelegationReminder(AgentMiddleware):
    """Append :data:`REMINDER` to every result of ``tool`` the orchestrator gets.

    Parameters
    ----------
    tool:
        The tool whose results carry the reminder. ``read_file`` in practice.
    """

    def __init__(self, *, tool: str = "read_file") -> None:
        super().__init__()
        self.tool = tool
        self._reminded = 0

    # ----- LangChain v1 hooks -----

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return self._remind(request, handler(request))

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return self._remind(request, await handler(request))

    # ----- implementation -----

    def _remind(self, request: Any, result: Any) -> Any:
        call = getattr(request, "tool_call", None) or getattr(request, "call", None)
        if call is None or _call_field(call, "name") != self.tool:
            return result
        reminded = _with_reminder(result)
        if reminded is not result:
            self._reminded += 1
        return reminded

    @property
    def reminded_count(self) -> int:
        """Direct reads the orchestrator made this session — each one came back
        with the reminder attached."""
        return self._reminded


def _with_reminder(result: Any) -> Any:
    """``result`` with the reminder appended, or ``result`` itself when it is not
    a shape that carries text the model reads (a ``Command``, say)."""
    try:
        from langchain_core.messages import ToolMessage
    except Exception:  # pragma: no cover - langchain always present in practice
        return result
    if not isinstance(result, ToolMessage):
        return result
    content = result.content
    if isinstance(content, str):
        new_content: Any = f"{content}\n\n{REMINDER}" if content else REMINDER
    elif isinstance(content, list):
        new_content = [*content, {"type": "text", "text": REMINDER}]
    else:
        return result
    return result.model_copy(update={"content": new_content})
