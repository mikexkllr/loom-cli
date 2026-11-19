"""Strip specific tools from a model request before it reaches the model.

Used to enforce airgap mode: deepagents' ``FilesystemMiddleware`` is required
scaffolding (create_deep_agent always injects it, wiring ``ls``/``read_file``/
``write_file``/``edit_file``/``glob``/``grep``) so the orchestrator can't opt
out of getting a ``read_file`` tool just by omitting it from its own ``tools``
list. This middleware removes named tools at the last mile instead.

Hiding the schema is only half of it. A tool the allowlist could not leave
out — ``read_file``, which ``FilesystemMiddleware`` insists on constructing —
is still registered with the tool node, and a model that names it anyway (the
system prompt mentions it, and every model has seen ``read_file`` before) gets
it executed. So an excluded tool is also refused at the call.
"""

from __future__ import annotations

from typing import Any, Callable

try:
    from langchain.agents.middleware import AgentMiddleware
except Exception:  # pragma: no cover
    class AgentMiddleware:  # type: ignore[no-redef]
        pass


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None
    return getattr(tool, "name", None)


class ToolExclusionMiddleware(AgentMiddleware):
    """Filters ``excluded`` tool names out of every model request."""

    def __init__(self, excluded: set[str] | frozenset[str]) -> None:
        super().__init__()
        self._excluded = frozenset(excluded)

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        if self._excluded:
            request = request.override(tools=[t for t in request.tools if _tool_name(t) not in self._excluded])
        return handler(request)

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        if self._excluded:
            request = request.override(tools=[t for t in request.tools if _tool_name(t) not in self._excluded])
        return await handler(request)

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        refusal = self._refuse(request)
        return refusal if refusal is not None else handler(request)

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        refusal = self._refuse(request)
        return refusal if refusal is not None else await handler(request)

    def _refuse(self, request: Any) -> Any | None:
        """A ToolMessage refusing an excluded tool, or None to let the call run."""
        call = getattr(request, "tool_call", None) or getattr(request, "call", None)
        name = _tool_name(call) if call is not None else None
        if name is None or name not in self._excluded:
            return None
        call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        content = f"[policy] `{name}` is not available to this agent in this run."
        try:
            from langchain_core.messages import ToolMessage

            # "success", like every other policy refusal: the system working
            # as designed is not a tool failure (see PolicyMiddleware._blocked).
            return ToolMessage(content=content, tool_call_id=call_id or "policy", name=name)
        except Exception:  # pragma: no cover - langchain always present in practice
            return content
