"""Strip specific tools from a model request before it reaches the model.

Used to enforce airgap mode: deepagents' ``FilesystemMiddleware`` is required
scaffolding (create_deep_agent always injects it, wiring ``ls``/``read_file``/
``write_file``/``edit_file``/``glob``/``grep``) so the orchestrator can't opt
out of getting a ``read_file`` tool just by omitting it from its own ``tools``
list. This middleware removes named tools at the last mile instead.
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
