"""PolicyMiddleware — enforce permissions and run hooks around each tool call.

Wraps tool execution (LangChain v1 ``wrap_tool_call``): checks the permission
decision, resolves "ask" via an injectable confirm callback (the REPL supplies
an interactive prompt; headless runs default to deny), runs pre/post hooks, and
blocks the tool cleanly instead of crashing the agent.

Defensive against API drift: if the request object doesn't expose a recognizable
tool name/args, the call passes through untouched.
"""

from __future__ import annotations

import contextvars
from typing import Any, Callable

from loom.core import hooks as hooks_engine
from loom.core import permissions as perm_engine
from loom.core.permissions import Decision
from loom.core.settings import Settings

try:
    from langchain.agents.middleware import AgentMiddleware
except Exception:  # pragma: no cover
    class AgentMiddleware:  # type: ignore[no-redef]
        pass


# Set by the REPL to prompt the user on "ask" decisions. Signature:
#   confirm(tool_name: str, tool_input: dict, reason: str) -> bool
# Default denies (safe for non-interactive/headless runs).
confirm_callback: contextvars.ContextVar[Callable[[str, dict, str], bool]] = contextvars.ContextVar(
    "loom_confirm_callback", default=lambda name, inp, reason: False
)

# When true, "ask" auto-approves (the REPL's /yolo mode, or --yes flag).
auto_approve: contextvars.ContextVar[bool] = contextvars.ContextVar("loom_auto_approve", default=False)


class PolicyMiddleware(AgentMiddleware):
    def __init__(self, settings: Settings, cwd: str = ".") -> None:
        super().__init__()
        self.settings = settings
        self.cwd = cwd

    # --- LangChain v1 hooks (sync + async) ---
    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        gate = self._gate(request)
        if gate is not None:
            return gate  # blocked/denied → short-circuit with a ToolMessage
        result = handler(request)
        self._post(request)
        return result

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        gate = self._gate(request)
        if gate is not None:
            return gate
        result = await handler(request)
        self._post(request)
        return result

    # --- shared gate/post logic ---
    def _gate(self, request: Any) -> Any | None:
        """Returns a ToolMessage to short-circuit, or None to proceed."""
        name, args, _ = self._extract(request)
        if name is None:
            return None

        decision = perm_engine.check(name, args, self.settings.permissions)
        if decision is Decision.deny:
            return self._blocked(request, f"Permission denied for `{name}` by policy.")
        if decision is Decision.ask and not auto_approve.get():
            if not confirm_callback.get()(name, args, "requires approval"):
                return self._blocked(request, f"User declined `{name}`.")

        pre = hooks_engine.pre_tool_use(self.settings.hooks, name, args, self.cwd)
        if pre.blocked:
            return self._blocked(request, f"Blocked by pre_tool_use hook: {pre.block_reason}")
        return None

    def _post(self, request: Any) -> None:
        name, args, _ = self._extract(request)
        if name is not None:
            hooks_engine.post_tool_use(self.settings.hooks, name, args, self.cwd)

    # --- helpers (aligned to ToolCallRequest.call = {name, args, id}) ---
    @staticmethod
    def _extract(request: Any) -> tuple[str | None, dict, str]:
        call = getattr(request, "call", None) or getattr(request, "tool_call", None)
        if isinstance(call, dict):
            args = call.get("args", {}) or {}
            return call.get("name"), args if isinstance(args, dict) else {}, call.get("id", "")
        # Older/alternate shapes.
        name = getattr(request, "tool_name", None) or getattr(request, "name", None)
        args = getattr(request, "args", None) or getattr(request, "tool_input", None) or {}
        if name:
            return name, args if isinstance(args, dict) else {}, ""
        return None, {}, ""

    def _blocked(self, request: Any, message: str) -> Any:
        _, _, tool_call_id = self._extract(request)
        try:
            from langchain_core.messages import ToolMessage

            return ToolMessage(content=f"[policy] {message}", tool_call_id=tool_call_id or "policy")
        except Exception:
            return f"[policy] {message}"
