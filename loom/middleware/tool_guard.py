"""ToolErrorGuard — a failing tool costs one step, never the turn.

LangGraph's default ``handle_tool_errors`` converts exactly one kind of
exception:

    def _default_handle_tool_errors(e):
        if isinstance(e, ToolInvocationError):   # bad arguments from the model
            return e.message
        raise e

Everything else is re-raised out of the tool node and takes the whole run with
it. deepagents patched that for its *own* filesystem tools (langchain-ai/
deepagents#927, PR #994) by wrapping each one in try/except — but that covers
three call sites in one file. MCP tools, ``web_search``, ``consult``, the
``task`` delegation and anything a user adds are all still on the default, so a
defect in any of them ends the session rather than the step.

This guard sits on the one seam every tool call passes through, so the rule
holds for tools nobody has written yet. It is deliberately **separate from**
:class:`~loom.middleware.policy.PolicyMiddleware` and installed
unconditionally: crash safety is not a permissions feature, and the policy gate
is skipped entirely on the bare-``LoomConfig`` back-compat path — which is
exactly where an unprotected tool crash would be hardest to explain.

It is also installed *outermost*, so it catches a failure in the policy gate
itself (a hook that blows up, a confirm callback that raises) rather than only
failures below it.

Four things can go wrong with a tool call, and they need different people:

``params``  the model called it wrongly — a string where an int belongs.
            Routine from smaller local models; the model is told which argument
            so it can retry. A signal about schema or model size, not a defect.
``policy``  Loom refused. Handled by the policy gate, never seen here.
``failed``  the tool ran and reported failure — file not found, non-zero exit.
``crash``   the tool raised. Someone has to fix it; reported with a stack.
"""

from __future__ import annotations

from typing import Any, Callable

try:
    from langchain.agents.middleware import AgentMiddleware
except Exception:  # pragma: no cover

    class AgentMiddleware:  # type: ignore[no-redef]
        pass


# LangGraph raises these to steer the graph — an `interrupt()` inside a tool, a
# parent Command. They travel as exceptions but they are control flow, and
# turning one into a tool result would silently break human-in-the-loop.
try:
    from langgraph.errors import GraphBubbleUp

    CONTROL_FLOW: tuple[type[BaseException], ...] = (GraphBubbleUp,)
except Exception:  # pragma: no cover - langgraph API drift

    class _NeverRaised(Exception):
        pass

    CONTROL_FLOW = (_NeverRaised,)


def call_identity(request: Any) -> tuple[str, str]:
    """``(tool_name, tool_call_id)`` across the request shapes LangChain uses.

    Kept tolerant on purpose: this runs on the failure path, and a guard that
    raises while handling a failure is worse than the failure.
    """
    call = getattr(request, "call", None) or getattr(request, "tool_call", None)
    if isinstance(call, dict):
        return str(call.get("name") or "?"), str(call.get("id") or "")
    name = getattr(request, "tool_name", None) or getattr(request, "name", None)
    return (str(name) if name else "?"), ""


def leading_error_type(content: Any) -> str:
    """``"FileNotFoundError"`` out of ``"FileNotFoundError: /Users/…/x.py"``.

    A deliberately narrow read: the token must look like a Python exception
    class name before it is sent, so a tool whose error message starts with a
    path or a snippet of the user's code contributes nothing rather than
    leaking its first word. The bare words ``Error``/``Exception`` pass that
    shape test but name nothing, so they read as unknown too.
    """
    head = str(content or "").strip().split(":", 1)[0].strip()
    if not head or len(head) > 60 or not head.isidentifier() or not head[0].isupper():
        return "unknown"
    if head in ("Error", "Exception"):
        return "unknown"
    return head if head.endswith(("Error", "Exception")) else "unknown"


# One report per (tool, category, error class) per process. A coding agent
# guesses paths and greps for things that aren't there — routinely, by design —
# so reporting every error result would bury the one genuine failure under a
# thousand "file not found"s. The class is in the key so that a session's first
# benign miss cannot claim the only slot and hide every later, different
# failure in the same tool. Crashes are never deduped here: they go through
# `report` with a stack, and Sentry groups those itself.
_reported: set[tuple[str, str, str]] = set()


def _reset_for_tests() -> None:
    _reported.clear()


class ToolErrorGuard(AgentMiddleware):
    """Catch what the tool node would let through, and tell Sentry about it."""

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        try:
            result = handler(request)
        except CONTROL_FLOW:
            raise
        except Exception as exc:
            return self._crashed(request, exc)
        self._classify(request, result)
        return result

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        try:
            result = await handler(request)
        except CONTROL_FLOW:
            raise
        except Exception as exc:
            return self._crashed(request, exc)
        self._classify(request, result)
        return result

    # ------------------------------------------------------------------ crash
    def _crashed(self, request: Any, exc: BaseException) -> Any:
        """Report the defect, then hand the model something it can act on.

        The model gets the exception type and message so it can change course.
        Sentry gets the stack; it does not get the message, which routinely
        quotes a path or a line of the user's file.
        """
        name, tool_call_id = call_identity(request)
        try:
            from loom.core import telemetry

            telemetry.report("tool.crash", exc, tool=name, category="bug")
        except Exception:
            pass
        return _tool_message(
            tool_call_id,
            f"[error] `{name}` failed with {type(exc).__name__}: {exc}. This is a fault in "
            "the tool, not in how you called it — try a different approach rather than "
            "repeating the same call.",
            status="error",
        )

    # --------------------------------------------------------------- classify
    def _classify(self, request: Any, result: Any) -> None:
        """A tool that returned failure instead of raising."""
        if str(getattr(result, "status", "")) != "error":
            return
        name, _ = call_identity(request)
        content = str(getattr(result, "content", "") or "")
        # LangGraph's TOOL_INVOCATION_ERROR_TEMPLATE — the model got the
        # arguments wrong, which is a different conversation from a defect.
        params = content.startswith("Error invoking tool ")
        category = "params" if params else "failed"
        error_type = "ToolInvocationError" if params else leading_error_type(content)

        kind = (name, category, error_type)
        if kind in _reported:
            return
        _reported.add(kind)
        try:
            from loom.core import telemetry

            telemetry.capture_message(
                f"tool {category}: {name}",
                where=f"tool.{category}",
                tool=name,
                category=category,
                error_type=error_type,
            )
        except Exception:
            pass


def _tool_message(tool_call_id: str, content: str, status: str = "success") -> Any:
    try:
        from langchain_core.messages import ToolMessage

        return ToolMessage(content=content, tool_call_id=tool_call_id or "tool", status=status)
    except Exception:  # pragma: no cover - langchain API drift
        return content
