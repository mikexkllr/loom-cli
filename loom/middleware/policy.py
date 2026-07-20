"""PolicyMiddleware — enforce permissions and run hooks around each tool call.

Wraps tool execution (LangChain v1 ``wrap_tool_call``): checks the permission
decision, resolves "ask" via an injectable confirm callback (the REPL supplies
an interactive prompt; headless runs default to deny), runs pre/post hooks, and
blocks the tool cleanly instead of crashing the agent.

Defensive against API drift: if the request object doesn't expose a recognizable
tool name/args, the call passes through untouched.
"""

from __future__ import annotations

from typing import Any, Callable

from loom.core import hooks as hooks_engine
from loom.core import permissions as perm_engine
from loom.core.permissions import Decision
from loom.core.settings import Settings
from loom.core.slot import Slot
from loom.tools.sandbox import resolve_in_sandbox

try:
    from langchain.agents.middleware import AgentMiddleware
except Exception:  # pragma: no cover

    class AgentMiddleware:  # type: ignore[no-redef]
        pass


# Set by the REPL to prompt the user on "ask" decisions. Signature:
#   confirm(tool_name: str, tool_input: dict, reason: str) -> bool | (bool, str)
# The tuple form carries decline feedback ("what to do instead"), which is
# routed back to the model in the blocking ToolMessage so it can adjust
# course. Default denies (safe for non-interactive/headless runs).
# NOTE: these are process-global Slots, NOT contextvars — LangGraph runs tool
# calls in worker threads, where a contextvar reverts to its default and the
# approval prompt would silently never appear (auto-deny).
confirm_callback: Slot[Callable[[str, dict, str], "bool | tuple"]] = Slot(lambda name, inp, reason: False)


def _normalize_confirm(result: "bool | tuple") -> tuple[bool, str]:
    """(approved, feedback) from either a bare bool or a (bool, str) tuple."""
    if isinstance(result, tuple):
        approved = bool(result[0]) if result else False
        feedback = str(result[1]) if len(result) > 1 and result[1] else ""
        return approved, feedback
    return bool(result), ""

# When true, "ask" auto-approves (the REPL's /yolo mode, or --yes flag).
auto_approve: Slot[bool] = Slot(False)

# Claude Code-style "accept edits" mode: file writes auto-approve, everything
# else (shell, etc.) still asks.
auto_approve_edits: Slot[bool] = Slot(False)

_EDIT_TOOLS = {"write_file", "edit_file"}


def _normalize_path_arg(args: dict, cwd: str) -> dict:
    """Normalize the path argument for permission, /undo, and diff preview hooks.

    Both deepagents' built-in ``FilesystemMiddleware`` tools and Loom's custom
    filesystem tools use ``file_path`` and absolute virtual paths such as
    ``/cs_ai_quiz/quiz.py``. We resolve those against the sandbox root and add a
    ``path`` key relative to the current working directory, keeping both keys in
    ``args`` so the tool call still receives ``file_path`` while Loom's
    permission specifiers (e.g. ``write_file(secrets/**)``), /undo snapshots, and
    diff previews all work.
    """
    if "path" not in args and "file_path" in args:
        args = {**args, "path": args["file_path"]}
    path = args.get("path")
    if path is None:
        return args
    from pathlib import Path

    try:
        target = resolve_in_sandbox(str(path))
        rel = str(target.resolve().relative_to(Path(cwd).resolve()))
        args = {**args, "path": rel}
    except Exception:
        # Best-effort: if we can't resolve, at least strip a leading ``/`` so
        # permission globs that expect relative paths (``src/**``) still match
        # virtual absolute paths (``/src/foo.py``).
        p = str(path)
        if p.startswith("/"):
            args = {**args, "path": p.lstrip("/")}
    return args


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
        self._snapshot(request)
        result = handler(request)
        self._post(request)
        return result

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        gate = self._gate(request)
        if gate is not None:
            return gate
        self._snapshot(request)
        result = await handler(request)
        self._post(request)
        return result

    def _snapshot(self, request: Any) -> None:
        """Record the pre-write file state so /undo can roll the turn back.

        ``delete`` is snapshotted too (single files restore; directory deletes
        are best-effort — the copy fails silently and /undo skips them).
        """
        name, args, _ = self._extract(request)
        if name in ("write_file", "edit_file", "delete") and args.get("path"):
            try:
                from loom.core import undo

                undo.snapshot(self.cwd, args["path"])
            except Exception:
                pass  # snapshots are best-effort; never block the write

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
            if not (name in _EDIT_TOOLS and auto_approve_edits.get()):
                approved, feedback = _normalize_confirm(confirm_callback.get()(name, args, "requires approval"))
                if not approved:
                    msg = f"User declined `{name}`."
                    if feedback:
                        msg += f" The user says to do this instead: {feedback}"
                    return self._blocked(request, msg)

        pre = hooks_engine.pre_tool_use(self.settings.hooks, name, args, self.cwd)
        if pre.blocked:
            return self._blocked(request, f"Blocked by pre_tool_use hook: {pre.block_reason}")
        return None

    def _post(self, request: Any) -> None:
        name, args, _ = self._extract(request)
        if name is not None:
            hooks_engine.post_tool_use(self.settings.hooks, name, args, self.cwd)

    # --- helpers (aligned to ToolCallRequest.call = {name, args, id}) ---
    def _extract(self, request: Any) -> tuple[str | None, dict, str]:
        call = getattr(request, "call", None) or getattr(request, "tool_call", None)
        if isinstance(call, dict):
            args = call.get("args", {}) or {}
            args = args if isinstance(args, dict) else {}
            return call.get("name"), _normalize_path_arg(args, self.cwd), call.get("id", "")
        # Older/alternate shapes.
        name = getattr(request, "tool_name", None) or getattr(request, "name", None)
        args = getattr(request, "args", None) or getattr(request, "tool_input", None) or {}
        if name:
            return name, _normalize_path_arg(args if isinstance(args, dict) else {}, self.cwd), ""
        return None, {}, ""

    def _blocked(self, request: Any, message: str) -> Any:
        _, _, tool_call_id = self._extract(request)
        try:
            from langchain_core.messages import ToolMessage

            return ToolMessage(content=f"[policy] {message}", tool_call_id=tool_call_id or "policy")
        except Exception:
            return f"[policy] {message}"
