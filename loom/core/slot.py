"""Process-global mutable slots with a ContextVar-compatible set/get API.

These used to be ``contextvars.ContextVar``s, but LangGraph executes nodes —
and parallel subagent tool calls — in worker threads, where a fresh context
means ``.get()`` silently returned the default: the REPL's confirm callback
fell back to headless auto-deny (tools blocked without ever showing the
approval prompt) and /undo snapshots lost their turn id. A Loom process hosts
exactly one interactive session, so plain process-global state is the correct
scope; the GIL makes the single-reference get/set safe.
"""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Slot(Generic[T]):
    def __init__(self, default: T) -> None:
        self._value = default

    def set(self, value: T) -> T:
        """Set the value; returns the previous value as a reset token
        (ContextVar-style ``token = slot.set(x) … slot.reset(token)``)."""
        prev = self._value
        self._value = value
        return prev

    def reset(self, token: T) -> None:
        self._value = token

    def get(self) -> T:
        return self._value
