"""Stop a turn's background work once the turn is over.

Ctrl-C lands in the main thread, but delegated work does not live there: a
subagent runs inside the ``task`` tool on a LangGraph worker thread, and
abandoning the stream does not stop that thread. In a live run a subagent made
23 more model calls (about a million tokens) in the five minutes after the
interrupt, and it could just as well have kept editing files.

Every model and tool call starts through the callbacks of the run that spawned
it, subagents included, so one handler per turn can refuse them all: once the
turn is cancelled, the next model or tool call that starts under it raises
:class:`TurnCancelled` instead. The call already in flight finishes; nothing
after it begins.
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class TurnCancelled(Exception):
    """Raised into a model or tool call that starts after its turn was cancelled."""


class TurnCancel(BaseCallbackHandler):
    """One per turn. ``cancel()`` is final: a new turn gets a new handler, so
    work left over from an interrupted turn can never be revived by the next."""

    # Refusing the call is the whole point, so the exception must reach it.
    raise_error = True

    def __init__(self) -> None:
        super().__init__()
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def _refuse(self) -> None:
        if self._event.is_set():
            raise TurnCancelled("the turn was interrupted; not starting new work")

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self._refuse()

    def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
        self._refuse()

    def on_tool_start(self, serialized: Any, input_str: str, **kwargs: Any) -> None:
        self._refuse()
