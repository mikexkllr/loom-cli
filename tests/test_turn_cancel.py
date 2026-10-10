"""An interrupted turn must not keep working in the background.

In a live run, Ctrl-C ended the stream while a general-purpose subagent kept
going on its worker thread: 23 more model calls, about a million tokens, in the
next five minutes. These tests pin the fix: once a turn is cancelled, no model
or tool call starts under it, subagents included.
"""

import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from loom.core import ollama
from loom.core.cancel import TurnCancel, TurnCancelled
from loom.core.config import LoomConfig
from loom.core.ollama import OllamaStatus
from loom.core.settings import Permissions, Settings

ROLES = ("explorer", "editor", "bash", "searcher", "reviewer", "general-purpose", "tester")


def test_nothing_is_refused_before_cancel():
    c = TurnCancel()
    c.on_chat_model_start({}, [])
    c.on_tool_start({}, "")
    assert not c.cancelled


def test_every_new_call_is_refused_after_cancel():
    c = TurnCancel()
    c.cancel()
    for start in (lambda: c.on_chat_model_start({}, []), lambda: c.on_llm_start({}, []),
                  lambda: c.on_tool_start({}, "")):
        with pytest.raises(TurnCancelled):
            start()


CALLS: list = []


class _Scripted(BaseChatModel):
    """The orchestrator delegates once; the subagent keeps calling `ls` forever,
    like a subagent in the middle of a long investigation."""

    tag: str = "orch"
    on_sub_call: object = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        CALLS.append(self.tag)  # module-level: pydantic copies list fields per instance
        if self.tag == "sub":
            if self.on_sub_call:
                self.on_sub_call()
            msg = AIMessage(content="", tool_calls=[{"name": "ls", "args": {"path": "/"}, "id": f"ls{len(CALLS)}"}])
        elif CALLS.count("orch") == 1:
            msg = AIMessage(content="", tool_calls=[{"name": "task", "id": "t1",
                            "args": {"description": "map the repo", "subagent_type": "explorer"}}])
        else:
            msg = AIMessage(content="done")
        msg.usage_metadata = {"input_tokens": 10, "output_tokens": 1, "input_token_details": {}}
        msg.response_metadata = {"model_name": "scripted"}
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_a_cancelled_turn_stops_its_subagent(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama, "status", lambda cfg: OllamaStatus(True, True, ["qwen3:4b"], "http://x"))
    CALLS.clear()
    log = CALLS
    cancel = TurnCancel()
    tools_run: list = []

    class _Tools(TurnCancel):  # records tool starts that got past the cancel check
        raise_error = False

        def on_tool_start(self, serialized, input_str, **kwargs):
            tools_run.append((serialized or {}).get("name"))

    sub = _Scripted(tag="sub", on_sub_call=cancel.cancel)  # the user presses Ctrl-C now
    monkeypatch.setattr("loom.core.orchestrator.build_model", lambda *a, **k: _Scripted(tag="orch"))
    monkeypatch.setattr("loom.subagents.base.build_model", lambda *a, **k: sub)

    from loom.core.orchestrator import build_orchestrator

    settings = Settings(models=LoomConfig(orchestrator="claude-sonnet-5", subagents={r: "ollama/qwen3:4b" for r in ROLES}),
                        permissions=Permissions(default_mode="allow"))
    bundle = build_orchestrator(settings, cwd=str(tmp_path))
    try:
        bundle.agent.invoke({"messages": [("user", "map it")]},
                            config={"callbacks": [cancel, _Tools()], "recursion_limit": 30})
    except TurnCancelled:
        pass  # the abandoned run may surface it; what matters is what did not run
    assert log.count("sub") == 1, log  # no second subagent model call
    assert "ls" not in tools_run  # the subagent's next tool call never started
    assert log.count("orch") == 1, log  # nor did the orchestrator carry on


def test_run_turn_cancels_its_handler_on_interrupt(tmp_path):
    from types import SimpleNamespace

    from loom.core import settings as st
    from loom.ui.repl import Session

    s = Session(st.Settings(), cwd=str(tmp_path))
    seen: list = []

    class InterruptedAgent:
        def stream(self, inputs, config=None, **kw):
            seen.extend(cb for cb in config["callbacks"] if isinstance(cb, TurnCancel))
            raise KeyboardInterrupt

    s.bundle = SimpleNamespace(agent=InterruptedAgent(), persistent=False, fallbacks={})
    s.run_turn("hello")
    assert len(seen) == 1 and seen[0].cancelled
    assert not any(isinstance(cb, TurnCancel) for cb in s._run_config()["callbacks"])
