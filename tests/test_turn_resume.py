"""A turn that fails partway is resumed, not re-sent.

Found with a scripted model on the real graph: when a model call failed in the
middle of a turn, the retry submitted the user's prompt a second time. On a
persistent thread that appended a duplicate of the prompt *after* the work the
turn had already finished — so the model saw the same request twice, with its
own completed delegation in between, and as often as not did the work again.
"""

import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from loom.core import ollama
from loom.core.config import LoomConfig
from loom.core.ollama import OllamaStatus
from loom.core.settings import Permissions, Settings
from loom.ui.repl import Session

ROLES = ("explorer", "editor", "bash", "searcher", "reviewer", "general-purpose", "tester")

CALLS: list = []
SEEN_BY_ORCHESTRATOR: list = []


class _Scripted(BaseChatModel):
    """Delegates once, fails once, then answers."""

    tag: str = "orch"

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        CALLS.append(self.tag)
        if self.tag == "sub":
            msg = AIMessage(content="report: one file, src/app.py")
        else:
            SEEN_BY_ORCHESTRATOR.append([m.type for m in messages])
            turn = CALLS.count("orch")
            if turn == 1:
                msg = AIMessage(content="", tool_calls=[{"name": "task", "id": "t1",
                                "args": {"description": "map the repo", "subagent_type": "explorer"}}])
            elif turn == 2:
                raise RuntimeError("upstream 500: transient")
            else:
                msg = AIMessage(content="all done")
        msg.usage_metadata = {"input_tokens": 10, "output_tokens": 1, "input_token_details": {}}
        msg.response_metadata = {"model_name": "scripted"}
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr(ollama, "status", lambda cfg: OllamaStatus(True, True, ["qwen3:4b"], "http://x"))
    monkeypatch.setattr("loom.core.orchestrator.build_model", lambda *a, **k: _Scripted(tag="orch"))
    monkeypatch.setattr("loom.subagents.base.build_model", lambda *a, **k: _Scripted(tag="sub"))
    monkeypatch.setattr("loom.core.advisor.build_model", lambda *a, **k: _Scripted(tag="advisor"))
    CALLS.clear()
    SEEN_BY_ORCHESTRATOR.clear()
    settings = Settings(
        models=LoomConfig(orchestrator="claude-sonnet-5", subagents={r: "ollama/qwen3:4b" for r in ROLES}),
        permissions=Permissions(default_mode="allow"),
        mcp_servers={},
    )
    return Session(settings, cwd=str(tmp_path))


@pytest.mark.parametrize("streaming", [True, False])
def test_a_failed_step_is_resumed_without_repeating_the_prompt(session, capsys, streaming):
    session.settings.ui.streaming = streaming

    assert session.run_turn("map it") == "all done"
    assert session.last_turn_failed is False
    out = capsys.readouterr().out
    assert "resuming where it stopped" in out

    # The delegation ran once: the retry picked up at the step that failed.
    assert CALLS == ["orch", "sub", "orch", "orch"]
    # And the model was asked once. Before, its third call saw the prompt twice.
    assert [types.count("human") for types in SEEN_BY_ORCHESTRATOR] == [1, 1, 1]

    state = session.bundle.agent.get_state({"configurable": {"thread_id": session.thread_id}})
    thread = [m.type for m in state.values["messages"]]
    assert thread == ["human", "ai", "tool", "ai"]
    assert not state.next  # nothing left pending


def test_the_next_turn_carries_on_from_a_clean_thread(session):
    session.run_turn("map it")
    CALLS.clear()
    assert session.run_turn("and now?") == "all done"
    state = session.bundle.agent.get_state({"configurable": {"thread_id": session.thread_id}})
    assert [m.type for m in state.values["messages"]].count("human") == 2


def test_the_transcript_survives_the_fleet_being_rebuilt(session, tmp_path, capsys):
    """Found in a terminal run: `/settings …` then `/export` said "nothing to
    export yet" one command after a finished turn. Every /model, /settings and
    mode switch drops the agent until the next turn, and the transcript was
    only ever read through the live one."""
    from loom.ui import slash

    session.run_turn("map it")
    assert [m.type for m in session.transcript()] == ["human", "ai", "tool", "ai"]

    session.rebuild()  # what /model, /settings, /plan, /local and /airgap all do
    assert session.bundle is None
    assert [m.type for m in session.transcript()] == ["human", "ai", "tool", "ai"]

    capsys.readouterr()
    slash.dispatch(session, "/export out.md")
    assert "exported" in capsys.readouterr().out
    exported = (tmp_path / "out.md").read_text()
    assert "map it" in exported and "all done" in exported

    session.reset()  # /clear: a new thread has nothing in it, rebuilt or not
    assert session.transcript() == []
