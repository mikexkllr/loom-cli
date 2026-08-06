"""The three ways a tool call goes wrong, driven through a real tool node.

A tool call fails in one of three ways, and they want different treatment:

1. **params** — the model called it wrongly (a string where an int belongs).
   Routine from smaller local models. The model must be told so it can retry.
2. **policy** — Loom refused: a deny rule, a declined prompt, a blocking hook.
   Working as designed. The model gets the reason and picks another route.
3. **bug** — the tool itself raised. Someone has to fix it.

All three must cost one step, never the turn, and all three must be
distinguishable afterwards. This exercises the real `ToolNode` wrapper path
rather than a stand-in handler, because the thing that was broken lived in the
seam: LangGraph's default `handle_tool_errors` converts only case 1, and case 3
escaped the graph and killed the run.
"""

from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")
pytest.importorskip("pydantic")

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode

from loom.core import settings as st
from loom.core import telemetry
from loom.middleware import policy
from loom.middleware.policy import PolicyMiddleware


@tool
def crashy(path: str, count: int) -> str:
    """Raises the way a genuine defect would."""
    raise ZeroDivisionError("a real bug inside the tool")


class _State(TypedDict):
    messages: Annotated[list, add_messages]


@pytest.fixture
def reports(monkeypatch):
    """Collect what would have gone to Sentry, and reset the dedupe set."""
    sent: list[tuple] = []
    monkeypatch.setattr(
        telemetry, "report", lambda where, exc, **tags: sent.append((where, type(exc).__name__, tags))
    )
    monkeypatch.setattr(
        telemetry, "capture_message", lambda msg, where="", **tags: sent.append((where, msg, tags))
    )
    policy._reported_tool_errors.clear()
    return sent


def _agent(settings=None, cwd="."):
    middleware = PolicyMiddleware(
        settings or st.Settings(permissions=st.Permissions(default_mode="allow")), cwd=cwd
    )
    graph = StateGraph(_State)
    graph.add_node("tools", ToolNode([crashy], wrap_tool_call=middleware.wrap_tool_call))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile()


def _call(agent, **args):
    message = AIMessage(
        content="", tool_calls=[{"name": "crashy", "args": args, "id": "1", "type": "tool_call"}]
    )
    return agent.invoke({"messages": [message]})["messages"][-1]


def test_wrong_parameter_types_come_back_to_the_model(reports):
    """Case 1. A small model handing a string to an int argument is a fact of
    life, not a defect — it has to be told, and it has to be able to retry."""
    result = _call(_agent(), path="x.py", count="not-an-int")

    assert result.status == "error"
    assert "count" in str(result.content)  # names the argument it got wrong
    assert reports == [
        ("tool.params", "tool params: crashy", {"tool": "crashy", "category": "params", "error_type": "ToolInvocationError"})
    ]


def test_a_tool_that_raises_does_not_end_the_turn(reports):
    """Case 3, and the one that was broken. LangGraph's default handler
    converts only `ToolInvocationError`; anything else propagates out of the
    tool node and ends the run."""
    result = _call(_agent(), path="x.py", count=1)

    assert result.status == "error"
    assert "ZeroDivisionError" in str(result.content)
    assert (
        "tool.crash",
        "ZeroDivisionError",
        {"tool": "crashy", "category": "bug"},
    ) in reports


def test_a_policy_denial_is_not_reported_as_a_bug(reports):
    """Case 2. A refusal is the system working; putting every "no" the user
    says into the crash stream would bury the failures that are real."""
    settings = st.Settings(permissions=st.Permissions(default_mode="deny"))
    result = _call(_agent(settings), path="x.py", count=1)

    assert "[policy]" in str(result.content)
    assert result.status == "success", "a denial is not an error the model should treat as a fault"
    assert reports == []


def test_all_three_are_distinguishable_in_sentry(reports):
    """The point of tagging: 'a small model fumbles arguments' and 'our tool
    crashes' must not land in the same bucket."""
    agent = _agent()
    _call(agent, path="x.py", count="not-an-int")
    _call(agent, path="x.py", count=1)

    categories = {tags.get("category") for _where, _what, tags in reports}
    assert categories == {"params", "bug"}


def test_a_benign_failure_does_not_mask_a_later_different_one(reports, tmp_path):
    """The dedupe key includes the category and the exception class. Keyed on
    the tool alone, the session's first 'file not found' would have claimed the
    only slot and hidden every other failure in that tool."""
    from types import SimpleNamespace

    middleware = PolicyMiddleware(
        st.Settings(permissions=st.Permissions(default_mode="allow")), cwd=str(tmp_path)
    )

    def call(content):
        middleware.wrap_tool_call(
            SimpleNamespace(call={"name": "read_file", "args": {}, "id": "x"}),
            lambda _r: SimpleNamespace(status="error", content=content),
        )

    call("FileNotFoundError: nope.py")
    call("FileNotFoundError: also-nope.py")  # same class → collapsed
    call("PermissionError: locked.py")  # different class → reported

    kinds = [tags["error_type"] for _where, _msg, tags in reports]
    assert kinds == ["FileNotFoundError", "PermissionError"]
