"""The ways a tool call fails, driven through a real tool node.

A tool call fails in one of four ways, and they want different treatment:

1. **params** — the model called it wrongly (a string where an int belongs).
   Routine from smaller local models. The model must be told so it can retry.
2. **policy** — Loom refused: a deny rule, a declined prompt, a blocking hook.
   Working as designed. The model gets the reason and picks another route.
3. **failed** — the tool ran and reported failure. Usually the model's aim.
4. **crash** — the tool itself raised. Someone has to fix it.

All four must cost one step, never the turn, and all four must be
distinguishable afterwards. These drive the real `ToolNode` wrapper path rather
than a stand-in handler, because the thing that was broken lived in that seam:
LangGraph's default `handle_tool_errors` converts only case 1, and case 4
escaped the compiled graph and ended the run.
"""

from types import SimpleNamespace
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
from loom.middleware import tool_guard
from loom.middleware.policy import PolicyMiddleware
from loom.middleware.tool_guard import ToolErrorGuard


@tool
def crashy(path: str, count: int) -> str:
    """Raises the way a genuine defect would."""
    raise ZeroDivisionError("a real bug inside the tool")


class _State(TypedDict):
    messages: Annotated[list, add_messages]


@pytest.fixture(autouse=True)
def reports(monkeypatch):
    """Collect what would have gone to Sentry, and reset the dedupe set."""
    sent: list[tuple] = []
    monkeypatch.setattr(
        telemetry, "report", lambda where, exc, **tags: sent.append((where, type(exc).__name__, tags))
    )
    monkeypatch.setattr(
        telemetry, "capture_message", lambda msg, where="", **tags: sent.append((where, msg, tags))
    )
    tool_guard._reset_for_tests()
    return sent


def _agent(settings=None):
    """The production stack order: the guard outermost, the policy gate under it."""
    middleware = [ToolErrorGuard()]
    if settings is not None:
        middleware.append(PolicyMiddleware(settings, cwd="."))

    def wrap(request, handler):
        def chain(index):
            if index == len(middleware):
                return handler
            return lambda req: middleware[index].wrap_tool_call(req, chain(index + 1))

        return chain(0)(request)

    graph = StateGraph(_State)
    graph.add_node("tools", ToolNode([crashy], wrap_tool_call=wrap))
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
        (
            "tool.params",
            "tool params: crashy",
            {"tool": "crashy", "category": "params", "error_type": "ToolInvocationError"},
        )
    ]


def test_a_tool_that_raises_does_not_end_the_turn(reports):
    """Case 4, and the one that was broken. LangGraph's default handler
    converts only `ToolInvocationError`; anything else propagates out of the
    tool node and ends the run."""
    result = _call(_agent(), path="x.py", count=1)

    assert result.status == "error"
    assert "ZeroDivisionError" in str(result.content)
    assert not str(result.content).startswith("[policy]"), "a crash is not a policy decision"
    assert ("tool.crash", "ZeroDivisionError", {"tool": "crashy", "category": "bug"}) in reports


def test_the_guard_holds_without_any_settings(reports):
    """The gap this middleware exists to close.

    The crash net used to live inside `PolicyMiddleware`, which is only
    installed when `Settings` are wired — so on the bare-`LoomConfig`
    back-compat path a tool defect still killed the run. Crash safety is not a
    permissions feature.
    """
    result = _call(_agent(settings=None), path="x.py", count=1)

    assert result.status == "error"
    assert "ZeroDivisionError" in str(result.content)
    assert any(where == "tool.crash" for where, _what, _tags in reports)


def test_a_policy_denial_is_not_reported_as_a_bug(reports):
    """Case 2. A refusal is the system working; putting every "no" the user
    says into the crash stream would bury the failures that are real."""
    settings = st.Settings(permissions=st.Permissions(default_mode="deny"))
    result = _call(_agent(settings), path="x.py", count=1)

    assert "[policy]" in str(result.content)
    assert result.status == "success", "a denial is not a fault the model should treat as one"
    assert reports == []


def test_control_flow_signals_are_not_tool_failures():
    """An `interrupt()` inside a tool travels as an exception but is control
    flow — converting one into a tool result would break human-in-the-loop."""
    from langgraph.errors import GraphBubbleUp

    guard = ToolErrorGuard()

    def handler(_req):
        raise GraphBubbleUp()

    with pytest.raises(GraphBubbleUp):
        guard.wrap_tool_call(SimpleNamespace(call={"name": "grep", "args": {}, "id": "x"}), handler)


def test_all_the_cases_are_distinguishable_in_sentry(reports):
    """The point of tagging: 'a small model fumbles arguments' and 'our tool
    crashes' must not land in the same bucket."""
    agent = _agent()
    _call(agent, path="x.py", count="not-an-int")
    _call(agent, path="x.py", count=1)

    assert {tags.get("category") for _where, _what, tags in reports} == {"params", "bug"}


def test_a_failure_result_is_reported_without_its_body(reports):
    """Tools usually *return* their failure rather than raising. Report it —
    but send only the tool name, the category and the exception class, never
    the message, which routinely quotes a path or a line of the user's file."""
    guard = ToolErrorGuard()
    guard.wrap_tool_call(
        SimpleNamespace(call={"name": "read_file", "args": {}, "id": "x"}),
        lambda _r: SimpleNamespace(status="error", content="FileNotFoundError: /Users/me/secret.py"),
    )
    where, msg, tags = reports[0]
    assert (where, msg) == ("tool.failed", "tool failed: read_file")
    assert tags == {"tool": "read_file", "category": "failed", "error_type": "FileNotFoundError"}
    assert "secret.py" not in str(reports)


def test_a_benign_failure_does_not_mask_a_later_different_one(reports):
    """The dedupe key includes the category and the exception class. Keyed on
    the tool alone, the session's first 'file not found' would have claimed the
    only slot and hidden every other failure in that tool."""
    guard = ToolErrorGuard()

    def call(content):
        guard.wrap_tool_call(
            SimpleNamespace(call={"name": "read_file", "args": {}, "id": "x"}),
            lambda _r: SimpleNamespace(status="error", content=content),
        )

    call("FileNotFoundError: nope.py")
    call("FileNotFoundError: also-nope.py")  # same class → collapsed
    call("PermissionError: locked.py")  # different class → reported

    assert [tags["error_type"] for _w, _m, tags in reports] == ["FileNotFoundError", "PermissionError"]


def test_a_successful_call_is_never_reported(reports):
    guard = ToolErrorGuard()
    guard.wrap_tool_call(
        SimpleNamespace(call={"name": "grep", "args": {}, "id": "x"}),
        lambda _r: SimpleNamespace(status="success", content="3 matches"),
    )
    assert reports == []
