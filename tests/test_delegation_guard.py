"""The orchestrator's per-turn read budget.

Prompt wording does not stop a strong model from reading a dozen files itself;
removing the tool does. These tests pin that the budget counts the current turn
only, that it revokes `read_file` and nothing else, and that the system prompt
tells the model the rule (so the tool vanishing reads as intended behaviour
rather than a broken harness).
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("langchain")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from loom.middleware.delegation_guard import DelegationGuard


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


def _request(messages, tools=("read_file", "task", "write_todos")):
    req = SimpleNamespace(messages=list(messages), tools=[_Tool(t) for t in tools])

    def override(**kw):
        return SimpleNamespace(
            messages=kw.get("messages", req.messages), tools=kw.get("tools", req.tools)
        )

    req.override = override
    return req


def _reads(n: int):
    """One AI message per read_file call, each with its tool result."""
    out = []
    for i in range(n):
        out.append(AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": str(i)}]))
        out.append(ToolMessage(content="…", tool_call_id=str(i)))
    return out


def _run(guard, req):
    seen = {}

    def handler(r):
        seen["req"] = r
        return "ok"

    guard.wrap_model_call(req, handler)
    return [t.name for t in seen["req"].tools]


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_counts_only_the_current_turn():
    guard = DelegationGuard(2)
    messages = [
        HumanMessage("first turn"),
        *_reads(5),
        HumanMessage("second turn"),
        *_reads(1),
    ]
    assert guard.spent(messages) == 1


def test_other_tools_do_not_consume_the_budget():
    guard = DelegationGuard(2)
    messages = [
        HumanMessage("go"),
        AIMessage(content="", tool_calls=[{"name": "task", "args": {}, "id": "a"}]),
        AIMessage(content="", tool_calls=[{"name": "write_todos", "args": {}, "id": "b"}]),
    ]
    assert guard.spent(messages) == 0


def test_counts_seeded_tuple_messages():
    """Loom's non-persistent path seeds state with ("user", text) tuples."""
    guard = DelegationGuard(2)
    assert guard.spent([("user", "go"), *_reads(3)]) == 3


# ---------------------------------------------------------------------------
# Revocation
# ---------------------------------------------------------------------------


def test_read_file_survives_under_budget():
    guard = DelegationGuard(3)
    tools = _run(guard, _request([HumanMessage("go"), *_reads(2)]))
    assert "read_file" in tools
    assert guard.blocked_count == 0


def test_read_file_is_revoked_at_budget():
    guard = DelegationGuard(3)
    tools = _run(guard, _request([HumanMessage("go"), *_reads(3)]))
    assert "read_file" not in tools
    # Delegation itself must survive — the whole point is to force it.
    assert {"task", "write_todos"} <= set(tools)
    assert guard.blocked_count == 1


def test_zero_budget_revokes_immediately():
    guard = DelegationGuard(0)
    assert "read_file" not in _run(guard, _request([HumanMessage("go")]))


def test_negative_budget_disables_the_guard():
    guard = DelegationGuard(-1)
    assert "read_file" in _run(guard, _request([HumanMessage("go"), *_reads(50)]))
    assert guard.blocked_count == 0


def test_already_absent_tool_is_not_counted_as_blocked():
    """Airgap strips read_file elsewhere; the guard must not inflate its count."""
    guard = DelegationGuard(0)
    _run(guard, _request([HumanMessage("go")], tools=("task",)))
    assert guard.blocked_count == 0


# ---------------------------------------------------------------------------
# The prompt has to state the rule the middleware enforces
# ---------------------------------------------------------------------------


def test_prompt_states_the_budget():
    from loom.core.orchestrator import orchestrator_system_prompt

    prompt = orchestrator_system_prompt(4)
    assert "4 direct" in prompt
    assert "withdrawn" in prompt


def test_prompt_drops_read_file_when_budget_is_zero():
    from loom.core.orchestrator import orchestrator_system_prompt

    prompt = orchestrator_system_prompt(0)
    assert "no `read_file` tool" in prompt


def test_prompt_handles_uncapped_budget():
    from loom.core.orchestrator import orchestrator_system_prompt

    assert "uncapped" in orchestrator_system_prompt(-1)


def test_config_rejects_nonsense_budgets():
    from loom.core.config import LoomConfig

    assert LoomConfig(orchestrator_read_budget=0).orchestrator_read_budget == 0
    assert LoomConfig(orchestrator_read_budget=-1).orchestrator_read_budget == -1
    with pytest.raises(ValueError):
        LoomConfig(orchestrator_read_budget=-2)


# ---------------------------------------------------------------------------
# Refusing the call the model makes anyway
#
# Withdrawing the tool from the model's schema is not enforcement on its own:
# ToolNode still holds every tool the agent was built with, so an over-budget
# `read_file` emitted regardless — which is what a model does after watching four
# such calls succeed in its visible history — used to execute and return the file.
# ---------------------------------------------------------------------------


def _tool_request(messages, call_id="x", name="read_file"):
    call = {"name": name, "args": {"file_path": "a.py"}, "id": call_id}
    return SimpleNamespace(
        tool_call=call,
        state={"messages": [*messages, AIMessage(content="", tool_calls=[call])]},
    )


def _run_tool(guard, req):
    """(result, ran) — ran is False when the guard short-circuited."""
    ran = []

    def handler(r):
        ran.append(r)
        return "FILE CONTENTS"

    result = guard.wrap_tool_call(req, handler)
    return result, bool(ran)


def test_a_call_within_budget_runs():
    guard = DelegationGuard(4)
    result, ran = _run_tool(guard, _tool_request([HumanMessage("go"), *_reads(2)]))
    assert ran and result == "FILE CONTENTS"
    assert guard.refused_count == 0


def test_the_last_call_inside_the_budget_still_runs():
    """Budget 4 with 3 already spent: this is the 4th, and it must not be refused."""
    guard = DelegationGuard(4)
    _, ran = _run_tool(guard, _tool_request([HumanMessage("go"), *_reads(3)]))
    assert ran
    assert guard.refused_count == 0


def test_an_over_budget_call_is_refused_not_executed():
    guard = DelegationGuard(4)
    result, ran = _run_tool(guard, _tool_request([HumanMessage("go"), *_reads(4)]))
    assert not ran, "the tool ran despite the budget being spent"
    assert guard.refused_count == 1
    assert isinstance(result, ToolMessage)
    assert "read budget" in result.content
    assert "task" in result.content, "the refusal must name the way forward"


def test_the_refusal_answers_the_call_it_refused():
    """A ToolMessage whose id does not match leaves the graph with a dangling call."""
    guard = DelegationGuard(0)
    result, _ = _run_tool(guard, _tool_request([HumanMessage("go")], call_id="abc123"))
    assert result.tool_call_id == "abc123"


def test_other_tools_are_never_refused():
    guard = DelegationGuard(0)
    _, ran = _run_tool(guard, _tool_request([HumanMessage("go")], name="task"))
    assert ran


def test_a_negative_budget_refuses_nothing():
    guard = DelegationGuard(-1)
    _, ran = _run_tool(guard, _tool_request([HumanMessage("go"), *_reads(20)]))
    assert ran
    assert guard.refused_count == 0


def test_a_parallel_batch_degrades_in_order():
    """Three reads in one message with 2 of 4 spent: the first two are inside the
    budget and must run. Totalling the batch would refuse all three, including
    calls the orchestrator was entitled to make."""
    guard = DelegationGuard(4)
    calls = [{"name": "read_file", "args": {}, "id": f"p{i}"} for i in range(3)]
    state = {"messages": [HumanMessage("go"), *_reads(2), AIMessage(content="", tool_calls=calls)]}
    outcomes = []
    for call in calls:
        req = SimpleNamespace(tool_call=call, state=state)
        _, ran = _run_tool(guard, req)
        outcomes.append(ran)
    assert outcomes == [True, True, False]
    assert guard.refused_count == 1


def test_refusals_are_counted_separately_from_revocations():
    """They answer different questions: how often the model was told no, versus
    how often it reached for a tool it could no longer see."""
    guard = DelegationGuard(1)
    _run(guard, _request([HumanMessage("go"), *_reads(1)]))       # revoked
    _run_tool(guard, _tool_request([HumanMessage("go"), *_reads(1)]))  # refused
    assert (guard.blocked_count, guard.refused_count) == (1, 1)


def test_a_missing_state_does_not_crash_the_call():
    guard = DelegationGuard(4)
    req = SimpleNamespace(tool_call={"name": "read_file", "args": {}, "id": "z"}, state=None)
    _, ran = _run_tool(guard, req)
    assert ran, "with no history to count, the call must be allowed through"
