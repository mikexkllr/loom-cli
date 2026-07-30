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
