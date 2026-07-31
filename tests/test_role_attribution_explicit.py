"""Roles that the run tree cannot place, and the two that it got wrong.

`UsageTracker` attributes a model call by walking the callback run tree up to the
nearest `task`. That works for subagents, whose graphs really do nest under the
tool run — and silently fails for a model invoked *by hand* from inside a tool.
The config such a tool is handed belongs to the tool **node**, so the call lands
beside the tool run rather than under it, the walk sails past the role, and the
tokens are credited to the orchestrator.

Two callers are in that position, and both were misattributed: the advisor's
`consult`, and `/compact`'s summariser. They now say who they are outright.
"""

import pytest

pytest.importorskip("langchain")

from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from loom.core.config import LoomConfig
from loom.core.usage import ROLE_KEY, UsageTracker, role_metadata


def _response(model: str, inp: int, out: int):
    msg = AIMessage(content="…")
    msg.usage_metadata = {"input_tokens": inp, "output_tokens": out, "input_token_details": {}}
    msg.response_metadata = {"model_name": model}
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def _tracker(**config) -> UsageTracker:
    t = UsageTracker(LoomConfig(**config))
    t.start_turn()
    return t


def _sibling_call(tracker, role: str | None, model: str, inp: int, out: int, parent):
    """A model call parented to the tool NODE, not to any `task`/`consult` run —
    exactly where a hand-invoked model lands."""
    run_id = uuid4()
    metadata = {"ls_provider": "anthropic"}
    if role:
        metadata[ROLE_KEY] = role
    tracker.on_chat_model_start({}, [], run_id=run_id, parent_run_id=parent, metadata=metadata)
    tracker.on_llm_end(_response(model, inp, out), run_id=run_id)
    return run_id


def _roles(tracker) -> set[str]:
    return {a.role for a in tracker.turn.actors}


# ---------------------------------------------------------------------------


def test_role_metadata_carries_the_rest_of_the_config_through():
    """Dropping the callbacks to add a role would make the call invisible instead
    of misattributed — a worse trade."""
    tracker = _tracker()
    original = {"callbacks": [tracker], "configurable": {"thread_id": "t1"},
                "metadata": {"existing": 1}, "tags": ["x"]}
    merged = role_metadata(original, "advisor")
    assert merged["callbacks"] == [tracker]
    assert merged["configurable"] == {"thread_id": "t1"}
    assert merged["tags"] == ["x"]
    assert merged["metadata"]["existing"] == 1
    assert merged["metadata"][ROLE_KEY] == "advisor"
    assert ROLE_KEY not in original.get("metadata", {}), "the caller's config must not be mutated"


def test_role_metadata_handles_no_config_at_all():
    assert role_metadata(None, "advisor")["metadata"][ROLE_KEY] == "advisor"


def test_a_sibling_call_without_a_claim_is_credited_to_the_orchestrator():
    """The bug, pinned: this is what the tree walk does on its own."""
    tracker = _tracker(advisor="claude-opus-4-8")
    parent = uuid4()
    tracker.on_chain_start({}, {}, run_id=parent, parent_run_id=None)
    _sibling_call(tracker, None, "claude-opus-4-8", 3_000, 250, parent)
    assert _roles(tracker) == {"orchestrator"}


def test_a_claimed_sibling_call_is_credited_to_its_own_role():
    tracker = _tracker(advisor="claude-opus-4-8")
    parent = uuid4()
    tracker.on_chain_start({}, {}, run_id=parent, parent_run_id=None)
    _sibling_call(tracker, "advisor", "claude-opus-4-8", 3_000, 250, parent)
    assert _roles(tracker) == {"advisor"}
    advisor = next(a for a in tracker.turn.actors if a.role == "advisor")
    assert tracker.turn.cost_of(advisor, tracker.turn.actors[advisor]) > 0


def test_an_explicit_claim_beats_the_tree():
    """A claim is direct evidence; the walk is inference. When both are available
    and disagree, the claim is the one that knows."""
    tracker = _tracker()
    task_run = uuid4()
    tracker.on_tool_start({"name": "task"}, "", run_id=task_run, parent_run_id=None,
                          inputs={"subagent_type": "explorer"})
    _sibling_call(tracker, "advisor", "claude-opus-4-8", 100, 10, task_run)
    assert _roles(tracker) == {"advisor"}


def test_an_unclaimed_call_under_a_task_still_belongs_to_the_subagent():
    tracker = _tracker()
    task_run = uuid4()
    tracker.on_tool_start({"name": "task"}, "", run_id=task_run, parent_run_id=None,
                          inputs={"subagent_type": "explorer"})
    _sibling_call(tracker, None, "qwen3.5:4b", 100, 10, task_run)
    assert _roles(tracker) == {"explorer"}


# ---------------------------------------------------------------------------
# Compaction is housekeeping, not the orchestrator's own work
# ---------------------------------------------------------------------------


def _with_compaction():
    tracker = _tracker(orchestrator="claude-sonnet-5")
    parent = uuid4()
    tracker.on_chain_start({}, {}, run_id=parent, parent_run_id=None)
    _sibling_call(tracker, None, "claude-sonnet-5", 10_000, 500, parent)       # orchestrator
    task_run = uuid4()
    tracker.on_tool_start({"name": "task"}, "", run_id=task_run, parent_run_id=parent,
                          inputs={"subagent_type": "explorer"})
    _sibling_call(tracker, None, "qwen3.5:4b", 40_000, 400, task_run)          # explorer
    return tracker, parent


def test_compaction_gets_its_own_row():
    tracker, parent = _with_compaction()
    _sibling_call(tracker, "compaction", "claude-sonnet-5", 8_000, 300, parent)
    assert "compaction" in _roles(tracker)


def test_compaction_is_not_counted_as_a_delegated_role():
    """It is not a subagent doing work, and counting it as one would make every
    /compact read as another delegation in the receipt."""
    tracker, parent = _with_compaction()
    before = tracker.turn.delegations()
    _sibling_call(tracker, "compaction", "claude-sonnet-5", 8_000, 300, parent)
    assert tracker.turn.delegations() == before


def test_compaction_does_not_move_the_orchestrators_share():
    """The number the whole receipt leads with must not shift because the user
    compacted their transcript."""
    tracker, parent = _with_compaction()
    before = tracker.turn.orchestrator_share()
    _sibling_call(tracker, "compaction", "claude-sonnet-5", 8_000, 300, parent)
    assert tracker.turn.orchestrator_share() == pytest.approx(before)


def test_compaction_is_still_charged_for():
    tracker, parent = _with_compaction()
    before = tracker.turn.cloud_cost
    _sibling_call(tracker, "compaction", "claude-sonnet-5", 8_000, 300, parent)
    assert tracker.turn.cloud_cost > before
    assert tracker.turn.overhead_tokens() == 8_300


def test_the_three_buckets_partition_every_token():
    """No token counted twice, none dropped — the invariant that makes the
    delegation ratio meaningful."""
    tracker, parent = _with_compaction()
    _sibling_call(tracker, "compaction", "claude-sonnet-5", 8_000, 300, parent)
    u = tracker.turn
    total = sum(usage.total_tokens for usage in u.actors.values())
    assert u.orchestrator_tokens() + u.delegated_tokens() + u.overhead_tokens() == total
