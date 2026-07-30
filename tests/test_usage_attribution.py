"""Cache-aware pricing, run-tree role attribution, and an honest baseline.

Three accounting bugs the receipt used to have, pinned so they stay fixed:

* every input token was priced at the uncached rate, overstating a cached
  conversation's cost by most of an order of magnitude;
* usage was keyed by model name, which cannot tell the orchestrator apart from a
  subagent sharing its model — the one number a delegating architecture needs;
* "saved vs all-cloud" was priced against ``config.orchestrator`` even when that
  was a local model, inventing a saving in local-only runs.
"""

import uuid

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langchain_core")

from loom.core.config import LoomConfig
from loom.core.usage import (
    DEFAULT_CLOUD_REFERENCE,
    Actor,
    UsageTracker,
    cost_usd,
    price_entry,
    price_for,
)


def _config(**kw):
    defaults = dict(
        orchestrator="claude-sonnet-4-6",
        subagents={"editor": "ollama/deepseek-coder:14b", "general-purpose": "ollama/qwen3:14b"},
    )
    defaults.update(kw)
    return LoomConfig(**defaults)


# ---------------------------------------------------------------------------
# Cache-aware pricing
# ---------------------------------------------------------------------------


def test_uncached_pricing_is_unchanged():
    assert price_for("claude-sonnet-4-6") == (3.0, 15.0)
    assert cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000) == pytest.approx(18.0)


def test_cache_reads_are_billed_at_a_tenth():
    """input_tokens is the provider's total, cached included — so a fully cached
    prompt costs a tenth of what the old math charged."""
    full = cost_usd("claude-sonnet-4-6", 100_000, 0)
    cached = cost_usd("claude-sonnet-4-6", 100_000, 0, cache_read_tokens=100_000)
    assert cached == pytest.approx(full * 0.1)


def test_cache_writes_carry_a_premium():
    p = price_entry("claude-sonnet-4-6")
    assert p.cache_write == 1.25
    written = cost_usd("claude-sonnet-4-6", 10_000, 0, cache_write_tokens=10_000)
    assert written == pytest.approx(10_000 * 3.0 * 1.25 / 1e6)


def test_mixed_cache_split_adds_up():
    cost = cost_usd(
        "claude-sonnet-4-6", 100_000, 1_000, cache_read_tokens=80_000, cache_write_tokens=5_000
    )
    expected = (
        15_000 * 3.0  # uncached remainder
        + 80_000 * 3.0 * 0.1
        + 5_000 * 3.0 * 1.25
        + 1_000 * 15.0
    ) / 1e6
    assert cost == pytest.approx(expected)


def test_openai_has_no_cache_write_premium():
    assert price_entry("gpt-5.6-sol").cache_write == 1.0


def test_cached_counts_never_exceed_the_total():
    """A provider reporting more cached than total tokens must not go negative."""
    assert cost_usd("claude-sonnet-4-6", 10, 0, cache_read_tokens=10_000) > 0


# ---------------------------------------------------------------------------
# Role attribution through the run tree
# ---------------------------------------------------------------------------


class _Msg:
    def __init__(self, inp, out, model, cache_read=0, cache_write=0):
        self.usage_metadata = {
            "input_tokens": inp,
            "output_tokens": out,
            "input_token_details": {"cache_read": cache_read, "cache_creation": cache_write},
        }
        self.response_metadata = {"model_name": model}


class _Gen:
    def __init__(self, message):
        self.message = message


class _Result:
    def __init__(self, message):
        self.generations = [[_Gen(message)]]


def _llm(tracker, *, run_id, parent, model, inp, out, provider="", **cache):
    tracker.on_chat_model_start(
        {}, [], run_id=run_id, parent_run_id=parent, metadata={"ls_provider": provider}
    )
    tracker.on_llm_end(_Result(_Msg(inp, out, model, **cache)), run_id=run_id)


def test_model_calls_default_to_the_orchestrator():
    t = UsageTracker(_config())
    t.start_turn()
    _llm(t, run_id=uuid.uuid4(), parent=None, model="claude-sonnet-4-6", inp=1_000, out=100)
    assert [a.role for a in t.turn.actors] == ["orchestrator"]


def test_calls_under_a_task_belong_to_that_subagent():
    t = UsageTracker(_config())
    t.start_turn()
    root = uuid.uuid4()
    task = uuid.uuid4()
    nested_chain = uuid.uuid4()
    llm = uuid.uuid4()

    t.on_chain_start({}, {}, run_id=root, parent_run_id=None)
    t.on_tool_start({"name": "task"}, "", run_id=task, parent_run_id=root, inputs={"subagent_type": "explorer"})
    t.on_chain_start({}, {}, run_id=nested_chain, parent_run_id=task)
    _llm(t, run_id=llm, parent=nested_chain, model="qwen3:4b", inp=40_000, out=500, provider="ollama")

    (actor,) = t.turn.actors
    assert actor == Actor("explorer", "qwen3:4b", True)


def test_consult_is_attributed_to_the_advisor():
    t = UsageTracker(_config())
    t.start_turn()
    tool = uuid.uuid4()
    t.on_tool_start({"name": "consult"}, "", run_id=tool, parent_run_id=None, inputs={})
    _llm(t, run_id=uuid.uuid4(), parent=tool, model="claude-opus-4-8", inp=2_000, out=400)
    assert [a.role for a in t.turn.actors] == ["advisor"]


def test_two_roles_on_one_model_stay_separate():
    """The case model-name keying could never express: a subagent that fell back
    to the same cloud model the orchestrator runs on."""
    t = UsageTracker(_config())
    t.start_turn()
    task = uuid.uuid4()
    _llm(t, run_id=uuid.uuid4(), parent=None, model="claude-sonnet-4-6", inp=5_000, out=500)
    t.on_tool_start({"name": "task"}, "", run_id=task, parent_run_id=None, inputs={"subagent_type": "editor"})
    _llm(t, run_id=uuid.uuid4(), parent=task, model="claude-sonnet-4-6", inp=20_000, out=2_000)

    roles = {a.role for a in t.turn.actors}
    assert roles == {"orchestrator", "editor"}
    # ...while the by-model view still merges them for the pricing table.
    assert set(t.turn.cloud) == {"claude-sonnet-4-6"}
    assert t.turn.cloud["claude-sonnet-4-6"].input_tokens == 25_000


def test_a_task_without_a_subagent_type_falls_back_to_general_purpose():
    t = UsageTracker(_config())
    t.start_turn()
    task = uuid.uuid4()
    t.on_tool_start({"name": "task"}, "", run_id=task, parent_run_id=None, inputs={})
    _llm(t, run_id=uuid.uuid4(), parent=task, model="qwen3:14b", inp=100, out=10, provider="ollama")
    assert [a.role for a in t.turn.actors] == ["general-purpose"]


def test_attribution_state_resets_each_turn():
    t = UsageTracker(_config())
    t.start_turn()
    task = uuid.uuid4()
    t.on_tool_start({"name": "task"}, "", run_id=task, parent_run_id=None, inputs={"subagent_type": "explorer"})
    t.start_turn()
    # The stale task run must not claim this turn's orchestrator call.
    _llm(t, run_id=uuid.uuid4(), parent=None, model="claude-sonnet-4-6", inp=10, out=1)
    assert [a.role for a in t.turn.actors] == ["orchestrator"]


def test_a_cycle_in_the_run_tree_terminates():
    t = UsageTracker(_config())
    t.start_turn()
    a, b = uuid.uuid4(), uuid.uuid4()
    t._parent[a] = b
    t._parent[b] = a
    assert t.role_for(a) == "orchestrator"


# ---------------------------------------------------------------------------
# The delegation ratio — the number that answers "is the orchestrator doing too
# much?"
# ---------------------------------------------------------------------------


def test_orchestrator_share_and_delegation_count():
    t = UsageTracker(_config())
    t.start_turn()
    t.turn.add("claude-sonnet-4-6", False, 8_000, 2_000, role="orchestrator")
    t.turn.add("qwen3:4b", True, 60_000, 1_000, role="explorer")
    t.turn.add("qwen3:14b", True, 28_000, 1_000, role="editor")

    assert t.turn.orchestrator_tokens() == 10_000
    assert t.turn.delegated_tokens() == 90_000
    assert t.turn.orchestrator_share() == pytest.approx(0.10)
    assert t.turn.delegations() == 2
    assert "orchestrator 10% of tokens, 2 delegated roles" in t.receipt()


def test_an_orchestrator_only_turn_reports_no_delegations():
    t = UsageTracker(_config())
    t.start_turn()
    t.turn.add("claude-sonnet-4-6", False, 40_000, 3_000, role="orchestrator")
    assert t.turn.orchestrator_share() == pytest.approx(1.0)
    assert t.turn.delegations() == 0
    assert "delegated role" not in t.receipt()


def test_orchestrator_cost_excludes_subagents():
    t = UsageTracker(_config())
    t.start_turn()
    t.turn.add("claude-sonnet-4-6", False, 10_000, 1_000, role="orchestrator")
    t.turn.add("claude-opus-4-8", False, 10_000, 1_000, role="advisor")
    assert t.turn.orchestrator_cost() == pytest.approx(cost_usd("claude-sonnet-4-6", 10_000, 1_000))
    assert t.turn.cloud_cost > t.turn.orchestrator_cost()


# ---------------------------------------------------------------------------
# Local/cloud classification
# ---------------------------------------------------------------------------


def test_provider_metadata_beats_the_name_heuristic():
    """An Ollama tag without a colon (`gpt-oss`, `llama3.2`) looks like a cloud
    name; misreading it invents a charge."""
    t = UsageTracker(_config())
    t.start_turn()
    _llm(t, run_id=uuid.uuid4(), parent=None, model="gpt-oss", inp=50_000, out=2_000, provider="ollama")
    assert t.turn.cloud_cost == 0.0
    assert t.turn.local_share() == pytest.approx(1.0)


def test_the_config_outranks_an_unrecognized_provider_tag():
    """A model the user assigned with an `ollama/` prefix is local, whatever the
    run's provider tag says. Letting the tag win billed free subagent tokens as
    cloud whenever the local model was served by anything not tagged "ollama"."""
    config = _config(subagents={"explorer": "ollama/qwen3:4b"})
    t = UsageTracker(config)
    t.start_turn()
    _llm(
        t,
        run_id=uuid.uuid4(),
        parent=None,
        model="qwen3:4b",
        inp=40_000,
        out=400,
        provider="something-else",
    )
    assert t.turn.cloud_cost == 0.0


def test_an_unknown_provider_on_an_unconfigured_model_stays_cloud():
    t = UsageTracker(_config())
    assert not t._is_local("some-hosted-model", provider="openai")


def test_name_heuristic_still_applies_without_provider_metadata():
    t = UsageTracker(_config())
    assert t._is_local("qwen3:14b")
    assert t._is_local("llama3.2:3b")
    assert not t._is_local("claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# The all-cloud baseline
# ---------------------------------------------------------------------------


def test_reference_is_the_cloud_orchestrator_when_there_is_one():
    t = UsageTracker(_config())
    assert t.cloud_reference() == "claude-sonnet-4-6"


def test_reference_falls_through_to_another_billed_role():
    config = _config(orchestrator="ollama/qwen3:14b", advisor="claude-opus-4-8")
    assert UsageTracker(config).cloud_reference() == "claude-opus-4-8"


def test_all_local_config_uses_a_named_default_not_a_local_model():
    """Pricing local tokens against a local orchestrator reported a saving over
    a model that was never billed."""
    config = _config(
        orchestrator="ollama/qwen3:14b",
        advisor="ollama/qwen3:14b",
        cloud_fallback="ollama/qwen3:14b",
        escalation_model="ollama/qwen3:14b",
    )
    t = UsageTracker(config)
    assert t.cloud_reference() == DEFAULT_CLOUD_REFERENCE
    t.start_turn()
    t.turn.add("qwen3:14b", True, 50_000, 2_000, role="explorer")
    # The saving is a stated counterfactual against a named cloud model...
    assert f"vs all-cloud on {DEFAULT_CLOUD_REFERENCE}" in t.receipt()
    # ...and nothing was actually billed.
    assert t.turn.cloud_cost == 0.0


def test_savings_uses_uncached_local_pricing():
    t = UsageTracker(_config())
    t.start_turn()
    t.turn.add("qwen3:14b", True, 72_000, 18_000, role="explorer")
    expected = (72_000 * 3.0 + 18_000 * 15.0) / 1e6
    assert t.turn.savings("claude-sonnet-4-6") == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def test_receipt_surfaces_the_cached_share():
    t = UsageTracker(_config())
    t.start_turn()
    t.turn.add("claude-sonnet-4-6", False, 120_000, 2_000, role="orchestrator", cache_read=110_000)
    receipt = t.receipt()
    assert "cached" in receipt
    # And the price reflects the discount rather than 120k at full rate.
    assert t.turn.cloud_cost < cost_usd("claude-sonnet-4-6", 120_000, 2_000) / 2


def test_empty_receipt_is_empty():
    t = UsageTracker(_config())
    t.start_turn()
    assert t.receipt() == ""


def test_accounting_never_raises_on_a_malformed_response():
    t = UsageTracker(_config())
    t.start_turn()
    t.on_llm_end(object(), run_id=uuid.uuid4())
    t.on_tool_start(None, "", run_id=None)
    assert t.receipt() == ""
