"""Pricing table, local/cloud classification, and receipt math."""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langchain_core")

from loom.core.config import LoomConfig
from loom.core.usage import UsageTracker, cost_usd, price_for


def _config():
    return LoomConfig(
        orchestrator="claude-sonnet-4-6",
        subagents={"editor": "ollama/deepseek-coder:14b", "general": "ollama/qwen3:14b"},
    )


def test_prices_longest_prefix_wins():
    assert price_for("claude-sonnet-4-6") == (3.0, 15.0)
    assert price_for("claude-opus-4-8") == (5.0, 25.0)
    assert price_for("claude-haiku-4-5-20251001") == (1.0, 5.0)
    assert price_for("claude-fable-5") == (10.0, 50.0)
    assert price_for("totally-unknown-model") == (3.0, 15.0)  # conservative fallback


def test_cost_math():
    # 1M in + 1M out on sonnet = $3 + $15
    assert cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000) == pytest.approx(18.0)


def test_tracker_classifies_and_prices():
    t = UsageTracker(_config())
    t.start_turn()
    t.turn.add("claude-sonnet-4-6", False, 10_000, 2_000)
    t.turn.add("qwen3:14b", True, 80_000, 9_000)
    t.session.add("claude-sonnet-4-6", False, 10_000, 2_000)
    t.session.add("qwen3:14b", True, 80_000, 9_000)

    assert t.turn.cloud_cost == pytest.approx((10_000 * 3 + 2_000 * 15) / 1e6)
    # all-cloud estimate prices the local tokens at orchestrator rates on top
    est = t.turn.all_cloud_estimate("claude-sonnet-4-6")
    assert est > t.turn.cloud_cost
    receipt = t.receipt()
    assert "cloud" in receipt and "local" in receipt and "vs all-cloud" in receipt


def test_local_share_and_savings():
    t = UsageTracker(_config())
    t.start_turn()
    t.turn.add("claude-sonnet-4-6", False, 8_000, 2_000)
    t.turn.add("qwen3:14b", True, 72_000, 18_000)
    # 90k of 100k tokens ran locally.
    assert t.turn.local_share() == pytest.approx(0.9)
    # Savings = what the local tokens would have cost on the reference model.
    expected = (72_000 * 3 + 18_000 * 15) / 1e6
    assert t.turn.savings("claude-sonnet-4-6") == pytest.approx(expected)
    receipt = t.receipt()
    assert "90% local" in receipt
    assert f"saved ~${expected:.3f}" in receipt


def test_all_cloud_turn_reports_no_savings():
    t = UsageTracker(_config())
    t.start_turn()
    t.turn.add("claude-sonnet-4-6", False, 1_000, 500)
    assert t.turn.local_share() == 0.0
    assert t.turn.savings("claude-sonnet-4-6") == 0.0
    assert "saved" not in t.receipt()


def test_is_local_by_name_and_shape():
    t = UsageTracker(_config())
    assert t._is_local("qwen3:14b")  # from config
    assert t._is_local("llama3.2:3b")  # ollama-shaped tag
    assert not t._is_local("claude-sonnet-4-6")


def test_empty_receipt_is_empty():
    t = UsageTracker(_config())
    t.start_turn()
    assert t.receipt() == ""


# ------------------------------------------------ concurrency + live config


def _llm_end(tracker, model, inp, out):
    """Feed one finished model call through the real callback hook."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    msg = AIMessage(content="x")
    msg.usage_metadata = {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
    msg.response_metadata = {"model_name": model}
    tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))


def test_parallel_model_calls_are_all_counted():
    """Parallel subagents finish model calls on separate worker threads, and a
    receipt is a sum of `+=` on shared counters."""
    import sys
    import threading

    tracker = UsageTracker(_config())
    tracker.start_turn()
    threads_n, calls_n = 8, 400
    start = threading.Barrier(threads_n)

    def work():
        start.wait()
        for _ in range(calls_n):
            _llm_end(tracker, "claude-sonnet-4-6", 3, 1)

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # make an unguarded `+=` lose updates reliably
    try:
        threads = [threading.Thread(target=work) for _ in range(threads_n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old)

    total = threads_n * calls_n
    for bucket in (tracker.turn, tracker.session):
        (usage,) = bucket.cloud.values()
        assert (usage.calls, usage.input_tokens, usage.output_tokens) == (total, total * 3, total)


def test_swapping_the_config_updates_what_counts_as_local():
    """`/model` and `/setup` reload the settings mid-session. The set of local
    model names was computed once in __init__, so a model assigned to Ollama
    afterwards — one whose tag has no colon to give it away — kept being priced
    as a cloud call."""
    tracker = UsageTracker(LoomConfig(orchestrator="claude-sonnet-4-6"))
    assert tracker._is_local("gpt-oss") is False

    tracker.config = LoomConfig(orchestrator="claude-sonnet-4-6", subagents={"editor": "ollama/gpt-oss"})
    assert tracker._is_local("gpt-oss") is True

    tracker.start_turn()
    _llm_end(tracker, "gpt-oss", 1_000, 100)
    assert tracker.turn.cloud_cost == 0
    assert tracker.turn.local_share() == 1.0


def test_an_inherited_role_is_priced_as_the_model_it_inherits():
    """The reviewer runs on the advisor's model unless it is given its own. The
    tracker resolved it against `general-purpose` instead, so a free-tier
    advisor's reviewer was billed at the unknown-model default."""
    config = LoomConfig(
        orchestrator="claude-sonnet-4-6",
        advisor="zen/deepseek-v4-flash-free",
        subagents={"general-purpose": "ollama/qwen3:14b"},
    )
    tracker = UsageTracker(config)
    assert tracker._role_model("reviewer") == "zen/deepseek-v4-flash-free"
    assert tracker._role_model("explorer") == "ollama/qwen3:14b"
    # The gateway reports the upstream name; the configured `-free` id prices it.
    assert tracker._billing_model("reviewer", "deepseek-v4-flash") == "zen/deepseek-v4-flash-free"
