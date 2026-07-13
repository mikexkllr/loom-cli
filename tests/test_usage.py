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
    assert "cloud" in receipt and "local" in receipt and "all-cloud est." in receipt


def test_is_local_by_name_and_shape():
    t = UsageTracker(_config())
    assert t._is_local("qwen3:14b")  # from config
    assert t._is_local("llama3.2:3b")  # ollama-shaped tag
    assert not t._is_local("claude-sonnet-4-6")


def test_empty_receipt_is_empty():
    t = UsageTracker(_config())
    t.start_turn()
    assert t.receipt() == ""
