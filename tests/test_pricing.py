"""What a call is priced against — the question behind every figure in the receipt.

The cost code has to price the model that *actually ran*, which is not always the
model the config named (a cloud fallback) and not always the model the provider
reported (a gateway serving a free tier under the upstream model's name). Getting
that wrong is not a rounding error in either direction:

* Prefer the config blindly and a billed cloud fallback prices as free local work.
* Prefer the report blindly and OpenCode Zen's ``deepseek-v4-flash-free`` — which
  answers reporting ``deepseek-v4-flash``, suffix stripped — bills at the
  unknown-model default. A live run on nothing but free models invoiced $0.032.

So: the report wins whenever Loom recognises it, and the config fills the gap when
it does not.
"""

import pytest

from loom.core.config import LoomConfig
from loom.core.usage import (
    Actor,
    ModelUsage,
    TurnUsage,
    UsageTracker,
    cost_usd,
    is_free_model,
    price_entry,
)

# ---------------------------------------------------------------------------
# Free tiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    ["deepseek-v4-flash-free", "zen/deepseek-v4-flash-free", "mimo-v2.5-free",
     "north-mini-code-free", "some-model:free"],
)
def test_a_free_tier_costs_nothing(model):
    assert is_free_model(model)
    p = price_entry(model)
    assert (p.inp, p.out) == (0.0, 0.0)
    assert not p.estimated, "zero is known, not guessed"
    assert cost_usd(model, 1_000_000, 1_000_000) == 0.0


@pytest.mark.parametrize("model", ["deepseek-v4-flash", "claude-sonnet-5", "freeform-model"])
def test_a_paid_model_is_not_mistaken_for_a_free_one(model):
    assert not is_free_model(model)


def test_the_suffix_is_matched_at_the_end_not_anywhere():
    """"free" inside a name means nothing; only the tier suffix does."""
    assert not is_free_model("freedom-1")
    assert not is_free_model("free-lunch-pro")


# ---------------------------------------------------------------------------
# Provider prefixes and estimates
# ---------------------------------------------------------------------------


def test_a_provider_prefix_does_not_hide_a_known_price():
    """Prices are keyed on bare model names, but model strings carry a provider."""
    assert price_entry("zen/claude-haiku-4-5") == price_entry("claude-haiku-4-5")
    assert price_entry("anthropic:claude-sonnet-5") == price_entry("claude-sonnet-5")


def test_a_known_model_is_not_flagged_as_an_estimate():
    assert not price_entry("claude-sonnet-5").estimated
    assert not price_entry("gpt-5.6-terra").estimated


def test_an_unknown_model_is_flagged_as_an_estimate():
    """It still gets a conservative price — a receipt with a hole in it is worse —
    but the UI has to be able to say the number is a guess."""
    p = price_entry("big-pickle")
    assert p.estimated
    assert (p.inp, p.out) == (3.0, 15.0)


def test_a_dated_model_id_still_matches_its_family():
    assert price_entry("claude-sonnet-4-5-20250929") == price_entry("claude-sonnet-4-5")


# ---------------------------------------------------------------------------
# Which model a row is billed as
# ---------------------------------------------------------------------------


def _tracker(**config) -> UsageTracker:
    return UsageTracker(LoomConfig(**config))


def test_a_recognised_report_is_priced_as_reported():
    """The config is not consulted at all here: the report is what ran."""
    t = _tracker(orchestrator="claude-sonnet-5")
    assert t._billing_model("orchestrator", "claude-haiku-4-5") == "claude-haiku-4-5"


def test_a_cloud_fallback_is_billed_even_though_the_config_says_local():
    """The role was configured local and ran on the cloud because Ollama was
    absent. Pricing the configured model here would report the billed session as
    free — the exact failure the report-first rule exists to prevent."""
    t = _tracker(subagents={"explorer": "ollama/qwen3.5:4b"})
    assert t._billing_model("explorer", "claude-haiku-4-5") == "claude-haiku-4-5"
    assert cost_usd("claude-haiku-4-5", 10_000, 1_000) > 0


def test_an_unrecognised_report_falls_back_to_the_configured_model():
    """OpenCode Zen's free tier: configured `-free`, reported without it."""
    t = _tracker(subagents={"explorer": "zen/deepseek-v4-flash-free"})
    assert t._billing_model("explorer", "deepseek-v4-flash") == "zen/deepseek-v4-flash-free"


def test_the_advisor_and_orchestrator_resolve_their_own_config_entries():
    t = _tracker(orchestrator="zen/big-pickle-free", advisor="zen/nemotron-3-ultra-free")
    assert t._billing_model("orchestrator", "mystery-model") == "zen/big-pickle-free"
    assert t._billing_model("advisor", "mystery-model") == "zen/nemotron-3-ultra-free"


def test_compaction_is_billed_on_the_orchestrator_model():
    """`/compact` summarises with the orchestrator's model."""
    t = _tracker(orchestrator="zen/deepseek-v4-flash-free")
    assert t._billing_model("compaction", "deepseek-v4-flash") == "zen/deepseek-v4-flash-free"


def test_an_unrecognised_role_resolves_through_the_configs_own_fallback_chain():
    """With the report unknown, any answer is a guess — so use the one Loom
    already defines: `LoomConfig.model_for` inherits an unassigned role from
    general-purpose, then the orchestrator."""
    t = _tracker(orchestrator="claude-sonnet-5", subagents={"general-purpose": "ollama/q:9b"})
    assert t._billing_model("mystery-role", "mystery-model") == "ollama/q:9b"
    bare = _tracker(orchestrator="claude-sonnet-5", subagents={})
    assert bare._billing_model("mystery-role", "mystery-model") == "claude-sonnet-5"


# ---------------------------------------------------------------------------
# How it reaches the receipt
# ---------------------------------------------------------------------------


def test_an_actor_prices_against_billed_as_when_it_has_one():
    free = Actor("explorer", "deepseek-v4-flash", False, "zen/deepseek-v4-flash-free")
    usage = ModelUsage(input_tokens=100_000, output_tokens=5_000, calls=3)
    assert TurnUsage.cost_of(free, usage) == 0.0
    # Without the config's answer the same tokens would have been invoiced.
    reported_only = Actor("explorer", "deepseek-v4-flash", False)
    assert TurnUsage.cost_of(reported_only, usage) > 0


def test_a_local_actor_is_free_regardless_of_what_it_is_billed_as():
    local = Actor("explorer", "qwen3.5:4b", True, "claude-opus-4-8")
    assert TurnUsage.cost_of(local, ModelUsage(input_tokens=999_999)) == 0.0


def test_estimates_are_reported_so_the_ui_can_mark_them():
    turn = TurnUsage()
    turn.add("big-pickle", False, 1_000, 100, role="orchestrator", billed_as="zen/big-pickle")
    assert turn.has_estimates()
    turn2 = TurnUsage()
    turn2.add("claude-sonnet-5", False, 1_000, 100, role="orchestrator")
    assert not turn2.has_estimates()


def test_a_free_row_is_not_an_estimate():
    turn = TurnUsage()
    turn.add("deepseek-v4-flash", False, 1_000, 100, role="explorer",
             billed_as="zen/deepseek-v4-flash-free")
    assert not turn.has_estimates()
    assert turn.cloud_cost == 0.0


def test_a_local_row_is_never_an_estimate():
    turn = TurnUsage()
    turn.add("qwen3.5:4b", True, 40_000, 500, role="explorer")
    assert not turn.has_estimates()


def test_the_receipt_marks_an_estimated_total():
    t = _tracker(orchestrator="zen/big-pickle")
    t.start_turn()
    t.turn.add("big-pickle", False, 10_000, 500, role="orchestrator", billed_as="zen/big-pickle")
    t.session.add("big-pickle", False, 10_000, 500, role="orchestrator", billed_as="zen/big-pickle")
    assert "~$" in t.receipt()


def test_the_receipt_does_not_mark_a_known_total():
    t = _tracker(orchestrator="claude-sonnet-5")
    t.start_turn()
    for bucket in (t.turn, t.session):
        bucket.add("claude-sonnet-5", False, 10_000, 500, role="orchestrator")
    receipt = t.receipt()
    assert "$" in receipt and "~$" not in receipt


def test_a_free_session_reports_zero_not_an_estimate():
    """The live regression: a whole session on OpenCode Zen's free models used to
    read $0.032."""
    t = _tracker(
        orchestrator="zen/deepseek-v4-flash-free",
        subagents={"explorer": "zen/deepseek-v4-flash-free"},
    )
    t.start_turn()
    for bucket in (t.turn, t.session):
        bucket.add("deepseek-v4-flash", False, 9_496, 271, role="orchestrator",
                   cache_read=9_216, billed_as="zen/deepseek-v4-flash-free")
        bucket.add("deepseek-v4-flash", False, 15_577, 838, role="explorer",
                   cache_read=14_976, billed_as="zen/deepseek-v4-flash-free")
    assert t.turn.cloud_cost == 0.0
    assert not t.turn.has_estimates()
    assert "$0.000 cloud" in t.receipt()


# ---------------------------------------------------------------------------
# The two OpenCode gateways side by side
#
# Both serve `deepseek-v4-flash`: free on Zen's `-free` tier, paid on Go's
# subscription. The reported name is identical, so only the configured string
# tells them apart — which is the whole reason the config is consulted at all.
# ---------------------------------------------------------------------------


def test_the_same_served_model_prices_differently_per_gateway():
    zen = _tracker(subagents={"explorer": "zen/deepseek-v4-flash-free"})
    go = _tracker(subagents={"explorer": "go/deepseek-v4-flash"})
    assert price_entry(zen._billing_model("explorer", "deepseek-v4-flash")).inp == 0.0
    assert price_entry(go._billing_model("explorer", "deepseek-v4-flash")).inp > 0.0


def test_a_go_model_is_charged_but_marked_as_an_estimate():
    """Go is a flat subscription, so per-token cost genuinely does not apply.
    Loom still shows a figure — a blank receipt is worse — but marks it."""
    p = price_entry("go/glm-5.2")
    assert p.estimated and p.inp > 0


def test_a_mixed_fleet_totals_only_the_paid_rows():
    """The live shape: a Go orchestrator delegating to a free Zen explorer."""
    turn = TurnUsage()
    turn.add("glm-5.2", False, 22_724, 267, role="orchestrator",
             cache_read=4_352, billed_as="go/glm-5.2")
    turn.add("deepseek-v4-flash", False, 4_798, 442, role="explorer",
             cache_read=2_304, billed_as="zen/deepseek-v4-flash-free")
    rows = {a.role: turn.cost_of(a, u) for a, u in turn.actors.items()}
    assert rows["explorer"] == 0.0
    assert rows["orchestrator"] > 0.0
    assert turn.cloud_cost == pytest.approx(rows["orchestrator"])
    assert turn.has_estimates(), "the Go row must carry the ~ marker"


def test_every_figure_showing_a_cost_carries_the_same_marker():
    """A number marked estimated in one place and billed in another is worse than
    either — it reads as two different figures. Turn total, session total and the
    /status row all derive from the same flag."""
    t = _tracker(orchestrator="go/glm-5.2")
    t.start_turn()
    for bucket in (t.turn, t.session):
        bucket.add("glm-5.2", False, 22_724, 267, role="orchestrator",
                   cache_read=4_352, billed_as="go/glm-5.2")
    receipt = t.receipt()
    assert receipt.count("~$") >= 2, receipt   # turn figure and session figure
    assert t.turn.has_estimates() and t.session.has_estimates()


def test_a_session_estimate_shows_even_when_the_turn_is_clean():
    """An earlier turn on an unpriced model still makes the running total a
    guess, however exact this turn happened to be."""
    t = _tracker(orchestrator="claude-sonnet-5")
    t.session.add("glm-5.2", False, 1_000, 100, role="orchestrator", billed_as="go/glm-5.2")
    t.start_turn()
    for bucket in (t.turn, t.session):
        bucket.add("claude-sonnet-5", False, 5_000, 200, role="orchestrator")
    receipt = t.receipt()
    assert "· session ~$" in receipt, receipt
    assert not t.turn.has_estimates()
