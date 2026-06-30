"""Advisor risk model + threshold gating (no model calls)."""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langchain_core")

from loom.core.advisor import RiskLevel, ReviewVerdict, should_consult_advisor


def test_risk_ordering():
    assert RiskLevel.high.at_least(RiskLevel.low)
    assert RiskLevel.medium.at_least(RiskLevel.medium)
    assert not RiskLevel.low.at_least(RiskLevel.high)


def test_threshold_gating():
    # threshold=high => only high risk consults
    assert should_consult_advisor(RiskLevel.high, "high")
    assert not should_consult_advisor(RiskLevel.medium, "high")
    # threshold=low => everything consults
    assert should_consult_advisor(RiskLevel.low, "low")


def test_review_verdict_schema():
    v = ReviewVerdict(risk=RiskLevel.medium, approved=False, summary="check this")
    assert v.risk == RiskLevel.medium
    assert v.approved is False
    assert v.issues == []
