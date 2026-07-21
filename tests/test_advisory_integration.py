"""Advisory pattern integration: local agent -> consult -> advisor guidance.

Tests that:
1. ``make_consult_tool`` builds a callable LangChain tool wired to the advisor model.
2. Invoking the tool sends the right prompt structure to the advisor model and
   returns its content verbatim.
3. The ``should_consult_advisor`` threshold gate correctly decides whether to act
   on a reviewer's risk verdict.
4. The full loop: reviewer returns a risk level, threshold gate fires, orchestrator
   calls consult, advisor replies with go/caution/stop guidance.
"""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langchain_core")

from unittest.mock import MagicMock, patch

from loom.core.advisor import (
    RiskLevel,
    ReviewVerdict,
    make_consult_tool,
    should_consult_advisor,
)
from loom.core.config import LoomConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**kw) -> LoomConfig:
    defaults = dict(
        orchestrator="ollama/qwen3:4b",
        advisor="claude-opus-4-8",
        advisor_threshold="medium",
    )
    defaults.update(kw)
    return LoomConfig(**defaults)

def _fake_advisor_response(text: str):
    """Return a mock LangChain message-like object with .content."""
    msg = MagicMock()
    msg.content = text
    return msg

# ---------------------------------------------------------------------------
# 1. make_consult_tool produces a tool with the right metadata
# ---------------------------------------------------------------------------

def test_consult_tool_is_registered_with_correct_name():
    config = _config()
    with patch("loom.core.advisor.build_model") as mock_build:
        mock_build.return_value = MagicMock()
        tool = make_consult_tool(config)

    assert tool.name == "consult"
    assert "Advisor" in tool.description or "advisor" in tool.description.lower()

# ---------------------------------------------------------------------------
# 2. Calling the tool routes the prompt to the advisor model
# ---------------------------------------------------------------------------

def test_consult_tool_invokes_advisor_model_with_question_and_context():
    config = _config()
    mock_model = MagicMock()
    mock_model.invoke.return_value = _fake_advisor_response("GO — safe to proceed.")

    with patch("loom.core.advisor.build_model", return_value=mock_model):
        consult = make_consult_tool(config)

    result = consult.invoke({
        "question": "Should I delete the production database?",
        "context_summary": "Running a cleanup script; db is backed up.",
    })

    # The advisor model must have been called exactly once
    mock_model.invoke.assert_called_once()
    call_messages = mock_model.invoke.call_args[0][0]  # positional arg: list of (role, text)

    # System prompt is first, human turn is second
    assert call_messages[0][0] == "system"
    assert "ADVISE" in call_messages[0][1] or "Advisor" in call_messages[0][1]

    assert call_messages[1][0] == "human"
    assert "Should I delete the production database?" in call_messages[1][1]
    assert "Running a cleanup script" in call_messages[1][1]

    # The tool returns the advisor's content
    assert result == "GO — safe to proceed."

# ---------------------------------------------------------------------------
# 3. Threshold gate: only escalate when risk >= threshold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("risk,threshold,expected", [
    (RiskLevel.high,   "high",   True),
    (RiskLevel.medium, "high",   False),
    (RiskLevel.low,    "high",   False),
    (RiskLevel.high,   "medium", True),
    (RiskLevel.medium, "medium", True),
    (RiskLevel.low,    "medium", False),
    (RiskLevel.high,   "low",    True),
    (RiskLevel.medium, "low",    True),
    (RiskLevel.low,    "low",    True),
])
def test_should_consult_advisor_threshold_matrix(risk, threshold, expected):
    assert should_consult_advisor(risk, threshold) == expected

# ---------------------------------------------------------------------------
# 4. Full loop: reviewer verdict -> gate -> consult -> advisory response
# ---------------------------------------------------------------------------

def test_full_advisory_loop_high_risk_fires_consult():
    """Simulate the orchestrator's advisory loop:
    - The reviewer returns a HIGH-risk verdict.
    - The threshold is 'medium' (high >= medium => consult fires).
    - The consult tool is called; advisor says 'STOP'.
    - The orchestrator receives 'STOP' guidance and can decide to halt.
    """
    # Step 1: reviewer produces a verdict
    verdict = ReviewVerdict(
        risk=RiskLevel.high,
        approved=False,
        summary="Deletes production data without backup.",
    )

    # Step 2: threshold gate
    config = _config(advisor_threshold="medium")
    should_fire = should_consult_advisor(verdict.risk, config.advisor_threshold)
    assert should_fire, "High risk must trigger consult at medium threshold"

    # Step 3: consult the advisor (mocked)
    mock_model = MagicMock()
    advisory_text = "STOP — this change is irreversible without a verified backup."
    mock_model.invoke.return_value = _fake_advisor_response(advisory_text)

    with patch("loom.core.advisor.build_model", return_value=mock_model):
        consult = make_consult_tool(config)

    guidance = consult.invoke({
        "question": "Reviewer flagged HIGH risk on data deletion. Proceed?",
        "context_summary": verdict.summary,
    })

    # Step 4: orchestrator receives stop signal
    assert "STOP" in guidance
    assert "irreversible" in guidance

def test_full_advisory_loop_low_risk_skips_consult_at_high_threshold():
    """When risk is low and threshold is high, the gate stays closed — no advisor call needed."""
    verdict = ReviewVerdict(
        risk=RiskLevel.low,
        approved=True,
        summary="Trivial docstring update.",
    )

    config = _config(advisor_threshold="high")
    should_fire = should_consult_advisor(verdict.risk, config.advisor_threshold)

    assert not should_fire, "Low risk must NOT trigger consult at high threshold"
    # No model is built; if the gate is respected, no API calls happen.

def test_consult_returns_string_when_model_returns_non_content_object():
    """Fallback: if the advisor model returns something without .content, str() is used."""
    config = _config()
    mock_model = MagicMock()
    # Return an object without a meaningful .content attribute
    mock_response = MagicMock(spec=[])  # spec=[] => no attributes at all
    mock_model.invoke.return_value = mock_response

    with patch("loom.core.advisor.build_model", return_value=mock_model):
        consult = make_consult_tool(config)

    # Should not raise; falls back to str(response)
    result = consult.invoke({
        "question": "Anything?",
        "context_summary": "Minimal context.",
    })
    assert isinstance(result, str)