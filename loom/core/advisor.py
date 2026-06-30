"""Advisor / critic pattern (build step 4).

Two distinct mechanisms, both mirroring Claude Code / Codex:

* ``consult`` — an on-demand tool the orchestrator calls at decision gates. The
  strongest cloud model returns ~500 tokens of guidance and *never acts*.
* ``ReviewVerdict`` + :func:`review_prompt` — structure for the ``reviewer``
  subagent, which runs after significant writes and returns a risk level.

Both are intentionally cheap to wire: ``consult`` is a closure over config so the
orchestrator just gets a ready tool; the reviewer is a normal subagent whose
``response_format`` is the ``ReviewVerdict`` schema.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from loom.core.config import LoomConfig
from loom.core.model_router import build_model

# ----------------------------------------------------------------------------
# Risk model (shared by reviewer + advisor-threshold gating)
# ----------------------------------------------------------------------------


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

    def at_least(self, other: "RiskLevel") -> bool:
        order = {RiskLevel.low: 0, RiskLevel.medium: 1, RiskLevel.high: 2}
        return order[self] >= order[other]


class ReviewIssue(BaseModel):
    severity: RiskLevel
    description: str
    location: str = Field(default="", description="file:line if known")


class ReviewVerdict(BaseModel):
    """Structured output of the reviewer subagent."""

    risk: RiskLevel = Field(description="Overall risk of the reviewed change")
    approved: bool = Field(description="True if safe to proceed without human sign-off")
    issues: list[ReviewIssue] = Field(default_factory=list)
    summary: str = Field(default="", description="One-paragraph reviewer summary")


# ----------------------------------------------------------------------------
# consult() tool
# ----------------------------------------------------------------------------

_CONSULT_SYSTEM = (
    "You are Loom's Advisor — the strongest model available, consulted only at "
    "hard decision gates. You ADVISE, you do not act. Read the orchestrator's "
    "question and context summary, then return at most ~500 tokens of concrete "
    "guidance: the recommended approach, the main risk to watch, and a clear "
    "go / caution / stop signal. Be decisive. Do not request tools or files."
)


def make_consult_tool(config: LoomConfig) -> Callable:
    """Build the ``consult`` tool bound to the configured advisor model.

    The returned tool is added to the orchestrator's tool set. Calling it invokes
    the advisor model once and returns its guidance text — the orchestrator
    decides whether to follow it.
    """
    advisor_model = build_model(config.advisor, config)

    @tool
    def consult(question: str, context_summary: str) -> str:
        """Consult the Advisor (strongest model) for guidance on a hard decision.

        Call this before major work, when stuck after repeated failures, before
        destructive operations, or before declaring a task done. Provide the
        decision ``question`` and a compact ``context_summary`` of the task so
        far. Returns ~500 tokens of guidance. The Advisor never acts — you stay
        in control of what to do with the advice.
        """
        messages = [
            ("system", _CONSULT_SYSTEM),
            ("human", f"DECISION:\n{question}\n\nCONTEXT SO FAR:\n{context_summary}"),
        ]
        response = advisor_model.invoke(messages)
        return getattr(response, "content", str(response))

    return consult


# ----------------------------------------------------------------------------
# reviewer prompt + threshold gating
# ----------------------------------------------------------------------------

REVIEW_SYSTEM = (
    "You are Loom's Reviewer, a fast critic dispatched after code is written. "
    "Read the changed files and assess risk. Return a ReviewVerdict: an overall "
    "risk level (low/medium/high), an approval decision, a list of concrete "
    "issues with severity and location, and a one-paragraph summary. Be "
    "skeptical but proportionate — flag real correctness/security/data-loss "
    "risks, not style nits. High risk or unapproved => the orchestrator will "
    "stop and ask the human."
)


def review_prompt(changed_files: list[str], task: str) -> str:
    files = "\n".join(f"- {f}" for f in changed_files) or "(none reported)"
    return (
        f"Task that produced the change:\n{task}\n\n"
        f"Files reported changed:\n{files}\n\n"
        "Read these files, judge the risk, and return your ReviewVerdict."
    )


def should_consult_advisor(risk: RiskLevel, threshold: str) -> bool:
    """Given a risk assessment and the configured advisor threshold, decide
    whether the advisor should be consulted automatically."""
    gate = RiskLevel(threshold)
    return risk.at_least(gate)
