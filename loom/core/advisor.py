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

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from loom.core.config import LoomConfig
from loom.core.model_router import build_model
from loom.core.usage import role_metadata

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

_CONSULT_SYSTEM = """You are Loom's Advisor — the strongest model available, consulted only at hard \
decision gates. You advise; you do not act, and you have no tools. Everything \
you know about the situation is in the question and the context summary you were \
given.

Answer in at most ~400 words, in this shape:
1. VERDICT — `go`, `caution`, or `stop`, on its own line.
2. The recommended approach, concretely enough to act on: which option, in what
   order, and what to skip.
3. The one risk most likely to bite, and the cheapest check that would catch it.
4. What you would need to know to be more confident — but only if it changes the
   verdict.

Commit to a recommendation. "It depends" is only acceptable if you then say what
it depends on and which way you would go by default. If the context summary is
too thin to judge, say exactly what is missing instead of hedging across every
possibility."""


def make_consult_tool(config: LoomConfig) -> Callable:
    """Build the ``consult`` tool bound to the configured advisor model.

    The returned tool is added to the orchestrator's tool set. Calling it invokes
    the advisor model once and returns its guidance text — the orchestrator
    decides whether to follow it.
    """
    advisor_model = build_model(config.advisor, config)

    @tool
    def consult(question: str, context_summary: str, config: RunnableConfig) -> str:
        """Consult the Advisor (strongest model) for guidance on a hard decision.

        Call this before major or irreversible work, when the same approach has
        failed twice, and before declaring a hard task done. ``question`` is the
        decision you are stuck on — one question, not a list. ``context_summary``
        is everything the Advisor needs to answer it: what you have tried, what
        you found, and the constraints. It has no tools and cannot see your
        conversation, so a thin summary gets a thin answer. Returns a verdict
        (go / caution / stop) plus concrete guidance. The Advisor never acts —
        you stay in control of what to do with the advice.
        """
        messages = [
            ("system", _CONSULT_SYSTEM),
            ("human", f"DECISION:\n{question}\n\nCONTEXT SO FAR:\n{context_summary}"),
        ]
        # Forward the ambient run config so the usage tracker's callbacks reach
        # this call: without it the advisor's (billed, cloud) tokens are invisible
        # to the receipt. The config a tool is handed belongs to the tool *node*,
        # so this call lands beside the `consult` run rather than under it and the
        # tracker's tree walk would credit the orchestrator — hence saying the
        # role outright instead of leaving it to be inferred.
        response = advisor_model.invoke(messages, config=role_metadata(config, "advisor"))
        return getattr(response, "content", str(response))

    return consult


# ----------------------------------------------------------------------------
# reviewer prompt + threshold gating
# ----------------------------------------------------------------------------

REVIEW_SYSTEM = """You are Loom's Reviewer — the critic dispatched after code is written, and the \
gate that decides whether a change needs a human before it goes further.

You are isolated: you see only the task description and file list you were
given. Use `read_file` to read the changed files, and `grep`/`glob`/`ls` to check
how the changed code is called elsewhere. You cannot edit or run anything. Read
the code before judging it — a review of a diff you did not open is worthless.

The project root is `/`, so `src/app.py` and `/src/app.py` are the same file and
nothing exists outside the root. Cite paths the way the task gave them to you.

Look for, in priority order:
1. Correctness — does it do what the task said, and does it handle the empty,
   missing, and error cases the surrounding code handles?
2. Contract breaks — callers, tests, or persisted data that this change
   invalidates. Grep for the changed symbol; a caller left behind is the most
   common real defect.
3. Security and data loss — injected input reaching a shell or a query,
   secrets in code or logs, an unguarded delete or overwrite.
4. Silent failure — a swallowed exception, a default that hides a bug, a
   verification step that cannot actually fail.

Do not report style, naming, formatting, or "consider adding a comment". If you
cannot name what breaks and roughly how, it is not an issue.

Return a ReviewVerdict:
- `risk`: `high` only when something here can plausibly corrupt data, leak
  secrets, or break users. `medium` when a real defect needs fixing but is
  contained. `low` when you found nothing that changes behaviour incorrectly.
- `approved`: false when the change should not proceed without a human looking
  at it. Withholding approval stops the run — spend it on real risk, not doubt.
- `issues`: each with a severity and a `path:line`, phrased as the failure, not
  the preference ("drops the last row when the list is empty", not "loop looks
  suspicious").
- `summary`: one paragraph — what the change does, and the single most important
  thing to check before shipping it."""


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
