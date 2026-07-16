"""reviewer — cloud critic that risk-rates code changes (build step 4).

Runs on a cheap cloud model (Haiku / gpt-4o-mini). Dispatched after significant
writes. Returns a structured :class:`ReviewVerdict` via ``response_format`` so
the orchestrator can gate on risk programmatically.
"""

from loom.core.advisor import REVIEW_SYSTEM, ReviewVerdict
from loom.subagents.base import SubagentSpec
from loom.tools import grep, read_file


def spec() -> SubagentSpec:
    return SubagentSpec(
        name="reviewer",
        description=(
            "Cloud critic. Dispatch after a significant code write to get a risk "
            "rating (low/medium/high), an approve/flag decision, and a list of "
            "issues. High risk => surface to the human before continuing."
        ),
        system_prompt=REVIEW_SYSTEM,
        tools=[read_file, grep],
        mode="read-only",
    )


# The reviewer is the one subagent that returns structured output. We expose the
# schema here so the orchestrator can attach it as response_format.
RESPONSE_FORMAT = ReviewVerdict
SPEC = spec()
