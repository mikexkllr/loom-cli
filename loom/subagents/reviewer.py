"""reviewer — critic that risk-rates code changes (build step 4).

Runs on the Advisor's model unless config assigns the role one of its own:
the reviewer is the gate that decides whether a change needs human sign-off,
so it should be as sharp as the model you'd consult about that same change,
not a cheaper afterthought. Dispatched after significant writes. Returns a
structured :class:`ReviewVerdict` via ``response_format`` so the orchestrator
can gate on risk programmatically.
"""

from loom.core.advisor import REVIEW_SYSTEM, ReviewVerdict
from loom.subagents.base import READ_FS_TOOLS, SubagentSpec


def spec() -> SubagentSpec:
    return SubagentSpec(
        name="reviewer",
        description=(
            "Critic. Dispatch after a significant write — pass it the task and "
            "the list of changed files. Reads the change and returns a risk "
            "rating (low/medium/high), an approve/flag decision, and located "
            "issues. High risk or no approval means stop and ask the human."
        ),
        system_prompt=REVIEW_SYSTEM,
        tools=[],
        fs_tools=READ_FS_TOOLS,
        inherits="advisor",
    )


# The reviewer is the one subagent that returns structured output. We expose the
# schema here so the orchestrator can attach it as response_format.
RESPONSE_FORMAT = ReviewVerdict
SPEC = spec()
