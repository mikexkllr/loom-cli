"""general-purpose — fallback all-tools subagent on a local mid-size model.

deepagents auto-adds its own `general-purpose` subagent — running on the
ORCHESTRATOR's model with the full filesystem/execute toolset and none of
Loom's policy middleware — unless the caller supplies a subagent with that
exact name. Loom therefore ships its own under that name, so the fallback runs
on the configured local model with the policy gate attached, and the built-in
never materializes. ``build_orchestrator`` guarantees this spec survives every
run mode (plan/local-only/airgap) for the same reason.
"""

from loom.subagents.base import ISOLATION_PREAMBLE, SubagentSpec
from loom.tools import web_search

SPEC = SubagentSpec(
    name="general-purpose",
    description=(
        "General-purpose fallback for tasks that mix reading, editing, and "
        "running commands and don't fit a specialized subagent. Has every tool."
    ),
    system_prompt=ISOLATION_PREAMBLE
    + (
        "You handle multi-step tasks that span exploring, editing, and running "
        "commands. Plan briefly, act, verify your own work, then return a "
        "summary of what you changed and how you confirmed it. Keep the "
        "orchestrator's context clean — summarize, don't dump."
    ),
    tools=[web_search],
    mode="write",
)
