"""general — fallback all-tools subagent on a local mid-size model.

deepagents auto-adds a `general-purpose` subagent when none is supplied; Loom
ships its own so the fallback runs on the configured local model. The filesystem,
terminal, and todo tools come from deepagents; Loom only adds web_search here.
"""

from loom.subagents.base import ISOLATION_PREAMBLE, SubagentSpec
from loom.tools import web_search

SPEC = SubagentSpec(
    name="general",
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
