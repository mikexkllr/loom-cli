"""bash — runs sandboxed shell commands on a local mid-size model."""

from loom.subagents.base import ISOLATION_PREAMBLE, SubagentSpec
from loom.tools import execute, write_file

SPEC = SubagentSpec(
    name="bash",
    description=(
        "Runs build/test/lint/git commands in a sandboxed shell and reports the "
        "outcome. Use for anything that needs a process to run. Returns exit "
        "status and a distilled summary — not raw logs."
    ),
    system_prompt=ISOLATION_PREAMBLE
    + (
        "You execute shell commands to accomplish the task (run tests, build, "
        "install, git ops). Commands run in the project/worktree root with a "
        "120s timeout. Read the output yourself and report only the verdict: did "
        "it pass/fail, the key error if any, and next-step suggestion. The raw "
        "log stays with you — quarantine it from the orchestrator's context."
    ),
    tools=[execute, write_file],
    mode="write",
)
