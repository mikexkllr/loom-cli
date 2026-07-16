"""explorer — read-only codebase reconnaissance on a local small model."""

from loom.subagents.base import ISOLATION_PREAMBLE, SubagentSpec

SPEC = SubagentSpec(
    name="explorer",
    description=(
        "Read-only codebase explorer. Use to map structure, locate where "
        "something is implemented, or gather file paths and short excerpts. "
        "Returns a summary of findings, never full file dumps."
    ),
    system_prompt=ISOLATION_PREAMBLE
    + (
        "Your job is reconnaissance. Use ls/glob/grep to navigate and read_file "
        "sparingly. Report: the files that matter, the relevant symbols/line "
        "ranges, and a one-paragraph synthesis. Do NOT modify anything."
    ),
    tools=[],
    mode="read-only",
)
