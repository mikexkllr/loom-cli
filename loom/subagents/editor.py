"""editor — applies bounded code edits on a local mid-size model."""

from loom.subagents.base import ISOLATION_PREAMBLE, SubagentSpec
from loom.tools import WRITE_FS

SPEC = SubagentSpec(
    name="editor",
    description=(
        "Applies a specific, well-scoped code change to named files. Give it the "
        "exact files and the intended edit. Returns a summary of what changed."
    ),
    system_prompt=ISOLATION_PREAMBLE
    + (
        "You make precise edits to the files named in your task. Read a file "
        "before editing it. Prefer edit_file with a unique old_string over "
        "rewriting whole files. Keep changes minimal and matched to the "
        "surrounding code style. Report each file you changed and the gist of "
        "the change. Do not run shell commands."
    ),
    tools=WRITE_FS,
    mode="write",
)
