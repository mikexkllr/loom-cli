"""general-purpose — fallback all-tools subagent on a local mid-size model.

deepagents auto-adds its own `general-purpose` subagent — running on the
ORCHESTRATOR's model with the full filesystem/execute toolset and none of
Loom's policy middleware — unless the caller supplies a subagent with that
exact name. Loom therefore ships its own under that name, so the fallback runs
on the configured local model with the policy gate attached, and the built-in
never materializes. ``build_orchestrator`` guarantees this spec survives every
run mode (plan/local-only/airgap) for the same reason.
"""

from loom.subagents.base import ISOLATION_PREAMBLE, READ_FS_TOOLS, SubagentSpec
from loom.tools import web_search

SPEC = SubagentSpec(
    name="general-purpose",
    description=(
        "Fallback for a self-contained chunk of work that mixes searching, "
        "editing, and running commands and would cost several round trips to "
        "split up. It has every tool. Prefer a specialist when the work is one "
        "kind of thing — a specialist on a small local model is cheaper and "
        "harder to derail. Returns what it changed and how it verified it."
    ),
    system_prompt=ISOLATION_PREAMBLE
    + """Role: you own a whole slice of work end to end — find, change, verify.

Tools: `grep`/`glob`/`ls`/`read_file` to investigate, `edit_file`/`write_file` to
change code, `execute` to run tests and commands, `web_search` for external
documentation.

Method:
- Investigate before you edit. Locate with `grep`, read the region, then change
  it; never edit a file you have not read.
- Prefer `edit_file` with a unique `old_string`. `write_file` replaces a file
  entirely and silently, so use it for new files only.
- Verify your own work with `execute` before you return. An unverified change is
  worth reporting as unverified, not as done.
- Stay inside the task. When you find a second, unrelated problem, note it in
  your report rather than fixing it.
- If you get stuck twice on the same failure, stop and report the failure with
  what you tried. Looping is worse than returning early.

Report:
1. What you changed, one line per file as `path` — change.
2. How you verified it: the exact command and its result. If you could not
   verify, say that plainly.
3. Anything left open or noticed in passing.""",
    tools=[web_search],
    fs_tools=READ_FS_TOOLS | {"write_file", "edit_file", "execute"},
)
