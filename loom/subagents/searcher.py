"""searcher — read-only code + optional web search on a local small model."""

from loom.subagents.base import ISOLATION_PREAMBLE, SubagentSpec
from loom.tools import SEARCH

SPEC = SubagentSpec(
    name="searcher",
    description=(
        "Searches the codebase (grep/glob) and, when enabled, the web for "
        "answers to a focused question. Returns the answer plus the references "
        "it relied on."
    ),
    system_prompt=ISOLATION_PREAMBLE
    + (
        "You answer a focused lookup question. Search the code with grep/glob; "
        "if web_search is available and the answer is external (API docs, error "
        "meanings), use it. Return a direct answer and cite the file:line or URL "
        "you used. Keep it short."
    ),
    tools=SEARCH,
    mode="read-only",
)
