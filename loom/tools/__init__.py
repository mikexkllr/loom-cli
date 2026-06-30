"""Concrete tool implementations Loom subagents draw from.

Each tool is a plain LangChain ``@tool`` so subagent tool sets can be assembled
explicitly (matching the spec's per-agent capability table) rather than relying
on implicit middleware injection. Tools operate relative to a configurable root
so subagents in isolated git worktrees stay sandboxed to their tree.
"""

from loom.tools.filesystem import edit_file, glob_tool, grep_tool, ls, read_file, write_file
from loom.tools.shell import execute
from loom.tools.search import web_search

# Capability bundles referenced by subagent definitions.
READ_ONLY_FS = [ls, read_file, glob_tool, grep_tool]
WRITE_FS = [read_file, write_file, edit_file]
SEARCH = [grep_tool, glob_tool, web_search]
ALL_TOOLS = [ls, read_file, write_file, edit_file, glob_tool, grep_tool, execute, web_search]

__all__ = [
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob_tool",
    "grep_tool",
    "execute",
    "web_search",
    "READ_ONLY_FS",
    "WRITE_FS",
    "SEARCH",
    "ALL_TOOLS",
]
