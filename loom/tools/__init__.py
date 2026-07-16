"""Concrete tool implementations Loom subagents draw from.

Each tool is a plain LangChain ``@tool`` so subagent tool sets can be assembled
explicitly (matching the spec's per-agent capability table) rather than relying
on implicit middleware injection. Tools operate relative to a configurable root
so subagents in isolated git worktrees stay sandboxed to their tree.

The tool names and signatures intentionally match the deepagents
``FilesystemMiddleware`` built-ins so Loom's custom implementations shadow them
cleanly and the deepagents filesystem system prompt stays valid.
"""

from loom.tools.filesystem import edit_file, glob, grep, ls, read_file, write_file
from loom.tools.shell import execute
from loom.tools.search import web_search

# Capability bundles referenced by subagent definitions.
READ_ONLY_FS = [ls, read_file, glob, grep]
WRITE_FS = [read_file, write_file, edit_file]
SEARCH = [grep, glob, web_search]
ALL_TOOLS = [ls, read_file, write_file, edit_file, glob, grep, execute, web_search]

__all__ = [
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",
    "web_search",
    "READ_ONLY_FS",
    "WRITE_FS",
    "SEARCH",
    "ALL_TOOLS",
]
