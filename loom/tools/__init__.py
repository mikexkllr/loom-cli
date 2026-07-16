"""Additional tools that deepagents does not provide out of the box.

Filesystem, shell, and todo tools are supplied by deepagents'
``FilesystemMiddleware`` and ``TodoListMiddleware``; Loom only defines the
``web_search`` tool here.
"""

from loom.tools.search import web_search

__all__ = ["web_search"]
