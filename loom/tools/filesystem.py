"""Filesystem tools: ls, read_file, write_file, edit_file, glob, grep.

Sandboxed to the active root (see :mod:`loom.tools.sandbox`). Outputs are kept
compact; large reads are the orchestrator's cue to delegate to a subagent so the
bulk never lands in the main context window.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from langchain_core.tools import tool

from loom.tools.sandbox import get_root, resolve_in_sandbox

_MAX_READ_BYTES = 400_000


@tool
def ls(path: str = ".") -> str:
    """List files and directories at ``path`` (relative to the project root)."""
    target = resolve_in_sandbox(path)
    if not target.exists():
        return f"error: {path} does not exist"
    if target.is_file():
        return target.name
    entries = []
    for child in sorted(target.iterdir()):
        suffix = "/" if child.is_dir() else ""
        entries.append(child.name + suffix)
    return "\n".join(entries) if entries else "(empty)"


@tool
def read_file(path: str) -> str:
    """Read and return the full text contents of a file."""
    target = resolve_in_sandbox(path)
    if not target.is_file():
        return f"error: {path} is not a file"
    data = target.read_bytes()
    if len(data) > _MAX_READ_BYTES:
        return (
            f"error: {path} is {len(data)} bytes (> {_MAX_READ_BYTES}). "
            "Use grep to find the relevant region, or read it in a subagent."
        )
    return data.decode("utf-8", errors="replace")


@tool
def write_file(path: str, content: str) -> str:
    """Write ``content`` to ``path``, creating parent directories as needed."""
    target = resolve_in_sandbox(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace the first exact occurrence of ``old_string`` with ``new_string``.

    Fails if ``old_string`` is absent or ambiguous (appears more than once),
    mirroring a precise single-edit semantics.
    """
    target = resolve_in_sandbox(path)
    if not target.is_file():
        return f"error: {path} is not a file"
    text = target.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        return f"error: old_string not found in {path}"
    if count > 1:
        return f"error: old_string appears {count} times in {path}; make it unique"
    target.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
    return f"edited {path}"


@tool
def glob_tool(pattern: str) -> str:
    """Find files matching a glob ``pattern`` (e.g. ``**/*.py``)."""
    root = get_root()
    matches = [str(p.relative_to(root)) for p in root.glob(pattern) if p.is_file()]
    matches.sort()
    if not matches:
        return "(no matches)"
    return "\n".join(matches[:200])


@tool
def grep_tool(pattern: str, path: str = ".", glob: str = "**/*") -> str:
    """Search files under ``path`` for a regex ``pattern``.

    Returns up to 100 ``file:line:text`` hits. ``glob`` narrows which files are
    scanned (default: every file).
    """
    base = resolve_in_sandbox(path)
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"error: invalid regex: {exc}"

    root = get_root()
    results: list[str] = []
    files = [base] if base.is_file() else base.rglob("*")
    for fp in files:
        if not fp.is_file():
            continue
        if not fnmatch.fnmatch(str(fp.relative_to(root)), glob):
            continue
        try:
            for i, line in enumerate(fp.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    results.append(f"{fp.relative_to(root)}:{i}:{line.strip()[:200]}")
                    if len(results) >= 100:
                        return "\n".join(results) + "\n... (truncated at 100 hits)"
        except OSError:
            continue
    return "\n".join(results) if results else "(no matches)"
