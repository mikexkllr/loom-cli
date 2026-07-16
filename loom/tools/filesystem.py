"""Filesystem tools: ls, read_file, write_file, edit_file, glob, grep.

Sandboxed to the active root (see :mod:`loom.tools.sandbox`). Outputs are kept
compact; large reads are the orchestrator's cue to delegate to a subagent so the
bulk never lands in the main context window.

The tool signatures intentionally mirror the deepagents ``FilesystemMiddleware``
built-ins (``file_path``, ``offset``/``limit``, ``replace_all``, ``output_mode``)
so the deepagents filesystem system prompt and the Loom tool implementations
agree on parameter names and behavior.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable

from langchain_core.tools import tool

from loom.tools.sandbox import get_root, resolve_in_sandbox

_MAX_READ_BYTES = 400_000
_MAX_READ_LINES = 100


def _format_lines(lines: list[str], start_line: int = 1) -> str:
    """Return ``cat -n`` style output with 1-based line numbers."""
    return "\n".join(f"{i}\t{line}" for i, line in enumerate(lines, start_line))


@tool
def ls(path: str = ".") -> str:
    """List files and directories at ``path`` (absolute virtual or relative)."""
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
def read_file(file_path: str, offset: int = 0, limit: int = 100) -> str:
    """Read up to ``limit`` lines from ``file_path`` starting at ``offset``.

    ``offset`` is 0-indexed. Returns ``cat -n`` style line-numbered output so the
    model can refer to exact lines. For large files, read in chunks using
    ``offset`` and ``limit``.
    """
    target = resolve_in_sandbox(file_path)
    if not target.is_file():
        return f"error: {file_path} is not a file"
    try:
        data = target.read_bytes()
    except OSError as exc:
        return f"error: could not read {file_path}: {exc}"
    if len(data) > _MAX_READ_BYTES:
        return (
            f"error: {file_path} is {len(data)} bytes (> {_MAX_READ_BYTES}). "
            "Use grep to find the relevant region, or read it in a subagent."
        )
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if offset < 0:
        offset = 0
    if limit < 0:
        limit = _MAX_READ_LINES
    page = lines[offset : offset + limit]
    if not page:
        return "(empty)"
    return _format_lines(page, start_line=offset + 1)


@tool
def write_file(file_path: str, content: str) -> str:
    """Write ``content`` to ``file_path``, creating parent directories as needed.

    Overwrites the file if it already exists. Prefer ``edit_file`` for changing
    existing files.
    """
    target = resolve_in_sandbox(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {file_path}"


@tool
def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace ``old_string`` with ``new_string`` in ``file_path``.

    By default ``old_string`` must occur exactly once. Set ``replace_all=True``
    to replace every occurrence.
    """
    target = resolve_in_sandbox(file_path)
    if not target.is_file():
        return f"error: {file_path} is not a file"
    text = target.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        return f"error: old_string not found in {file_path}"
    if not replace_all and count > 1:
        return f"error: old_string appears {count} times in {file_path}; make it unique or use replace_all=True"
    replacements = -1 if replace_all else 1
    target.write_text(text.replace(old_string, new_string, replacements), encoding="utf-8")
    return f"edited {file_path} ({count} replacement{'s' if count != 1 else ''})"


@tool
def glob(pattern: str, path: str | None = None) -> str:
    """Find files matching ``pattern`` under ``path`` (defaults to project root)."""
    base = resolve_in_sandbox(path or ".")
    root = get_root()
    matches = sorted(str(p.relative_to(root)) for p in base.rglob(pattern) if p.is_file())
    if not matches:
        return "(no matches)"
    return "\n".join(matches[:200])


def _iter_files(base: Path, pattern: str | None) -> Iterable[tuple[Path, str, int, str]]:
    """Yield (file_path, root_relative_path, line_number, line_text) for ``grep``."""
    root = get_root()
    if base.is_file():
        files = [base]
        glob_filter = "**/*"
    else:
        files = base.rglob("*")
        glob_filter = pattern or "**/*"

    for fp in files:
        if not fp.is_file():
            continue
        try:
            rel = str(fp.relative_to(root))
        except ValueError:
            rel = str(fp)
        if not fnmatch.fnmatch(rel, glob_filter):
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            yield fp, rel, i, line


@tool
def grep(
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    output_mode: str = "files_with_matches",
) -> str:
    """Search files under ``path`` for the literal ``pattern``.

    ``output_mode`` controls the result format:
      - ``files_with_matches``: file paths only (default)
      - ``content``: ``file:line:text`` for each match
      - ``count``: ``file: count`` totals
    """
    if output_mode not in {"files_with_matches", "content", "count"}:
        return f"error: invalid output_mode '{output_mode}' (choose files_with_matches, content, or count)"

    base = resolve_in_sandbox(path or ".")
    results: list[str] = []
    counts: dict[str, int] = {}
    seen_files: set[str] = set()
    limit = 100

    for fp, rel, i, line in _iter_files(base, glob):
        if pattern not in line:
            continue
        if output_mode == "files_with_matches":
            if rel not in seen_files:
                seen_files.add(rel)
                results.append(rel)
        elif output_mode == "content":
            results.append(f"{rel}:{i}:{line.strip()[:200]}")
        else:  # count
            counts[rel] = counts.get(rel, 0) + 1

        if len(results) >= limit:
            break

    if output_mode == "count":
        if not counts:
            return "(no matches)"
        return "\n".join(f"{rel}: {c}" for rel, c in sorted(counts.items()))

    if not results:
        return "(no matches)"
    output = "\n".join(results)
    if len(results) >= limit:
        output += "\n... (truncated at 100 hits)"
    return output
