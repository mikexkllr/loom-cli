"""Repo map + @file mentions — orient the orchestrator without exploration turns.

``repo_map`` builds a compact tree of the project (via ``git ls-files`` when
available, else a bounded walk) that the REPL prepends to the first message.
``expand_mentions`` inlines ``@path`` references from the user's prompt so
"look at @src/api.py" carries the file with it.
"""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from pathlib import Path

_MAX_MAP_LINES = 60
_MAX_MENTION_BYTES = 100_000
_MAX_MENTION_LINES = 400

_MENTION_RE = re.compile(r"@([A-Za-z0-9_\-./]+)")


def _git_files(root: Path) -> list[str] | None:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, timeout=10
        )
        if out.returncode != 0:
            return None
        return [line for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        return None


def _walk_files(root: Path, cap: int = 2000) -> list[str]:
    skip = {".git", "node_modules", ".venv", "venv", "__pycache__", ".loom", "dist", "build"}
    files: list[str] = []
    for path in root.rglob("*"):
        if len(files) >= cap:
            break
        if path.is_file() and not any(part in skip for part in path.parts):
            files.append(str(path.relative_to(root)))
    return files


def repo_map(root: str | Path = ".") -> str:
    """A compact, capped directory summary: top-level entries with file counts
    and the most useful filenames spelled out."""
    root = Path(root).resolve()
    files = _git_files(root)
    if files is None:
        files = _walk_files(root)
    if not files:
        return ""

    by_top: dict[str, list[str]] = defaultdict(list)
    for f in files:
        top = f.split("/", 1)[0] if "/" in f else "."
        by_top[top].append(f)

    lines = [f"{len(files)} files"]
    for top in sorted(by_top, key=lambda t: (-len(by_top[t]), t)):
        group = by_top[top]
        if top == ".":
            for f in sorted(group)[:12]:
                lines.append(f)
            continue
        # Show the directory with count, plus a few representative files.
        lines.append(f"{top}/ ({len(group)} files)")
        for f in sorted(group)[:6]:
            lines.append(f"  {f}")
        if len(group) > 6:
            lines.append(f"  … +{len(group) - 6} more")
        if len(lines) >= _MAX_MAP_LINES:
            lines.append("…")
            break
    return "\n".join(lines[: _MAX_MAP_LINES + 1])


def expand_mentions(text: str, root: str | Path = ".") -> str:
    """Append the contents of every existing ``@path`` file mentioned in text."""
    root = Path(root).resolve()
    attachments: list[str] = []
    seen: set[str] = set()
    for match in _MENTION_RE.finditer(text):
        rel = match.group(1).rstrip(".")
        if rel in seen:
            continue
        path = (root / rel).resolve()
        try:
            path.relative_to(root)  # stay inside the sandbox
        except ValueError:
            continue
        if not path.is_file() or path.stat().st_size > _MAX_MENTION_BYTES:
            continue
        seen.add(rel)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = content.splitlines()
        if len(lines) > _MAX_MENTION_LINES:
            content = "\n".join(lines[:_MAX_MENTION_LINES]) + f"\n… (+{len(lines) - _MAX_MENTION_LINES} lines truncated)"
        attachments.append(f"[Attached file: {rel}]\n```\n{content}\n```")
    if not attachments:
        return text
    return text + "\n\n" + "\n\n".join(attachments)
