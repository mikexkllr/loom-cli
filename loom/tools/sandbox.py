"""Path sandboxing shared by the filesystem and shell tools.

All tool file access is confined to a *root* directory (the project, or a
subagent's git worktree). This keeps an isolated write agent from wandering
outside its tree, and makes the tools safe to hand to a local model.
"""

from __future__ import annotations

import contextvars
from pathlib import Path

# The active sandbox root for the current execution context. The orchestrator /
# subagent runner sets this; tools read it. A ContextVar (not a global) so
# parallel worktree agents each see their own root.
_ROOT: contextvars.ContextVar[Path] = contextvars.ContextVar("loom_sandbox_root", default=Path.cwd())


def set_root(path: str | Path) -> None:
    _ROOT.set(Path(path).resolve())


def get_root() -> Path:
    return _ROOT.get()


def resolve_in_sandbox(relative_or_abs: str) -> Path:
    """Resolve a tool-supplied path and assert it stays under the sandbox root.

    Supports both ordinary relative paths and *virtual absolute* paths
    (e.g. ``/cs_ai_quiz/quiz.py``) produced by deepagents' filesystem tools.
    A real absolute path that is already under the root is allowed unchanged;
    any other absolute path is treated as virtual and anchored to the root.

    Raises ``ValueError`` on traversal outside the root — surfaced back to the
    model as a tool error so it can correct course.
    """
    root = get_root()
    candidate = Path(relative_or_abs)
    if candidate.is_absolute():
        # First, treat it as a real absolute path; resolve and verify it is
        # under the root. If it is not, treat the leading ``/`` as a virtual
        # root and anchor the remainder under the sandbox root.
        resolved = candidate.resolve()
        if root not in resolved.parents and resolved != root:
            virtual = candidate.relative_to("/")
            resolved = (root / virtual).resolve()
        full = resolved
    else:
        full = (root / candidate).resolve()
    if root not in full.parents and full != root:
        raise ValueError(
            f"Path {relative_or_abs!r} escapes the sandbox root {root}. "
            "Tools may only touch files under the project / worktree root."
        )
    return full
