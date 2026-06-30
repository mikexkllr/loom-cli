"""Git worktree isolation for parallel write agents (build step 7).

When several write-capable subagents run concurrently, each gets its own git
worktree so their edits never collide (mirrors Codex's recommendation). The
worktree is created off the current HEAD on a throwaway branch; on success the
orchestrator can merge it back, and it is cleaned up afterwards.

If the project isn't a git repo, worktree creation degrades to a plain temp
directory copy is *not* attempted — instead we signal the caller to run the
agent in-place, since isolation can't be guaranteed.
"""

from __future__ import annotations

import contextlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


def is_git_repo(path: str | Path = ".") -> bool:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(path),
            capture_output=True,
            text=True,
        )
        return out.returncode == 0 and out.stdout.strip() == "true"
    except FileNotFoundError:
        return False


@dataclass
class Worktree:
    path: Path
    branch: str
    created: bool  # False => fell back to in-place (no git)


@contextlib.contextmanager
def worktree_for(agent_name: str, repo: str | Path = "."):
    """Context manager yielding an isolated :class:`Worktree` for an agent.

    Usage::

        with worktree_for("editor") as wt:
            tools.sandbox.set_root(wt.path)
            ...  # run the write subagent against wt.path

    Cleans up the worktree on exit. If not a git repo, yields an in-place
    worktree (``created=False``) pointing at the repo root.
    """
    repo = Path(repo).resolve()
    if not is_git_repo(repo):
        yield Worktree(path=repo, branch="", created=False)
        return

    branch = f"loom/{agent_name}-{_short_id()}"
    wt_path = Path(tempfile.mkdtemp(prefix=f"loom-{agent_name}-"))
    subprocess.run(
        ["git", "worktree", "add", "-b", branch, str(wt_path), "HEAD"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    try:
        yield Worktree(path=wt_path, branch=branch, created=True)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(wt_path)],
            cwd=str(repo),
            capture_output=True,
        )
        subprocess.run(["git", "branch", "-D", branch], cwd=str(repo), capture_output=True)


def _short_id() -> str:
    # Avoid Math.random-style nondeterminism concerns; derive from a temp name.
    return Path(tempfile.mktemp()).name[-6:]
