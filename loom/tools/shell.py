"""Sandboxed shell execution tool.

``execute`` runs a command with the working directory pinned to the sandbox
root and a hard timeout. It is intentionally the only tool that can mutate the
world beyond the filesystem, so it is handed only to the ``bash`` and
``general`` subagents.
"""

from __future__ import annotations

import subprocess

from langchain_core.tools import tool

from loom.tools.sandbox import get_root

_TIMEOUT_SECONDS = 120
_MAX_OUTPUT_CHARS = 8000


@tool
def execute(command: str) -> str:
    """Run a shell ``command`` in the project/worktree root and return output.

    Captures stdout+stderr, enforces a 120s timeout, and truncates very large
    output (the orchestrator should delegate noisy commands to a subagent so the
    bulk never reaches the main context).
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(get_root()),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {_TIMEOUT_SECONDS}s"

    output = (proc.stdout or "") + (proc.stderr or "")
    if len(output) > _MAX_OUTPUT_CHARS:
        head = output[: _MAX_OUTPUT_CHARS // 2]
        tail = output[-_MAX_OUTPUT_CHARS // 2 :]
        output = f"{head}\n... (truncated {len(output) - _MAX_OUTPUT_CHARS} chars) ...\n{tail}"

    return f"[exit {proc.returncode}]\n{output}".strip()
