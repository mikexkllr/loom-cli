"""Artifact store + context-compaction helpers (build step 6).

Large tool outputs are written to ``.loom/artifacts/`` and replaced in-context
with a short path reference, so noisy output never bloats the orchestrator's
window. Subagent transcripts are stored under a separate namespace so main-thread
compaction never touches them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loom.core.config import LoomConfig
from loom.core.model_router import estimate_tokens


class ArtifactStore:
    """Offloads oversized strings to disk, returning a reference token."""

    def __init__(self, root: str | Path = ".loom/artifacts") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def offload(self, content: str, *, label: str = "output") -> str:
        """Persist ``content`` and return a path reference to put in-context."""
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        path = self.root / f"{label}-{digest}.txt"
        path.write_text(content, encoding="utf-8")
        tokens = estimate_tokens(content)
        return (
            f"[artifact://{path} | {tokens} tokens offloaded | "
            f"read with read_file('{path}') if you need the detail]"
        )

    def maybe_offload(self, content: str, config: LoomConfig, *, label: str = "output") -> str:
        """Offload only if ``content`` exceeds the configured token budget."""
        if estimate_tokens(content) <= config.artifact_offload_tokens:
            return content
        return self.offload(content, label=label)

    def read(self, ref_or_path: str) -> str:
        path = ref_or_path.replace("artifact://", "").strip("[] ")
        return Path(path).read_text(encoding="utf-8")


def summarization_middleware(config: LoomConfig):
    """Build the deepagents SummarizationMiddleware tuned to Loom's thresholds.

    Auto-compacts the orchestrator context at ``compaction_threshold`` of the
    window. Returns ``None`` (and the orchestrator skips it) if deepagents'
    middleware isn't importable, so the rest of Loom still works.
    """
    try:
        from deepagents.middleware.summarization import SummarizationMiddleware
    except Exception:
        try:
            from langchain.agents.middleware import SummarizationMiddleware  # type: ignore
        except Exception:
            return None

    window = config.context_window_for(config.orchestrator, default=200_000)
    trigger = int(window * config.compaction_threshold)
    try:
        return SummarizationMiddleware(
            model=config.orchestrator,
            max_tokens_before_summary=trigger,
        )
    except TypeError:
        # API drift in kwargs — fall back to defaults.
        return SummarizationMiddleware(model=config.orchestrator)
