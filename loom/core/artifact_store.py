"""Artifact store + context-compaction helpers (build step 6).

Large tool outputs are written to ``.loom/artifacts/`` and replaced in-context
with a short path reference, so noisy output never bloats the orchestrator's
window. Subagent transcripts are stored under a separate namespace so main-thread
compaction never touches them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

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


def compaction_trigger(config: LoomConfig, model_string: str, *, default: int = 200_000) -> int:
    """Token count at which this model's context should auto-compact.

    ``compaction_threshold`` of the model's real window, floored so that a tiny
    or mis-detected window can't set a trigger the very first message trips.
    """
    window = config.context_window_for(model_string, default=default)
    return max(4_000, int(window * config.compaction_threshold))


def summarization_middleware(
    config: LoomConfig,
    backend: Any,
    *,
    model: Any | None = None,
    model_string: str | None = None,
):
    """Build the deepagents SummarizationMiddleware tuned to Loom's thresholds.

    Auto-compacts at ``compaction_threshold`` of the model's real context window,
    and offloads the evicted history through ``backend`` — which for Loom routes
    ``/conversation_history/`` under ``.loom/sessions/``, so a compacted turn
    stays re-readable instead of being dropped.

    Two reasons this overrides the instance ``create_deep_agent`` builds itself
    (it does so by name match, see :func:`loom.core.orchestrator.build_orchestrator`):
    ``compaction_threshold`` is a documented Loom knob and nothing read it before,
    and deepagents' own default derives the trigger from the model's published
    profile — which ChatOllama has none of, leaving a local orchestrator with a
    flat 170K-token trigger it can never reach before overflowing a 32K window.

    Returns ``None`` (and the caller keeps deepagents' default) if the middleware
    isn't importable or rejects these arguments, so API drift degrades instead of
    breaking the run.
    """
    try:
        from deepagents.middleware.summarization import SummarizationMiddleware
    except Exception:
        try:
            from langchain.agents.middleware import SummarizationMiddleware  # type: ignore
        except Exception:
            return None

    model_string = model_string or config.orchestrator
    trigger = compaction_trigger(config, model_string)

    if model is None:
        # deepagents resolves a bare string through init_chat_model, which does
        # not understand Loom's "ollama/tag" form (nor its endpoint or num_ctx).
        # Build the model with Loom's own router instead of handing over a string
        # it will reject.
        from loom.core.model_router import build_model

        try:
            model = build_model(model_string, config)
        except Exception:
            return None

    try:
        return SummarizationMiddleware(model, backend=backend, trigger=("tokens", trigger))
    except (TypeError, ValueError):
        return None
