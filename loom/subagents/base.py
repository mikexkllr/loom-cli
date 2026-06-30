"""Shared helper for assembling deepagents subagent definitions.

A Loom subagent is a deepagents subagent dict:

    {"name", "description", "system_prompt", "tools", "model", "middleware"}

with two Loom-specific conventions:
  * ``model`` is a concrete LangChain model instance built from config (so local
    Ollama models carry their base_url / num_ctx), not just a string.
  * local subagents get a :class:`PromptSizeGuard` so an oversized prompt
    escalates to the cloud instead of failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from loom.core.config import LoomConfig
from loom.core.model_router import build_model
from loom.middleware.prompt_size_guard import PromptSizeGuard


@dataclass(frozen=True)
class SubagentSpec:
    """Static description of a subagent, independent of config/model wiring."""

    name: str
    description: str
    system_prompt: str
    tools: list[Callable] = field(default_factory=list)
    mode: str = "read-only"  # "read-only" | "write"

    def build(self, config: LoomConfig) -> dict[str, Any]:
        """Resolve this spec against config into a deepagents subagent dict."""
        model_string = config.subagents.get(self.name, config.subagents.get("general", config.orchestrator))
        model = build_model(model_string, config)

        middleware: list[Any] = []
        if config.is_local(model_string):
            middleware.append(PromptSizeGuard(model_string, config))

        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": list(self.tools),
            "model": model,
            "middleware": middleware,
        }


# Shared preamble injected into every subagent: the isolation contract.
ISOLATION_PREAMBLE = (
    "You are an isolated Loom subagent. You run in your own fresh context window "
    "and see only the task the orchestrator handed you plus the file paths it "
    "named. When finished, return a TERSE summary of what you found or did — not "
    "raw file contents or full logs. The orchestrator depends on your summary "
    "staying small to protect its own context. Never ask the user questions; if "
    "blocked, return what you learned and why you stopped.\n\n"
)
