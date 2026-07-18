"""Shared helper for assembling deepagents subagent definitions.

A Loom subagent is a deepagents subagent dict:

    {"name", "description", "system_prompt", "tools", "model", "middleware"}

with three Loom-specific conventions:
  * ``model`` is a concrete LangChain model instance built from config (so local
    Ollama models carry their base_url / num_ctx), not just a string.
  * local subagents get a :class:`PromptSizeGuard` so an oversized prompt
    escalates to the cloud instead of failing.
  * every subagent gets its own :class:`PolicyMiddleware` (when Settings are
    available) and a :class:`ToolExclusionMiddleware` matching its declared
    capabilities. deepagents builds a fresh middleware stack per subagent —
    the orchestrator's middleware does NOT propagate down — so permissions,
    hooks, /undo snapshots, and read-only enforcement must be attached here,
    where the write/execute tools actually run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from loom.core.config import LoomConfig
from loom.core.model_router import build_model
from loom.middleware.prompt_size_guard import PromptSizeGuard

if TYPE_CHECKING:
    from loom.core.settings import Settings

# deepagents' FilesystemMiddleware injects all of these into every subagent
# (execute via the sandbox-capable backend). Read-only subagents get them
# stripped by ToolExclusionMiddleware — the prompt alone is not enforcement.
WRITE_TOOLS = frozenset({"write_file", "edit_file", "delete", "execute"})


@dataclass(frozen=True)
class SubagentSpec:
    """Static description of a subagent, independent of config/model wiring."""

    name: str
    description: str
    system_prompt: str
    tools: list[Callable] = field(default_factory=list)
    mode: str = "read-only"  # "read-only" | "write"
    # Extra tools to strip beyond what ``mode`` implies (e.g. the editor
    # writes files but must not run shell commands).
    excluded_tools: frozenset[str] = frozenset()

    def build(
        self,
        config: LoomConfig,
        settings: "Settings | None" = None,
        cwd: str = ".",
        *,
        model_string: str | None = None,
        extra_excluded: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        """Resolve this spec against config into a deepagents subagent dict.

        ``settings`` enables the per-subagent policy gate (permissions, hooks,
        undo snapshots). ``model_string`` overrides the config-assigned model
        (used to pin ``general-purpose`` to a local model in local-only/airgap
        runs). ``extra_excluded`` adds run-mode exclusions (e.g. plan mode
        strips write tools from every subagent).
        """
        if model_string is None:
            model_string = config.subagents.get(
                self.name, config.subagents.get("general-purpose", config.orchestrator)
            )
        model = build_model(model_string, config)

        middleware: list[Any] = []

        excluded: set[str] = set(self.excluded_tools) | set(extra_excluded)
        if self.mode == "read-only":
            excluded |= WRITE_TOOLS
        if excluded:
            from loom.middleware.tool_exclusion import ToolExclusionMiddleware

            middleware.append(ToolExclusionMiddleware(frozenset(excluded)))

        if config.is_local(model_string):
            middleware.append(PromptSizeGuard(model_string, config))

        if settings is not None:
            from loom.middleware.policy import PolicyMiddleware

            middleware.append(PolicyMiddleware(settings, cwd=cwd))

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
