"""Shared helper for assembling deepagents subagent definitions.

A Loom subagent is a deepagents subagent dict:

    {"name", "description", "system_prompt", "tools", "model", "middleware"}

with these Loom-specific conventions:
  * ``model`` is a concrete LangChain model instance built from config (so local
    Ollama models carry their base_url / num_ctx), not just a string.
  * each spec declares ``fs_tools`` — the filesystem/shell tools it is allowed
    to see. deepagents >= 0.7 accepts a caller-supplied
    ``FilesystemMiddleware(tools=[...])`` in a subagent's ``middleware`` list
    and uses it *instead of* the default one (matched by ``.name``), so the
    allowlist is structural: the tools the role has no business calling are
    never created, and their schemas never reach the model.
  * a :class:`ToolExclusionMiddleware` backs the same allowlist at the last
    mile. It is not redundant: ``FilesystemMiddleware`` requires ``read_file``
    in any allowlist, so "no filesystem at all" (the hardened
    ``general-purpose`` in airgap mode) can only be expressed by removing the
    tool after injection.
  * local subagents get a :class:`PromptSizeGuard` so an oversized prompt
    escalates to a roomier model instead of failing, plus a window-aware
    summarization middleware — deepagents' own default trigger for a model with
    no published profile is a flat 170K tokens, which a 32K Ollama model would
    never reach before overflowing.
  * every subagent gets its own :class:`PolicyMiddleware` (when Settings are
    available). deepagents builds a fresh middleware stack per subagent — the
    orchestrator's middleware does NOT propagate down — so permissions,
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

# Every filesystem/shell tool deepagents' FilesystemMiddleware can inject.
ALL_FS_TOOLS = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute"}
)
# Recon: navigate and read, mutate nothing.
READ_FS_TOOLS = frozenset({"ls", "read_file", "glob", "grep"})
# The mutating half. Read-only subagents (and plan mode) lose all of these.
WRITE_TOOLS = frozenset({"write_file", "edit_file", "delete", "execute"})
# Recon + authoring, no shell and no recursive delete.
EDIT_FS_TOOLS = READ_FS_TOOLS | {"write_file", "edit_file"}

# ``delete`` is deliberately absent from every spec. deepagents 0.7 hands the
# model a *recursive* delete whenever the backend supports it, and classifies it
# as a write — so any rule that permits writing a directory also permits erasing
# that whole subtree. Nothing Loom delegates needs it: bash can `rm` through
# `execute`, which the policy gate prompts for by name.


@dataclass(frozen=True)
class SubagentSpec:
    """Static description of a subagent, independent of config/model wiring."""

    name: str
    description: str
    system_prompt: str
    # Extra tools beyond the filesystem set (web_search, MCP browser tools).
    tools: list[Callable] = field(default_factory=list)
    # The filesystem/shell tools this role may see. Everything in
    # ``ALL_FS_TOOLS`` outside this set is both un-injected and stripped.
    fs_tools: frozenset[str] = READ_FS_TOOLS
    # Role this spec takes its model from when config assigns it none. An
    # explicit ``subagents[name]`` entry always overrides it.
    inherits: str = "general-purpose"

    @property
    def mode(self) -> str:
        """``"write"`` if this role can mutate the tree or run commands."""
        return "write" if self.fs_tools & WRITE_TOOLS else "read-only"

    def build(
        self,
        config: LoomConfig,
        settings: "Settings | None" = None,
        cwd: str = ".",
        *,
        model_string: str | None = None,
        extra_excluded: frozenset[str] = frozenset(),
        ladder: tuple[tuple[str, int], ...] = (),
        backend: Any | None = None,
    ) -> dict[str, Any]:
        """Resolve this spec against config into a deepagents subagent dict.

        ``settings`` enables the per-subagent policy gate (permissions, hooks,
        undo snapshots). ``model_string`` overrides the config-assigned model
        (used to pin ``general-purpose`` to a local model in local-only/airgap
        runs). ``extra_excluded`` narrows the declared ``fs_tools`` further —
        plan mode passes ``WRITE_TOOLS``, and the hardened ``general-purpose``
        passes every filesystem tool. ``ladder`` lists the local models the
        daemon actually serves, so an oversized prompt escalates to a roomier
        *local* model before it reaches for the cloud. ``backend`` is the
        orchestrator's storage backend; passing it lets this spec install its
        own tool-allowlisted ``FilesystemMiddleware`` over deepagents' default
        without losing the real filesystem underneath.
        """
        if model_string is None:
            model_string = config.model_for(self.name, self.inherits)
        model = build_model(model_string, config)
        is_local = config.is_local(model_string)

        # What this role is allowed to touch on this run.
        allowed = frozenset(self.fs_tools) - frozenset(extra_excluded)
        excluded = set(ALL_FS_TOOLS - allowed)

        middleware: list[Any] = []

        # First and unconditional — see the orchestrator's copy. A subagent
        # that dies on a tool defect costs the orchestrator a whole delegation
        # and returns nothing usable, so the guarantee matters more here, not
        # less, and it must not depend on Settings being wired.
        from loom.middleware.tool_guard import ToolErrorGuard

        middleware.append(ToolErrorGuard())

        if backend is not None:
            fs = _filesystem_middleware(allowed, backend, model_string, config)
            if fs is not None:
                middleware.append(fs)
            if is_local:
                summarizer = _summarization_middleware(model, backend, model_string, config)
                if summarizer is not None:
                    middleware.append(summarizer)

        if excluded:
            from loom.middleware.tool_exclusion import ToolExclusionMiddleware

            middleware.append(ToolExclusionMiddleware(frozenset(excluded)))

        if is_local:
            middleware.append(PromptSizeGuard(model_string, config, ladder))

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


# ---------------------------------------------------------------------------
# Middleware overrides (name-matched replacements for deepagents' defaults)
# ---------------------------------------------------------------------------


def _filesystem_middleware(
    allowed: frozenset[str], backend: Any, model_string: str, config: LoomConfig
) -> Any | None:
    """A ``FilesystemMiddleware`` restricted to ``allowed`` and tuned to the
    size of the model behind it.

    Replaces deepagents' default instance (same ``.name``) so the excluded tools
    are never injected. ``read_file`` is forced into the list because the
    constructor rejects one without it; the ToolExclusionMiddleware alongside
    strips it again when the role really is meant to have none.
    """
    # As in the orchestrator: None hands this subagent deepagents' default
    # filesystem middleware — every tool, including the ones this role's
    # allowlist deliberately withholds. Worth hearing about.
    from loom.core import telemetry

    try:
        from deepagents.middleware.filesystem import FilesystemMiddleware
    except Exception as exc:  # pragma: no cover - deepagents API drift
        telemetry.report("subagent.fs_middleware.import", exc)
        return None

    is_local = config.is_local(model_string)
    window = config.context_window_for(model_string, default=32_768)
    try:
        return FilesystemMiddleware(
            backend=backend,
            tools=sorted(allowed | {"read_file"}),
            # A 1000-match grep or a 20K-token tool result is fine for a
            # 200K-window cloud model and fatal for a 4B local one. Scale both
            # to the window so a noisy result gets offloaded to disk instead of
            # eating the whole context.
            grep_max_count=200 if is_local else 1000,
            tool_token_limit_before_evict=max(2_000, window // 8) if is_local else 20_000,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - deepagents API drift
        telemetry.report("subagent.fs_middleware.build", exc)
        return None


def _summarization_middleware(
    model: Any, backend: Any, model_string: str, config: LoomConfig
) -> Any | None:
    """Window-aware auto-compaction for a local subagent.

    deepagents derives its trigger from the model's published profile; ChatOllama
    has none, so it falls back to a flat 170K tokens — a threshold a 32K local
    model can never reach, leaving it to overflow instead of compacting. Trigger
    on this model's real window instead.
    """
    from loom.core.artifact_store import summarization_middleware

    return summarization_middleware(config, backend, model=model, model_string=model_string)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# deepagents >= 0.7 ships an empty base prompt and injects no tool-usage prose
# (it used to describe the filesystem and task tools for us). Every word a
# subagent reads about how to behave now comes from here.

ISOLATION_PREAMBLE = """You are a Loom subagent — an isolated worker with your own fresh context window.

Your contract:
- You see only the task the orchestrator wrote for you. There is no conversation
  history, no user, and no follow-up turn. You cannot ask a question; if you are
  blocked, finish anyway and report what you learned and where you stopped.
- The orchestrator sees ONLY your final message. Your tool calls, the file
  contents you read, and the logs you produced are discarded when you return.
  Anything the orchestrator needs must be written in that final message.
- That final message is a report, not a transcript. Lead with the answer, cite
  `path:line` instead of pasting code, and quote at most a few lines when the
  exact text *is* the answer. Never return a file dump, a full log, or a
  directory listing — protecting the orchestrator's context is the entire reason
  you exist.
- Do exactly the task you were given. Do not widen the scope, refactor code you
  were not asked about, or start the next step yourself.

Paths: the project root is `/`. `src/app.py` and `/src/app.py` name the same
file, so treat the two spellings as equivalent and never report one as a
correction to the other. Nothing exists outside the root — there is no home
directory and no system tree, so do not go looking in one.

"""
