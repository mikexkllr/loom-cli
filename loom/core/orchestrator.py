"""Orchestrator assembly (build step 3).

Wires the cloud orchestrator model + the six subagents + the consult tool +
summarization middleware into a single deepagents agent via
``create_deep_agent``. The orchestrator plans, decomposes, and routes — it never
touches raw tool output; subagents quarantine that.

Run modes:
  * normal     — full local/cloud fleet.
  * plan       — read-only: only explorer/searcher/reviewer, no writes.
  * local_only — no cloud calls at all: orchestrator runs on a local model, the
                 cloud reviewer/advisor are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loom.core.advisor import make_consult_tool
from loom.core.artifact_store import summarization_middleware
from loom.core.config import LoomConfig
from loom.core.model_router import build_model
from loom.subagents import build_all_subagents
from loom.tools import read_file

ORCHESTRATOR_SYSTEM = """You are Loom, a hybrid local/cloud multi-agent coding orchestrator.

Core principle: subagents do not make you smarter — they PROTECT YOUR CONTEXT.
Quarantine noisy work (reading large files, running commands, broad searches)
inside subagents that return only short summaries. Your context stays clean for
planning and synthesis.

How to work:
1. Decompose the task and call write_todos to track the plan.
2. Route each bounded subtask to the right subagent via the `task` tool:
   - explorer  : read-only recon, "where/how is X implemented"
   - searcher  : focused code/web lookups
   - editor    : apply a specific code change to named files
   - bash      : run tests/builds/git and report the verdict
   - reviewer  : after a significant write, get a risk rating
   - general   : multi-step work that mixes the above
3. Never read large files or run noisy commands yourself — delegate. You may use
   read_file only for small, targeted confirmations.
4. Consult the Advisor with `consult(question, context_summary)` at decision
   gates: before major/destructive work, after repeated failures, or before
   declaring done. The Advisor only advises; you decide.
5. After the reviewer flags HIGH risk or withholds approval, STOP and surface it
   to the human before continuing.

Keep your messages tight. Synthesize subagent summaries; do not echo their raw
output back."""

PLAN_SUFFIX = """

PLAN MODE: This is a read-only planning pass. Do NOT edit files or run mutating
commands. Use explorer/searcher to investigate, then produce a concrete,
ordered implementation plan and stop. The user will approve before any writes."""

LOCAL_ONLY_SUFFIX = """

LOCAL-ONLY MODE: No cloud calls are permitted. The Advisor and cloud reviewer are
unavailable. Rely on local subagents and your own judgment."""

# Read-only subagents permitted in plan mode.
_PLAN_SUBAGENTS = {"explorer", "searcher", "reviewer"}


@dataclass
class OrchestratorBundle:
    """The compiled agent plus metadata the CLI needs to render status."""

    agent: Any
    model_string: str
    subagent_names: list[str]
    mode: str
    persistent: bool = False  # True if a checkpointer is active (resume/thread state)


def build_orchestrator(
    settings: "Settings | LoomConfig",
    *,
    plan: bool = False,
    local_only: bool = False,
    advisor_threshold: str | None = None,
    cwd: str = ".",
    checkpointer: Any | None = None,
) -> OrchestratorBundle:
    """Construct the orchestrator agent for the requested run mode.

    Accepts a full :class:`Settings` (preferred — applies env, permissions, and
    hooks) or a bare :class:`LoomConfig` (model routing only, back-compat).
    """
    from deepagents import create_deep_agent

    from loom.core.settings import Settings

    if isinstance(settings, Settings):
        loom_settings = settings
        config = settings.models
        settings.apply_env()  # inject configured env vars before any model call
    else:
        loom_settings = None
        config = settings

    if advisor_threshold is not None:
        config = config.model_copy(update={"advisor_threshold": advisor_threshold})

    # ----- pick the orchestrator model -----
    if local_only:
        # Fall back to the general local model so nothing hits the cloud.
        orch_model_string = config.subagents.get("general", config.orchestrator)
    else:
        orch_model_string = config.orchestrator
    orch_model = build_model(orch_model_string, config)

    # ----- assemble subagents -----
    subagents = build_all_subagents(config)
    if plan:
        subagents = [s for s in subagents if s["name"] in _PLAN_SUBAGENTS]
    if local_only:
        # Drop any cloud-backed subagent (e.g. reviewer on Haiku).
        subagents = [s for s in subagents if config.is_local(config.subagents.get(s["name"], ""))]

    # ----- orchestrator tools -----
    tools: list[Any] = [read_file]
    if not local_only:
        tools.append(make_consult_tool(config))

    # ----- system prompt (kept prefix-stable for prompt caching) -----
    system = ORCHESTRATOR_SYSTEM
    if plan:
        system += PLAN_SUFFIX
    if local_only:
        system += LOCAL_ONLY_SUFFIX

    # ----- middleware: deepagents defaults + our summarization tuning -----
    middleware: list[Any] = []
    summ = summarization_middleware(config)
    if summ is not None:
        middleware.append(summ)
    if loom_settings is not None:
        # Permission + hook enforcement around every tool call.
        from loom.middleware.policy import PolicyMiddleware

        middleware.append(PolicyMiddleware(loom_settings, cwd=cwd))

    kwargs: dict[str, Any] = dict(
        model=orch_model,
        tools=tools,
        system_prompt=system,
        subagents=subagents,
        middleware=middleware,
    )

    # Optional LangGraph persistence: pass a checkpointer if create_deep_agent
    # supports it, so the REPL keeps thread state and can resume across runs.
    persistent = False
    if checkpointer is not None:
        try:
            agent = create_deep_agent(**kwargs, checkpointer=checkpointer)
            persistent = True
        except TypeError:
            agent = create_deep_agent(**kwargs)  # older signature — no persistence
    else:
        agent = create_deep_agent(**kwargs)

    return OrchestratorBundle(
        agent=agent,
        persistent=persistent,
        model_string=orch_model_string,
        subagent_names=[s["name"] for s in subagents],
        mode="plan" if plan else ("local-only" if local_only else "normal"),
    )
