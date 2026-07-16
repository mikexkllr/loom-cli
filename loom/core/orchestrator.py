"""Orchestrator assembly (build step 3).

Wires the cloud orchestrator model + the subagent fleet + the consult tool +
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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loom.core.settings import Settings

from loom.core.advisor import make_consult_tool
from loom.core.config import LoomConfig
from loom.core.model_router import build_model
from loom.subagents import build_all_subagents
from loom.tools.sandbox import get_root

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
   - tester    : drive a real browser (Playwright) through a user journey and
                 report PASS/FAIL per step
   - general   : multi-step work that mixes the above
3. Never read large files or run noisy commands yourself — delegate. You may use
   read_file only for small, targeted confirmations.
4. MANDATORY end-to-end verification: whenever the work changes anything a user
   can see or interact with through a frontend (a web page, UI component, form,
   route, or an API a visible screen depends on), you MUST verify it from the
   user's perspective before declaring the task done — unit tests and code
   review are NOT sufficient. Concretely: have bash start (or confirm) the dev
   server, then route a tester task that names the URL and the exact user
   journey to walk (what to click/type and what must visibly happen at each
   step), covering both the changed behavior and the surrounding happy path.
   If the tester reports FAIL, treat the task as not done: fix and re-test.
   Skip this only when the change has no user-visible surface (pure backend
   library code, tests, docs, tooling) — and say explicitly that you skipped
   it and why.
5. Consult the Advisor with `consult(question, context_summary)` at decision
   gates: before major/destructive work, after repeated failures, or before
   declaring done. The Advisor only advises; you decide.
6. After the reviewer flags HIGH risk or withholds approval, STOP and surface it
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

AIRGAP_SUFFIX = """

AIRGAP MODE: Raw source code must NEVER enter your context or leave this
machine. You have no read_file tool — delegate ALL file reading to local
subagents and work from their distilled summaries only. Never ask a subagent
to return raw file contents; ask for summaries, signatures, and line
references. Cloud escalation is disabled."""

NO_TESTER_SUFFIX = """

NOTE: The tester subagent is unavailable in this run (no browser/MCP tools
connected). Skip rule 4's browser verification, state that end-to-end testing
was skipped, and tell the user how to verify manually."""

# Read-only subagents permitted in plan mode.
_PLAN_SUBAGENTS = {"explorer", "searcher", "reviewer"}


def apply_cloud_fallback(config: LoomConfig) -> tuple[LoomConfig, dict[str, str]]:
    """Reroute local roles to ``config.cloud_fallback`` when Ollama can't serve them.

    Returns the (possibly rewritten) config and a map of role -> original
    local model for every role that was rerouted. No network is touched when
    the config has no local roles at all.
    """
    from loom.core import ollama
    from loom.core.model_router import resolve

    local_roles = {role: m for role, m in config.subagents.items() if config.is_local(m)}
    orch_local = config.is_local(config.orchestrator)
    if not local_roles and not orch_local:
        return config, {}

    try:
        status = ollama.status(config)
        available = set(status.models) if status.running else set()
    except Exception:
        available = set()

    def _served(model: str) -> bool:
        name = resolve(model).name
        return name in available or f"{name}:latest" in available

    fallbacks: dict[str, str] = {}
    subagents = dict(config.subagents)
    for role, model in local_roles.items():
        if not _served(model):
            subagents[role] = config.cloud_fallback
            fallbacks[role] = model
    update: dict[str, Any] = {"subagents": subagents}
    if orch_local and not _served(config.orchestrator):
        update["orchestrator"] = config.cloud_fallback
        fallbacks["orchestrator"] = config.orchestrator
    if not fallbacks:
        return config, {}
    return config.model_copy(update=update), fallbacks


def _require_ollama(config: LoomConfig, mode: str) -> None:
    """local-only / airgap cannot fall back to the cloud — fail fast instead
    of dying mid-run with connection errors."""
    from loom.core import ollama

    local_models = [m for m in config.all_models().values() if config.is_local(m)]
    if not local_models:
        return
    status = ollama.status(config)
    if not status.running:
        raise RuntimeError(
            f"{mode} mode needs local models, but the Ollama daemon isn't reachable "
            f"at {config.ollama_endpoint}. {ollama.INSTALL_HINT}"
        )


@dataclass
class OrchestratorBundle:
    """The compiled agent plus metadata the CLI needs to render status."""

    agent: Any
    model_string: str
    subagent_names: list[str]
    mode: str
    persistent: bool = False  # True if a checkpointer is active (resume/thread state)
    # role -> original local model, for every role rerouted to the cloud
    # because Ollama couldn't serve it this session.
    fallbacks: dict[str, str] = field(default_factory=dict)
    # PromptSizeGuard instances, so the UI can report escalation counts.
    guards: list[Any] = field(default_factory=list)


def build_orchestrator(
    settings: "Settings | LoomConfig",
    *,
    plan: bool = False,
    local_only: bool = False,
    airgap: bool = False,
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
    if airgap:
        # Cloud escalation would ship raw prompts (file contents) to the cloud;
        # an unbuildable escalation model makes the guard fall through to local.
        config = config.model_copy(update={"escalation_model": ""})

    # ----- Ollama availability -----
    fallbacks: dict[str, str] = {}
    if local_only or airgap:
        _require_ollama(config, "local-only" if local_only else "airgap")
    else:
        # No Ollama? Run the local roles on a cheap cloud model this session
        # rather than failing mid-run (the REPL surfaces this loudly).
        config, fallbacks = apply_cloud_fallback(config)

    # ----- pick the orchestrator model -----
    if local_only:
        # Fall back to the general local model so nothing hits the cloud.
        orch_model_string = config.subagents.get("general", config.orchestrator)
    else:
        orch_model_string = config.orchestrator
    orch_model = build_model(orch_model_string, config)

    # ----- MCP tools (Playwright browser etc.) -----
    # Sessions are process-wide singletons so the browser survives rebuilds.
    mcp_tools: list[Any] = []
    if loom_settings is not None and loom_settings.mcp_servers and not plan:
        from loom.core.mcp import get_mcp_tools

        mcp_tools = get_mcp_tools(loom_settings)

    # ----- assemble subagents -----
    subagents = build_all_subagents(config)
    if plan:
        subagents = [s for s in subagents if s["name"] in _PLAN_SUBAGENTS]
    if local_only or airgap:
        # Drop any cloud-backed subagent (e.g. reviewer on Haiku). In airgap
        # mode only local subagents may touch raw code.
        subagents = [s for s in subagents if config.is_local(config.subagents.get(s["name"], ""))]

    # The tester only exists when browser MCP tools actually connected.
    browser_tools = [t for t in mcp_tools if t.name.startswith("browser_")]
    has_tester = False
    for sub in list(subagents):
        if sub["name"] != "tester":
            continue
        if browser_tools:
            sub["tools"] = list(sub["tools"]) + browser_tools
            has_tester = True
        else:
            subagents.remove(sub)

    # Any other MCP tools (non-browser servers the user added) go to general.
    other_mcp = [t for t in mcp_tools if not t.name.startswith("browser_")]
    if other_mcp:
        for sub in subagents:
            if sub["name"] == "general":
                sub["tools"] = list(sub["tools"]) + other_mcp

    # ----- orchestrator tools -----
    # Airgap: the (cloud) orchestrator loses read_file so raw file contents
    # can never enter its context — only subagent summaries.
    # deepagents' FilesystemMiddleware injects ls/read_file/write_file/edit_file/
    # glob/grep/execute automatically. The orchestrator may use read_file (via
    # the deepagents schema, with virtual absolute paths) and the read-only
    # ls/glob/grep tools; write_file/edit_file/execute are removed below.
    tools: list[Any] = []
    if not local_only and not airgap:
        # consult sends the question+context to a cloud advisor; in airgap and
        # local-only modes no orchestrator-originated data may leave the machine.
        tools.append(make_consult_tool(config))

    # ----- system prompt (kept prefix-stable for prompt caching) -----
    system = ORCHESTRATOR_SYSTEM
    if plan:
        system += PLAN_SUFFIX
    if local_only:
        system += LOCAL_ONLY_SUFFIX
    if airgap:
        system += AIRGAP_SUFFIX
    if not has_tester and not plan:
        system += NO_TESTER_SUFFIX

    # ----- middleware: deepagents defaults (incl. SummarizationMiddleware) -----
    middleware: list[Any] = []
    if loom_settings is not None:
        # Permission + hook enforcement around every tool call.
        from loom.middleware.policy import PolicyMiddleware

        # Airgap: harden the policy gate so that even if a file tool slips
        # through the tool-exclusion middleware, the policy gate rejects it.
        if airgap:
            from loom.core.settings import Permissions

            policy_settings = loom_settings.model_copy(
                update={
                    "permissions": Permissions(
                        default_mode="deny",
                        allow=["task", "write_todos"],
                        deny=[
                            "read_file",
                            "write_file",
                            "edit_file",
                            "execute",
                            "ls",
                            "glob",
                            "grep",
                        ],
                    )
                }
            )
        else:
            policy_settings = loom_settings
        middleware.append(PolicyMiddleware(policy_settings, cwd=cwd))

    # deepagents always injects FilesystemMiddleware (ls/read_file/write_file/
    # edit_file/glob/grep/execute). Use a composite backend so large tool results
    # and summarization offloads are persisted under ``.loom/`` rather than in
    # the project root, while the default sandbox still runs shell commands in
    # the project/worktree root. Loom's own filesystem/shell tools have been
    # removed; filesystem and terminal calls now go through deepagents.
    from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend

    sessions_dir = Path(cwd) / ".loom" / "sessions"
    artifacts_dir = Path(cwd) / ".loom" / "artifacts"
    backend = CompositeBackend(
        default=LocalShellBackend(root_dir=get_root(), virtual_mode=True, inherit_env=True),
        routes={
            "/conversation_history/": FilesystemBackend(
                root_dir=sessions_dir / "conversation_history", virtual_mode=True
            ),
            "/large_tool_results/": FilesystemBackend(
                root_dir=artifacts_dir / "large_tool_results", virtual_mode=True
            ),
        },
    )

    # Strip dangerous/file tools from the orchestrator request. The middleware
    # injects them unconditionally, but the orchestrator should only read and
    # delegate; writes and shell execution belong to subagents.
    from loom.middleware.tool_exclusion import ToolExclusionMiddleware

    excluded_tools = {"write_file", "edit_file", "execute"}
    if airgap:
        # Airgap: no orchestrator tool may touch the filesystem or run a process.
        excluded_tools = {"read_file", "write_file", "edit_file", "execute", "ls", "glob", "grep"}
    middleware.append(ToolExclusionMiddleware(excluded_tools))

    kwargs: dict[str, Any] = dict(
        model=orch_model,
        tools=tools,
        system_prompt=system,
        subagents=subagents,
        middleware=middleware,
        backend=backend,
    )

    # Optional LangGraph persistence: pass a checkpointer if create_deep_agent
    # supports it, so the REPL keeps thread state and can resume across runs.
    def _build_agent(kw: dict[str, Any]) -> tuple[Any, bool]:
        if checkpointer is not None:
            try:
                return create_deep_agent(**kw, checkpointer=checkpointer), True
            except TypeError:
                return create_deep_agent(**kw), False  # older signature — no persistence
        return create_deep_agent(**kw), False

    agent, persistent = _build_agent(kwargs)

    from loom.middleware.prompt_size_guard import PromptSizeGuard

    guards = [
        m for s in subagents for m in s.get("middleware", []) if isinstance(m, PromptSizeGuard)
    ]

    return OrchestratorBundle(
        agent=agent,
        persistent=persistent,
        fallbacks=fallbacks,
        guards=guards,
        model_string=orch_model_string,
        subagent_names=[s["name"] for s in subagents],
        mode="plan"
        if plan
        else ("local-only" if local_only else ("airgap" if airgap else "normal")),
    )
