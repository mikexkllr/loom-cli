"""Orchestrator assembly (build step 3).

Wires the cloud orchestrator model + the subagent fleet + the consult tool +
Loom's middleware stack into a single deepagents agent via
``create_deep_agent``. The orchestrator plans, decomposes, and routes — it never
touches raw tool output; subagents quarantine that.

deepagents >= 0.7 changed what this module is responsible for:

* The built-in prompts are gone. ``BASE_AGENT_PROMPT`` is empty and the
  filesystem/task/todo tool-usage prose is no longer injected, so every word the
  orchestrator reads about how to behave comes from
  :func:`orchestrator_system_prompt` here and from the subagent prompts in
  :mod:`loom.subagents`.
* ``TodoListMiddleware`` is no longer a default, so ``write_todos`` only exists
  because this module adds it back — with Loom's own planning prose rather than
  LangChain's.
* A caller-supplied middleware instance replaces the default with the same
  ``.name``. That is how the orchestrator gets a ``FilesystemMiddleware``
  restricted to a single tool, and a window-aware summarizer, instead of
  filtering the defaults after the fact.

Run modes:
  * normal     — full local/cloud fleet.
  * plan       — read-only: only explorer/searcher/reviewer, no writes.
  * local_only — no cloud calls at all: orchestrator runs on a local model, the
                 cloud reviewer/advisor are dropped.
  * airgap     — as local_only, plus no raw source may enter the orchestrator's
                 context at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loom.core.settings import Settings

from loom.core.advisor import make_consult_tool
from loom.core.config import LoomConfig
from loom.core.local_pool import LocalPool, RolePlan
from loom.core.model_router import build_model
from loom.subagents import ALL_FS_TOOLS, build_all_subagents
from loom.tools.sandbox import get_root

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# Kept as a prefix-stable template: everything that varies by run mode is
# appended as a suffix so the provider's prompt cache keeps hitting.

_ORCHESTRATOR_SYSTEM_TEMPLATE = """You are Loom, a hybrid local/cloud multi-agent coding orchestrator.

You do not do the work. You decide what the work is, hand each piece to the right
subagent, and turn what comes back into an answer. Subagents are not smarter than
you — they are disposable context. Every file they read, command they run, and log
they print dies with them, and only a short report reaches you. That is what keeps
your own window clean enough to hold a whole task, which makes routing work away
from yourself the job itself, not an optimization.

## Your tools

- `task(description, subagent_type)` — delegate. This is your primary verb.
- `write_todos` — the plan, for anything past a couple of steps.
- `read_file` — {read_phrase}
- `consult(question, context_summary)` — the strongest available model, for
  judgment calls. It advises; it cannot act.

You have no `ls`, `glob`, `grep`, `write_file`, `edit_file`, `delete`, or shell.
They are not missing — they are quarantined inside subagents deliberately. To
learn what is in this repository, you ask; you do not look.

## The fleet

- `explorer` — read-only recon: where is X, how does Y work, what depends on Z.
- `searcher` — one focused lookup, in this code or on the web.
- `editor` — applies a specific change to named files. Cannot run anything.
- `bash` — runs tests/builds/lint/git/servers and reports the verdict. Cannot edit.
- `reviewer` — risk-rates a finished change; returns a structured verdict.
- `tester` — drives a real browser through a user journey.
- `general-purpose` — one self-contained slice that mixes the above.

Default to the local specialists. They are free and private; `consult` and any
cloud-backed role costs money, so spend it on judgment and on knowledge that is
genuinely outside this machine — not on work a local model can do.

## How to delegate

Route by the kind of work, not by its size. "It is only one file" is not a reason
to do it yourself.

Write every task as if to a competent stranger who cannot see this conversation,
because that is exactly what it is. Each one needs:
- the goal stated as the outcome you want,
- the inputs it must not have to rediscover: paths, symbol names, the exact
  command, the URL,
- what "done" looks like and what to report back,
- what not to touch, whenever there is an obvious way to overreach.

Weak: "look at the auth code."
Strong: "In loom/core/permissions.py, find where deny rules are matched against a
tool call and report the precedence between allow/deny/ask, with line numbers."

Launch independent tasks in the SAME message so they run concurrently: recon
across three subsystems is three parallel `task` calls, not three round trips.
Serialize only when one task truly needs another's output.

Never send the same work to two subagents to compare, and never redo a subagent's
work yourself because you would like to see it directly. If a report is too thin
to act on, send it back with a sharper question.

## Reading

{reading_rule}

## Verifying

Whenever the work changes anything a user can see or interact with through a
frontend — a page, component, form, route, or an API a visible screen depends on
— you MUST verify it from the user's perspective before calling the task done.
Unit tests and code review are not sufficient. Have `bash` start or confirm the
dev server, then give `tester` the URL and the exact journey to walk: what to
click and type, and what must visibly appear at each step, covering both the
changed behaviour and the happy path around it. A FAIL means the task is not
done: fix it and re-test.

Skip this only when the change has no user-visible surface (library code, tests,
docs, tooling) — and say that you skipped it and why.

For everything else, "verified" means a subagent ran something and reported the
result. If nothing was run, the change is unverified; say so.

## Judgment gates

- `consult` before major or irreversible work, after the same approach has failed
  twice, and before declaring a hard task done.
- Dispatch `reviewer` after any significant write, with the task and the list of
  changed files.
- If the reviewer returns HIGH risk or withholds approval, STOP and surface it to
  the human before doing anything else.

## Answering

Your final message is the only thing the user reads. Lead with the outcome, name
the files that changed, and state plainly what was verified and what was not.
Synthesize the reports — never paste one back verbatim, and never narrate your
own routing ("now I will delegate to explorer"); the user already sees the tool
calls. Keep it tight."""

_READING_RULE = """`read_file` is for confirmation, never for investigation: the \
exact path a subagent just named, or the region of a change you are about to
approve. Anything larger — mapping how something works, tracing a flow, reading
several files, anything that begins with "let me look around" — is a job for
`explorer`: spawn it with `task` and work from its summary. Every `read_file`
result ends with a one-line reminder of this rule; it is not an error and not new
information.

The project root is `/`, so `src/app.py` and `/src/app.py` are the same file.
Nothing exists outside the root."""

_READING_RULE_NONE = """You have no `read_file` tool. Every fact about this \
codebase reaches you through a subagent's report. If you need to see a specific
region of a file, ask `explorer` for that region and what to note about it."""

_READ_PHRASE = "confirm one specific spot; anything larger goes to `explorer` (see Reading)."
_READ_PHRASE_NONE = "unavailable in this run (see Reading)."

# deepagents 0.7 dropped TodoListMiddleware from the defaults, so `write_todos`
# is only present because Loom adds it back — and its prompt is Loom's to write.
# LangChain's default is ~600 words of generic task-tracking prose that says
# nothing about delegation; this replaces it with the same idea expressed in
# terms of the fleet.
TODO_SYSTEM_PROMPT = """## Planning with `write_todos`

Keep a todo list for any task with three or more steps, or more than one
delegation. Write each item as the work plus the subagent that will do it
("explorer: locate where deny rules are matched"), so the plan and the routing
are one artifact instead of two.

Mark an item `in_progress` before you delegate it and `completed` as soon as its
report comes back — never batch completions, and never leave the list with
nothing in progress while work continues. When a report changes the plan, revise
the list; that revision is the reason to keep one.

Skip the list for questions and single-step requests: writing it costs tokens and
buys nothing there. Never call `write_todos` twice in one message, and never let
it be your last act in a turn — the turn ends with your answer to the user, not
with a checklist update."""

PLAN_SUFFIX = """

PLAN MODE: This is a read-only planning pass. Do NOT edit files or run mutating
commands. Use explorer/searcher to investigate, then produce a concrete, ordered
implementation plan and stop. Your FINAL message must contain the complete plan —
numbered steps, the files each step touches, the subagent you will route it to,
and how you will verify the result. The user is then asked to approve it, and on
approval you implement exactly that plan."""

LOCAL_ONLY_SUFFIX = """

LOCAL-ONLY MODE: No cloud calls are permitted. The Advisor (`consult`) and the
cloud reviewer are unavailable, so the judgment gates fall to you: state your
reasoning at the points where you would have consulted, and flag anything you are
not confident about instead of resolving it silently."""

AIRGAP_SUFFIX = """

AIRGAP MODE: Raw source code must NEVER enter your context or leave this machine.
You have no `read_file` tool — delegate ALL file reading to local subagents and
work from their distilled summaries only. Never ask a subagent to return raw file
contents; ask for summaries, signatures, and line references. Cloud escalation is
disabled."""

NO_TESTER_SUFFIX = """

NOTE: The tester subagent is unavailable in this run (no browser/MCP tools
connected). Skip the browser verification under Verifying, state explicitly that
end-to-end testing was skipped, and tell the user how to verify it manually."""

GRAPH_SUFFIX = """

KNOWLEDGE GRAPH: A Graphify code knowledge graph of this repo is connected
(query_graph / get_node / shortest_path). For structure questions — "where is X
defined", "what connects A to B", "what depends on Y", "what would break if I
change Z" — query the graph FIRST: it answers from the pre-built index with
file:line citations at a fraction of the tokens a grep-and-read sweep costs, and
it does not spend a delegation. Explorer and searcher hold the same tools; tell
them to prefer the graph too. Fall back to a subagent when you need exact code
bodies or the graph lacks the detail."""


def orchestrator_system_prompt(can_read: bool = True) -> str:
    """The base system prompt, with the reading rule filled in.

    The rule announces the reminder :class:`DelegationReminder` attaches to every
    ``read_file`` result, so the model reads it as the harness working rather than
    as something new to react to. ``can_read=False`` is airgap: no ``read_file``
    at all, so no reminder and no path conventions either.
    """
    if can_read:
        reading, phrase = _READING_RULE, _READ_PHRASE
    else:
        reading, phrase = _READING_RULE_NONE, _READ_PHRASE_NONE
    return _ORCHESTRATOR_SYSTEM_TEMPLATE.format(read_phrase=phrase, reading_rule=reading)


# Subagents permitted in plan mode. general-purpose stays (built read-only in
# plan mode) because dropping it would let deepagents auto-add its own
# write-capable default under that reserved name.
_PLAN_SUBAGENTS = {"explorer", "searcher", "reviewer", "general-purpose"}

# Every filesystem/shell tool deepagents' FilesystemMiddleware can inject.
_ALL_FS_TOOLS = ALL_FS_TOOLS

# The only filesystem tool the orchestrator ever holds. Writes, shell, and broad
# search belong to subagents; a single targeted read stays so the orchestrator
# can confirm a path a subagent named without a second round trip.
_ORCHESTRATOR_FS_TOOLS = frozenset({"read_file"})


def _orchestrator_fs_tools(*, airgap: bool) -> frozenset[str]:
    """The filesystem tools the orchestrator is allowed to see."""
    return frozenset() if airgap else _ORCHESTRATOR_FS_TOOLS


def _orchestrator_excluded_tools(*, airgap: bool) -> set[str]:
    """Tools stripped from the orchestrator's own request (subagents are
    unaffected — they get their own middleware stack).

    write/edit/delete/execute always: those belong to subagents. glob/grep/ls
    too — browsing and broad search are recon, which stays quarantined in
    explorer/searcher. That quarantine used to be prompt-only ("delegate, don't
    investigate yourself"), which the orchestrator model can and did ignore:
    strong cloud models (e.g. gpt-5.5) map the tree with ls and sweep files
    themselves instead of routing recon to a local explorer, defeating the
    context-quarantine design. Removing the tools makes the split structural
    instead of advisory. read_file stays available for the small targeted
    confirmations the system prompt calls for — and every result it returns ends
    with a reminder to hand anything larger to explorer
    (:class:`DelegationReminder`). Airgap strips every filesystem tool, no exception.

    As of deepagents 0.7 this is the second of two layers: the orchestrator's
    ``FilesystemMiddleware`` is built with a matching allowlist so most of these
    tools are never injected in the first place. This still matters, because that
    allowlist cannot express "no read_file" — the constructor rejects it — which
    is exactly what airgap mode needs.
    """
    return set(_ALL_FS_TOOLS - _orchestrator_fs_tools(airgap=airgap))


def _ensure_general_purpose(
    subagents: list[dict[str, Any]],
    config: LoomConfig,
    settings: Any,
    cwd: str,
    *,
    read_only: bool,
    ladder: tuple[tuple[str, int], ...] = (),
    backend: Any | None = None,
) -> list[dict[str, Any]]:
    """Guarantee a subagent named ``general-purpose`` survives every run mode.

    deepagents auto-adds its own general-purpose subagent — orchestrator model,
    full filesystem/execute toolset, none of Loom's policy middleware — whenever
    no spec carries that exact name. If mode filtering dropped ours (e.g. it was
    assigned a cloud model in local-only/airgap), rebuild it pinned to a local
    model; with no local model available, keep the name claimed but strip every
    filesystem/shell tool so it can never touch code.
    """
    if any(s["name"] == "general-purpose" for s in subagents):
        return subagents

    from loom.subagents import SPECS, WRITE_TOOLS

    spec = SPECS["general-purpose"]
    candidates = [
        config.subagents.get("general-purpose", ""),
        *config.subagents.values(),
        config.orchestrator,
    ]
    local_model = next((m for m in candidates if m and config.is_local(m)), None)
    if local_model is not None:
        sub = spec.build(
            config,
            settings,
            cwd,
            model_string=local_model,
            extra_excluded=WRITE_TOOLS if read_only else frozenset(),
            ladder=ladder,
            backend=backend,
        )
    else:
        sub = spec.build(
            config,
            settings,
            cwd,
            extra_excluded=_ALL_FS_TOOLS,
            ladder=ladder,
            backend=backend,
        )
    return [*subagents, sub]


def apply_cloud_fallback(config: LoomConfig, pool: "LocalPool | None" = None) -> RolePlan:
    """Resolve local roles Ollama can't serve — another local model first.

    A role whose exact tag isn't pulled used to go straight to the billed
    ``cloud_fallback``, even on a machine with other local models loaded and
    idle. Now the cloud is the second choice: see
    :func:`loom.core.local_pool.plan_local_roles`. No network is touched when
    the config has no local roles at all.
    """
    from loom.core.local_pool import build_pool, plan_local_roles

    if not any(config.is_local(m) for m in config.all_models().values()):
        return RolePlan(config, {}, {})
    return plan_local_roles(config, pool or build_pool(config))


def _require_ollama(config: LoomConfig, mode: str, pool: "LocalPool | None" = None) -> None:
    """local-only / airgap cannot fall back to the cloud — fail fast instead
    of dying mid-run with connection errors."""
    from loom.core import ollama
    from loom.core.local_pool import build_pool

    local_models = [m for m in config.all_models().values() if config.is_local(m)]
    if not local_models:
        return
    if not (pool or build_pool(config)).running:
        raise RuntimeError(
            f"{mode} mode needs local models, but the Ollama daemon isn't reachable "
            f"at {config.ollama_endpoint}. {ollama.INSTALL_HINT}"
        )


def _build_backend(cwd: str) -> tuple[Any, list[tuple[str, Any]]]:
    """``(backend, skill_sources)`` — the composite storage backend shared by the
    orchestrator and every subagent, plus the skill routes mounted into it.

    Shell commands and ordinary file paths land in the project (or worktree)
    root; deepagents' own offload paths — evicted conversation history and
    oversized tool results — are routed under ``.loom/`` so compaction artifacts
    never litter the user's tree.

    Built before the subagents because each of them installs its own
    tool-restricted ``FilesystemMiddleware`` over this exact backend instance; a
    subagent that built one without it would silently fall back to an in-memory
    StateBackend and lose the real filesystem.
    """
    from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend

    from loom.core.skills import skill_sources

    sessions_dir = Path(cwd) / ".loom" / "sessions"
    artifacts_dir = Path(cwd) / ".loom" / "artifacts"
    routes: dict[str, Any] = {
        "/conversation_history/": FilesystemBackend(
            root_dir=sessions_dir / "conversation_history", virtual_mode=True
        ),
        "/large_tool_results/": FilesystemBackend(
            root_dir=artifacts_dir / "large_tool_results", virtual_mode=True
        ),
    }
    # Agent skills (Anthropic SKILL.md pattern via deepagents): packaged →
    # user (~/.loom/skills) → project (.loom/skills), later wins. Each layer
    # mounts read-only into the virtual filesystem under /skills/.
    sources = skill_sources(cwd)
    for route, real_dir in sources:
        routes[route] = FilesystemBackend(root_dir=real_dir, virtual_mode=True)
    backend = CompositeBackend(
        default=LocalShellBackend(root_dir=get_root(), virtual_mode=True, inherit_env=True),
        routes=routes,
    )
    return backend, sources


def _orchestrator_filesystem_middleware(
    backend: Any, config: LoomConfig, model_string: str, *, airgap: bool
) -> Any | None:
    """A ``FilesystemMiddleware`` holding only what the orchestrator may use.

    Replaces deepagents' default (matched by ``.name``), so the tools the
    orchestrator must not have are never constructed and their schemas never
    reach the model. In airgap mode the allowlist is still ``read_file`` — the
    constructor requires it — and :class:`ToolExclusionMiddleware` removes it at
    the last mile.
    """
    # Returning None here does not mean "no filesystem middleware" — it means
    # deepagents installs its *default* one, with the full toolset the
    # allowlist exists to withhold. A silent None is the tool quarantine
    # quietly switching itself off, so both drift guards report.
    from loom.core import telemetry

    try:
        from deepagents.middleware.filesystem import FilesystemMiddleware
    except Exception as exc:  # pragma: no cover - deepagents API drift
        telemetry.report("orchestrator.fs_middleware.import", exc)
        return None
    allowed = _orchestrator_fs_tools(airgap=airgap) | {"read_file"}
    window = config.context_window_for(model_string, default=200_000)
    try:
        return FilesystemMiddleware(
            backend=backend,
            tools=sorted(allowed),
            # A read the orchestrator makes is meant to be a small confirmation.
            # Evict anything larger to disk rather than letting one file fill the
            # window the whole design exists to protect.
            tool_token_limit_before_evict=max(2_000, min(20_000, window // 10)),
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - deepagents API drift
        telemetry.report("orchestrator.fs_middleware.build", exc)
        return None


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
    # role -> original local model, for every role covered by a *different*
    # local model instead. Still free and private — worth telling the user
    # (their config asked for something else) but not a warning.
    substitutions: dict[str, str] = field(default_factory=dict)
    # The config as actually resolved for this run: detected context windows
    # filled in and unserved roles reassigned. The UI reads it to report what
    # each role is really running on, rather than what the file asked for.
    active_config: LoomConfig | None = None
    # PromptSizeGuard instances, so the UI can report escalation counts.
    guards: list[Any] = field(default_factory=list)
    # The orchestrator's DelegationReminder (None in airgap, which has no
    # read_file), so /status and /cost can report how often it read files itself.
    delegation_reminder: Any | None = None


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
    # One probe of the daemon feeds all three local-first decisions below:
    # what each model's real context window is, which roles need standing in
    # for, and what the prompt-size guards can escalate to without going cloud.
    from loom.core.local_pool import (
        build_pool,
        detect_context_windows,
        escalation_ladder,
        plan_local_roles,
    )

    pool = build_pool(config)
    # Ask Ollama for real context lengths before anything reads a window: the
    # ladder, the fallback planner, the num_ctx we hand ChatOllama, and every
    # subagent's summarization trigger all depend on them, and a blind 32K guess
    # sends work to the cloud that the model could have held.
    config = detect_context_windows(config, pool)

    if local_only or airgap:
        _require_ollama(config, "local-only" if local_only else "airgap", pool)
        # Cloud is off the table here, but a missing tag can still be covered
        # by another local model rather than failing the role outright.
        role_plan = plan_local_roles(config, pool, allow_cloud=False)
    else:
        # A model that isn't pulled falls to another *local* model first, and
        # only then to a cheap cloud model for the session rather than failing
        # mid-run (the REPL surfaces both, loudly for the billed one).
        role_plan = apply_cloud_fallback(config, pool)
    config = role_plan.config
    fallbacks, substitutions = role_plan.cloud, role_plan.substituted
    ladder = escalation_ladder(config, pool)

    # ----- pick the orchestrator model -----
    if local_only:
        # Fall back to the general-purpose local model so nothing hits the cloud.
        orch_model_string = config.subagents.get("general-purpose", config.orchestrator)
    else:
        orch_model_string = config.orchestrator
    orch_model = build_model(orch_model_string, config)

    # ----- MCP tools (Playwright browser etc.) -----
    # Sessions are process-wide singletons so the browser survives rebuilds.
    mcp_tools: list[Any] = []
    if loom_settings is not None and loom_settings.mcp_servers and not plan:
        from loom.core.mcp import get_mcp_tools

        mcp_tools = get_mcp_tools(loom_settings)

    # ----- storage backend (shared by orchestrator and every subagent) -----
    backend, skill_source_routes = _build_backend(cwd)

    # ----- assemble subagents -----
    # Subagents carry their own FilesystemMiddleware (tool allowlist),
    # PolicyMiddleware (permissions/hooks/undo), ToolExclusionMiddleware and
    # PromptSizeGuard: deepagents builds a fresh middleware stack per subagent,
    # so nothing from the orchestrator's stack applies down there.
    # In airgap mode subagents keep the NORMAL settings — they must read and
    # edit files locally; only the orchestrator gets the hardened deny policy.
    # Cloud-backed roles (e.g. a reviewer trailing a cloud advisor) are dropped
    # by the builder itself in the no-cloud modes, before their models are
    # constructed — building one validates credentials this machine is entitled
    # not to have. In airgap mode this doubles as the rule that only local
    # subagents may touch raw code.
    subagents = build_all_subagents(
        config,
        loom_settings,
        cwd,
        read_only=plan,
        ladder=ladder,
        backend=backend,
        local_only=local_only or airgap,
    )
    if plan:
        subagents = [s for s in subagents if s["name"] in _PLAN_SUBAGENTS]
    # deepagents auto-adds an unrestricted general-purpose subagent if the name
    # is absent — never let mode filtering open that hole.
    subagents = _ensure_general_purpose(
        subagents, config, loom_settings, cwd, read_only=plan, ladder=ladder, backend=backend
    )

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

    # Graphify's read-only graph-query tools also go to the recon subagents —
    # explorer/searcher answer "where/how" questions from the graph instead of
    # grep-and-read sweeps.
    from loom.core.graphify import graph_tools_from

    graph_tools = graph_tools_from(mcp_tools)
    if graph_tools:
        for sub in subagents:
            if sub["name"] in ("explorer", "searcher"):
                sub["tools"] = list(sub["tools"]) + graph_tools

    # Any other MCP tools (non-browser servers the user added) go to
    # general-purpose.
    other_mcp = [t for t in mcp_tools if not t.name.startswith("browser_")]
    if other_mcp:
        for sub in subagents:
            if sub["name"] == "general-purpose":
                sub["tools"] = list(sub["tools"]) + other_mcp

    # ----- orchestrator tools -----
    tools: list[Any] = []
    if not local_only and not airgap:
        # consult sends the question+context to a cloud advisor; in airgap and
        # local-only modes no orchestrator-originated data may leave the machine.
        tools.append(make_consult_tool(config))
    # The orchestrator queries the knowledge graph directly — structure answers
    # without spawning a subagent or reading files. Not in airgap: graph nodes
    # carry code identifiers/snippets, which must not reach a cloud orchestrator.
    if graph_tools and not airgap:
        tools.extend(graph_tools)

    # ----- system prompt (kept prefix-stable for prompt caching) -----
    system = orchestrator_system_prompt(can_read=not airgap)
    if plan:
        system += PLAN_SUFFIX
    if local_only:
        system += LOCAL_ONLY_SUFFIX
    if airgap:
        system += AIRGAP_SUFFIX
    if not has_tester and not plan:
        system += NO_TESTER_SUFFIX
    # Airgap: the orchestrator has no graph tools (nodes carry code
    # identifiers) — explorer/searcher still hold them, no prompt needed.
    if graph_tools and not airgap:
        system += GRAPH_SUFFIX

    # ----- middleware -----
    # Ordering note: deepagents merges these into its own stack by `.name` —
    # a matching name replaces the default in place, a new name is spliced in
    # after the core middleware and before the prompt-caching tail.
    middleware: list[Any] = []

    # Unconditional and first, so it wraps everything below it — including the
    # policy gate, whose own hooks and confirm callback can throw. A tool that
    # raises must cost one step, not the turn, and that is not a permissions
    # feature: the policy gate is skipped entirely on the bare-LoomConfig path,
    # which is exactly where an unprotected crash is hardest to explain.
    from loom.middleware.tool_guard import ToolErrorGuard

    middleware.append(ToolErrorGuard())

    fs_middleware = _orchestrator_filesystem_middleware(
        backend, config, orch_model_string, airgap=airgap
    )
    if fs_middleware is not None:
        middleware.append(fs_middleware)

    summarizer = _orchestrator_summarization_middleware(orch_model, backend, config, orch_model_string)
    if summarizer is not None:
        middleware.append(summarizer)

    # write_todos, with Loom's planning prose instead of LangChain's. Not given
    # to subagents on purpose: their work is bounded by the task they were
    # handed, and a todo tool on a small local model buys tracking overhead
    # without buying a plan.
    todo_middleware = _todo_middleware()
    if todo_middleware is not None:
        middleware.append(todo_middleware)

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
                        deny=sorted(_ALL_FS_TOOLS),
                    )
                }
            )
        else:
            policy_settings = loom_settings
        middleware.append(PolicyMiddleware(policy_settings, cwd=cwd))

    # Strip whatever the filesystem allowlist could not express (airgap's
    # read_file) and any non-filesystem tool that must not reach the model.
    from loom.middleware.tool_exclusion import ToolExclusionMiddleware

    excluded_tools = _orchestrator_excluded_tools(airgap=airgap)
    middleware.append(ToolExclusionMiddleware(excluded_tools))

    # The one read tool that remains carries a nudge on every result: anything
    # larger than a targeted check goes to explorer
    # (see loom/middleware/delegation_reminder.py). Airgap has no read_file.
    delegation_reminder = None
    if not airgap:
        from loom.middleware.delegation_reminder import DelegationReminder

        delegation_reminder = DelegationReminder()
        middleware.append(delegation_reminder)

    kwargs: dict[str, Any] = dict(
        model=orch_model,
        tools=tools,
        system_prompt=system,
        subagents=subagents,
        middleware=middleware,
        backend=backend,
    )
    if skill_source_routes:
        # Progressive disclosure: only name+description hit the prompt; the
        # agent reads a skill's full SKILL.md when the task matches.
        kwargs["skills"] = [route for route, _ in skill_source_routes]

    # LangGraph persistence: the REPL keeps thread state and can resume across
    # runs (checkpointer is a stable create_deep_agent parameter in >=0.6).
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    agent = create_deep_agent(**kwargs)
    persistent = checkpointer is not None

    from loom.middleware.prompt_size_guard import PromptSizeGuard

    guards = [
        m for s in subagents for m in s.get("middleware", []) if isinstance(m, PromptSizeGuard)
    ]

    return OrchestratorBundle(
        agent=agent,
        persistent=persistent,
        fallbacks=fallbacks,
        substitutions=substitutions,
        active_config=config,
        guards=guards,
        delegation_reminder=delegation_reminder,
        model_string=orch_model_string,
        subagent_names=[s["name"] for s in subagents],
        mode="plan"
        if plan
        else ("local-only" if local_only else ("airgap" if airgap else "normal")),
    )


def _todo_middleware() -> Any | None:
    """``TodoListMiddleware`` carrying Loom's planning prompt.

    deepagents 0.7 removed it from the defaults, which silently deleted
    ``write_todos``; without this the orchestrator prompt would point at a tool
    that does not exist.
    """
    try:
        from langchain.agents.middleware import TodoListMiddleware
    except Exception:  # pragma: no cover - langchain API drift
        return None
    try:
        return TodoListMiddleware(system_prompt=TODO_SYSTEM_PROMPT)
    except TypeError:  # pragma: no cover - langchain API drift
        return TodoListMiddleware()


def _orchestrator_summarization_middleware(
    model: Any, backend: Any, config: LoomConfig, model_string: str
) -> Any | None:
    """Auto-compaction at Loom's configured threshold rather than deepagents'.

    Two reasons to override the default. Loom exposes ``compaction_threshold``
    and until now nothing read it. And deepagents derives its trigger from the
    model's published profile — which a local orchestrator (ChatOllama) does not
    have, so it falls back to a flat 170K tokens that a 32K model can never reach
    before overflowing.
    """
    from loom.core.artifact_store import summarization_middleware

    return summarization_middleware(config, backend, model=model, model_string=model_string)
