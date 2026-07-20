"""Slash commands for the REPL (Claude Code-style ``/command``).

Each handler takes the live :class:`~loom.ui.repl.Session` and the argument
string, and returns ``True`` if the loop should continue (always, except
``/exit``). Handlers render directly to ``session.console``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from loom.ui.repl import Session

# name -> (help text, handler)
_REGISTRY: dict[str, tuple[str, Callable[["Session", str], bool]]] = {}


def command(name: str, help_text: str):
    def deco(fn: Callable[["Session", str], bool]):
        _REGISTRY[name] = (help_text, fn)
        return fn

    return deco


def dispatch(session: "Session", line: str) -> bool:
    """Handle a ``/command``. Returns False only to signal exit."""
    parts = line[1:].split(maxsplit=1)
    name = parts[0] if parts else ""
    args = parts[1] if len(parts) > 1 else ""
    aliases = {"quit": "exit", "q": "exit", "?": "help", "h": "help", "config": "settings"}
    name = aliases.get(name, name)
    entry = _REGISTRY.get(name)
    if entry is None:
        session.console.print(f"[loom.warn]unknown command:[/loom.warn] /{name} — try /help")
        return True
    return entry[1](session, args)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@command("help", "Show this help")
def _help(session: "Session", args: str) -> bool:
    table = Table(title="Loom commands", show_header=True, header_style="loom.accent")
    table.add_column("Command")
    table.add_column("What it does")
    for name, (help_text, _) in sorted(_REGISTRY.items()):
        table.add_row(f"/{name}", help_text)
    table.add_row("<text>", "Send a task to the orchestrator")
    session.console.print(table)
    return True


@command("exit", "Quit Loom")
def _exit(session: "Session", args: str) -> bool:
    session.console.print("[loom.dim]bye[/loom.dim]")
    return False


@command("clear", "Reset the conversation")
def _clear(session: "Session", args: str) -> bool:
    session.reset()
    session.console.print("[loom.dim]conversation cleared[/loom.dim]")
    return True


@command("plan", "Toggle plan mode: read-only planning, then approve & execute")
def _plan(session: "Session", args: str) -> bool:
    choice = args.strip().lower()
    turn_on = choice == "on" if choice in ("on", "off") else not session.plan
    session.set_mode("plan" if turn_on else "default")
    if session.plan:
        session.console.print(
            "plan mode: [loom.accent]on[/loom.accent] — read-only; no edits, no shell writes.\n"
            "[loom.dim]when the plan is ready you'll be asked to approve it — approving "
            "executes it immediately (Shift+Tab cycles modes)[/loom.dim]"
        )
    else:
        session.console.print("plan mode: [loom.accent]off[/loom.accent]")
    return True


@command("local", "Toggle local-only mode (no cloud calls)")
def _local(session: "Session", args: str) -> bool:
    session.local_only = not session.local_only
    session.rebuild()
    session.console.print(f"local-only: [loom.accent]{'on' if session.local_only else 'off'}[/loom.accent]")
    return True


@command("yolo", "Toggle full auto-approve (shorthand for /mode yolo)")
def _yolo(session: "Session", args: str) -> bool:
    if session.yolo:
        session.set_mode("default")
        session.console.print("auto-approve: [loom.warn]off[/loom.warn] (mode: default)")
    else:
        session.set_mode("yolo")
        session.console.print("auto-approve: [loom.warn]ON — every tool runs without asking[/loom.warn]")
    return True


@command("mode", "Show or set the mode: /mode [default|accept-edits|plan|yolo] (Shift+Tab cycles)")
def _mode(session: "Session", args: str) -> bool:
    choice = args.strip().lower()
    aliases = {
        "normal": "default",
        "edits": "accept-edits",
        "accept_edits": "accept-edits",
    }
    choice = aliases.get(choice, choice)
    if not choice:
        choice = session.cycle_approval_mode()
    elif choice in ("default", "accept-edits", "plan", "yolo"):
        session.set_mode(choice)
    else:
        session.console.print(
            f"[loom.err]unknown mode:[/loom.err] {choice} (default | accept-edits | plan | yolo)"
        )
        return True
    desc = {
        "default": "every ask-tool prompts you",
        "accept-edits": "file edits auto-approve; shell and the rest still ask",
        "plan": "read-only planning — approve the plan to execute it",
        "yolo": "everything auto-approves",
    }[choice]
    session.console.print(f"mode: [loom.accent]{choice}[/loom.accent] — {desc}")
    return True


def parse_loop_args(args: str) -> tuple[int, str, str | None]:
    """``/loop [N] <prompt> [--until "cmd"]`` → (max_iters, prompt, until)."""
    until = None
    if args.strip().startswith("--until "):
        until = args.strip()[len("--until ") :].strip().strip("\"'") or None
        args = ""
    elif " --until " in args:
        args, _, until = args.partition(" --until ")
        until = until.strip().strip("\"'") or None
    parts = args.split(maxsplit=1)
    max_iters = 10
    if parts and parts[0].isdigit():
        max_iters = max(1, min(int(parts[0]), 100))
        args = parts[1] if len(parts) > 1 else ""
    return max_iters, args.strip(), until


@command("loop", "Iterate on a task until done: /loop [N] <task> [--until \"pytest -q\"]")
def _loop(session: "Session", args: str) -> bool:
    max_iters, prompt, until = parse_loop_args(args)
    if not prompt and not until:
        session.console.print(
            "usage: [loom.accent]/loop [N] <task> [--until \"check command\"][/loom.accent]\n"
            "[loom.dim]runs up to N iterations (default 10); stops when the agent reports\n"
            "LOOP_COMPLETE, or — with --until — when the check command exits 0.\n"
            "check failures are fed back into the next iteration.[/loom.dim]"
        )
        return True
    if not prompt:
        prompt = f"Make the check command `{until}` pass."
    session.run_loop(prompt, max_iters=max_iters, until=until)
    return True


_MODEL_ROLES = ("orchestrator", "advisor", "escalation")


def _model_roles(session: "Session") -> list[str]:
    from loom.subagents import SPECS

    return list(_MODEL_ROLES) + list(SPECS)


def _set_role_model(session: "Session", role: str, model: str) -> None:
    from loom.core import settings as st

    key = {
        "orchestrator": "models.orchestrator",
        "advisor": "models.advisor",
        "escalation": "models.escalation_model",
    }.get(role, f"models.subagents.{role}")
    st.set_value(key, model)
    session.reload_settings()
    session.rebuild()
    session.console.print(f"{role} → [loom.accent]{model}[/loom.accent]")
    _offer_pull_if_missing(session, model)


def _offer_pull_if_missing(session: "Session", model: str) -> None:
    """Local model just assigned but not installed? Offer to pull it now —
    otherwise the next build silently reroutes the role to the billed cloud
    fallback."""
    from loom.core import ollama
    from loom.core.model_router import resolve

    cfg = session.settings.models
    if not cfg.is_local(model):
        return
    tag = resolve(model).name
    st = ollama.status(cfg)
    if st.running and ollama.is_served(tag, st.models):
        return
    if not st.running:
        session.console.print(f"[loom.warn]{ollama.daemon_hint(st.endpoint)} — until then this role runs on {cfg.cloud_fallback} (billed).[/loom.warn]")
        return
    from rich.prompt import Confirm

    if Confirm.ask(f"  `{tag}` isn't pulled yet — pull it now?", default=True):
        if ollama.pull(tag, cfg.ollama_endpoint, session.console) == 0:
            session.console.print(f"[loom.subagent]✓ {tag} ready[/loom.subagent]")
            session.rebuild()
        else:
            session.console.print(f"[loom.warn]pull failed — this role runs on {cfg.cloud_fallback} (billed) until `{tag}` is pulled.[/loom.warn]")
    else:
        session.console.print(f"[loom.dim]skipped — this role runs on {cfg.cloud_fallback} (billed) until you pull `{tag}`.[/loom.dim]")


def _model_candidates(session: "Session") -> list[tuple[str, str]]:
    """(model string, where-label) pairs: installed local Ollama models first,
    then hardware-fitting recommendations that just need a pull, then each
    cloud provider's example models. For full provider/credential control,
    use /setup."""
    from loom.core import ollama
    from loom.core import providers as prov
    from loom.core import recommendations as rec

    st = ollama.status(session.settings.models)
    out: list[tuple[str, str]] = [(f"ollama/{tag}", "local · installed") for tag in st.models]
    seen = {m for m, _ in out}
    for r in rec.recommend_local_models(rec.detect_hardware()):
        model = f"ollama/{r.tag}"
        if model not in seen and not ollama.is_served(r.tag, st.models):
            out.append((model, "local · pulls on select"))
            seen.add(model)
    # Use the full prefixed string (not the bare model id) — some providers
    # (zen/go/custom/vertexai) need it to resolve to the right provider at all.
    for p in prov.cloud_providers():
        for m in p.example_models:
            model = p.model_string(m)
            if model not in seen:
                out.append((model, "cloud"))
                seen.add(model)
    return out


@command("model", "Show models, or set one: /model [role] [model] — /model editor picks interactively")
def _model(session: "Session", args: str) -> bool:
    cfg = session.settings.models
    parts = args.split()
    roles = _model_roles(session)

    if not parts:
        table = Table(show_header=True, header_style="loom.accent")
        for col in ("Role", "Model", "Where"):
            table.add_column(col)
        table.add_row("orchestrator", cfg.orchestrator, "local" if cfg.is_local(cfg.orchestrator) else "cloud")
        table.add_row("advisor", cfg.advisor, "cloud")
        table.add_row("escalation", cfg.escalation_model, "local" if cfg.is_local(cfg.escalation_model) else "cloud")
        for name, model in cfg.subagents.items():
            table.add_row(name, model, "local" if cfg.is_local(model) else "cloud")
        session.console.print(table)
        local = [m for m, where in _model_candidates(session) if where == "local · installed"]
        if local:
            session.console.print(f"[loom.dim]installed local models: {', '.join(m[7:] for m in local)}[/loom.dim]")
        session.console.print("[loom.dim]set: /model <role> <model> · pick interactively: /model <role>[/loom.dim]")
        return True

    if parts[0] not in roles:
        # Back-compat: /model <model> sets the orchestrator.
        _set_role_model(session, "orchestrator", parts[0])
        return True

    role = parts[0]
    if len(parts) > 1:
        _set_role_model(session, role, parts[1])
        return True

    # Interactive picker: installed local models, hardware-recommended pulls,
    # and common cloud models. Picking a not-yet-pulled local model offers to
    # download it on the spot.
    candidates = _model_candidates(session)
    if not candidates:
        session.console.print("[loom.warn]no models found — is the Ollama daemon running?[/loom.warn]")
        return True
    current = {
        "orchestrator": cfg.orchestrator,
        "advisor": cfg.advisor,
        "escalation": cfg.escalation_model,
    }.get(role) or cfg.subagents.get(role, "(inherit)")
    session.console.print(f"pick a model for [loom.accent]{role}[/loom.accent] (current: {current}):")
    for i, (model, where) in enumerate(candidates, 1):
        session.console.print(f"  [loom.accent]{i}[/loom.accent]  {model}  [loom.dim]({where})[/loom.dim]")
    from rich.prompt import Prompt

    choice = Prompt.ask("  number (or model name, empty to cancel)", default="")
    choice = choice.strip()
    if not choice:
        session.console.print("[loom.dim]cancelled[/loom.dim]")
        return True
    if choice.isdigit() and 1 <= int(choice) <= len(candidates):
        choice = candidates[int(choice) - 1][0]
    _set_role_model(session, role, choice)
    return True


@command("setup", "Run the setup wizard: configure providers/models for every role")
def _setup(session: "Session", args: str) -> bool:
    from loom.ui import onboarding

    roles = onboarding.ALL_ROLES
    if args.strip():
        requested = tuple(r for r in args.split() if r in onboarding.ALL_ROLES)
        if requested:
            roles = requested
    try:
        onboarding.run(session.console, root=session.cwd, roles=roles)
    except (KeyboardInterrupt, EOFError):
        session.console.print("\n[loom.dim]setup cancelled[/loom.dim]")
        return True
    session.reload_settings()
    session.rebuild()
    return True


@command("agents", "List subagents and their models")
def _agents(session: "Session", args: str) -> bool:
    from loom.subagents import describe_subagents

    table = Table(show_header=True, header_style="loom.accent")
    for col in ("Agent", "Model", "Where", "Mode"):
        table.add_column(col)
    for row in describe_subagents(session.settings.models):
        table.add_row(row["name"], row["model"], row["scope"], row["mode"])
    session.console.print(table)
    return True


@command("models", "Check local Ollama models (status)")
def _models(session: "Session", args: str) -> bool:
    from loom.core import ollama

    st = ollama.status(session.settings.models)
    if not st.running:
        hint = ollama.daemon_hint(st.endpoint) if st.installed else ollama.INSTALL_HINT
        session.console.print(f"[loom.err]{hint}[/loom.err]")
        return True
    missing = ollama.missing_models(session.settings.models)
    session.console.print(f"ollama daemon: running @ {st.endpoint}")
    session.console.print(
        "[loom.warn]missing:[/loom.warn] " + ", ".join(missing) + " — pull with `loom models pull`"
        if missing
        else "[loom.subagent]all models present[/loom.subagent]"
    )
    return True


@command("permissions", "Show the active permission rules")
def _permissions(session: "Session", args: str) -> bool:
    p = session.settings.permissions
    body = (
        f"default: {p.default_mode}\n"
        f"allow: {', '.join(p.allow) or '—'}\n"
        f"ask:   {', '.join(p.ask) or '—'}\n"
        f"deny:  {', '.join(p.deny) or '—'}"
    )
    session.console.print(Panel(body, title="permissions", border_style="loom.accent"))
    return True


@command("settings", "Show settings, or set one: /settings ui.theme light")
def _settings(session: "Session", args: str) -> bool:
    from loom.core import settings as st

    parts = args.split(maxsplit=1)
    if len(parts) == 2:
        st.set_value(parts[0], parts[1])
        session.reload_settings()
        session.rebuild()
        session.console.print(f"set [loom.accent]{parts[0]}[/loom.accent] = {parts[1]}")
    else:
        import json

        session.console.print(
            Panel(
                json.dumps(session.settings.model_dump(exclude={"models"}), indent=2),
                title="settings.json (user+project merged)",
                border_style="loom.dim",
            )
        )
    return True


@command("cwd", "Show the project root the agents are sandboxed to")
def _cwd(session: "Session", args: str) -> bool:
    session.console.print(str(session.cwd))
    return True


@command("status", "Show version, models, modes, MCP, and session usage")
def _status(session: "Session", args: str) -> bool:
    from loom import __version__
    from loom.core.mcp import mcp_status

    cfg = session.settings.models
    modes = [
        m
        for m, on in (
            ("plan", session.plan),
            ("local-only", session.local_only),
            ("airgap", session.airgap),
        )
        if on
    ]
    if session.approval_mode != "default":
        modes.append(session.approval_mode)
    mcp_line = ", ".join(
        f"{r['name']} ({r['state']}{', ' + str(len(r['tools'])) + ' tools' if r['tools'] else ''})"
        for r in mcp_status(session.settings)
    ) or "—"
    u = session.usage
    table = Table.grid(padding=(0, 2))
    table.add_row("[loom.dim]version[/loom.dim]", f"loom v{__version__}")
    table.add_row("[loom.dim]cwd[/loom.dim]", str(session.cwd))
    def _with_badge(role: str, model: str) -> str:
        origin = session.model_origin(role)
        model, is_local = origin if origin else (model, cfg.is_local(model))
        return f"{'⌂' if is_local else '☁'} {model} ({'local' if is_local else 'cloud'})"

    table.add_row("[loom.dim]orchestrator[/loom.dim]", _with_badge("orchestrator", cfg.orchestrator))
    table.add_row("[loom.dim]advisor[/loom.dim]", _with_badge("advisor", cfg.advisor))
    local_tags = session.local_model_tags()
    table.add_row(
        "[loom.dim]local models[/loom.dim]",
        ("⌂ " + ", ".join(local_tags)) if local_tags else "— (all roles on cloud)",
    )
    table.add_row("[loom.dim]mode[/loom.dim]", ", ".join(modes) or "normal")
    table.add_row("[loom.dim]permissions[/loom.dim]", f"default: {session.settings.permissions.default_mode}")
    table.add_row("[loom.dim]mcp[/loom.dim]", mcp_line)
    table.add_row("[loom.dim]memory[/loom.dim]", str(session.memory_path() or "— (create with /init)"))
    table.add_row("[loom.dim]persistence[/loom.dim]", "sqlite (.loom/sessions.db)" if session.durable else "in-memory (no /resume across restarts)")
    escalations = sum(
        getattr(g, "escalation_count", 0) for g in getattr(session.bundle, "guards", []) or []
    ) if session.bundle is not None else 0
    table.add_row("[loom.dim]escalations[/loom.dim]", f"{escalations} local→cloud (prompt-size guard)")
    table.add_row(
        "[loom.dim]session[/loom.dim]",
        f"{u['turns']} turns · {u['input_tokens']} in / {u['output_tokens']} out tokens · ${u['cloud_cost']:.3f} cloud",
    )
    share = session.tracker.session.local_share()
    saved = session.tracker.session.savings(cfg.orchestrator)
    table.add_row(
        "[loom.dim]savings[/loom.dim]",
        f"{share:.0%} of tokens ran locally (free) · saved ~${saved:.2f} vs all-cloud",
    )
    session.console.print(Panel(table, title="status", border_style="loom.accent", expand=False))
    return True


@command("graphify", "Code knowledge graph (GraphRAG): /graphify [build|update|on|off|query <q>|path <a> <b>|explain <c>]")
def _graphify(session: "Session", args: str) -> bool:
    from loom.core import graphify
    from loom.core import mcp as mcp_mod
    from loom.core import settings as st

    parts = args.split(maxsplit=1)
    verb = parts[0] if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    def _set_server(enabled: bool) -> None:
        if enabled:
            # Pin the resolved binary path — a fresh `uv tool install` lands in
            # ~/.local/bin, which may not be on the PATH the MCP spawn inherits.
            st.set_value("mcp_servers.graphify.command", graphify.binary() or "graphify")
        st.set_value("mcp_servers.graphify.enabled", "true" if enabled else "false")
        session.reload_settings()
        # MCP sessions are a process-wide singleton — restart so the next turn
        # (re)connects with the new server set.
        mcp_mod.shutdown_mcp()
        session.rebuild()

    def _ensure_installed() -> bool:
        """Offer to install the graphify CLI on the spot; True when usable."""
        if graphify.installed():
            return True
        from rich.prompt import Confirm

        session.console.print(
            "[loom.warn]graphify isn't installed[/loom.warn] [loom.dim]— free, MIT, runs fully "
            "locally (tree-sitter); powers graph-RAG structure queries[/loom.dim]"
        )
        try:
            if not Confirm.ask(f"  install it now via `uv tool install {graphify.PYPI_NAME}`?", default=True):
                session.console.print(f"[loom.dim]{graphify.INSTALL_HINT}[/loom.dim]")
                return False
        except (EOFError, KeyboardInterrupt):
            return False
        ok, how = graphify.install()
        if ok:
            session.console.print(f"[loom.subagent]✓ graphify installed[/loom.subagent] [loom.dim]({how})[/loom.dim]")
        else:
            session.console.print(f"[loom.err]install failed ({how})[/loom.err] [loom.dim]{graphify.INSTALL_HINT}[/loom.dim]")
        return ok

    def _build(update: bool) -> None:
        session.console.print(f"[loom.accent]⏺ graphify {'--update' if update else ''} — indexing {session.cwd}[/loom.accent]")
        code = graphify.build(session.cwd, update=update)
        if code != 0:
            session.console.print(f"[loom.err]graphify exited with code {code}[/loom.err]")
            return
        detail = graphify.format_stats(graphify.graph_stats(session.cwd))
        session.console.print(f"[loom.subagent]✓ graph ready[/loom.subagent] [loom.dim]({detail or 'graphify-out/graph.json'})[/loom.dim]")
        srv = session.settings.mcp_servers.get("graphify")
        if srv is None or not srv.enabled:
            _set_server(True)
            session.console.print("[loom.dim]graphify MCP server enabled — graph tools connect on the next task[/loom.dim]")

    if verb in ("build", "update"):
        if _ensure_installed():
            _build(update=verb == "update")
        return True

    if verb in ("on", "off"):
        if verb == "on" and not graphify.graph_exists(session.cwd):
            session.console.print("[loom.warn]no graph yet — run /graphify build first[/loom.warn]")
            return True
        _set_server(verb == "on")
        session.console.print(f"graphify MCP server: [loom.accent]{verb}[/loom.accent]")
        return True

    if verb in ("query", "path", "explain"):
        if not _ensure_installed():
            return True
        if not graphify.graph_exists(session.cwd):
            session.console.print("[loom.warn]no graph yet — run /graphify build first[/loom.warn]")
            return True
        cli_args = [verb, *([rest] if verb != "path" else rest.split(maxsplit=1))]
        code, out = graphify.run_cli(session.cwd, *[a for a in cli_args if a])
        style = "loom.err" if code != 0 else None
        session.console.print(out or "(no output)", style=style)
        return True

    # No/unknown verb: status.
    stats = graphify.graph_stats(session.cwd)
    server = session.settings.mcp_servers.get("graphify")
    state = next((r["state"] for r in mcp_mod.mcp_status(session.settings) if r["name"] == "graphify"), "not configured")
    lines = [
        f"cli:    {'installed' if graphify.installed() else 'not installed — ' + graphify.INSTALL_HINT}",
        f"graph:  {graphify.format_stats(stats) or 'not built — /graphify build'}",
        f"server: {state if server is not None and server.enabled else 'disabled — /graphify on'}",
    ]
    lines.append(
        "[loom.dim]the orchestrator + explorer/searcher answer structure questions from the graph\n"
        "instead of glob/grep/read sweeps — fewer tokens, real file:line citations[/loom.dim]"
    )
    session.console.print(Panel("\n".join(lines), title="graphify — code knowledge graph", border_style="loom.accent", expand=False))
    # First run: walk through install + build right here instead of making the
    # user retype the verbs — Loom is a coding assistant, it sets itself up.
    if stats is None:
        from rich.prompt import Confirm

        if not _ensure_installed():
            return True
        try:
            if Confirm.ask("  build the knowledge graph for this repo now?", default=True):
                _build(update=False)
        except (EOFError, KeyboardInterrupt):
            pass
    return True


@command("skills", "List agent skills (SKILL.md folders: packaged, ~/.loom/skills, .loom/skills)")
def _skills(session: "Session", args: str) -> bool:
    from loom.core import skills as skills_mod

    found = skills_mod.list_skills(session.cwd)
    if not found:
        session.console.print(
            "[loom.dim]no skills found — add one at .loom/skills/<name>/SKILL.md (project) "
            "or ~/.loom/skills/<name>/SKILL.md (all projects)[/loom.dim]"
        )
        return True
    table = Table(show_header=True, header_style="loom.accent")
    for col in ("Skill", "Source", "Description"):
        table.add_column(col)
    for s in found:
        table.add_row(s["name"], s["source"], s["description"][:100])
    session.console.print(table)
    session.console.print(
        "[loom.dim]skills load via deepagents (progressive disclosure): only name+description "
        "enter the prompt; the agent reads the full SKILL.md when a task matches[/loom.dim]"
    )
    return True


@command("mcp", "List MCP servers, connection state, and their tools")
def _mcp(session: "Session", args: str) -> bool:
    from loom.core.mcp import mcp_status

    rows = mcp_status(session.settings)
    if not rows:
        session.console.print("[loom.dim]no MCP servers configured (settings.json → mcp_servers)[/loom.dim]")
        return True
    table = Table(show_header=True, header_style="loom.accent")
    for col in ("Server", "Transport", "Target", "State", "Tools"):
        table.add_column(col)
    for r in rows:
        tools = f"{len(r['tools'])}: {', '.join(r['tools'][:5])}{'…' if len(r['tools']) > 5 else ''}" if r["tools"] else "—"
        table.add_row(r["name"], r["transport"], r["target"], r["state"], tools)
    session.console.print(table)
    session.console.print("[loom.dim]servers connect on the first task; browser_* tools power the tester subagent[/loom.dim]")
    return True


@command("cost", "Show the session cost receipt (cloud vs free local tokens)")
def _cost(session: "Session", args: str) -> bool:
    t = session.tracker
    u = t.session
    session.console.print(f"session: [loom.accent]{t.turns}[/loom.accent] turns")
    table = Table(show_header=True, header_style="loom.accent")
    for col in ("Model", "Where", "In", "Out", "Cost"):
        table.add_column(col)
    from loom.core.usage import cost_usd

    for model, mu in u.cloud.items():
        table.add_row(model, "cloud", f"{mu.input_tokens:,}", f"{mu.output_tokens:,}", f"${cost_usd(model, mu.input_tokens, mu.output_tokens):.3f}")
    for model, mu in u.local.items():
        table.add_row(model, "local", f"{mu.input_tokens:,}", f"{mu.output_tokens:,}", "free")
    if table.row_count:
        session.console.print(table)
    receipt = t.receipt(turn=False)
    if receipt:
        session.console.print(f"[loom.dim]✻ {receipt}[/loom.dim]")
    else:
        session.console.print("[loom.dim]0 tokens spent so far[/loom.dim]")
    return True


@command("resume", "List past sessions, or resume one: /resume [n | thread-id]")
def _resume(session: "Session", args: str) -> bool:
    from loom.core import sessions as sessions_mod

    rows = sessions_mod.load_index(session.cwd)
    if not args.strip():
        if not rows:
            session.console.print("[loom.dim]no past sessions in this project[/loom.dim]")
            return True
        table = Table(show_header=True, header_style="loom.accent")
        for col in ("#", "Updated", "Turns", "Title", "Thread"):
            table.add_column(col)
        for i, row in enumerate(reversed(rows), 1):
            table.add_row(str(i), row["updated"], str(row.get("turns", "?")), row["title"], row["thread_id"])
        session.console.print(table)
        if not session.durable:
            session.console.print("[loom.warn]sessions.db unavailable (install langgraph-checkpoint-sqlite) — history won't survive restarts[/loom.warn]")
        session.console.print("[loom.dim]resume: /resume <#> or /resume <thread-id>[/loom.dim]")
        return True

    choice = args.strip()
    target = None
    if choice.isdigit():
        ordered = list(reversed(rows))
        if 1 <= int(choice) <= len(ordered):
            target = ordered[int(choice) - 1]["thread_id"]
    else:
        target = next((r["thread_id"] for r in rows if r["thread_id"] == choice), None)
    if target is None:
        session.console.print(f"[loom.err]no such session:[/loom.err] {choice}")
        return True
    session.thread_id = target
    session._memory_sent = True  # resumed thread already has its context
    session.console.print(f"resumed [loom.accent]{target}[/loom.accent] — continue where you left off")
    return True


@command("undo", "Roll back the file changes of the last turn")
def _undo(session: "Session", args: str) -> bool:
    from loom.core import undo as undo_mod

    restored = undo_mod.undo_last(session.cwd)
    if not restored:
        session.console.print("[loom.dim]nothing to undo (no snapshotted file writes)[/loom.dim]")
        return True
    for rel in restored:
        session.console.print(f"  ⎿ restored {rel}")
    session.console.print(f"[loom.accent]↩ rolled back {len(restored)} file(s) from the last turn[/loom.accent]")
    return True


@command("airgap", "Toggle airgap mode — raw code never reaches the cloud")
def _airgap(session: "Session", args: str) -> bool:
    session.airgap = not session.airgap
    session.rebuild()
    if session.airgap:
        session.console.print(
            "airgap: [loom.accent]on[/loom.accent] — cloud orchestrator plans from summaries only; "
            "local subagents do all file reading; cloud escalation disabled"
        )
    else:
        session.console.print("airgap: [loom.accent]off[/loom.accent]")
    return True


@command("compact", "Summarize the conversation and free up context")
def _compact(session: "Session", args: str) -> bool:
    transcript = session.transcript()
    if not transcript:
        session.console.print("[loom.dim]nothing to compact yet[/loom.dim]")
        return True
    lines = []
    for m in transcript:
        role = m[0] if isinstance(m, tuple) else getattr(m, "type", "?")
        content = m[1] if isinstance(m, tuple) else getattr(m, "content", "")
        if content and role in ("user", "human", "assistant", "ai"):
            lines.append(f"{role}: {str(content)[:2000]}")
    if not lines:
        session.console.print("[loom.dim]nothing to compact yet[/loom.dim]")
        return True

    from loom.core.model_router import build_model

    cfg = session.settings.models
    model_string = cfg.subagents.get("general-purpose", cfg.orchestrator) if session.local_only else cfg.orchestrator
    try:
        model = build_model(model_string, cfg)
        prompt = (
            "Summarize this coding-session transcript so work can continue "
            "seamlessly: goals, decisions, files touched, current state, and "
            "open next steps. Be concise but lose nothing load-bearing.\n\n"
            + "\n".join(lines)
        )
        summary = str(model.invoke(prompt, config={"callbacks": [session.tracker]}).content)
    except Exception as exc:
        session.console.print(f"[loom.err]compact failed:[/loom.err] {exc}")
        return True
    session.reset()
    session.pending_context = summary
    session.console.print("[loom.dim]✻ context compacted — summary will be carried into your next message[/loom.dim]")
    return True


@command("doctor", "Check the health of your Loom setup")
def _doctor(session: "Session", args: str) -> bool:
    import os
    import shutil
    import sys

    from loom.core import ollama
    from loom.core.mcp import mcp_status

    def row(ok: bool | None, label: str, detail: str) -> str:
        mark = "[loom.subagent]✓[/loom.subagent]" if ok else ("[loom.warn]•[/loom.warn]" if ok is None else "[loom.err]✗[/loom.err]")
        return f" {mark} {label}: {detail}"

    out = [row(sys.version_info >= (3, 11), "python", sys.version.split()[0])]

    cfg = session.settings.models
    st = ollama.status(cfg)
    if st.running:
        detail = f"running @ {st.endpoint}" + ("" if st.installed else " (remote — no local binary)")
        out.append(row(True, "ollama", detail))
        missing = ollama.missing_models(cfg)
        out.append(row(not missing, "local models", ", ".join(missing) + " missing" if missing else "all present"))
        if missing:
            out.append(row(None, "cloud fallback", f"local roles run on {cfg.cloud_fallback} (billed)"))
    else:
        out.append(row(False, "ollama", f"not reachable @ {st.endpoint}" + ("" if st.installed else ", binary not installed")))
        out.append(row(None, "cloud fallback", f"local roles run on {cfg.cloud_fallback} (billed)"))

    def effective_env(key: str) -> str | None:
        return os.environ.get(key) or session.settings.env.get(key)

    key_set = bool(
        effective_env("ANTHROPIC_API_KEY")
        or effective_env("ANTHROPIC_AUTH_TOKEN")
        or effective_env("AWS_BEARER_TOKEN_BEDROCK")
    )
    out.append(row(key_set, "anthropic_api_key", "set" if key_set else "not set"))

    out.append(row(bool(shutil.which("npx")), "npx", "found" if shutil.which("npx") else "not found (Playwright MCP needs Node)"))
    for r in mcp_status(session.settings):
        ok: bool | None = True if r["state"] == "connected" else (None if r["state"] in ("not connected", "disabled") else False)
        out.append(row(ok, f"mcp:{r['name']}", r["state"]))

    session.console.print(Panel("\n".join(out), title="doctor", border_style="loom.accent", expand=False))
    return True


@command("init", "Analyze the codebase and write a LOOM.md memory file")
def _init(session: "Session", args: str) -> bool:
    existing = session.memory_path()
    if existing is not None and existing.name == "LOOM.md":
        session.console.print(f"[loom.warn]{existing}[/loom.warn] already exists — edit it with /memory")
        return True
    session.run_turn(
        "Analyze this codebase and write a LOOM.md file in the project root: a "
        "concise memory file for AI coding agents. Include: what the project "
        "is, how to build/test/run it, architecture and key directories, and "
        "any conventions an agent must follow. Keep it under ~60 lines."
    )
    return True


@command("memory", "Show the project memory file (LOOM.md / CLAUDE.md)")
def _memory(session: "Session", args: str) -> bool:
    from rich.markdown import Markdown

    path = session.memory_path()
    if path is None:
        session.console.print("[loom.dim]no memory file (LOOM.md / CLAUDE.md / AGENTS.md) — create one with /init[/loom.dim]")
        return True
    session.console.print(Panel(Markdown(path.read_text(encoding="utf-8")), title=str(path), border_style="loom.dim"))
    session.console.print(f"[loom.dim]it is sent with your first message each session; edit: $EDITOR {path.name}[/loom.dim]")
    return True


@command("export", "Save the conversation to a markdown file: /export [path]")
def _export(session: "Session", args: str) -> bool:
    from datetime import datetime

    target = Path(args.strip()) if args.strip() else session.cwd / f"loom-session-{datetime.now():%Y%m%d-%H%M%S}.md"
    lines = ["# Loom session\n"]
    for m in session.transcript():
        role = m[0] if isinstance(m, tuple) else getattr(m, "type", "?")
        content = m[1] if isinstance(m, tuple) else getattr(m, "content", "")
        if content and role in ("user", "human", "assistant", "ai"):
            who = "You" if role in ("user", "human") else "Loom"
            lines.append(f"## {who}\n\n{content}\n")
    target.write_text("\n".join(lines), encoding="utf-8")
    session.console.print(f"exported → [loom.accent]{target}[/loom.accent]")
    return True


@command("hooks", "Show the configured tool hooks")
def _hooks(session: "Session", args: str) -> bool:
    h = session.settings.hooks
    table = Table(show_header=True, header_style="loom.accent")
    for col in ("Event", "Matcher", "Command"):
        table.add_column(col)
    for event in ("pre_tool_use", "post_tool_use", "user_prompt_submit", "stop"):
        for hook in getattr(h, event):
            table.add_row(event, hook.matcher, hook.command)
    if table.row_count:
        session.console.print(table)
    else:
        session.console.print("[loom.dim]no hooks configured (settings.json → hooks)[/loom.dim]")
    return True


@command("theme", "Show or set the UI theme: /theme dark|light|mono|auto")
def _theme(session: "Session", args: str) -> bool:
    if not args.strip():
        session.console.print(f"theme: [loom.accent]{session.settings.ui.theme}[/loom.accent] (dark, light, mono, auto)")
        return True
    from loom.core import settings as st

    try:
        st.set_value("ui.theme", args.strip())
    except Exception as exc:
        session.console.print(f"[loom.err]{exc}[/loom.err]")
        return True
    session.reload_settings()
    session.console.print(f"theme → [loom.accent]{args.strip()}[/loom.accent]")
    return True


@command("vim", "Toggle vim editing mode for the input line")
def _vim(session: "Session", args: str) -> bool:
    session.vim = not session.vim
    ps = getattr(session, "_prompt_session", None)
    if ps is not None:
        try:
            from prompt_toolkit.enums import EditingMode

            ps.editing_mode = EditingMode.VI if session.vim else EditingMode.EMACS
        except Exception:
            pass
    session.console.print(f"vim mode: [loom.accent]{'on' if session.vim else 'off'}[/loom.accent]")
    return True
