"""Slash commands for the REPL (Claude Code-style ``/command``).

Each handler takes the live :class:`~loom.ui.repl.Session` and the argument
string, and returns ``True`` if the loop should continue (always, except
``/exit``). Handlers render directly to ``session.console``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from rich.text import Text

from loom.ui import render
from loom.ui.render import ink

if TYPE_CHECKING:
    from loom.ui.repl import Session

# name -> (help text, handler)
_REGISTRY: dict[str, tuple[str, Callable[["Session", str], bool]]] = {}
# Commands grouped for /help, in the order they're most useful to a newcomer.
# A command missing from every group still shows, under "more".
_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("session", ("help", "status", "cost", "clear", "compact", "resume", "export", "exit")),
    ("modes", ("mode", "plan", "yolo", "local", "airgap", "loop", "vim")),
    ("fleet", ("model", "agents", "ollama", "setup", "skills", "mcp", "playwright", "graphify")),
    ("project", ("cwd", "memory", "init", "undo", "permissions", "hooks", "privacy", "settings", "theme", "doctor")),
]


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
    # `/models` (plural) reads as the config command's plural, so route it
    # there — the Ollama daemon/model health check is `/ollama`, not `/models`.
    aliases = {"quit": "exit", "q": "exit", "?": "help", "h": "help", "config": "settings", "models": "model"}
    name = aliases.get(name, name)
    entry = _REGISTRY.get(name)
    if entry is None:
        import difflib

        near = difflib.get_close_matches(name, _REGISTRY, n=1, cutoff=0.6)
        hint = f"did you mean [loom.warp]/{near[0]}[/loom.warp]?" if near else "try [loom.warp]/help[/loom.warp]"
        render.note(session.console, f"unknown command /{name} — {hint}", kind="warn")
        return True
    return entry[1](session, args)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@command("help", "Show this help")
def _help(session: "Session", args: str) -> bool:
    console = session.console
    g = ink(console)
    grouped = {name for _, names in _GROUPS for name in names}
    groups = [*_GROUPS]
    if extra := sorted(set(_REGISTRY) - grouped):
        groups.append(("more", tuple(extra)))

    console.print()
    for group, names in groups:
        render.rule(console, group)
        rows = [
            (f"/{name}", Text(_REGISTRY[name][0], style="loom.text")) for name in names if name in _REGISTRY
        ]
        console.print(render.kv(rows, key_style="loom.warp", justify="left"))
        console.print()
    render.note(
        console,
        f"type anything else to send it to the orchestrator {g.dot} "
        f"[loom.warp]@file[/loom.warp] pulls a file into context {g.dot} "
        f"[loom.warp]shift+tab[/loom.warp] cycles modes",
    )
    return True


@command("exit", "Quit Loom")
def _exit(session: "Session", args: str) -> bool:
    render.note(session.console, "bye")
    return False


@command("clear", "Reset the conversation")
def _clear(session: "Session", args: str) -> bool:
    session.reset()
    render.note(session.console, "conversation cleared", kind="good")
    return True


@command("plan", "Toggle plan mode: read-only planning, then approve & execute")
def _plan(session: "Session", args: str) -> bool:
    choice = args.strip().lower()
    turn_on = choice == "on" if choice in ("on", "off") else not session.plan
    session.set_mode("plan" if turn_on else "default")
    if session.plan:
        session.console.print(
            "plan mode: [loom.warp.b]on[/loom.warp.b] — read-only; no edits, no shell writes.\n"
            "[loom.muted]when the plan is ready you'll be asked to approve it — approving "
            "executes it immediately (Shift+Tab cycles modes)[/loom.muted]"
        )
    else:
        session.console.print("plan mode: [loom.warp.b]off[/loom.warp.b]")
    return True


@command("local", "Toggle local-only mode (no cloud calls)")
def _local(session: "Session", args: str) -> bool:
    session.local_only = not session.local_only
    session.rebuild()
    session.console.print(f"local-only: [loom.warp.b]{'on' if session.local_only else 'off'}[/loom.warp.b]")
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
            f"[loom.bad.b]unknown mode:[/loom.bad.b] {choice} (default | accept-edits | plan | yolo)"
        )
        return True
    desc = {
        "default": "every ask-tool prompts you",
        "accept-edits": "file edits auto-approve; shell and the rest still ask",
        "plan": "read-only planning — approve the plan to execute it",
        "yolo": "everything auto-approves",
    }[choice]
    session.console.print(f"mode: [loom.warp.b]{choice}[/loom.warp.b] — {desc}")
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
            "usage: [loom.warp.b]/loop [N] <task> [--until \"check command\"][/loom.warp.b]\n"
            "[loom.muted]runs up to N iterations (default 10); stops when the agent reports\n"
            "LOOP_COMPLETE, or — with --until — when the check command exits 0.\n"
            "check failures are fed back into the next iteration.[/loom.muted]"
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

    # set_value writes the user layer, which this project's .loom/settings.json
    # overrides if it names the same role. Reporting the assignment without
    # checking is how a setting silently doesn't apply.
    from loom.ui.onboarding import effective_model

    actual = effective_model(role, session.settings)
    if actual != model:
        render.note(
            session.console,
            f"[loom.warn]{role}[/loom.warn] is still [loom.text]{actual}[/loom.text] — "
            f"this project's .loom/settings.json overrides the user layer",
            kind="warn",
        )
        render.note(
            session.console,
            f"edit it there, or run [loom.warp]/setup {role}[/loom.warp] and save to the project",
            kind="tip",
        )
        return
    session.console.print(f"{role} → [loom.warp.b]{model}[/loom.warp.b]")
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

    if render.confirm(session.console, f"  `{tag}` isn't pulled yet — pull it now?"):
        if ollama.pull(tag, cfg.ollama_endpoint, session.console) == 0:
            session.console.print(f"[loom.good]✓ {tag} ready[/loom.good]")
            session.rebuild()
        else:
            session.console.print(f"[loom.warn]pull failed — this role runs on {cfg.cloud_fallback} (billed) until `{tag}` is pulled.[/loom.warn]")
    else:
        session.console.print(f"[loom.muted]skipped — this role runs on {cfg.cloud_fallback} (billed) until you pull `{tag}`.[/loom.muted]")


def _model_candidates(session: "Session") -> list[tuple[str, str]]:
    """(model string, where-label) pairs: installed local Ollama models
    first, then Loom's full curated local catalog (needs a pull — see
    recommendations.py; there's no stable public API to list every Ollama
    library model, so this is a hand-maintained snapshot), then each cloud
    provider's models — a live catalog fetch when the provider has one and
    either needs no credential or already has one on file (see
    model_catalog.py), else its hardcoded example models. For full
    provider/credential control, use /setup."""
    from loom.core import model_catalog as catalog
    from loom.core import ollama
    from loom.core import providers as prov
    from loom.core import recommendations as rec

    st = ollama.status(session.settings.models)
    out: list[tuple[str, str]] = [(f"ollama/{tag}", "local · installed") for tag in st.models]
    seen = {m for m, _ in out}
    hw = rec.detect_hardware()
    for r in rec.all_local_models():
        model = f"ollama/{r.tag}"
        if model in seen or ollama.is_served(r.tag, st.models):
            continue
        fit = "fits your hardware" if rec.fits_hardware(hw, r) else "may not fit your hardware"
        out.append((model, f"local · needs pull, {fit}"))
        seen.add(model)
    # Use the full prefixed string (not the bare model id) — some providers
    # (zen/go/custom/vertexai) need it to resolve to the right provider at all.
    env = dict(session.settings.env)
    for p in prov.cloud_providers():
        models, is_live = catalog.available_models(p, env)
        label = "cloud · live" if is_live else "cloud · example"
        for m in models:
            model = p.model_string(m)
            if model not in seen:
                out.append((model, label))
                seen.add(model)
    return out


@command("model", "Show models, or set one: /model [role] [model] — /model editor picks interactively")
def _model(session: "Session", args: str) -> bool:
    cfg = session.settings.models
    parts = args.split()
    roles = _model_roles(session)

    if not parts:
        console = session.console
        rows = [
            ("orchestrator", cfg.orchestrator, cfg.is_local(cfg.orchestrator), ""),
            ("advisor", cfg.advisor, cfg.is_local(cfg.advisor), ""),
            ("escalation", cfg.escalation_model, cfg.is_local(cfg.escalation_model), ""),
            *((n, m, cfg.is_local(m), "") for n, m in cfg.subagents.items()),
        ]
        console.print()
        render.rule(console, "models")
        console.print(render.fleet_table(console, rows))
        console.print()
        # Cheap local-only status line — no cloud network calls, unlike the
        # full _model_candidates() picker used below for interactive selection.
        from loom.core import ollama

        installed = ollama.status(session.settings.models).models
        if installed:
            render.note(console, f"installed locally: [loom.local]{', '.join(installed)}[/loom.local]")
        render.note(
            console,
            "[loom.warp]/model <role> <model>[/loom.warp] sets one "
            f"{ink(console).dot} [loom.warp]/model <role>[/loom.warp] picks interactively",
            kind="tip",
        )
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
    console = session.console
    candidates = _model_candidates(session)
    if not candidates:
        render.note(console, "no models found — is the Ollama daemon running?", kind="warn")
        return True
    current = {
        "orchestrator": cfg.orchestrator,
        "advisor": cfg.advisor,
        "escalation": cfg.escalation_model,
    }.get(role) or cfg.subagents.get(role, "(inherit)")
    console.print()
    render.rule(console, f"model for {role}")
    render.note(console, f"currently [loom.text]{current}[/loom.text]")
    console.print()
    render.choices(console, [(str(i), m, w) for i, (m, w) in enumerate(candidates, 1)])
    choice = render.ask(console, "number, model name, or empty to cancel", default="").strip()
    if not choice:
        render.note(console, "cancelled")
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
        # Re-picking one role's model is not the moment to re-ask someone what
        # they consent to sending — the full wizard is.
        settings = onboarding.run(session.console, root=session.cwd, roles=roles, privacy=(roles == onboarding.ALL_ROLES))
        onboarding.maybe_setup_playwright(session.console, settings)
    except (KeyboardInterrupt, EOFError):
        render.note(session.console, "setup cancelled")
        return True
    session.reload_settings()
    session.rebuild()
    return True


@command("agents", "List subagents and their models")
def _agents(session: "Session", args: str) -> bool:
    from loom.subagents import describe_subagents

    console = session.console
    console.print()
    render.rule(console, "fleet")
    table = render.table(
        console,
        ("agent", {"no_wrap": True}),
        ("", {"justify": "center", "no_wrap": True}),
        ("model", {"overflow": "ellipsis"}),
        ("can", {"no_wrap": True}),
    )
    for row in describe_subagents(session.settings.models):
        is_local = row["scope"] == "local"
        table.add_row(
            Text(row["name"], style="loom.text"),
            render.where(console, is_local),
            Text(row["model"], style="loom.local" if is_local else "loom.cloud"),
            Text(row["mode"], style="loom.muted" if row["mode"] == "read-only" else "loom.warn"),
        )
    console.print(table)
    console.print()
    render.note(console, "each subagent works in its own context and returns only a summary")
    return True


@command("ollama", "Ollama status; `/ollama install` sets it up, `/ollama rm <tag>` frees space")
def _ollama(session: "Session", args: str) -> bool:
    from loom.core import ollama

    console = session.console
    verb, _, rest = args.strip().partition(" ")
    if verb == "rm" and rest.strip():
        ok, message = ollama.remove(rest.strip(), session.settings.models.ollama_endpoint)
        render.note(console, message, kind="good" if ok else "bad")
        if ok:
            session.rebuild()
        return True

    if args.strip() in ("install", "setup", "start", "serve"):
        from loom.ui import onboarding

        if onboarding.ensure_ollama(console, session.settings.models):
            session.rebuild()  # local roles may now be servable
        return True

    st = ollama.status(session.settings.models)
    if not st.running:
        render.note(console, ollama.daemon_hint(st.endpoint) if st.installed else ollama.INSTALL_HINT, kind="bad")
        render.note(console, "[loom.warp]/ollama install[/loom.warp] does it for you", kind="tip")
        return True
    render.note(console, f"daemon running at {st.endpoint}", kind="good")
    missing = ollama.missing_models(session.settings.models)
    if missing:
        render.note(console, f"missing: {', '.join(missing)} — `loom models pull`", kind="warn")
    else:
        render.note(console, f"all {len(st.models)} configured model(s) downloaded", kind="good")
    return True


@command("playwright", "Check/install the Playwright MCP browser (`/playwright install`)")
def _playwright(session: "Session", args: str) -> bool:
    from loom.core import playwright_setup

    if args.strip() == "install":
        browser = "chromium"
        code = playwright_setup.install_browsers(session.console, browser)
        if code == 0:
            session.console.print(f"[loom.good]✓ {browser} installed[/loom.good]")
        return True

    st = playwright_setup.status()
    if not st.npx_available:
        session.console.print(f"[loom.bad.b]{playwright_setup.INSTALL_HINT}[/loom.bad.b]")
        return True
    if st.browsers_installed:
        session.console.print(f"playwright browser: [loom.good]installed[/loom.good] ({st.browsers_dir})")
    else:
        session.console.print(
            f"[loom.warn]no browser installed[/loom.warn] at {st.browsers_dir} — "
            "run `/playwright install` or `loom playwright install`"
        )
    return True


@command("permissions", "Show the active permission rules")
def _permissions(session: "Session", args: str) -> bool:
    console = session.console
    p = session.settings.permissions
    none = ink(console).dot

    def _rules(values, style: str) -> Text:
        return Text(", ".join(values), style=style) if values else Text(none, style="loom.line")

    console.print()
    render.rule(console, "permissions")
    console.print(
        render.kv(
            [
                ("default", Text(p.default_mode, style="loom.text")),
                ("allow", _rules(p.allow, "loom.good")),
                ("ask", _rules(p.ask, "loom.warn")),
                ("deny", _rules(p.deny, "loom.bad")),
            ]
        )
    )
    return True


@command("privacy", "What Loom may share: /privacy [set none|errors|full] [here] [setup]")
def _privacy(session: "Session", args: str) -> bool:
    from loom.core import telemetry as tel
    from loom.ui import privacy as privacy_mod

    console = session.console
    verb, _, rest = args.strip().partition(" ")

    if verb == "set":
        if privacy_mod.set_mode(console, rest.strip(), session.cwd):
            _reactivate_telemetry(session)
        return True
    if verb == "here":
        # Re-answer the per-project question regardless of the stored answer —
        # this verb exists for "I said yes here once and I've changed my mind".
        if tel.global_mode() == tel.DEFAULT_MODE:
            render.note(console, "privacy mode is none — set a mode first ([loom.warp]/privacy set errors[/loom.warp])", kind="warn")
            return True
        try:
            privacy_mod.ask_project(console, session.cwd)
        except (KeyboardInterrupt, EOFError):
            render.note(console, "left as it was")
            return True
        _reactivate_telemetry(session)
        return True
    if verb == "setup":
        try:
            privacy_mod.run(console, session.cwd, force_credentials=True)
        except (KeyboardInterrupt, EOFError):
            render.note(console, "cancelled — privacy left as it was")
            return True
        _reactivate_telemetry(session)
        return True
    if verb and verb != "status":
        render.note(console, f"unknown /privacy verb {verb!r} — try [loom.warp]/privacy[/loom.warp]", kind="warn")
        return True

    privacy_mod.describe(console, session.cwd)
    return True


def _reactivate_telemetry(session: "Session") -> None:
    """Re-read consent after a /privacy change so it takes effect this
    session, and say what actually changed.

    Callbacks are re-assembled per turn (``Session._run_config``) and crash
    reporting checks the mode at capture time, so a mode change needs no
    rebuild — but a mode that was off at startup has its SDKs cold, and a
    mode switched *down* leaves an initialized SDK inert rather than
    un-imported. The note says exactly that; anything vaguer would read as a
    promise either way."""
    from loom.core import telemetry as tel

    try:
        mode = tel.activate(session.cwd)
    except Exception:
        return
    info = tel.mode_info(mode)
    render.note(
        session.console,
        f"in effect now: [loom.warp]{info.label}[/loom.warp] — {info.blurb}",
        kind="good" if mode == "none" else "warn",
    )


@command("settings", "Show settings, or set one: /settings ui.theme light")
def _settings(session: "Session", args: str) -> bool:
    from loom.core import settings as st

    parts = args.split(maxsplit=1)
    if len(parts) == 2:
        st.set_value(parts[0], parts[1])
        session.reload_settings()
        session.rebuild()
        render.note(session.console, f"[loom.warp]{parts[0]}[/loom.warp] = {parts[1]}", kind="good")
    else:
        import json

        from rich.syntax import Syntax

        blob = json.dumps(session.settings.model_dump(exclude={"models"}), indent=2)
        session.console.print()
        session.console.print(
            render.card(
                session.console,
                Syntax(blob, "json", theme="ansi_dark", background_color="default", word_wrap=True),
                title="settings.json",
                subtitle="user + project, merged",
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
    console = session.console
    g = ink(console)
    u = session.usage
    su = session.tracker.session

    def _badge(role: str, model: str) -> Text:
        origin = session.model_origin(role)
        model, is_local = origin if origin else (model, cfg.is_local(model))
        return render.model_badge(console, model, is_local)

    local_tags = session.local_model_tags()
    budget = cfg.orchestrator_read_budget
    guards = (getattr(session.bundle, "guards", []) or []) if session.bundle is not None else []
    note = _read_budget_note(session)
    reference = session.tracker.cloud_reference()
    approx = "~" if u.get("cost_estimated") else ""

    console.print()
    render.rule(console, "setup")
    console.print(
        render.kv(
            [
                ("version", Text(f"loom v{__version__}", style="loom.text")),
                ("cwd", Text(str(session.cwd), style="loom.text")),
                ("orchestrator", _badge("orchestrator", cfg.orchestrator)),
                ("advisor", _badge("advisor", cfg.advisor)),
                (
                    "fleet",
                    Text(f"{g.local} " + ", ".join(local_tags), style="loom.local")
                    if local_tags
                    else Text("no local models — every role is billed", style="loom.warn"),
                ),
                ("mode", render.join(console, modes or ["default"], style="loom.text")),
                ("permissions", Text(session.settings.permissions.default_mode, style="loom.text")),
                ("mcp", Text(mcp_line, style="loom.text")),
                ("memory", Text(str(session.memory_path() or "none — /init writes one"), style="loom.text")),
                (
                    "sessions",
                    Text(
                        "sqlite (.loom/sessions.db)" if session.durable else "in-memory — /resume won't survive a restart",
                        style="loom.text" if session.durable else "loom.warn",
                    ),
                ),
            ]
        )
    )

    console.print()
    render.rule(console, "this session")
    console.print(
        render.kv(
            [
                (
                    "spent",
                    render.join(
                        console,
                        [
                            Text(f"{u['turns']} turns", style="loom.text"),
                            Text(f"{u['input_tokens']:,} in / {u['output_tokens']:,} out", style="loom.muted"),
                            Text(f"{approx}${u['cloud_cost']:.3f}", style="loom.cloud"),
                        ],
                    ),
                ),
                (
                    "saved",
                    render.join(
                        console,
                        [
                            Text(f"{su.local_share():.0%} of tokens ran free", style="loom.local"),
                            Text(f"~${su.savings(reference):.2f} vs all-cloud on {reference}", style="loom.muted"),
                        ],
                    ),
                ),
                (
                    "delegation",
                    render.join(
                        console,
                        [
                            Text(f"orchestrator held {su.orchestrator_share():.0%}", style="loom.text"),
                            Text(f"{su.delegations()} delegated role(s)", style="loom.muted"),
                            Text(f"read budget {'off' if budget < 0 else f'{budget}/turn'}", style="loom.muted"),
                            *([Text(note, style="loom.warn")] if note else []),
                        ],
                    ),
                ),
                (
                    "escalations",
                    render.join(
                        console,
                        [
                            Text(
                                f"{sum(getattr(x, 'local_escalation_count', 0) for x in guards)} local{g.arrow}local",
                                style="loom.local",
                            ),
                            Text(
                                f"{sum(getattr(x, 'escalation_count', 0) for x in guards)} local{g.arrow}cloud",
                                style="loom.cloud",
                            ),
                        ],
                    ),
                ),
            ]
        )
    )
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

        session.console.print(
            "[loom.warn]graphify isn't installed[/loom.warn] [loom.muted]— free, MIT, runs fully "
            "locally (tree-sitter); powers graph-RAG structure queries[/loom.muted]"
        )
        try:
            if not render.confirm(session.console, f"  install it now via `uv tool install {graphify.PYPI_NAME}`?"):
                session.console.print(f"[loom.muted]{graphify.INSTALL_HINT}[/loom.muted]")
                return False
        except (EOFError, KeyboardInterrupt):
            return False
        ok, how = graphify.install()
        if ok:
            session.console.print(f"[loom.good]✓ graphify installed[/loom.good] [loom.muted]({how})[/loom.muted]")
        else:
            session.console.print(f"[loom.bad.b]install failed ({how})[/loom.bad.b] [loom.muted]{graphify.INSTALL_HINT}[/loom.muted]")
        return ok

    def _build(update: bool) -> None:
        session.console.print(f"[loom.warp.b]⏺ graphify {'--update' if update else ''} — indexing {session.cwd}[/loom.warp.b]")
        code = graphify.build(session.cwd, update=update)
        if code != 0:
            session.console.print(f"[loom.bad.b]graphify exited with code {code}[/loom.bad.b]")
            return
        detail = graphify.format_stats(graphify.graph_stats(session.cwd))
        session.console.print(f"[loom.good]✓ graph ready[/loom.good] [loom.muted]({detail or 'graphify-out/graph.json'})[/loom.muted]")
        srv = session.settings.mcp_servers.get("graphify")
        if srv is None or not srv.enabled:
            _set_server(True)
            session.console.print("[loom.muted]graphify MCP server enabled — graph tools connect on the next task[/loom.muted]")

    if verb in ("build", "update"):
        if _ensure_installed():
            _build(update=verb == "update")
        return True

    if verb in ("on", "off"):
        if verb == "on" and not graphify.graph_exists(session.cwd):
            session.console.print("[loom.warn]no graph yet — run /graphify build first[/loom.warn]")
            return True
        _set_server(verb == "on")
        session.console.print(f"graphify MCP server: [loom.warp.b]{verb}[/loom.warp.b]")
        return True

    if verb in ("query", "path", "explain"):
        if not _ensure_installed():
            return True
        if not graphify.graph_exists(session.cwd):
            session.console.print("[loom.warn]no graph yet — run /graphify build first[/loom.warn]")
            return True
        cli_args = [verb, *([rest] if verb != "path" else rest.split(maxsplit=1))]
        code, out = graphify.run_cli(session.cwd, *[a for a in cli_args if a])
        style = "loom.bad.b" if code != 0 else None
        session.console.print(out or "(no output)", style=style)
        return True

    # No/unknown verb: status.
    stats = graphify.graph_stats(session.cwd)
    server = session.settings.mcp_servers.get("graphify")
    state = next((r["state"] for r in mcp_mod.mcp_status(session.settings) if r["name"] == "graphify"), "not configured")
    console = session.console
    console.print()
    render.rule(console, "graphify — code knowledge graph")
    console.print(
        render.kv(
            [
                (
                    "cli",
                    Text("installed", style="loom.good")
                    if graphify.installed()
                    else Text(f"not installed — {graphify.INSTALL_HINT}", style="loom.warn"),
                ),
                (
                    "graph",
                    Text(graphify.format_stats(stats), style="loom.text")
                    if graphify.format_stats(stats)
                    else Text("not built — /graphify build", style="loom.warn"),
                ),
                (
                    "server",
                    Text(state, style="loom.text")
                    if server is not None and server.enabled
                    else Text("disabled — /graphify on", style="loom.muted"),
                ),
            ]
        )
    )
    console.print()
    render.note(
        console,
        "the orchestrator, explorer and searcher answer structure questions from the graph instead "
        "of glob/grep/read sweeps — fewer tokens, real file:line citations",
    )
    # First run: walk through install + build right here instead of making the
    # user retype the verbs — Loom is a coding assistant, it sets itself up.
    if stats is None:

        if not _ensure_installed():
            return True
        try:
            if render.confirm(session.console, "  build the knowledge graph for this repo now?"):
                _build(update=False)
        except (EOFError, KeyboardInterrupt):
            pass
    return True


@command("skills", "List agent skills (SKILL.md folders: packaged, ~/.loom/skills, .loom/skills)")
def _skills(session: "Session", args: str) -> bool:
    from loom.core import skills as skills_mod

    console = session.console
    found = skills_mod.list_skills(session.cwd)
    if not found:
        render.note(
            console,
            "no skills yet — add one at .loom/skills/<name>/SKILL.md (this project) "
            "or ~/.loom/skills/<name>/SKILL.md (every project)",
        )
        return True
    console.print()
    render.rule(console, "skills")
    table = render.table(
        console,
        ("skill", {"no_wrap": True}),
        ("source", {"no_wrap": True}),
        ("what it's for", {"overflow": "fold"}),
    )
    for s in found:
        table.add_row(
            Text(s["name"], style="loom.warp"),
            Text(s["source"], style="loom.muted"),
            Text(s["description"][:100], style="loom.text"),
        )
    console.print(table)
    console.print()
    render.note(
        console,
        "progressive disclosure: only name and description enter the prompt; "
        "the agent reads the full SKILL.md when a task matches",
    )
    return True


@command("mcp", "List MCP servers, connection state, and their tools")
def _mcp(session: "Session", args: str) -> bool:
    from loom.core.mcp import mcp_status

    console = session.console
    g = ink(console)
    rows = mcp_status(session.settings)
    if not rows:
        render.note(console, "no MCP servers configured (settings.json → mcp_servers)")
        return True
    console.print()
    render.rule(console, "mcp servers")
    table = render.table(
        console,
        ("server", {"no_wrap": True}),
        ("transport", {"no_wrap": True}),
        ("target", {"overflow": "ellipsis"}),
        ("state", {"no_wrap": True}),
        ("tools", {"overflow": "ellipsis"}),
    )
    for r in rows:
        connected = r["state"] == "connected"
        tools = (
            f"{len(r['tools'])}: {', '.join(r['tools'][:5])}{g.ellipsis if len(r['tools']) > 5 else ''}"
            if r["tools"]
            else g.dot
        )
        table.add_row(
            Text(r["name"], style="loom.text"),
            Text(r["transport"], style="loom.muted"),
            Text(r["target"], style="loom.muted"),
            Text(r["state"], style="loom.good" if connected else "loom.muted"),
            Text(tools, style="loom.muted"),
        )
    console.print(table)
    console.print()
    render.note(console, "servers connect on the first task; browser_* tools power the tester subagent")
    return True


@command("cost", "Show the session cost receipt, broken down by who spent it")
def _cost(session: "Session", args: str) -> bool:
    console = session.console
    g = ink(console)
    t = session.tracker
    u = t.session

    console.print()
    render.rule(console, f"cost {g.dot} {t.turns} turns")
    # Rows are per *actor*, not per model: the point of a delegating
    # architecture is knowing whether the orchestrator or the fleet spent the
    # tokens, and two roles can share one model.
    table = render.table(
        console,
        ("role", {"no_wrap": True}),
        ("model", {"overflow": "ellipsis"}),
        ("calls", {"justify": "right", "no_wrap": True}),
        ("in", {"justify": "right", "no_wrap": True}),
        ("cached", {"justify": "right", "no_wrap": True}),
        ("out", {"justify": "right", "no_wrap": True}),
        ("cost", {"justify": "right", "no_wrap": True}),
    )
    for actor, mu in u.rows():
        cost = u.cost_of(actor, mu)
        if actor.is_local:
            # Stated as free, in the local colour. This number is the argument
            # for the whole architecture; burying it as "$0.000" wastes it.
            price = Text("free", style="loom.local")
        else:
            # "~" marks a model Loom has no published price for, charged at the
            # Sonnet-tier default. Printing it bare read as a real bill.
            price = Text(f"{'~' if u.is_estimated(actor) else ''}${cost:.3f}", style="loom.cloud")
        table.add_row(
            Text(actor.role, style="loom.text"),
            render.model_badge(console, actor.model, actor.is_local),
            Text(f"{mu.calls:,}", style="loom.muted"),
            Text(f"{mu.input_tokens:,}", style="loom.muted"),
            Text(f"{mu.cache_read_tokens:,}" if mu.cache_read_tokens else g.dot, style="loom.muted"),
            Text(f"{mu.output_tokens:,}", style="loom.muted"),
            price,
        )
    if not table.row_count:
        render.note(console, "nothing spent yet — no model has been called this session")
        return True

    console.print(table)
    console.print()
    if u.has_estimates():
        render.note(console, "~ no published price for that model; charged here at Sonnet-tier rates")
    console.print(
        render.kv(
            [
                (
                    "delegation",
                    render.join(
                        console,
                        [
                            Text(f"orchestrator held {u.orchestrator_share():.0%}", style="loom.text"),
                            Text(f"${u.orchestrator_cost():.3f}", style="loom.cloud"),
                            Text(f"{u.delegations()} role(s) worked in their own context", style="loom.muted"),
                        ],
                    ),
                ),
                ("read budget", Text(_read_budget_note(session) or "never hit", style="loom.muted")),
            ]
        )
    )
    receipt = t.receipt(turn=False)
    if receipt:
        render.note(console, f"[loom.muted]{receipt}[/loom.muted]", kind="tip")
    return True


def _read_budget_note(session: "Session") -> str:
    """What the read budget actually did, as a bare fragment both callers frame
    themselves — /status already labels the row "read budget", and repeating the
    phrase there read like a stutter."""
    guard = getattr(session.bundle, "delegation_guard", None) if session.bundle else None
    blocked = getattr(guard, "blocked_count", 0) or 0
    refused = getattr(guard, "refused_count", 0) or 0
    if not blocked and not refused:
        return ""
    parts = [f"withheld on {blocked} model call(s)"] if blocked else []
    if refused:
        # Worth naming separately: the model reached for a tool it could no
        # longer see, which is why the budget is enforced at the tool too.
        parts.append(f"{refused} over-budget read(s) refused")
    return ", ".join(parts)


@command("resume", "List past sessions, or resume one: /resume [n | thread-id]")
def _resume(session: "Session", args: str) -> bool:
    from loom.core import sessions as sessions_mod

    console = session.console
    rows = sessions_mod.load_index(session.cwd)
    if not args.strip():
        if not rows:
            render.note(console, "no past sessions in this project")
            return True
        console.print()
        render.rule(console, "sessions")
        table = render.table(
            console,
            ("#", {"justify": "right", "no_wrap": True}),
            ("updated", {"no_wrap": True}),
            ("turns", {"justify": "right", "no_wrap": True}),
            ("title", {"overflow": "ellipsis"}),
            ("thread", {"no_wrap": True}),
        )
        for i, row in enumerate(reversed(rows), 1):
            table.add_row(
                Text(str(i), style="loom.warp"),
                Text(row["updated"], style="loom.muted"),
                Text(str(row.get("turns", "?")), style="loom.muted"),
                Text(row["title"], style="loom.text"),
                Text(row["thread_id"], style="loom.line"),
            )
        console.print(table)
        console.print()
        if not session.durable:
            render.note(
                console,
                "sessions.db unavailable (install langgraph-checkpoint-sqlite) — history won't survive restarts",
                kind="warn",
            )
        render.note(console, "[loom.warp]/resume <#>[/loom.warp] or [loom.warp]/resume <thread-id>[/loom.warp]", kind="tip")
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
        render.note(session.console, f"no such session: {choice}", kind="bad")
        return True
    session.thread_id = target
    session._memory_sent = True  # resumed thread already has its context
    render.note(
        session.console,
        f"resumed [loom.warp]{target}[/loom.warp] — continue where you left off",
        kind="good",
    )
    return True


@command("undo", "Roll back the file changes of the last turn")
def _undo(session: "Session", args: str) -> bool:
    from loom.core import undo as undo_mod

    console = session.console
    restored = undo_mod.undo_last(session.cwd)
    if not restored:
        render.note(console, "nothing to undo (no snapshotted file writes)")
        return True
    g = ink(console)
    for rel in restored:
        console.print(Text(f"  {g.result} ", style="loom.line") + Text(str(rel), style="loom.muted"))
    render.note(console, f"rolled back {len(restored)} file(s) from the last turn", kind="good")
    return True


@command("airgap", "Toggle airgap mode — raw code never reaches the cloud")
def _airgap(session: "Session", args: str) -> bool:
    session.airgap = not session.airgap
    session.rebuild()
    if session.airgap:
        session.console.print(
            "airgap: [loom.warp.b]on[/loom.warp.b] — cloud orchestrator plans from summaries only; "
            "local subagents do all file reading; cloud escalation disabled"
        )
    else:
        session.console.print("airgap: [loom.warp.b]off[/loom.warp.b]")
    return True


@command("compact", "Summarize the conversation and free up context")
def _compact(session: "Session", args: str) -> bool:
    transcript = session.transcript()
    if not transcript:
        session.console.print("[loom.muted]nothing to compact yet[/loom.muted]")
        return True
    lines = []
    for m in transcript:
        role = m[0] if isinstance(m, tuple) else getattr(m, "type", "?")
        content = m[1] if isinstance(m, tuple) else getattr(m, "content", "")
        if content and role in ("user", "human", "assistant", "ai"):
            lines.append(f"{role}: {str(content)[:2000]}")
    if not lines:
        session.console.print("[loom.muted]nothing to compact yet[/loom.muted]")
        return True

    from loom.core.model_router import build_model
    from loom.core.usage import role_metadata

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
        # Billed to "compaction", not the orchestrator: it is housekeeping on the
        # transcript, and folding it into the orchestrator's share would make the
        # delegation ratio look worse every time the user compacts.
        summary = str(
            model.invoke(
                prompt, config=role_metadata({"callbacks": [session.tracker]}, "compaction")
            ).content
        )
    except Exception as exc:
        session.console.print(f"[loom.bad.b]compact failed:[/loom.bad.b] {exc}")
        return True
    session.reset()
    session.pending_context = summary
    session.console.print("[loom.muted]✻ context compacted — summary will be carried into your next message[/loom.muted]")
    return True


@command("doctor", "Check your setup; `/doctor probe` also calls every configured model")
def _doctor(session: "Session", args: str) -> bool:
    import os
    import sys

    if args.strip() in ("probe", "models", "verify"):
        return _probe(session)

    from loom.core import ollama
    from loom.core.mcp import mcp_status

    console = session.console
    g = ink(console)

    def row(ok: bool | None, label: str, detail: str):
        mark, style = (g.ok, "loom.good") if ok else ((g.pending, "loom.warn") if ok is None else (g.fail, "loom.bad"))
        return (
            Text(f"{mark} ", style=style) + Text(label, style="loom.text"),
            Text(detail, style="loom.muted" if ok else style),
        )

    out = [row(sys.version_info >= (3, 11), "python", sys.version.split()[0])]

    cfg = session.settings.models
    st = ollama.status(cfg)
    if st.running:
        detail = f"running @ {st.endpoint}" + ("" if st.installed else " (remote — no local binary)")
        out.append(row(True, "ollama", detail))
        missing = ollama.missing_models(cfg)
        out.append(row(not missing, "local models", ", ".join(missing) + " missing" if missing else "all present"))
        if missing:
            # A missing tag only costs money if nothing local can cover it.
            from loom.core.local_pool import build_pool, plan_local_roles

            plan = plan_local_roles(cfg, build_pool(cfg, st))
            if plan.cloud:
                out.append(row(None, "cloud fallback", f"{', '.join(sorted(plan.cloud))} run on {cfg.cloud_fallback} (billed)"))
            if plan.substituted:
                out.append(row(None, "local fallback", f"{', '.join(sorted(plan.substituted))} run on another local model (free)"))
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

    from loom.core import playwright_setup

    pw = playwright_setup.status()
    out.append(row(pw.npx_available, "npx", "found" if pw.npx_available else "not found (Playwright MCP needs Node)"))
    if pw.npx_available:
        out.append(
            row(
                pw.browsers_installed,
                "playwright browsers",
                "installed" if pw.browsers_installed else "missing — run `loom playwright install` or `/playwright install`",
            )
        )
    for r in mcp_status(session.settings):
        ok: bool | None = True if r["state"] == "connected" else (None if r["state"] in ("not connected", "disabled") else False)
        out.append(row(ok, f"mcp:{r['name']}", r["state"]))

    from loom.ui import privacy as privacy_mod

    out.append(row(*privacy_mod.doctor_row(session.cwd)))

    console.print()
    render.rule(console, "doctor")
    console.print(render.kv(out, justify="left"))
    render.note(console, "[loom.warp]/doctor probe[/loom.warp] also calls every configured model", kind="tip")
    return True


def _probe(session: "Session") -> bool:
    """Call every configured model once and report what answered.

    The offline checks above prove a key is *present*; only a real call proves
    it is *accepted for that model*. Those differ more often than you'd
    think — wrong region, not opted in, not on your plan, retired, misspelt.
    """
    from loom.core import preflight

    console = session.console
    g = ink(console)
    cfg_models = session.settings.models
    session.settings.apply_env()
    plan = {
        "orchestrator": cfg_models.orchestrator,
        "advisor": cfg_models.advisor,
        "escalation": cfg_models.escalation_model,
        **cfg_models.subagents,
    }
    console.print()
    render.rule(console, "probe")
    render.note(console, "one tiny prompt per distinct model — a few tokens each")
    table = render.table(
        console,
        ("", {"no_wrap": True}),
        ("roles", {"overflow": "ellipsis"}),
        ("", {"justify": "center", "no_wrap": True}),
        ("model", {"no_wrap": True, "overflow": "ellipsis"}),
        ("", {"overflow": "fold"}),
    )
    failed = []
    with render.Working(console, "probing models"):
        results = list(preflight.check_plan(plan, cfg_models))
    for roles, check in results:
        if not check.ok:
            failed.append(check)
        mark, style = (g.ok, "loom.good") if check.ok else (
            (g.pending, "loom.warn") if check.state in ("not-pulled", "offline") else (g.fail, "loom.bad")
        )
        is_local = cfg_models.is_local(check.model)
        table.add_row(
            Text(mark, style=style),
            Text(", ".join(sorted(roles)), style="loom.text"),
            render.where(console, is_local),
            Text(check.model, style="loom.local" if is_local else "loom.cloud"),
            Text(check.detail, style="loom.muted" if check.ok else style),
        )
    console.print(table)
    console.print()
    for check in failed:
        if check.hint:
            render.note(console, f"[loom.text]{check.model}[/loom.text] — {check.hint}", kind="tip")
    if not failed:
        render.note(console, "every configured model answered", kind="good")
    else:
        render.note(
            console,
            f"{len(failed)} model(s) won't answer — [loom.warp]/model <role>[/loom.warp] to change one, "
            "or [loom.warp]/setup[/loom.warp] to redo it",
            kind="warn",
        )
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

    console = session.console
    path = session.memory_path()
    if path is None:
        render.note(console, "no memory file (LOOM.md / CLAUDE.md / AGENTS.md) — create one with /init")
        return True
    console.print()
    console.print(
        render.card(
            console,
            Markdown(path.read_text(encoding="utf-8")),
            title=path.name,
            subtitle="sent with your first message each session",
            expand=True,
        )
    )
    render.note(console, f"edit: $EDITOR {path.name}")
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
    session.console.print(f"exported → [loom.warp.b]{target}[/loom.warp.b]")
    return True


@command("hooks", "Show the configured tool hooks")
def _hooks(session: "Session", args: str) -> bool:
    console = session.console
    h = session.settings.hooks
    table = render.table(
        console,
        ("event", {"no_wrap": True}),
        ("matcher", {"no_wrap": True}),
        ("command", {"overflow": "fold"}),
    )
    for event in ("pre_tool_use", "post_tool_use", "user_prompt_submit", "stop"):
        for hook in getattr(h, event):
            table.add_row(
                Text(event, style="loom.warp"),
                Text(hook.matcher, style="loom.text"),
                Text(hook.command, style="loom.muted"),
            )
    if not table.row_count:
        render.note(console, "no hooks configured (settings.json → hooks)")
        return True
    console.print()
    render.rule(console, "hooks")
    console.print(table)
    return True


@command("theme", "Show or set the UI theme: /theme loom|loom-light|phosphor|mono")
def _theme(session: "Session", args: str) -> bool:
    from loom.ui.theme import PALETTES, THEME_NAMES, active_theme_name

    console = session.console
    choice = args.strip()
    if not choice:
        active = active_theme_name(session.settings.ui)
        console.print()
        render.rule(console, "themes")
        rows = []
        for name in THEME_NAMES:
            palette = PALETTES.get(name)
            blurb = "no colour — pipes, CI logs, e-ink" if palette is None else (
                "the default: warm local, cool cloud"
                if name == "loom"
                else "for light terminals"
                if name == "loom-light"
                else "a CRT in a basement"
            )
            mark = f"{ink(console).ok} " if name == active else "  "
            rows.append(
                (
                    Text(mark, style="loom.warp") + Text(name, style="loom.bright" if name == active else "loom.text"),
                    Text(blurb, style="loom.muted"),
                )
            )
        console.print(render.kv(rows, justify="left"))
        console.print()
        render.note(console, f"set: [loom.warp]/theme <name>[/loom.warp] {ink(console).dot} configured as `{session.settings.ui.theme}`", kind="tip")
        return True

    from loom.core import settings as st

    try:
        st.set_value("ui.theme", choice)
    except Exception as exc:
        render.note(console, str(exc), kind="bad")
        return True
    session.reload_settings()
    render.note(session.console, f"theme → [loom.warp]{choice}[/loom.warp]", kind="good")
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
    session.console.print(f"vim mode: [loom.warp.b]{'on' if session.vim else 'off'}[/loom.warp.b]")
    return True
