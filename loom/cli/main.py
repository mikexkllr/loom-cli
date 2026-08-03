"""Loom CLI — Typer app with Rich streaming output (build step 5).

    loom "refactor the auth module to use JWT"
    loom --plan "add pagination to all list endpoints"
    loom --local-only "explain this function"
    loom --advisor-threshold high "redesign the DB schema"
    loom config set orchestrator gpt-4o
    loom config show
    loom agents list
    loom models status | list | pull

Heavy deps (deepagents/langchain) are imported lazily inside the run path so
inspection commands (config / agents / models) work without the full stack.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.text import Text

from loom.core import config as cfg
from loom.core import settings as settings_mod
from loom.core.settings import UISettings
from loom.tools import sandbox
from loom.ui import render
from loom.ui.render import ink
from loom.ui.theme import make_console

# Themed from the start: subcommands print `loom.*` markup long before any
# settings are loaded, and an unthemed Console can't resolve those names.
console = make_console(UISettings())


def _retheme(root: str = ".") -> None:
    """Adopt the project's theme once we know which project we're in."""
    global console
    try:
        console = make_console(settings_mod.load_settings(root).ui)
    except Exception:
        pass  # a broken settings.json is the loader's problem to report, not ours


class _PromptOrCommandGroup(typer.core.TyperGroup):
    """Let the root command take either a free-form task or a subcommand.

    Click parses the group's ``[PROMPT]`` argument before resolving
    subcommands, so ``loom models status`` would otherwise become the task
    "models". If the first non-option token names a known subcommand, insert
    an empty prompt placeholder so the subcommand resolves normally.

    Otherwise we are in prompt form, and options are allowed to follow the task
    text. Click switches that off for groups so a subcommand's own flags survive
    to be parsed by the subcommand (``loom models pull --all``) — but in prompt
    form there is no subcommand to shield, and leaving it off made the natural
    ``loom "fix the tests" --yolo`` fail with "No such command '--yolo'".
    """

    def parse_args(self, ctx, args):
        first = next((a for a in args if not a.startswith("-")), None)
        if first is not None and first in self.commands:
            idx = args.index(first)
            args = [*args[:idx], "", *args[idx:]]
        else:
            ctx.allow_interspersed_args = True
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=_PromptOrCommandGroup,
    add_completion=False,
    help="Loom — hybrid local/cloud multi-agent CLI coding assistant.",
    no_args_is_help=False,  # no args -> launch the interactive REPL
)
config_app = typer.Typer(help="View and edit model-routing configuration.")
settings_app = typer.Typer(help="View and edit settings.json (permissions/hooks/env/ui).")
agents_app = typer.Typer(help="Inspect registered subagents.")
models_app = typer.Typer(help="Manage local Ollama models.")
playwright_app = typer.Typer(help="Set up the Playwright MCP browser (used by the tester subagent).")
app.add_typer(config_app, name="config")
app.add_typer(settings_app, name="settings")
app.add_typer(agents_app, name="agents")
app.add_typer(models_app, name="models")
app.add_typer(playwright_app, name="playwright")


# ----------------------------------------------------------------------------
# Main task entry
# ----------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Argument(None, help="The task to run. Omit to open the interactive REPL."),
    plan: bool = typer.Option(False, "--plan", help="Plan-first, read-only exploration before any writes."),
    local_only: bool = typer.Option(False, "--local-only", help="No cloud calls — local models only."),
    airgap: bool = typer.Option(False, "--airgap", help="Raw code never reaches the cloud: local subagents read files, the cloud orchestrator sees only summaries."),
    yolo: bool = typer.Option(False, "--yolo", help="Auto-approve tools that would otherwise ask."),
    accept_edits: bool = typer.Option(False, "--accept-edits", help="Auto-approve file edits only; shell still asks."),
    loop: int = typer.Option(0, "--loop", help="Loop mode: iterate on the task up to N times until done."),
    until: Optional[str] = typer.Option(None, "--until", help="Loop stop condition: shell command that must exit 0."),
    advisor_threshold: Optional[str] = typer.Option(
        None, "--advisor-threshold", help="When to auto-consult the advisor: low | medium | high."
    ),
    root: str = typer.Option(".", "--root", help="Project root the agents are sandboxed to."),
) -> None:
    """Run a task, or (with no task and no subcommand) open the interactive UI."""
    if ctx.invoked_subcommand is not None:
        return

    _maybe_offer_update()

    sandbox.set_root(root)
    settings = settings_mod.load_settings(root)

    if not prompt:
        from loom.ui import repl

        repl.run(settings, cwd=root, plan=plan, local_only=local_only, yolo=yolo, airgap=airgap)
        raise typer.Exit()

    _run_task(
        settings, prompt, plan=plan, local_only=local_only, airgap=airgap, yolo=yolo,
        accept_edits=accept_edits, loop=loop, until=until,
        advisor_threshold=advisor_threshold, root=root,
    )


@app.command("chat")
def chat(
    plan: bool = typer.Option(False, "--plan"),
    local_only: bool = typer.Option(False, "--local-only"),
    airgap: bool = typer.Option(False, "--airgap"),
    yolo: bool = typer.Option(False, "--yolo"),
    root: str = typer.Option(".", "--root"),
) -> None:
    """Open the interactive Loom REPL (same as running `loom` with no task)."""
    sandbox.set_root(root)
    from loom.ui import repl

    repl.run(settings_mod.load_settings(root), cwd=root, plan=plan, local_only=local_only, yolo=yolo, airgap=airgap)


def _maybe_offer_update() -> None:
    """Startup-only, throttled update check for binary installs (see
    loom/core/update.py). Only reached for the REPL / one-shot task paths —
    subcommands like `doctor` or `config show` return before this runs, so
    they stay fast and offline-safe. A broken network never blocks startup:
    check_for_startup() swallows its own errors and returns None."""
    import sys as _sys

    from loom.core import update as update_mod

    result = update_mod.check_for_startup()
    if result is None:
        return

    if not _sys.stdin.isatty():
        console.print(f"[loom.warn]update available[/loom.warn] ({result.asset}) — run [loom.bright]loom update[/loom.bright]")
        return

    console.print(f"[loom.warn]a newer loom build is available[/loom.warn] for {result.asset}.")

    try:
        want_update = render.confirm(console, "Update now before continuing?", default=False)
    except (KeyboardInterrupt, EOFError):
        console.print()
        want_update = False

    if not want_update:
        console.print("[loom.muted]continuing with the current version — run `loom update` anytime[/loom.muted]")
        return

    try:
        update_mod.apply_and_relaunch(result, console=console, argv=_sys.argv[1:])
    except SystemExit:
        raise
    except Exception as exc:
        console.print(f"[loom.bad.b]update failed:[/loom.bad.b] {exc} — continuing with the current version")


def _run_task(
    settings,
    prompt: str,
    *,
    plan: bool,
    local_only: bool,
    airgap: bool = False,
    yolo: bool,
    accept_edits: bool = False,
    loop: int = 0,
    until: Optional[str] = None,
    advisor_threshold,
    root: str,
) -> None:
    """Headless task run — same Session engine (rendering, receipts, loop) as
    the REPL, minus the input loop."""
    from loom.ui.repl import Session

    settings.apply_env()
    if advisor_threshold is not None:
        settings.models = settings.models.model_copy(update={"advisor_threshold": advisor_threshold})

    session = Session(settings, cwd=root, plan=plan, local_only=local_only, yolo=yolo, airgap=airgap)
    session.accept_edits = accept_edits

    try:
        bundle = session.ensure_bundle()
    except ModuleNotFoundError as exc:
        render.note(session.console, f"missing dependency: {exc} — install with `uv sync`", kind="bad")
        raise typer.Exit(1)
    except RuntimeError as exc:  # e.g. local-only without Ollama
        render.note(session.console, str(exc), kind="bad")
        raise typer.Exit(1)

    out = session.console
    if settings.ui.show_fleet_panel:
        out.print()
        render.rule(out, f"fleet {ink(out).dot} {bundle.mode}")
        out.print(render.fleet_table(out, _fleet_rows(settings.models, bundle), header=False))
    out.print()
    render.rule(out, "task")
    out.print(render.kv([("", Text(prompt, style="loom.bright"))]))
    out.print()

    if loop > 0:
        session.run_loop(prompt, max_iters=loop, until=until)
    else:
        session.run_turn(prompt)


def _fleet_rows(config: cfg.LoomConfig, bundle):
    """(role, model, is_local, note) for the roster a headless run prints
    before it starts — the same shape the REPL's welcome card uses."""
    rows = [("orchestrator", bundle.model_string, config.is_local(bundle.model_string), "")]
    for name in bundle.subagent_names:
        model = config.subagents.get(name, "(inherit)")
        rows.append((name, model, config.is_local(model), ""))
    if bundle.mode != "local-only":
        rows.append(("advisor", config.advisor, config.is_local(config.advisor), "on demand"))
    return rows


# ----------------------------------------------------------------------------
# config subcommands
# ----------------------------------------------------------------------------


@config_app.command("show")
def config_show(root: str = typer.Option(".", "--root")) -> None:
    """Print the effective model routing (config.yaml defaults + settings.json overrides)."""
    import yaml
    from rich.syntax import Syntax

    _retheme(root)
    models = settings_mod.load_settings(root).models
    console.print()
    console.print(
        render.card(
            console,
            Syntax(
                yaml.safe_dump(models.model_dump(), sort_keys=False),
                "yaml",
                theme="ansi_dark",
                background_color="default",
            ),
            title="model routing",
            subtitle="config.yaml + settings.json, merged",
        )
    )


@config_app.command("set")
def config_set(key: str, value: str, root: str = typer.Option(".", "--root")) -> None:
    """Set a model-routing value, e.g. `loom config set orchestrator claude-opus-4-8`.

    Writes to ``~/.loom/settings.json`` — the layer that overrides
    ``config.yaml`` — so the change takes effect and isn't shadowed by a
    ``models`` block a prior ``/setup`` wrote there. Same destination as
    ``/model`` and ``loom settings set models.<key>``.
    """
    try:
        settings_mod.set_value(f"models.{key}", value, root)
    except Exception as exc:
        console.print(f"[loom.bad.b]invalid:[/loom.bad.b] {exc}")
        raise typer.Exit(1)
    console.print(f"[loom.good]set[/loom.good] {key} = {value} [loom.muted]in {settings_mod.USER_SETTINGS_PATH}[/loom.muted]")


@config_app.command("path")
def config_path() -> None:
    """Print the config files: config.yaml (defaults) and settings.json (overrides)."""
    cfg.ensure_user_config()
    console.print(f"defaults:  {cfg.USER_CONFIG_PATH}")
    console.print(f"overrides: {settings_mod.USER_SETTINGS_PATH}")


# ----------------------------------------------------------------------------
# settings subcommands (permissions / hooks / env / ui)
# ----------------------------------------------------------------------------


@settings_app.command("show")
def settings_show(
    section: Optional[str] = typer.Argument(None, help="Only show one section: permissions|hooks|env|ui"),
    root: str = typer.Option(".", "--root"),
) -> None:
    """Print the merged settings (all layers), or one section."""
    import json

    from rich.syntax import Syntax

    _retheme(root)
    settings = settings_mod.load_settings(root)
    data = settings.model_dump(exclude={"models"})
    if section:
        if section not in data:
            render.note(console, f"unknown section: {section} (try permissions|hooks|env|ui)", kind="bad")
            raise typer.Exit(1)
        data = {section: data[section]}
    console.print()
    console.print(
        render.card(
            console,
            Syntax(json.dumps(data, indent=2), "json", theme="ansi_dark", background_color="default"),
            title=section or "settings.json",
            subtitle="all layers, merged",
        )
    )


@settings_app.command("set")
def settings_set(key: str, value: str, root: str = typer.Option(".", "--root")) -> None:
    """Set a settings value, e.g. `loom settings set ui.theme light`
    or `loom settings set permissions.default_mode allow`."""
    try:
        settings_mod.set_value(key, value, root)
    except Exception as exc:
        console.print(f"[loom.bad.b]invalid:[/loom.bad.b] {exc}")
        raise typer.Exit(1)
    console.print(f"[loom.good]set[/loom.good] {key} = {value}")


@settings_app.command("path")
def settings_path() -> None:
    """Print the path to the user settings.json file."""
    console.print(str(settings_mod.USER_SETTINGS_PATH))


@settings_app.command("init")
def settings_init(root: str = typer.Option(".", "--root")) -> None:
    """Write a starter .loom/settings.json into the current project."""
    import json

    target = settings_mod.project_settings_paths(root)[0]
    if target.exists():
        console.print(f"[loom.warn]exists:[/loom.warn] {target}")
        raise typer.Exit()
    target.parent.mkdir(parents=True, exist_ok=True)
    starter = settings_mod._read_json(settings_mod.DEFAULT_SETTINGS_PATH)
    target.write_text(json.dumps(starter, indent=2), encoding="utf-8")
    console.print(f"[loom.good]created[/loom.good] {target}")


# ----------------------------------------------------------------------------
# agents subcommands
# ----------------------------------------------------------------------------


@agents_app.command("list")
def agents_list() -> None:
    """Show registered subagents and their assigned models."""
    from loom.subagents import describe_subagents

    config = cfg.load_config()
    console.print()
    render.rule(console, "fleet")
    table = render.table(
        console,
        ("agent", {"no_wrap": True}),
        ("", {"justify": "center", "no_wrap": True}),
        ("model", {"overflow": "ellipsis"}),
        ("can", {"no_wrap": True}),
        ("tools", {"overflow": "fold"}),
    )
    for row in describe_subagents(config):
        is_local = row["scope"] == "local"
        table.add_row(
            Text(row["name"], style="loom.text"),
            render.where(console, is_local),
            Text(row["model"], style="loom.local" if is_local else "loom.cloud"),
            Text(row["mode"], style="loom.muted" if row["mode"] == "read-only" else "loom.warn"),
            Text(row["tools"], style="loom.muted"),
        )
    console.print(table)


# ----------------------------------------------------------------------------
# models subcommands
# ----------------------------------------------------------------------------


@models_app.command("status")
def models_status() -> None:
    """Check the Ollama daemon and which required models are installed."""
    from loom.core import ollama

    config = cfg.load_config()
    st = ollama.status(config)
    # What matters is a reachable daemon at the configured endpoint — the
    # binary is optional (the endpoint may be a remote host).
    if not st.running:
        render.note(console, ollama.daemon_hint(st.endpoint) if st.installed else ollama.INSTALL_HINT, kind="bad")
        raise typer.Exit(1)
    binary = "installed" if st.installed else "no local binary (remote daemon is fine)"
    render.note(console, f"ollama running at {st.endpoint} {ink(console).dot} {binary}", kind="good")
    missing = ollama.missing_models(config)
    if missing:
        render.note(console, f"missing: {', '.join(missing)} — run `loom models pull`", kind="warn")
    else:
        render.note(console, "all required local models are installed", kind="good")


@models_app.command("list")
def models_list() -> None:
    """List required local models and whether each is installed."""
    from loom.core import ollama

    config = cfg.load_config()
    st = ollama.status(config)
    have = set(st.models)
    g = ink(console)
    console.print()
    render.rule(console, "local models")
    table = render.table(
        console,
        ("", {"no_wrap": True}),
        ("model", {"overflow": "ellipsis"}),
        ("", {"overflow": "fold"}),
    )
    for tag in ollama.required_local_models(config):
        installed = tag in have
        table.add_row(
            Text(g.ok if installed else g.fail, style="loom.good" if installed else "loom.bad"),
            Text(tag, style="loom.local" if installed else "loom.muted"),
            Text("" if installed else "not pulled", style="loom.muted"),
        )
    console.print(table)


@models_app.command("install")
def models_install(root: str = typer.Option(".", "--root")) -> None:
    """Install Ollama if missing, start its daemon, and pull the configured
    local models — the whole local side of a Loom setup in one command."""
    from loom.ui import onboarding

    _retheme(root)
    settings = settings_mod.load_settings(root)
    if not onboarding.ensure_ollama(console, settings.models):
        raise typer.Exit(1)
    models_pull(None)


@models_app.command("serve")
def models_serve(root: str = typer.Option(".", "--root")) -> None:
    """Start the Ollama daemon in the background and wait for it to answer."""
    from loom.core import ollama_setup

    _retheme(root)
    endpoint = settings_mod.load_settings(root).models.ollama_endpoint
    if ollama_setup.is_up(endpoint):
        render.note(console, f"already running at {endpoint}", kind="good")
        return
    with render.Working(console, "starting the Ollama daemon"):
        ok = ollama_setup.serve(endpoint)
    if not ok:
        render.note(console, f"didn't come up — see {ollama_setup.log_path()}", kind="bad")
        raise typer.Exit(1)
    render.note(console, f"daemon running at {endpoint}", kind="good")


@models_app.command("rm")
def models_rm(
    model: str = typer.Argument(..., help="Model tag to delete, e.g. gemma4:e4b"),
    root: str = typer.Option(".", "--root"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Delete a downloaded local model and free its disk space."""
    from loom.core import ollama

    _retheme(root)
    config = settings_mod.load_settings(root).models
    sizes = ollama.installed_sizes(config)
    size = next((v for k, v in sizes.items() if k == model or k.split(":")[0] == model), 0)
    if size:
        render.note(console, f"{model} is using [loom.local]{ollama.human_size(size)}[/loom.local]")
    # Deleting weights is slow to undo (a multi-gigabyte re-download), so it
    # asks first unless told not to.
    if not yes and not render.confirm(console, f"  delete `{model}`?", default=False):
        render.note(console, "kept")
        return
    ok, message = ollama.remove(model, config.ollama_endpoint)
    render.note(console, message, kind="good" if ok else "bad")
    if not ok:
        raise typer.Exit(1)
    for role, configured in config.all_models().items():
        if config.is_local(configured) and configured.endswith(model):
            render.note(
                console,
                f"[loom.warn]{role}[/loom.warn] still points at it — "
                f"`loom config set {role} <model>` or `loom models pull {model}`",
                kind="warn",
            )
            break


@models_app.command("pull")
def models_pull(
    model: Optional[str] = typer.Argument(None, help="Specific model tag; omit to pull all missing.")
) -> None:
    """Pull local models through the Ollama daemon's HTTP API (works with
    remote endpoints; no ollama binary needed)."""
    from loom.core import ollama

    config = cfg.load_config()
    targets = [model] if model else ollama.missing_models(config)
    if not targets:
        render.note(console, "nothing to pull — all required models present", kind="good")
        return
    for tag in targets:
        render.note(console, f"pulling [loom.local]{tag}[/loom.local] from {config.ollama_endpoint}", kind="tip")
        code = ollama.pull(tag, config.ollama_endpoint, console)
        if code != 0:
            render.note(console, f"pull failed for {tag}", kind="bad")
            raise typer.Exit(code)
        render.note(console, tag, kind="good")
    render.note(console, f"{len(targets)} model(s) ready — all free from here on", kind="good")


@playwright_app.command("status")
def playwright_status() -> None:
    """Check for npx and whether Playwright's browser binaries are installed."""
    from loom.core import playwright_setup

    st = playwright_setup.status()
    if not st.npx_available:
        render.note(console, playwright_setup.INSTALL_HINT, kind="bad")
        raise typer.Exit(1)
    if st.browsers_installed:
        render.note(console, f"playwright browser installed ({st.browsers_dir})", kind="good")
    else:
        render.note(
            console,
            f"no browser installed at {st.browsers_dir} — run `loom playwright install`",
            kind="warn",
        )
        raise typer.Exit(1)


@playwright_app.command("install")
def playwright_install(
    browser: str = typer.Argument("chromium", help="Browser to install: chromium, firefox, or webkit.")
) -> None:
    """Download the browser binary the Playwright MCP server (and `tester`
    subagent) drives — a one-time step `npx @playwright/mcp` doesn't do for
    you."""
    from loom.core import playwright_setup

    code = playwright_setup.install_browsers(console, browser)
    if code != 0:
        render.note(console, "install failed", kind="bad")
        raise typer.Exit(code)
    render.note(console, f"{browser} installed", kind="good")


@app.command("doctor")
def doctor(
    root: str = typer.Option(".", "--root"),
    probe: bool = typer.Option(
        False, "--probe", help="Also call every configured model once to prove it answers."
    ),
) -> None:
    """Health-check the Loom setup: python, ollama, API keys, npx, MCP.

    ``--probe`` adds a real round-trip per configured model. The offline
    checks prove a key is present; only a call proves it is accepted for that
    model — a distinction that otherwise surfaces mid-task.
    """
    import os
    import sys as _sys

    from loom.core import ollama
    from loom.core.mcp import mcp_status

    _retheme(root)
    settings = settings_mod.load_settings(root)
    config = settings.models

    def effective_env(key: str) -> str | None:
        """Value ``key`` would resolve to without mutating the real process
        env (doctor is read-only): real env wins, else settings.json's env."""
        return os.environ.get(key) or settings.env.get(key)

    g = ink(console)

    def row(ok, label, detail):
        mark, style = (g.ok, "loom.good") if ok else ((g.pending, "loom.warn") if ok is None else (g.fail, "loom.bad"))
        return (
            Text(f"{mark} ", style=style) + Text(label, style="loom.text"),
            Text(detail, style="loom.muted" if ok else style),
        )

    lines = [row(_sys.version_info >= (3, 11), "python", _sys.version.split()[0])]
    st = ollama.status(config)
    if st.running:
        detail = f"running @ {st.endpoint}" + ("" if st.installed else " (remote — no local binary)")
        lines.append(row(True, "ollama", detail))
        missing = ollama.missing_models(config)
        lines.append(row(not missing, "local models", ", ".join(missing) + " missing" if missing else "all present"))
        if missing:
            # A missing tag only costs money if nothing local can cover it.
            from loom.core.local_pool import build_pool, plan_local_roles

            plan = plan_local_roles(config, build_pool(config, st))
            if plan.cloud:
                lines.append(row(None, "cloud fallback", f"{', '.join(sorted(plan.cloud))} will run on {config.cloud_fallback} (billed)"))
            if plan.substituted:
                lines.append(row(None, "local fallback", f"{', '.join(sorted(plan.substituted))} will run on another local model (free)"))
    else:
        lines.append(row(False, "ollama", f"not reachable @ {st.endpoint}" + ("" if st.installed else ", binary not installed")))
        lines.append(row(None, "cloud fallback", f"local roles will run on {config.cloud_fallback} (billed)"))
    key_set = bool(
        effective_env("ANTHROPIC_API_KEY")
        or effective_env("ANTHROPIC_AUTH_TOKEN")
        or effective_env("AWS_BEARER_TOKEN_BEDROCK")
    )
    lines.append(row(key_set, "anthropic_api_key", "set" if key_set else "not set"))
    from loom.core import playwright_setup

    pw = playwright_setup.status()
    lines.append(row(pw.npx_available, "npx", "found" if pw.npx_available else "not found (Playwright MCP needs Node)"))
    if pw.npx_available:
        lines.append(
            row(
                pw.browsers_installed,
                "playwright browsers",
                "installed" if pw.browsers_installed else "missing — run `loom playwright install`",
            )
        )
    for r in mcp_status(settings):
        ok = True if r["state"] == "connected" else (None if r["state"] in ("not connected", "disabled") else False)
        lines.append(row(ok, f"mcp:{r['name']}", r["state"]))
    console.print()
    render.rule(console, "doctor")
    console.print(render.kv(lines, justify="left"))
    if not probe:
        render.note(console, "`loom doctor --probe` also calls every configured model", kind="tip")
        return
    _probe_models(settings)


def _probe_models(settings) -> None:
    from loom.core import preflight

    g = ink(console)
    models = settings.models
    settings.apply_env()
    plan = {
        "orchestrator": models.orchestrator,
        "advisor": models.advisor,
        "escalation": models.escalation_model,
        **models.subagents,
    }
    console.print()
    render.rule(console, "probe")
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
        results = list(preflight.check_plan(plan, models))
    for roles, check in results:
        if not check.ok:
            failed.append(check)
        mark, style = (g.ok, "loom.good") if check.ok else (
            (g.pending, "loom.warn") if check.state in ("not-pulled", "offline") else (g.fail, "loom.bad")
        )
        is_local = models.is_local(check.model)
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
    if failed:
        render.note(console, f"{len(failed)} model(s) won't answer", kind="warn")
        raise typer.Exit(1)
    render.note(console, "every configured model answered", kind="good")


@app.command("update")
def update() -> None:
    """Check for and install a newer build (frozen binary installs only —
    source installs get a `git pull && uv sync` hint instead)."""
    from loom.core import update as update_mod

    if not update_mod.is_frozen():
        console.print(
            "[loom.warn]running from source[/loom.warn] — update with:\n"
            "  [loom.bright]git pull && uv sync[/loom.bright]"
        )
        raise typer.Exit()

    console.print(f"[loom.warp]checking[/loom.warp] latest release of {update_mod.REPO} …")
    try:
        result = update_mod.check()
    except Exception as exc:
        console.print(f"[loom.bad.b]update check failed:[/loom.bad.b] {exc}")
        raise typer.Exit(1)

    if result.up_to_date:
        console.print(f"[loom.good]✓ up to date[/loom.good] ({result.asset}, sha256 {result.current_sha256[:12]}…)")
        raise typer.Exit()

    console.print(f"[loom.warn]update available[/loom.warn] for {result.asset}")
    try:
        update_mod.apply(result, console=console)
    except Exception as exc:
        console.print(f"[loom.bad.b]update failed:[/loom.bad.b] {exc}")
        raise typer.Exit(1)


@app.command("setup")
def setup(
    root: str = typer.Option(".", "--root"),
    scope: Optional[str] = typer.Option(None, "--scope", help="Skip the scope prompt: user | project."),
) -> None:
    """Run the interactive setup wizard: pick providers/models for every role."""
    from loom.ui import onboarding

    try:
        settings = onboarding.run(console, root=root, scope=scope)
        onboarding.maybe_setup_playwright(console, settings)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[loom.muted]setup cancelled[/loom.muted]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
