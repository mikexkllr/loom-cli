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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from loom.core import config as cfg
from loom.core import settings as settings_mod
from loom.tools import sandbox

console = Console()


class _PromptOrCommandGroup(typer.core.TyperGroup):
    """Let the root command take either a free-form task or a subcommand.

    Click parses the group's ``[PROMPT]`` argument before resolving
    subcommands, so ``loom models status`` would otherwise become the task
    "models". If the first non-option token names a known subcommand, insert
    an empty prompt placeholder so the subcommand resolves normally.
    """

    def parse_args(self, ctx, args):
        first = next((a for a in args if not a.startswith("-")), None)
        if first is not None and first in self.commands:
            idx = args.index(first)
            args = [*args[:idx], "", *args[idx:]]
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
app.add_typer(config_app, name="config")
app.add_typer(settings_app, name="settings")
app.add_typer(agents_app, name="agents")
app.add_typer(models_app, name="models")


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
    advisor_threshold: Optional[str] = typer.Option(
        None, "--advisor-threshold", help="When to auto-consult the advisor: low | medium | high."
    ),
    root: str = typer.Option(".", "--root", help="Project root the agents are sandboxed to."),
) -> None:
    """Run a task, or (with no task and no subcommand) open the interactive UI."""
    if ctx.invoked_subcommand is not None:
        return

    sandbox.set_root(root)
    settings = settings_mod.load_settings(root)

    if not prompt:
        from loom.ui import repl

        repl.run(settings, cwd=root, plan=plan, local_only=local_only, yolo=yolo, airgap=airgap)
        raise typer.Exit()

    _run_task(
        settings, prompt, plan=plan, local_only=local_only, airgap=airgap, yolo=yolo,
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


def _run_task(settings, prompt: str, *, plan: bool, local_only: bool, airgap: bool = False, yolo: bool, advisor_threshold, root: str) -> None:
    from loom.core.orchestrator import build_orchestrator
    from loom.middleware import policy

    settings.apply_env()
    policy.auto_approve.set(yolo)

    config = settings.models
    try:
        bundle = build_orchestrator(
            settings, plan=plan, local_only=local_only, airgap=airgap,
            advisor_threshold=advisor_threshold, cwd=root,
        )
    except ModuleNotFoundError as exc:
        console.print(
            f"[red]Missing dependency:[/red] {exc}. Install with [bold]pip install -e .[/bold]"
        )
        raise typer.Exit(1)

    if bundle.fallbacks:
        roles = ", ".join(sorted(bundle.fallbacks))
        console.print(
            f"[yellow]⚠ Ollama unavailable — {roles} running on {config.cloud_fallback} "
            f"this session (billed). Start Ollama and `loom models pull` to go hybrid.[/yellow]"
        )
    console.print(_fleet_panel(config, bundle))
    console.print(Panel(prompt, title="Task", border_style="cyan"))

    from loom.core.usage import UsageTracker

    tracker = UsageTracker(config)
    tracker.start_turn()
    run_config = {"callbacks": [tracker]}
    inputs = {"messages": [("user", prompt)]}
    try:
        _stream(bundle.agent, inputs, run_config)
    except Exception as exc:  # streaming API drift — fall back to invoke
        console.print(f"[yellow]streaming unavailable ({exc}); running synchronously…[/yellow]")
        result = bundle.agent.invoke(inputs, config=run_config)
        _print_final(result)
    receipt = tracker.receipt(turn=False)
    if receipt:
        console.print(f"[dim]✻ {receipt}[/dim]")


def _stream(agent, inputs, run_config=None) -> None:
    """Stream graph updates and render them as they arrive."""
    last = None
    for chunk in agent.stream(inputs, config=run_config, stream_mode="updates"):
        last = chunk
        for node, update in (chunk or {}).items():
            messages = (update or {}).get("messages") if isinstance(update, dict) else None
            if not messages:
                console.print(f"[dim]· {node}[/dim]")
                continue
            msg = messages[-1]
            text = getattr(msg, "content", "")
            if text:
                console.print(Panel(str(text), title=f"[bold]{node}[/bold]", border_style="green"))
            for call in getattr(msg, "tool_calls", []) or []:
                name = call.get("name", "?") if isinstance(call, dict) else getattr(call, "name", "?")
                console.print(f"  [magenta]→ {name}[/magenta]")
    if last is not None:
        console.rule("[dim]done[/dim]")


def _print_final(result) -> None:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if messages:
        console.print(Panel(str(getattr(messages[-1], "content", messages[-1])), title="Result"))


def _fleet_panel(config: cfg.LoomConfig, bundle) -> Panel:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Role")
    table.add_column("Model")
    table.add_column("Where")
    table.add_row("orchestrator", bundle.model_string, "local" if config.is_local(bundle.model_string) else "cloud")
    for name in bundle.subagent_names:
        model = config.subagents.get(name, "(inherit)")
        table.add_row(name, model, "local" if config.is_local(model) else "cloud")
    if bundle.mode != "local-only":
        table.add_row("advisor", config.advisor, "cloud (on-demand)")
    return Panel(table, title=f"Loom fleet · mode={bundle.mode}", border_style="blue")


# ----------------------------------------------------------------------------
# config subcommands
# ----------------------------------------------------------------------------


@config_app.command("show")
def config_show() -> None:
    """Print the active configuration."""
    config = cfg.load_config()
    import yaml

    console.print(Panel(yaml.safe_dump(config.model_dump(), sort_keys=False), title=str(cfg.USER_CONFIG_PATH)))


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Set a config value, e.g. `loom config set orchestrator gpt-4o`."""
    try:
        updated = cfg.set_value(key, value)
    except Exception as exc:
        console.print(f"[red]invalid:[/red] {exc}")
        raise typer.Exit(1)
    console.print(f"[green]set[/green] {key} = {value}")
    _ = updated


@config_app.command("path")
def config_path() -> None:
    """Print the path to the user config file."""
    cfg.ensure_user_config()
    console.print(str(cfg.USER_CONFIG_PATH))


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

    settings = settings_mod.load_settings(root)
    data = settings.model_dump(exclude={"models"})
    if section:
        if section not in data:
            console.print(f"[red]unknown section:[/red] {section} (try permissions|hooks|env|ui)")
            raise typer.Exit(1)
        data = {section: data[section]}
    console.print(Panel(json.dumps(data, indent=2), title="settings.json (merged)"))


@settings_app.command("set")
def settings_set(key: str, value: str, root: str = typer.Option(".", "--root")) -> None:
    """Set a settings value, e.g. `loom settings set ui.theme light`
    or `loom settings set permissions.default_mode allow`."""
    try:
        settings_mod.set_value(key, value, root)
    except Exception as exc:
        console.print(f"[red]invalid:[/red] {exc}")
        raise typer.Exit(1)
    console.print(f"[green]set[/green] {key} = {value}")


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
        console.print(f"[yellow]exists:[/yellow] {target}")
        raise typer.Exit()
    target.parent.mkdir(parents=True, exist_ok=True)
    starter = settings_mod._read_json(settings_mod.DEFAULT_SETTINGS_PATH)
    target.write_text(json.dumps(starter, indent=2), encoding="utf-8")
    console.print(f"[green]created[/green] {target}")


# ----------------------------------------------------------------------------
# agents subcommands
# ----------------------------------------------------------------------------


@agents_app.command("list")
def agents_list() -> None:
    """Show registered subagents and their assigned models."""
    from loom.subagents import describe_subagents

    config = cfg.load_config()
    table = Table(show_header=True, header_style="bold")
    for col in ("Agent", "Model", "Where", "Mode", "Tools"):
        table.add_column(col)
    for row in describe_subagents(config):
        where = f"[green]{row['scope']}[/green]" if row["scope"] == "local" else f"[cyan]{row['scope']}[/cyan]"
        table.add_row(row["name"], row["model"], where, row["mode"], row["tools"])
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
    if not st.installed:
        console.print(f"[red]✗[/red] {ollama.INSTALL_HINT}")
        raise typer.Exit(1)
    running = "[green]running[/green]" if st.running else "[red]not running[/red]"
    console.print(f"ollama: installed, daemon {running} at {st.endpoint}")
    missing = ollama.missing_models(config)
    if missing:
        console.print(f"[yellow]missing:[/yellow] {', '.join(missing)} — run [bold]loom models pull[/bold]")
    else:
        console.print("[green]✓ all required local models are installed[/green]")


@models_app.command("list")
def models_list() -> None:
    """List required local models and whether each is installed."""
    from loom.core import ollama

    config = cfg.load_config()
    st = ollama.status(config)
    have = set(st.models)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Installed")
    for tag in ollama.required_local_models(config):
        mark = "[green]✓[/green]" if tag in have else "[red]✗[/red]"
        table.add_row(tag, mark)
    console.print(table)


@models_app.command("pull")
def models_pull(
    model: Optional[str] = typer.Argument(None, help="Specific model tag; omit to pull all missing.")
) -> None:
    """Pull local models via Ollama (`ollama pull`)."""
    from loom.core import ollama

    config = cfg.load_config()
    targets = [model] if model else ollama.missing_models(config)
    if not targets:
        console.print("[green]nothing to pull — all required models present[/green]")
        return
    for tag in targets:
        console.print(f"[cyan]pulling[/cyan] {tag} …")
        try:
            code = ollama.pull(tag)
        except FileNotFoundError:
            console.print(f"[red]✗[/red] {ollama.INSTALL_HINT}")
            raise typer.Exit(1)
        if code != 0:
            console.print(f"[red]✗ pull failed for {tag} (exit {code})[/red]")
            raise typer.Exit(code)
    console.print("[green]✓ done[/green]")


@app.command("doctor")
def doctor(root: str = typer.Option(".", "--root")) -> None:
    """Health-check the Loom setup: python, ollama, API keys, npx, MCP."""
    import os
    import shutil as _shutil
    import sys as _sys

    from loom.core import ollama
    from loom.core.mcp import mcp_status

    settings = settings_mod.load_settings(root)
    config = settings.models

    def row(ok, label, detail) -> str:
        mark = "[green]✓[/green]" if ok else ("[yellow]•[/yellow]" if ok is None else "[red]✗[/red]")
        return f" {mark} {label}: {detail}"

    lines = [row(_sys.version_info >= (3, 11), "python", _sys.version.split()[0])]
    st = ollama.status(config)
    if st.installed:
        lines.append(row(st.running, "ollama", f"{'running' if st.running else 'not running'} @ {st.endpoint}"))
        missing = ollama.missing_models(config)
        lines.append(row(not missing, "local models", ", ".join(missing) + " missing" if missing else "all present"))
        if not st.running or missing:
            lines.append(row(None, "cloud fallback", f"local roles will run on {config.cloud_fallback} (billed)"))
    else:
        lines.append(row(False, "ollama", "not installed"))
        lines.append(row(None, "cloud fallback", f"local roles will run on {config.cloud_fallback} (billed)"))
    key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    lines.append(row(key_set, "anthropic_api_key", "set" if key_set else "not set"))
    lines.append(row(bool(_shutil.which("npx")), "npx", "found" if _shutil.which("npx") else "not found (Playwright MCP needs Node)"))
    for r in mcp_status(settings):
        ok = True if r["state"] == "connected" else (None if r["state"] in ("not connected", "disabled") else False)
        lines.append(row(ok, f"mcp:{r['name']}", r["state"]))
    console.print(Panel("\n".join(lines), title="loom doctor", border_style="blue"))


if __name__ == "__main__":
    app()
