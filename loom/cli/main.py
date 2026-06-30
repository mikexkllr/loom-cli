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
from loom.tools import sandbox

console = Console()
app = typer.Typer(
    add_completion=False,
    help="Loom — hybrid local/cloud multi-agent CLI coding assistant.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="View and edit configuration.")
agents_app = typer.Typer(help="Inspect registered subagents.")
models_app = typer.Typer(help="Manage local Ollama models.")
app.add_typer(config_app, name="config")
app.add_typer(agents_app, name="agents")
app.add_typer(models_app, name="models")


# ----------------------------------------------------------------------------
# Main task entry
# ----------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Argument(None, help="The task for Loom to perform."),
    plan: bool = typer.Option(False, "--plan", help="Plan-first, read-only exploration before any writes."),
    local_only: bool = typer.Option(False, "--local-only", help="No cloud calls — local models only."),
    advisor_threshold: Optional[str] = typer.Option(
        None, "--advisor-threshold", help="When to auto-consult the advisor: low | medium | high."
    ),
    root: str = typer.Option(".", "--root", help="Project root the agents are sandboxed to."),
) -> None:
    """Run a task. If a subcommand (config/agents/models) is given, defer to it."""
    if ctx.invoked_subcommand is not None:
        return
    if not prompt:
        console.print(ctx.get_help())
        raise typer.Exit()

    sandbox.set_root(root)
    config = cfg.load_config()
    _run_task(config, prompt, plan=plan, local_only=local_only, advisor_threshold=advisor_threshold)


def _run_task(config: cfg.LoomConfig, prompt: str, *, plan: bool, local_only: bool, advisor_threshold) -> None:
    from loom.core.orchestrator import build_orchestrator

    try:
        bundle = build_orchestrator(
            config, plan=plan, local_only=local_only, advisor_threshold=advisor_threshold
        )
    except ModuleNotFoundError as exc:
        console.print(
            f"[red]Missing dependency:[/red] {exc}. Install with [bold]pip install -e .[/bold]"
        )
        raise typer.Exit(1)

    console.print(_fleet_panel(config, bundle))
    console.print(Panel(prompt, title="Task", border_style="cyan"))

    inputs = {"messages": [("user", prompt)]}
    try:
        _stream(bundle.agent, inputs)
    except Exception as exc:  # streaming API drift — fall back to invoke
        console.print(f"[yellow]streaming unavailable ({exc}); running synchronously…[/yellow]")
        result = bundle.agent.invoke(inputs)
        _print_final(result)


def _stream(agent, inputs) -> None:
    """Stream graph updates and render them as they arrive."""
    last = None
    for chunk in agent.stream(inputs, stream_mode="updates"):
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


if __name__ == "__main__":
    app()
