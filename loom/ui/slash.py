"""Slash commands for the REPL (Claude Code-style ``/command``).

Each handler takes the live :class:`~loom.ui.repl.Session` and the argument
string, and returns ``True`` if the loop should continue (always, except
``/exit``). Handlers render directly to ``session.console``.
"""

from __future__ import annotations

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
    aliases = {"quit": "exit", "q": "exit", "?": "help", "h": "help"}
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


@command("plan", "Toggle plan mode (read-only, no writes)")
def _plan(session: "Session", args: str) -> bool:
    session.plan = not session.plan
    if session.plan:
        session.local_only = session.local_only
    session.rebuild()
    session.console.print(f"plan mode: [loom.accent]{'on' if session.plan else 'off'}[/loom.accent]")
    return True


@command("local", "Toggle local-only mode (no cloud calls)")
def _local(session: "Session", args: str) -> bool:
    session.local_only = not session.local_only
    session.rebuild()
    session.console.print(f"local-only: [loom.accent]{'on' if session.local_only else 'off'}[/loom.accent]")
    return True


@command("yolo", "Toggle auto-approve for tools that would ask")
def _yolo(session: "Session", args: str) -> bool:
    session.yolo = not session.yolo
    session.console.print(
        f"auto-approve: [loom.warn]{'ON — tools run without asking' if session.yolo else 'off'}[/loom.warn]"
    )
    return True


@command("model", "Show or set the orchestrator model: /model gpt-4o")
def _model(session: "Session", args: str) -> bool:
    if args.strip():
        from loom.core import settings as st

        st.set_value("models.orchestrator", args.strip())
        session.reload_settings()
        session.rebuild()
        session.console.print(f"orchestrator → [loom.accent]{args.strip()}[/loom.accent]")
    else:
        session.console.print(f"orchestrator: [loom.accent]{session.settings.models.orchestrator}[/loom.accent]")
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
    if not st.installed:
        session.console.print(f"[loom.err]{ollama.INSTALL_HINT}[/loom.err]")
        return True
    missing = ollama.missing_models(session.settings.models)
    state = "running" if st.running else "not running"
    session.console.print(f"ollama daemon: {state} @ {st.endpoint}")
    session.console.print(
        "[loom.warn]missing:[/loom.warn] " + ", ".join(missing) if missing else "[loom.subagent]all models present[/loom.subagent]"
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
