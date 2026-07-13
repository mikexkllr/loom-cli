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


@command("status", "Show version, models, modes, MCP, and session usage")
def _status(session: "Session", args: str) -> bool:
    from loom import __version__
    from loom.core.mcp import mcp_status

    cfg = session.settings.models
    modes = [m for m, on in (("plan", session.plan), ("local-only", session.local_only), ("yolo", session.yolo)) if on]
    mcp_line = ", ".join(
        f"{r['name']} ({r['state']}{', ' + str(len(r['tools'])) + ' tools' if r['tools'] else ''})"
        for r in mcp_status(session.settings)
    ) or "—"
    u = session.usage
    table = Table.grid(padding=(0, 2))
    table.add_row("[loom.dim]version[/loom.dim]", f"loom v{__version__}")
    table.add_row("[loom.dim]cwd[/loom.dim]", str(session.cwd))
    table.add_row("[loom.dim]orchestrator[/loom.dim]", cfg.orchestrator)
    table.add_row("[loom.dim]advisor[/loom.dim]", cfg.advisor)
    table.add_row("[loom.dim]mode[/loom.dim]", ", ".join(modes) or "normal")
    table.add_row("[loom.dim]permissions[/loom.dim]", f"default: {session.settings.permissions.default_mode}")
    table.add_row("[loom.dim]mcp[/loom.dim]", mcp_line)
    table.add_row("[loom.dim]memory[/loom.dim]", str(session.memory_path() or "— (create with /init)"))
    table.add_row("[loom.dim]session[/loom.dim]", f"{u['turns']} turns · {u['input_tokens']} in / {u['output_tokens']} out tokens")
    session.console.print(Panel(table, title="status", border_style="loom.accent", expand=False))
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


@command("cost", "Show token usage for this session")
def _cost(session: "Session", args: str) -> bool:
    u = session.usage
    session.console.print(
        f"session: [loom.accent]{u['turns']}[/loom.accent] turns · "
        f"[loom.accent]{u['input_tokens']:,}[/loom.accent] input / "
        f"[loom.accent]{u['output_tokens']:,}[/loom.accent] output tokens"
    )
    session.console.print("[loom.dim]local (ollama) tokens are free; cloud tokens are billed by your provider[/loom.dim]")
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
    model_string = cfg.subagents.get("general", cfg.orchestrator) if session.local_only else cfg.orchestrator
    try:
        model = build_model(model_string, cfg)
        prompt = (
            "Summarize this coding-session transcript so work can continue "
            "seamlessly: goals, decisions, files touched, current state, and "
            "open next steps. Be concise but lose nothing load-bearing.\n\n"
            + "\n".join(lines)
        )
        summary = str(model.invoke(prompt).content)
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

    st = ollama.status(session.settings.models)
    if st.installed:
        out.append(row(st.running, "ollama", f"{'running' if st.running else 'not running'} @ {st.endpoint}"))
        missing = ollama.missing_models(session.settings.models)
        out.append(row(not missing, "local models", ", ".join(missing) + " missing" if missing else "all present"))
    else:
        out.append(row(False, "ollama", "not installed"))

    for key in ("ANTHROPIC_API_KEY",):
        out.append(row(bool(os.environ.get(key)), key.lower(), "set" if os.environ.get(key) else "not set"))

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
