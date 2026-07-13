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


def _model_candidates(session: "Session") -> list[str]:
    """Installed local Ollama models first, then common cloud models."""
    from loom.core import ollama

    local = [f"ollama/{tag}" for tag in ollama.status(session.settings.models).models]
    cloud = ["claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-8"]
    return local + [c for c in cloud if c not in local]


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
        local = [c for c in _model_candidates(session) if c.startswith("ollama/")]
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

    # Interactive picker: installed ollama models + common cloud models.
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
    for i, model in enumerate(candidates, 1):
        where = "local" if model.startswith("ollama/") else "cloud"
        session.console.print(f"  [loom.accent]{i}[/loom.accent]  {model}  [loom.dim]({where})[/loom.dim]")
    from rich.prompt import Prompt

    choice = Prompt.ask("  number (or model name, empty to cancel)", default="")
    choice = choice.strip()
    if not choice:
        session.console.print("[loom.dim]cancelled[/loom.dim]")
        return True
    if choice.isdigit() and 1 <= int(choice) <= len(candidates):
        choice = candidates[int(choice) - 1]
    _set_role_model(session, role, choice)
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
    modes = [
        m
        for m, on in (
            ("plan", session.plan),
            ("local-only", session.local_only),
            ("airgap", session.airgap),
            ("yolo", session.yolo),
        )
        if on
    ]
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
    table.add_row("[loom.dim]persistence[/loom.dim]", "sqlite (.loom/sessions.db)" if session.durable else "in-memory (no /resume across restarts)")
    escalations = sum(
        getattr(g, "escalation_count", 0) for g in getattr(session.bundle, "guards", []) or []
    ) if session.bundle is not None else 0
    table.add_row("[loom.dim]escalations[/loom.dim]", f"{escalations} local→cloud (prompt-size guard)")
    table.add_row(
        "[loom.dim]session[/loom.dim]",
        f"{u['turns']} turns · {u['input_tokens']} in / {u['output_tokens']} out tokens · ${u['cloud_cost']:.3f} cloud",
    )
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
    model_string = cfg.subagents.get("general", cfg.orchestrator) if session.local_only else cfg.orchestrator
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
    if st.installed:
        out.append(row(st.running, "ollama", f"{'running' if st.running else 'not running'} @ {st.endpoint}"))
        missing = ollama.missing_models(cfg)
        out.append(row(not missing, "local models", ", ".join(missing) + " missing" if missing else "all present"))
        if not st.running or missing:
            out.append(row(None, "cloud fallback", f"local roles run on {cfg.cloud_fallback} (billed)"))
    else:
        out.append(row(False, "ollama", "not installed"))
        out.append(row(None, "cloud fallback", f"local roles run on {cfg.cloud_fallback} (billed)"))

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
