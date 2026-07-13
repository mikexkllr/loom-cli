"""The interactive Loom REPL — a chat UI that lives in the terminal.

Launched by ``loom`` with no task (or ``loom chat``). Uses prompt_toolkit for a
rich input line (history, key bindings, a live status toolbar) and Rich for
rendering the orchestrator/subagent stream, styled after Claude Code /
opencode: a compact welcome box, a bare ``>`` prompt, ``⏺`` bullets for
assistant text and tool calls, and ``⎿`` continuation lines for results.
Slash commands (``/help``, ``/model``, ``/mcp`` …) are handled without
touching the model.
"""

from __future__ import annotations

from pathlib import Path

from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.text import Text

from loom.core import settings as settings_mod
from loom.core.settings import Settings
from loom.middleware import policy
from loom.tools import sandbox
from loom.ui import slash
from loom.ui.theme import make_console

_HISTORY_FILE = settings_mod.cfg.USER_CONFIG_DIR / "history"

# Project memory files, first match wins (Claude Code reads CLAUDE.md; Loom's
# own is LOOM.md but we honor the ecosystem names too).
MEMORY_FILES = ("LOOM.md", "CLAUDE.md", "AGENTS.md")


def _make_checkpointer():
    """In-memory LangGraph checkpointer for thread persistence within a session.

    Returns None if langgraph isn't importable, in which case the REPL falls
    back to resending the full transcript each turn.
    """
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    except Exception:
        try:
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()
        except Exception:
            return None


class Session:
    """Mutable state for one interactive Loom session."""

    def __init__(self, settings: Settings, cwd: str = ".", *, plan=False, local_only=False, yolo=False) -> None:
        self.settings = settings
        self.cwd = Path(cwd).resolve()
        self.plan = plan
        self.local_only = local_only
        self.yolo = yolo
        self.vim = False
        self.console = make_console(settings.ui)
        self.messages: list = []
        self.bundle = None
        self.thread_id = "loom-repl"
        self.checkpointer = _make_checkpointer()
        self.usage = {"input_tokens": 0, "output_tokens": 0, "turns": 0}
        self.pending_context: str | None = None  # summary injected after /compact
        self._memory_sent = False
        sandbox.set_root(self.cwd)

    # ----- lifecycle -----
    def reset(self) -> None:
        self.messages = []
        self._memory_sent = False
        # New thread_id => the checkpointer's prior state is no longer referenced.
        self._reset_count = getattr(self, "_reset_count", 0) + 1
        self.thread_id = f"loom-repl-{self._reset_count}"

    def reload_settings(self) -> None:
        self.settings = settings_mod.load_settings(self.cwd)
        self.console = make_console(self.settings.ui)

    def rebuild(self) -> None:
        """Rebuild the orchestrator after a mode/model/settings change."""
        self.bundle = None  # lazily rebuilt on next turn

    def ensure_bundle(self):
        if self.bundle is None:
            from loom.core.orchestrator import build_orchestrator

            self.bundle = build_orchestrator(
                self.settings,
                plan=self.plan,
                local_only=self.local_only,
                cwd=str(self.cwd),
                checkpointer=self.checkpointer,
            )
        return self.bundle

    def _run_config(self):
        """LangGraph config carrying the thread_id (only when persistent)."""
        if self.bundle is not None and self.bundle.persistent:
            return {"configurable": {"thread_id": self.thread_id}}
        return None

    # ----- memory / context helpers -----
    def memory_path(self) -> Path | None:
        for name in MEMORY_FILES:
            p = self.cwd / name
            if p.exists():
                return p
        return None

    def _prepare_text(self, text: str) -> str:
        """Prepend one-time context: the project memory file and any /compact summary."""
        parts: list[str] = []
        if not self._memory_sent:
            mem = self.memory_path()
            if mem is not None:
                try:
                    parts.append(f"[Project memory — {mem.name}]\n{mem.read_text(encoding='utf-8')}")
                except OSError:
                    pass
            self._memory_sent = True
        if self.pending_context:
            parts.append(f"[Summary of the compacted earlier conversation]\n{self.pending_context}")
            self.pending_context = None
        parts.append(text)
        return "\n\n".join(parts)

    def transcript(self) -> list:
        """Best-effort transcript: from the graph state if persistent, else local."""
        if self.bundle is not None and self.bundle.persistent:
            try:
                state = self.bundle.agent.get_state({"configurable": {"thread_id": self.thread_id}})
                msgs = (state.values or {}).get("messages") or []
                if msgs:
                    return list(msgs)
            except Exception:
                pass
        return list(self.messages)

    # ----- a single turn -----
    def run_turn(self, text: str) -> None:
        policy.auto_approve.set(self.yolo)
        policy.confirm_callback.set(self._confirm)

        try:
            bundle = self.ensure_bundle()
        except ModuleNotFoundError as exc:
            self.console.print(f"[loom.err]missing dependency:[/loom.err] {exc} — run `pip install -e .`")
            return
        except Exception as exc:
            self.console.print(f"[loom.err]could not start orchestrator:[/loom.err] {exc}")
            return

        text = self._prepare_text(text)

        # With a checkpointer, the graph persists history under thread_id — send
        # only the new turn. Without one, resend the full local transcript.
        if bundle.persistent:
            inputs = {"messages": [("user", text)]}
        else:
            self.messages.append(("user", text))
            inputs = {"messages": list(self.messages)}

        self.usage["turns"] += 1
        run_config = self._run_config()
        try:
            self._stream(bundle.agent, inputs, run_config)
        except Exception as exc:
            self.console.print(f"[loom.warn]streaming unavailable ({exc}); running synchronously…[/loom.warn]")
            result = bundle.agent.invoke(inputs, config=run_config) if run_config else bundle.agent.invoke(inputs)
            self._absorb_result(result)

    def _confirm(self, tool_name: str, tool_input: dict, reason: str) -> bool:
        detail = ", ".join(f"{k}={str(v)[:60]}" for k, v in (tool_input or {}).items())
        self.console.print(
            Panel(f"[loom.tool]{tool_name}[/loom.tool]  {detail}", title=f"approve? ({reason})", border_style="loom.warn")
        )
        return Confirm.ask("  run this tool?", default=False)

    # ----- rendering (Claude Code style: ⏺ bullets + ⎿ results) -----

    def _track_usage(self, msg) -> None:
        meta = getattr(msg, "usage_metadata", None)
        if isinstance(meta, dict):
            self.usage["input_tokens"] += meta.get("input_tokens", 0) or 0
            self.usage["output_tokens"] += meta.get("output_tokens", 0) or 0

    @staticmethod
    def _call_args_brief(call) -> str:
        args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {}) or {}
        if not isinstance(args, dict):
            return str(args)[:80]
        brief = ", ".join(f"{k}: {str(v)[:50]}" for k, v in list(args.items())[:3])
        return brief[:100]

    def _print_tool_call(self, call, node: str) -> None:
        name = call.get("name", "?") if isinstance(call, dict) else getattr(call, "name", "?")
        line = Text()
        line.append("⏺ ", style="loom.tool")
        line.append(name, style="loom.tool")
        brief = self._call_args_brief(call)
        if brief:
            line.append(f"({brief})", style="loom.dim")
        if node not in ("agent", "model"):
            line.append(f"  [{node}]", style="loom.dim")
        self.console.print(line)

    def _print_tool_result(self, msg) -> None:
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            return
        first = content.splitlines()[0][:120]
        more = len(content.splitlines()) - 1
        suffix = f" … +{more} lines" if more > 0 else ""
        self.console.print(Text(f"  ⎿ {first}{suffix}", style="loom.dim"))

    def _print_assistant(self, text: str, node: str) -> None:
        bullet = Text("⏺ ", style="loom.agent" if node in ("agent", "model") else "loom.subagent")
        if node not in ("agent", "model"):
            bullet.append(f"[{node}] ", style="loom.dim")
        self.console.print(bullet, end="")
        try:
            self.console.print(Markdown(text))
        except Exception:
            self.console.print(text)
        self.console.print()

    def _stream(self, agent, inputs, run_config=None) -> None:
        ui = self.settings.ui
        final_text = None
        stream = (
            agent.stream(inputs, config=run_config, stream_mode="updates")
            if run_config
            else agent.stream(inputs, stream_mode="updates")
        )
        for chunk in stream:
            for node, update in (chunk or {}).items():
                msgs = (update or {}).get("messages") if isinstance(update, dict) else None
                if not msgs:
                    continue
                msg = msgs[-1]
                self._track_usage(msg)
                if getattr(msg, "type", "") == "tool":
                    if ui.show_tool_calls:
                        self._print_tool_result(msg)
                    continue
                for call in getattr(msg, "tool_calls", []) or []:
                    if ui.show_tool_calls:
                        self._print_tool_call(call, node)
                text = getattr(msg, "content", "")
                if text:
                    self._print_assistant(str(text), node)
                    final_text = str(text)
        # Only mirror into the local transcript when the graph isn't persisting.
        if final_text is not None and not (self.bundle and self.bundle.persistent):
            self.messages.append(("assistant", final_text))

    def _absorb_result(self, result) -> None:
        msgs = result.get("messages", []) if isinstance(result, dict) else []
        if msgs:
            text = str(getattr(msgs[-1], "content", msgs[-1]))
            self._print_assistant(text, "agent")
            self.messages.append(("assistant", text))


# ---------------------------------------------------------------------------
# Banner + toolbar
# ---------------------------------------------------------------------------


def _banner(session: Session) -> Panel:
    from loom import __version__

    cfg = session.settings.models
    body = Text()
    body.append("✻ Welcome to Loom!", style="loom.accent")
    body.append(f"  v{__version__}\n\n", style="loom.dim")
    body.append("  /help for help, /status for your current setup\n\n", style="loom.dim")
    body.append(f"  model: {cfg.orchestrator} · advisor: {cfg.advisor}\n", style="loom.dim")
    body.append(f"  cwd: {session.cwd}", style="loom.dim")
    return Panel(body, border_style="loom.accent", expand=False, padding=(0, 1))


def _toolbar(session: Session):
    modes = []
    if session.plan:
        modes.append("PLAN")
    if session.local_only:
        modes.append("LOCAL")
    if session.yolo:
        modes.append("YOLO")
    if session.vim:
        modes.append("VIM")
    mode_str = " ".join(modes) or "normal"
    return f" {session.settings.models.orchestrator} · {mode_str} · /help "


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(settings: Settings, cwd: str = ".", *, plan=False, local_only=False, yolo=False) -> None:
    session = Session(settings, cwd, plan=plan, local_only=local_only, yolo=yolo)
    if settings.ui.banner:
        session.console.print(_banner(session))

    prompt_session = _make_prompt_session()
    session._prompt_session = prompt_session

    while True:
        try:
            line = _read_line(prompt_session, session)
        except (EOFError, KeyboardInterrupt):
            session.console.print("\n[loom.dim]bye[/loom.dim]")
            break

        line = (line or "").strip()
        if not line:
            continue
        if line.startswith("/"):
            if not slash.dispatch(session, line):
                break
            continue

        try:
            session.run_turn(line)
        except KeyboardInterrupt:
            session.console.print("[loom.warn]⏹ interrupted[/loom.warn]")


def _make_prompt_session():
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory

        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        completer = WordCompleter([f"/{n}" for n in slash._REGISTRY], sentence=True)
        return PromptSession(history=FileHistory(str(_HISTORY_FILE)), completer=completer)
    except Exception:
        return None  # fall back to builtin input()


def _read_line(prompt_session, session: Session) -> str:
    if prompt_session is None:
        return input(f"{session.settings.ui.prompt_symbol} ")
    return prompt_session.prompt(
        f"{session.settings.ui.prompt_symbol} ",
        bottom_toolbar=lambda: _toolbar(session),
    )
