"""The interactive Loom REPL — a chat UI that lives in the terminal.

Launched by ``loom`` with no task (or ``loom chat``). Uses prompt_toolkit for a
rich input line (history, key bindings, a live status toolbar) and Rich for
rendering the orchestrator/subagent stream. Slash commands (``/help``, ``/plan``,
``/model`` …) are handled without touching the model.
"""

from __future__ import annotations

from pathlib import Path

from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from loom.core import settings as settings_mod
from loom.core.settings import Settings
from loom.middleware import policy
from loom.tools import sandbox
from loom.ui import slash
from loom.ui.theme import make_console

_HISTORY_FILE = settings_mod.cfg.USER_CONFIG_DIR / "history"


class Session:
    """Mutable state for one interactive Loom session."""

    def __init__(self, settings: Settings, cwd: str = ".", *, plan=False, local_only=False, yolo=False) -> None:
        self.settings = settings
        self.cwd = Path(cwd).resolve()
        self.plan = plan
        self.local_only = local_only
        self.yolo = yolo
        self.console = make_console(settings.ui)
        self.messages: list = []
        self.bundle = None
        self._exit = False
        sandbox.set_root(self.cwd)

    # ----- lifecycle -----
    def reset(self) -> None:
        self.messages = []

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
            )
        return self.bundle

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

        self.messages.append(("user", text))
        inputs = {"messages": list(self.messages)}
        try:
            self._stream(bundle.agent, inputs)
        except Exception as exc:
            self.console.print(f"[loom.warn]streaming unavailable ({exc}); running synchronously…[/loom.warn]")
            result = bundle.agent.invoke(inputs)
            self._absorb_result(result)

    def _confirm(self, tool_name: str, tool_input: dict, reason: str) -> bool:
        detail = ", ".join(f"{k}={str(v)[:60]}" for k, v in (tool_input or {}).items())
        self.console.print(
            Panel(f"[loom.tool]{tool_name}[/loom.tool]  {detail}", title=f"approve? ({reason})", border_style="loom.warn")
        )
        return Confirm.ask("  run this tool?", default=False)

    def _stream(self, agent, inputs) -> None:
        ui = self.settings.ui
        final_text = None
        for chunk in agent.stream(inputs, stream_mode="updates"):
            for node, update in (chunk or {}).items():
                msgs = (update or {}).get("messages") if isinstance(update, dict) else None
                if not msgs:
                    continue
                msg = msgs[-1]
                text = getattr(msg, "content", "")
                for call in getattr(msg, "tool_calls", []) or []:
                    if ui.show_tool_calls:
                        name = call.get("name", "?") if isinstance(call, dict) else getattr(call, "name", "?")
                        self.console.print(f"  [loom.tool]→ {name}[/loom.tool]")
                if text:
                    style = "loom.subagent" if node != "agent" else "loom.agent"
                    self.console.print(Panel(str(text), border_style=style, title=f"[{style}]{node}[/{style}]"))
                    final_text = str(text)
        if final_text is not None:
            self.messages.append(("assistant", final_text))

    def _absorb_result(self, result) -> None:
        msgs = result.get("messages", []) if isinstance(result, dict) else []
        if msgs:
            text = str(getattr(msgs[-1], "content", msgs[-1]))
            self.console.print(Panel(text, title="[loom.agent]loom[/loom.agent]", border_style="loom.agent"))
            self.messages.append(("assistant", text))


# ---------------------------------------------------------------------------
# Banner + toolbar
# ---------------------------------------------------------------------------


def _banner(session: Session) -> Panel:
    from loom import __version__

    t = Table.grid(padding=(0, 2))
    t.add_row("[loom.accent]Loom[/loom.accent]", f"v{__version__} · hybrid local/cloud agents")
    t.add_row("orchestrator", session.settings.models.orchestrator)
    t.add_row("advisor", session.settings.models.advisor)
    t.add_row("cwd", str(session.cwd))
    t.add_row("[loom.dim]tips[/loom.dim]", "/help for commands · /plan · /local · Ctrl-D to exit")
    return Panel(t, border_style="loom.accent", title="welcome")


def _toolbar(session: Session):
    modes = []
    if session.plan:
        modes.append("PLAN")
    if session.local_only:
        modes.append("LOCAL")
    if session.yolo:
        modes.append("YOLO")
    mode_str = " ".join(modes) or "normal"
    return f" {session.settings.ui.prompt_symbol}  model={session.settings.models.orchestrator}  mode={mode_str} "


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(settings: Settings, cwd: str = ".", *, plan=False, local_only=False, yolo=False) -> None:
    session = Session(settings, cwd, plan=plan, local_only=local_only, yolo=yolo)
    if settings.ui.banner:
        session.console.print(_banner(session))

    prompt_session = _make_prompt_session()

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
