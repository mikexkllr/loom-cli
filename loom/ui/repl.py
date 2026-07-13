"""The interactive Loom REPL — a chat UI that lives in the terminal.

Launched by ``loom`` with no task (or ``loom chat``). Uses prompt_toolkit for a
rich input line (history, key bindings, a live status toolbar) and Rich for
rendering the orchestrator/subagent stream, styled after Claude Code /
opencode: a compact welcome box, a bare ``>`` prompt, token-level streaming
with ``⏺`` bullets for assistant text and tool calls, ``⎿`` continuation
lines for results, and a cost receipt after every turn. Slash commands
(``/help``, ``/model``, ``/resume`` …) are handled without touching the model.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.text import Text

from loom.core import repomap
from loom.core import sessions as sessions_mod
from loom.core import settings as settings_mod
from loom.core import undo
from loom.core.settings import Settings
from loom.core.usage import UsageTracker
from loom.middleware import policy
from loom.tools import sandbox
from loom.ui import slash
from loom.ui.theme import make_console

_HISTORY_FILE = settings_mod.cfg.USER_CONFIG_DIR / "history"

# Project memory files, first match wins (Claude Code reads CLAUDE.md; Loom's
# own is LOOM.md but we honor the ecosystem names too).
MEMORY_FILES = ("LOOM.md", "CLAUDE.md", "AGENTS.md")


class Session:
    """Mutable state for one interactive Loom session."""

    def __init__(
        self,
        settings: Settings,
        cwd: str = ".",
        *,
        plan=False,
        local_only=False,
        yolo=False,
        airgap=False,
    ) -> None:
        self.settings = settings
        self.cwd = Path(cwd).resolve()
        self.plan = plan
        self.local_only = local_only
        self.yolo = yolo
        self.airgap = airgap
        self.vim = False
        self.console = make_console(settings.ui)
        self.messages: list = []
        self.bundle = None
        self.thread_id = sessions_mod.new_thread_id()
        self.checkpointer, self.durable = sessions_mod.make_checkpointer(self.cwd)
        self.tracker = UsageTracker(settings.models)
        self.pending_context: str | None = None  # summary injected after /compact
        self._memory_sent = False
        sandbox.set_root(self.cwd)

    # Back-compat view used by /status and tests.
    @property
    def usage(self) -> dict:
        ci, co = self.tracker.session.tokens(self.tracker.session.cloud)
        li, lo = self.tracker.session.tokens(self.tracker.session.local)
        return {
            "turns": self.tracker.turns,
            "input_tokens": ci + li,
            "output_tokens": co + lo,
            "cloud_cost": self.tracker.session.cloud_cost,
        }

    # ----- lifecycle -----
    def reset(self) -> None:
        self.messages = []
        self._memory_sent = False
        # New thread_id => the checkpointer's prior state is no longer referenced.
        self.thread_id = sessions_mod.new_thread_id()

    def reload_settings(self) -> None:
        self.settings = settings_mod.load_settings(self.cwd)
        self.console = make_console(self.settings.ui)
        self.tracker.config = self.settings.models

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
                airgap=self.airgap,
                cwd=str(self.cwd),
                checkpointer=self.checkpointer,
            )
        return self.bundle

    def _run_config(self):
        """LangGraph config: usage callbacks always, thread_id when persistent."""
        config: dict = {"callbacks": [self.tracker]}
        if self.bundle is not None and self.bundle.persistent:
            config["configurable"] = {"thread_id": self.thread_id}
        return config

    # ----- memory / context helpers -----
    def memory_path(self) -> Path | None:
        for name in MEMORY_FILES:
            p = self.cwd / name
            if p.exists():
                return p
        return None

    def _prepare_text(self, text: str) -> str:
        """Prepend one-time context (project memory, repo map, any /compact
        summary) and expand @file mentions."""
        parts: list[str] = []
        if not self._memory_sent:
            mem = self.memory_path()
            if mem is not None:
                try:
                    parts.append(f"[Project memory — {mem.name}]\n{mem.read_text(encoding='utf-8')}")
                except OSError:
                    pass
            try:
                tree = repomap.repo_map(self.cwd)
                if tree:
                    parts.append(f"[Repo map]\n{tree}")
            except Exception:
                pass
            self._memory_sent = True
        if self.pending_context:
            parts.append(f"[Summary of the compacted earlier conversation]\n{self.pending_context}")
            self.pending_context = None
        parts.append(repomap.expand_mentions(text, self.cwd))
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

        sessions_mod.record(self.cwd, self.thread_id, text)
        undo.current_turn_id.set(f"{self.thread_id}-t{self.tracker.turns + 1}")
        self.tracker.start_turn()
        text = self._prepare_text(text)

        # With a checkpointer, the graph persists history under thread_id — send
        # only the new turn. Without one, resend the full local transcript.
        if bundle.persistent:
            inputs = {"messages": [("user", text)]}
        else:
            self.messages.append(("user", text))
            inputs = {"messages": list(self.messages)}

        run_config = self._run_config()
        try:
            self._stream(bundle.agent, inputs, run_config)
        except KeyboardInterrupt:
            self.console.print("\n[loom.warn]⏹ interrupted — partial work may have landed; /undo rolls back this turn's file writes[/loom.warn]")
        except Exception as exc:
            self.console.print(f"[loom.warn]streaming unavailable ({exc}); running synchronously…[/loom.warn]")
            result = bundle.agent.invoke(inputs, config=run_config)
            self._absorb_result(result)
        finally:
            undo.current_turn_id.set("")
            receipt = self.tracker.receipt(turn=True)
            if receipt:
                self.console.print(Text(f"✻ {receipt}", style="loom.dim"))

    # ----- approval prompt with diff preview -----
    def _confirm(self, tool_name: str, tool_input: dict, reason: str) -> bool:
        detail = ", ".join(f"{k}={str(v)[:60]}" for k, v in (tool_input or {}).items())
        self.console.print(
            Panel(f"[loom.tool]{tool_name}[/loom.tool]  {detail}", title=f"approve? ({reason})", border_style="loom.warn")
        )
        diff = self._diff_for(tool_name, tool_input or {})
        if diff:
            self.console.print(diff)
        return Confirm.ask("  run this tool?", default=False)

    def _diff_for(self, tool_name: str, tool_input: dict) -> Text | None:
        """Unified diff preview for write_file / edit_file approvals."""
        path = tool_input.get("path")
        if not path:
            return None
        if tool_name == "edit_file":
            old = str(tool_input.get("old_string", ""))
            new = str(tool_input.get("new_string", ""))
        elif tool_name == "write_file":
            target = self.cwd / path if not Path(path).is_absolute() else Path(path)
            try:
                old = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
            except OSError:
                old = ""
            new = str(tool_input.get("content", ""))
        else:
            return None
        lines = list(
            difflib.unified_diff(old.splitlines(), new.splitlines(), fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="")
        )
        if not lines:
            return None
        out = Text()
        for line in lines[:80]:
            style = "loom.subagent" if line.startswith("+") else "loom.err" if line.startswith("-") else "loom.dim"
            out.append(line + "\n", style=style)
        if len(lines) > 80:
            out.append(f"… +{len(lines) - 80} more diff lines\n", style="loom.dim")
        return out

    # ----- rendering (Claude Code style: ⏺ bullets + ⎿ results) -----

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

    @staticmethod
    def _chunk_text(chunk) -> str:
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
            )
        return ""

    def _stream(self, agent, inputs, run_config) -> None:
        """Token-level streaming; falls back to per-update rendering when the
        installed langgraph doesn't support multi-mode streams."""
        if not self.settings.ui.streaming:
            self._stream_updates(agent.stream(inputs, config=run_config, stream_mode="updates"))
            return
        try:
            stream = agent.stream(inputs, config=run_config, stream_mode=["updates", "messages"])
            self._stream_multi(stream)
        except (TypeError, ValueError):
            self._stream_updates(agent.stream(inputs, config=run_config, stream_mode="updates"))

    def _stream_multi(self, stream) -> None:
        ui = self.settings.ui
        streamed: set[str] = set()  # finalized token-streamed texts (dedup vs updates)
        buf: list[str] = []
        final_text = None

        def finish_block() -> None:
            nonlocal buf
            if buf:
                streamed.add("".join(buf).strip())
                buf = []
                self.console.print("\n")

        for item in stream:
            if not (isinstance(item, tuple) and len(item) == 2):
                continue
            mode, payload = item
            if mode == "messages":
                chunk, _meta = payload
                if type(chunk).__name__ != "AIMessageChunk":
                    continue
                text = self._chunk_text(chunk)
                if not text:
                    continue
                if not buf:
                    self.console.print(Text("⏺ ", style="loom.agent"), end="")
                buf.append(text)
                self.console.print(text, end="", markup=False, highlight=False, soft_wrap=True)
                continue

            # updates mode — structure: tool calls, results, non-streamed text
            finish_block()
            for node, update in (payload or {}).items():
                msgs = (update or {}).get("messages") if isinstance(update, dict) else None
                if not msgs:
                    continue
                msg = msgs[-1]
                if getattr(msg, "type", "") == "tool":
                    if ui.show_tool_calls:
                        self._print_tool_result(msg)
                    continue
                for call in getattr(msg, "tool_calls", []) or []:
                    if ui.show_tool_calls:
                        self._print_tool_call(call, node)
                text = getattr(msg, "content", "")
                if text:
                    text = str(text) if isinstance(text, str) else self._chunk_text(msg)
                    if text.strip() and text.strip() not in streamed:
                        self._print_assistant(text, node)
                    final_text = text
        finish_block()
        if final_text is not None and not (self.bundle and self.bundle.persistent):
            self.messages.append(("assistant", final_text))

    def _stream_updates(self, stream) -> None:
        ui = self.settings.ui
        final_text = None
        for chunk in stream:
            for node, update in (chunk or {}).items():
                msgs = (update or {}).get("messages") if isinstance(update, dict) else None
                if not msgs:
                    continue
                msg = msgs[-1]
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
    if session.airgap:
        modes.append("AIRGAP")
    if session.yolo:
        modes.append("YOLO")
    if session.vim:
        modes.append("VIM")
    mode_str = " ".join(modes) or "normal"
    cost = session.tracker.session.cloud_cost
    return f" {session.settings.models.orchestrator} · {mode_str} · ${cost:.3f} · /help "


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(settings: Settings, cwd: str = ".", *, plan=False, local_only=False, yolo=False, airgap=False) -> None:
    session = Session(settings, cwd, plan=plan, local_only=local_only, yolo=yolo, airgap=airgap)
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
