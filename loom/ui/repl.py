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
        self.accept_edits = False
        self.airgap = airgap
        self.vim = False
        self._interrupted = False
        self.console = make_console(settings.ui)
        self.messages: list = []
        self.bundle = None
        self.thread_id = sessions_mod.new_thread_id()
        self.checkpointer, self.durable = sessions_mod.make_checkpointer(self.cwd)
        self.tracker = UsageTracker(settings.models)
        self.pending_context: str | None = None  # summary injected after /compact
        self._memory_sent = False
        sandbox.set_root(self.cwd)

    # ----- modes (Claude Code-style: default → accept-edits → plan → yolo)
    @property
    def approval_mode(self) -> str:
        if self.yolo:
            return "yolo"
        if self.accept_edits:
            return "accept-edits"
        return "default"

    @property
    def mode(self) -> str:
        """The Shift+Tab-cycled mode: plan wins over the approval modes."""
        return "plan" if self.plan else self.approval_mode

    def set_mode(self, mode: str) -> None:
        """Set the exclusive mode; entering/leaving plan rebuilds the agent
        (plan mode compiles a read-only orchestrator)."""
        was_plan = self.plan
        self.plan = mode == "plan"
        self.accept_edits = mode == "accept-edits"
        self.yolo = mode == "yolo"
        if self.plan != was_plan:
            self.rebuild()

    def cycle_approval_mode(self) -> str:
        order = ("default", "accept-edits", "plan", "yolo")
        nxt = order[(order.index(self.mode) + 1) % len(order)]
        self.set_mode(nxt)
        return nxt

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
            if self.bundle.fallbacks:
                roles = ", ".join(sorted(self.bundle.fallbacks))
                self.console.print(
                    f"[loom.warn]⚠ Ollama unavailable — {roles} running on "
                    f"{self.settings.models.cloud_fallback} this session (billed).[/loom.warn] "
                    f"[loom.dim]Start Ollama and `loom models pull` to go hybrid; /doctor for details.[/loom.dim]"
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
    def run_turn(self, text: str) -> str | None:
        """Run one turn; returns the final assistant text (None on failure)."""
        policy.auto_approve.set(self.yolo)
        policy.auto_approve_edits.set(self.accept_edits)
        policy.confirm_callback.set(self._confirm)
        self._interrupted = False

        try:
            bundle = self.ensure_bundle()
        except ModuleNotFoundError as exc:
            self.console.print(f"[loom.err]missing dependency:[/loom.err] {exc} — run `pip install -e .`")
            return None
        except Exception as exc:
            self.console.print(f"[loom.err]could not start orchestrator:[/loom.err] {exc}")
            return None

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
        final_text: str | None = None
        try:
            final_text = self._stream(bundle.agent, inputs, run_config)
        except KeyboardInterrupt:
            self._interrupted = True
            self.console.print("\n[loom.warn]⏹ interrupted — partial work may have landed; /undo rolls back this turn's file writes[/loom.warn]")
        except Exception as exc:
            self.console.print(f"[loom.warn]streaming unavailable ({exc}); running synchronously…[/loom.warn]")
            result = bundle.agent.invoke(inputs, config=run_config)
            final_text = self._absorb_result(result)
        finally:
            undo.current_turn_id.set("")
            receipt = self.tracker.receipt(turn=True)
            if receipt:
                self.console.print(Text(f"✻ {receipt}", style="loom.dim"))
        return final_text

    # ----- plan mode (Claude Code-style: plan → approve → execute) -----
    PLAN_EXECUTE_PROMPT = (
        "The plan you just presented is APPROVED and plan mode is now off. "
        "Implement the plan step by step, verifying as you go. If reality "
        "diverges from the plan, adapt and say so."
    )

    def offer_plan_execution(self) -> None:
        """After a planning turn, offer to approve the plan and execute it
        immediately — plan mode switches off and the same thread continues,
        so the orchestrator implements the plan it just wrote."""
        from rich.prompt import Prompt

        self.console.print(
            Panel(
                "[loom.accent]1[/loom.accent]  yes, and auto-accept edits\n"
                "[loom.accent]2[/loom.accent]  yes, and approve edits manually\n"
                "[loom.accent]3[/loom.accent]  no, keep planning",
                title="plan ready — execute it?",
                border_style="loom.accent",
                expand=False,
            )
        )
        try:
            choice = Prompt.ask("  choice", choices=["1", "2", "3"], default="3")
        except (EOFError, KeyboardInterrupt):
            choice = "3"
        if choice not in ("1", "2"):
            self.console.print(
                "[loom.dim]still in plan mode — refine the plan, or /plan to leave without executing[/loom.dim]"
            )
            return
        self.set_mode("accept-edits" if choice == "1" else "default")
        self.console.print(
            f"[loom.accent]✓ plan approved[/loom.accent] [loom.dim]— executing (mode: {self.mode})[/loom.dim]"
        )
        self.run_turn(self.PLAN_EXECUTE_PROMPT)

    # ----- loop mode -----
    LOOP_NOTE = (
        "\n\n[Loop mode] Work autonomously toward the goal. When the ENTIRE task is "
        "complete and verified, include the exact token LOOP_COMPLETE in your final "
        "message. Otherwise end with a one-line status of what remains."
    )

    def run_loop(self, prompt: str, max_iters: int = 10, until: str | None = None) -> None:
        """Iterate on a task until done: agent signals LOOP_COMPLETE, an
        optional ``until`` shell command exits 0, or max_iters is reached.
        Check failures are fed back into the next iteration."""
        import subprocess

        if self.approval_mode == "default":
            self.console.print(
                "[loom.dim]tip: loop mode pauses on every approval — /mode accept-edits or /yolo makes it autonomous[/loom.dim]"
            )
        next_prompt = prompt + self.LOOP_NOTE
        for i in range(1, max_iters + 1):
            self.console.print(f"[loom.accent]↻ loop {i}/{max_iters}[/loom.accent]")
            text = self.run_turn(next_prompt) or ""
            if self._interrupted:
                self.console.print("[loom.warn]loop stopped (interrupted)[/loom.warn]")
                return
            if until:
                check = subprocess.run(until, shell=True, cwd=self.cwd, capture_output=True, text=True)
                if check.returncode == 0:
                    self.console.print(f"[loom.subagent]✓ loop done — `{until}` passed after {i} iteration(s)[/loom.subagent]")
                    return
                tail = (check.stdout + check.stderr)[-2000:]
                next_prompt = (
                    f"The check command `{until}` still fails (exit {check.returncode}):\n"
                    f"```\n{tail}\n```\nKeep fixing until it passes." + self.LOOP_NOTE
                )
                continue
            if "LOOP_COMPLETE" in text:
                self.console.print(f"[loom.subagent]✓ loop done — agent reported complete after {i} iteration(s)[/loom.subagent]")
                return
            next_prompt = "Continue the loop task from where you left off." + self.LOOP_NOTE
        self.console.print(f"[loom.warn]loop ended after {max_iters} iterations without completing[/loom.warn]")

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
        # deepagents' own FilesystemMiddleware tools (used by the orchestrator
        # directly) key the target path as `file_path`; Loom's sandboxed tools
        # (used by subagents) key it as `path`.
        path = tool_input.get("path") or tool_input.get("file_path")
        if not path:
            return None
        try:
            target = sandbox.resolve_in_sandbox(str(path))
            display_path = target.relative_to(self.cwd)
        except Exception:
            return None
        if tool_name == "edit_file":
            old = str(tool_input.get("old_string", ""))
            new = str(tool_input.get("new_string", ""))
        elif tool_name == "write_file":
            try:
                old = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
            except OSError:
                old = ""
            new = str(tool_input.get("content", ""))
        else:
            return None
        lines = list(
            difflib.unified_diff(old.splitlines(), new.splitlines(), fromfile=f"a/{display_path}", tofile=f"b/{display_path}", lineterm="")
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

    def model_origin(self, node: str) -> tuple[str, bool] | None:
        """(model string, is_local) for a role / stream-node name, accounting
        for live Ollama fallbacks (a role whose local model is unreachable is
        actually running on the billed cloud fallback). None if unknown."""
        cfg = self.settings.models
        role = "orchestrator" if node in ("agent", "model") else node
        if role == "orchestrator":
            model = cfg.orchestrator
        elif role == "advisor":
            model = cfg.advisor
        elif role == "escalation":
            model = cfg.escalation_model
        else:
            model = cfg.subagents.get(role)
        if model is None:
            return None
        if role in (getattr(self.bundle, "fallbacks", None) or {}):
            return cfg.cloud_fallback, False
        return model, cfg.is_local(model)

    @staticmethod
    def _where_badge(is_local: bool) -> str:
        return "⌂ local" if is_local else "☁ cloud"

    @staticmethod
    def _call_args_brief(call) -> str:
        args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {}) or {}
        if not isinstance(args, dict):
            return str(args)[:80]
        brief = ", ".join(f"{k}: {str(v)[:50]}" for k, v in list(args.items())[:3])
        return brief[:100]

    def _print_tool_call(self, call, node: str) -> None:
        name = call.get("name", "?") if isinstance(call, dict) else getattr(call, "name", "?")
        args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {}) or {}
        line = Text()
        line.append("⏺ ", style="loom.tool")
        line.append(name, style="loom.tool")
        brief = self._call_args_brief(call)
        if brief:
            line.append(f"({brief})", style="loom.dim")
        # Delegation calls: show which model will do the work, and where it
        # runs (⌂ local / ☁ cloud), so billed calls are visible at a glance.
        target = None
        if name == "task" and isinstance(args, dict):
            target = args.get("subagent_type") or "general-purpose"
        elif name == "consult":
            target = "advisor"
        origin = self.model_origin(target) if target else None
        if origin:
            line.append(f"  → {origin[0]} ({self._where_badge(origin[1])})", style="loom.dim")
        if node not in ("agent", "model"):
            line.append(f"  [{self._node_label(node)}]", style="loom.dim")
        self.console.print(line)

    def _print_tool_result(self, msg) -> None:
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            return
        first = content.splitlines()[0][:120]
        more = len(content.splitlines()) - 1
        suffix = f" … +{more} lines" if more > 0 else ""
        self.console.print(Text(f"  ⎿ {first}{suffix}", style="loom.dim"))

    def _node_label(self, node: str) -> str:
        """Node name plus its ⌂ local / ☁ cloud badge when the model is known."""
        origin = self.model_origin(node)
        return f"{node} · {self._where_badge(origin[1])}" if origin else node

    def _print_assistant(self, text: str, node: str) -> None:
        bullet = Text("⏺ ", style="loom.agent" if node in ("agent", "model") else "loom.subagent")
        if node not in ("agent", "model"):
            bullet.append(f"[{self._node_label(node)}] ", style="loom.dim")
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

    def _stream(self, agent, inputs, run_config) -> str | None:
        """Token-level streaming; falls back to per-update rendering when the
        installed langgraph doesn't support multi-mode streams. Returns the
        final assistant text."""
        if not self.settings.ui.streaming:
            return self._stream_updates(agent.stream(inputs, config=run_config, stream_mode="updates"))
        try:
            stream = agent.stream(inputs, config=run_config, stream_mode=["updates", "messages"])
            return self._stream_multi(stream)
        except (TypeError, ValueError):
            return self._stream_updates(agent.stream(inputs, config=run_config, stream_mode="updates"))

    def _stream_multi(self, stream) -> str | None:
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
        return final_text

    def _stream_updates(self, stream) -> str | None:
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
        return final_text

    def _absorb_result(self, result) -> str | None:
        msgs = result.get("messages", []) if isinstance(result, dict) else []
        if msgs:
            text = str(getattr(msgs[-1], "content", msgs[-1]))
            self._print_assistant(text, "agent")
            self.messages.append(("assistant", text))
            return text
        return None


# ---------------------------------------------------------------------------
# Banner + toolbar
# ---------------------------------------------------------------------------


def _banner(session: Session) -> Panel:
    from loom import __version__

    cfg = session.settings.models
    o_badge = "⌂" if cfg.is_local(cfg.orchestrator) else "☁"
    a_badge = "⌂" if cfg.is_local(cfg.advisor) else "☁"
    body = Text()
    body.append("✻ Welcome to Loom!", style="loom.accent")
    body.append(f"  v{__version__}\n\n", style="loom.dim")
    body.append("  /help for help, /status for your current setup\n\n", style="loom.dim")
    body.append(f"  model: {o_badge} {cfg.orchestrator} · advisor: {a_badge} {cfg.advisor}\n", style="loom.dim")
    body.append(f"  cwd: {session.cwd}", style="loom.dim")
    return Panel(body, border_style="loom.accent", expand=False, padding=(0, 1))


def _toolbar(session: Session):
    modes = [session.approval_mode.upper()] if session.approval_mode != "default" else []
    if session.plan:
        modes.append("PLAN")
    if session.local_only:
        modes.append("LOCAL")
    if session.airgap:
        modes.append("AIRGAP")
    if session.vim:
        modes.append("VIM")
    mode_str = " ".join(modes) or "normal"
    cost = session.tracker.session.cloud_cost
    # model_origin is fallback-aware: if Ollama is down the orchestrator badge
    # flips to the billed ☁ cloud fallback rather than lying about being local.
    model, is_local = session.model_origin("orchestrator")
    where = "⌂" if is_local else "☁"
    return f" {where} {model} · {mode_str} · ${cost:.3f} · shift+tab: mode · /help "


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _setup_hint(session: Session) -> None:
    """First-run guidance when neither a cloud key nor Ollama is available."""
    import os

    cloud_keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")
    if any(os.environ.get(k) for k in cloud_keys):
        return
    try:
        from loom.core import ollama

        if ollama.status(session.settings.models).running:
            return
    except Exception:
        pass
    session.console.print(
        "[loom.warn]⚠ No cloud API key and no Ollama daemon found — tasks will fail.[/loom.warn]\n"
        "[loom.dim]  cloud:  export ANTHROPIC_API_KEY=…\n"
        "  local:  install Ollama (https://ollama.com) · loom models pull\n"
        "  check:  /doctor[/loom.dim]"
    )


def _maybe_run_onboarding(session: Session) -> None:
    """True first run (no settings.json anywhere yet): launch the setup
    wizard instead of silently falling back to packaged defaults. Falls back
    to the passive `_setup_hint` if the user cancels or it's not a first run."""
    from loom.ui import onboarding

    if not onboarding.needs_onboarding(session.cwd):
        _setup_hint(session)
        return
    session.console.print("[loom.dim]No settings.json found yet — let's configure your models (/setup to redo this later).[/loom.dim]")
    try:
        onboarding.run(session.console, root=session.cwd)
    except (KeyboardInterrupt, EOFError):
        session.console.print("\n[loom.dim]setup skipped — run /setup any time to configure models[/loom.dim]")
        return
    session.reload_settings()
    session.rebuild()


def run(settings: Settings, cwd: str = ".", *, plan=False, local_only=False, yolo=False, airgap=False) -> None:
    session = Session(settings, cwd, plan=plan, local_only=local_only, yolo=yolo, airgap=airgap)
    if settings.ui.banner:
        session.console.print(_banner(session))
    _maybe_run_onboarding(session)

    prompt_session = _make_prompt_session(session)
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
            reply = session.run_turn(line)
        except KeyboardInterrupt:
            session.console.print("[loom.warn]⏹ interrupted[/loom.warn]")
            continue
        if session.plan and reply and not session._interrupted:
            session.offer_plan_execution()


def _make_prompt_session(session: Session | None = None):
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings

        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        completer = WordCompleter([f"/{n}" for n in slash._REGISTRY], sentence=True)
        kb = KeyBindings()
        if session is not None:
            # Claude Code-style: Shift+Tab cycles default → accept-edits → yolo.
            @kb.add("s-tab")
            def _cycle(event) -> None:
                session.cycle_approval_mode()
                event.app.invalidate()  # refresh the toolbar

        return PromptSession(history=FileHistory(str(_HISTORY_FILE)), completer=completer, key_bindings=kb)
    except Exception:
        return None  # fall back to builtin input()


def _read_line(prompt_session, session: Session) -> str:
    if prompt_session is None:
        return input(f"{session.settings.ui.prompt_symbol} ")
    return prompt_session.prompt(
        f"{session.settings.ui.prompt_symbol} ",
        bottom_toolbar=lambda: _toolbar(session),
    )
