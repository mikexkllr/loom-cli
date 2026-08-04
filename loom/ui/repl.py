"""The interactive Loom REPL — a chat UI that lives in the terminal.

Launched by ``loom`` with no task (or ``loom chat``). prompt_toolkit drives the
input line (history, key bindings, the status line); Rich draws everything
else through :mod:`loom.ui.render`.

The transcript is drawn as a **weave**: the orchestrator holds a rail down the
left gutter, and every subagent it delegates to opens its own indented rail
beside it, coloured warm when the work is local and free and cool when it is
billed. A turn's shape — who did what, where, and what it cost — is legible
without reading any of the prose. Slash commands (``/help``, ``/model``,
``/resume`` …) are handled without touching the model.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from rich.text import Text

from loom.core import repomap
from loom.core import sessions as sessions_mod
from loom.core import settings as settings_mod
from loom.core import telemetry
from loom.core import undo
from loom.core.settings import Settings
from loom.core.usage import UsageTracker
from loom.middleware import policy
from loom.tools import sandbox
from loom.ui import banner as banner_mod
from loom.ui import render, slash
from loom.ui.render import Thread, Weave, ink
from loom.ui.theme import make_console

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
        # Tools approved with "don't ask again" — cleared when the session ends.
        self.session_allowed: set[str] = set()
        # Approval prompts can fire from LangGraph worker threads; serialize
        # them so two parallel tool calls never interleave on the terminal.
        import threading

        self._confirm_lock = threading.Lock()
        self._interrupted = False
        self.console = make_console(settings.ui)
        self.weave = Weave(self.console, flat=not settings.ui.weave)
        self.messages: list = []
        self.bundle = None
        # Subagent attribution: a nested graph streams under a "tools:<id>"
        # namespace whose id is LangGraph's internal task id, not the `task`
        # tool-call id — so it can't be matched back to the call directly.
        # Instead we bind namespaces to the subagent_types seen in `task` calls,
        # in delegation order (see _attribute_ns). Reset per turn in
        # _stream_multi.
        self._pending_subagents: list[str] = []
        self._ns_role: dict[str, str] = {}
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
        session = self.tracker.session
        ci, co = session.tokens(session.cloud)
        li, lo = session.tokens(session.local)
        return {
            "turns": self.tracker.turns,
            "input_tokens": ci + li,
            "output_tokens": co + lo,
            "cached_tokens": session.cache_read_tokens,
            "cloud_cost": session.cloud_cost,
            # True when any of that cost was priced against a model Loom has no
            # published price for. Shown wherever the figure is, or the same
            # number reads as billed in one place and estimated in another.
            "cost_estimated": session.has_estimates(),
            "orchestrator_share": session.orchestrator_share(),
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
        self.weave = Weave(self.console, flat=not self.settings.ui.weave)
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
            g = ink(self.console)
            if self.bundle.substitutions:
                # Still local, still free — a note, not a warning.
                roles = ", ".join(
                    f"{role} {g.arrow} {self._substituted_model(role)}"
                    for role in sorted(self.bundle.substitutions)
                )
                render.note(
                    self.console,
                    f"[loom.local]{g.local}[/loom.local] model not pulled — {roles} for this session "
                    f"(still free). [loom.muted]`loom models pull` to use your configured models.[/loom.muted]",
                )
            if self.bundle.fallbacks:
                roles = ", ".join(sorted(self.bundle.fallbacks))
                render.note(
                    self.console,
                    f"no local model available — {roles} running on "
                    f"[loom.cloud]{self.settings.models.cloud_fallback}[/loom.cloud] this session "
                    f"[loom.warn](billed)[/loom.warn].\n"
                    f"  [loom.muted]start Ollama and `loom models pull` to go hybrid "
                    f"{g.dot} /doctor for details[/loom.muted]",
                    kind="warn",
                )
        return self.bundle

    def _run_config(self):
        """LangGraph config: usage callbacks always, thread_id when persistent.

        In ``full`` privacy mode the Langfuse tracer rides along in the same
        callback list as the usage tracker — one more ``BaseCallbackHandler``,
        so it sees every model call the orchestrator and its subagents make
        without any of them knowing it is there. `telemetry.callbacks()`
        returns an empty list in every other mode, so this is a no-op unless
        the user opted in.
        """
        from loom.core import telemetry

        config: dict = {"callbacks": [self.tracker, *telemetry.callbacks()]}
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
            self.console.print(f"[loom.bad.b]missing dependency:[/loom.bad.b] {exc} — run `uv sync`")
            telemetry.report("bundle.dependency", exc)
            return None
        except Exception as exc:
            self.console.print(f"[loom.bad.b]could not start orchestrator:[/loom.bad.b] {exc}")
            telemetry.report("bundle.build", exc)
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
        self.weave.reset()
        # The gap before the first token is the one moment nothing is streaming.
        # The weave retires this the instant it has anything real to draw.
        working = render.Working(self.console, "weaving")
        self.weave.working = working
        working.start()
        try:
            final_text = self._stream(bundle.agent, inputs, run_config)
        except KeyboardInterrupt:
            self._interrupted = True
            self.weave.aside(
                Text(
                    "interrupted — partial work may have landed; /undo rolls back this turn's writes",
                    style="loom.warn",
                )
            )
        except Exception as exc:
            self.weave.aside(Text(f"streaming unavailable ({exc}); running synchronously…", style="loom.warn"))
            # Reported even though the run recovers: falling back to a
            # synchronous invoke hides a real provider/transport failure behind
            # a working turn, which is exactly how a regression here survives
            # for weeks.
            telemetry.report("turn.stream", exc)
            try:
                result = bundle.agent.invoke(inputs, config=run_config)
            except Exception as exc2:
                self.weave.aside(Text(f"model call failed: {exc2}", style="loom.bad.b"))
                telemetry.report("turn.invoke", exc2)
                return None
            final_text = self._absorb_result(result)
        finally:
            working.stop()
            self.weave.working = None
            undo.current_turn_id.set("")
            # Explicit end-of-turn marker: while it's absent, Loom is still
            # streaming — intermediate text is never the final answer.
            self.weave.close(self._receipt_text(), ok=not self._interrupted)
        return final_text

    def _receipt_text(self) -> Text:
        """The turn's receipt, with the money in the accent colour and
        everything free stated as free — the whole reason the fleet exists."""
        parts = [
            Text("turn interrupted" if self._interrupted else "turn complete", style="loom.muted")
        ]
        receipt = self.tracker.receipt(turn=True)
        if receipt:
            parts.append(Text(receipt, style="loom.muted"))
        return render.join(self.console, parts)

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
        self.console.print()
        render.rule(self.console, "plan ready", style="loom.warp")
        render.choices(
            self.console,
            [
                ("1", "execute it", "auto-accept edits"),
                ("2", "execute it", "approve each edit"),
                ("3", "keep planning", ""),
            ],
        )
        try:
            choice = render.ask(self.console, "", options=["1", "2", "3"], default="3")
        except (EOFError, KeyboardInterrupt):
            choice = "3"
        if choice not in ("1", "2"):
            render.note(self.console, "still in plan mode — /plan leaves it without executing")
            return
        self.set_mode("accept-edits" if choice == "1" else "default")
        render.note(self.console, f"plan approved — executing in [loom.warp]{self.mode}[/loom.warp] mode", kind="good")
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

        g = ink(self.console)
        if self.approval_mode == "default":
            render.note(
                self.console,
                "loop mode pauses on every approval — /mode accept-edits or /yolo makes it autonomous",
                kind="tip",
            )
        next_prompt = prompt + self.LOOP_NOTE
        for i in range(1, max_iters + 1):
            self.console.print()
            render.rule(self.console, f"{g.loop} loop {i}/{max_iters}", style="loom.line")
            text = self.run_turn(next_prompt) or ""
            if self._interrupted:
                render.note(self.console, "loop stopped (interrupted)", kind="warn")
                return
            if until:
                check = subprocess.run(until, shell=True, cwd=self.cwd, capture_output=True, text=True)
                if check.returncode == 0:
                    render.note(
                        self.console, f"loop done — `{until}` passed after {i} iteration(s)", kind="good"
                    )
                    return
                tail = (check.stdout + check.stderr)[-2000:]
                next_prompt = (
                    f"The check command `{until}` still fails (exit {check.returncode}):\n"
                    f"```\n{tail}\n```\nKeep fixing until it passes." + self.LOOP_NOTE
                )
                continue
            if "LOOP_COMPLETE" in text:
                render.note(
                    self.console, f"loop done — agent reported complete after {i} iteration(s)", kind="good"
                )
                return
            next_prompt = "Continue the loop task from where you left off." + self.LOOP_NOTE
        render.note(self.console, f"loop ended after {max_iters} iterations without completing", kind="warn")

    # ----- approval prompt with diff preview (Claude Code-style selector) -----
    def _confirm(self, tool_name: str, tool_input: dict, reason: str) -> "bool | tuple[bool, str]":
        if tool_name in self.session_allowed:
            return True
        with self._confirm_lock:
            return self._confirm_locked(tool_name, tool_input, reason)

    def _confirm_locked(self, tool_name: str, tool_input: dict, reason: str) -> "bool | tuple[bool, str]":
        if tool_name in self.session_allowed:  # approved while we waited on the lock
            return True
        self.weave.end_block()
        g = ink(self.console)
        head = Text(f"{g.pending} ", style="loom.warn.b")
        head.append(tool_name, style="loom.tool")
        body = [head]
        diff = self._diff_for(tool_name, tool_input or {})
        # With a diff to show, the raw old_string/new_string/content arguments
        # are the same bytes twice — the diff is the readable half.
        hidden = {"old_string", "new_string", "content"} if diff else set()
        for key, value in (tool_input or {}).items():
            if key in hidden:
                continue
            row = Text("  ")
            row.append(f"{key} ", style="loom.key")
            row.append(str(value)[:200].replace("\n", "↵"), style="loom.text")
            body.append(row)
        if diff:
            body.extend([Text(), diff])
        self.console.print()
        self.console.print(
            render.card(self.console, render.stack(*body), title="approve?", subtitle=reason, style="loom.warn")
        )
        render.choices(
            self.console,
            [
                ("1", "yes", ""),
                ("2", "yes, and don't ask again", f"{tool_name} {g.dot} this session"),
                ("3", "no", "and tell Loom what to do instead"),
            ],
        )
        try:
            choice = render.ask(self.console, "", options=["1", "2", "3"], default="1")
        except (EOFError, KeyboardInterrupt):
            return False
        if choice == "2":
            self.session_allowed.add(tool_name)
            return True
        if choice == "3":
            try:
                feedback = render.ask(self.console, "what should Loom do instead?", default="")
            except (EOFError, KeyboardInterrupt):
                feedback = ""
            return (False, feedback.strip())
        return True

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
        return render.diff(self.console, lines)

    # ----- rendering (Claude Code style: ⏺ bullets + ⎿ results) -----

    def model_origin(self, node: str) -> tuple[str, bool] | None:
        """(model string, is_local) for a role / stream-node name, accounting
        for what the role is *actually* running on this session: a role whose
        local model is unreachable may have been covered by another local model
        (still free) or, failing that, by the billed cloud fallback. None if
        unknown."""
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
        # The bundle's config carries the substituted local model; the settings
        # config still holds what the user asked for.
        stand_in = self._substituted_model(role)
        if stand_in is not None:
            return stand_in, True
        return model, cfg.is_local(model)

    def _substituted_model(self, role: str) -> str | None:
        """The local model actually serving ``role`` when its configured tag
        wasn't pulled, or None if the role runs as configured."""
        if role not in (getattr(self.bundle, "substitutions", None) or {}):
            return None
        active = getattr(self.bundle, "active_config", None)
        if active is None:
            return None
        return active.orchestrator if role == "orchestrator" else active.subagents.get(role)

    @staticmethod
    def _where_badge(is_local: bool) -> str:
        return "⌂ local" if is_local else "☁ cloud"

    def local_model_tags(self) -> list[str]:
        """Distinct local (Ollama) model tags assigned to any role, excluding
        roles currently live-fallen-back to the cloud — what's actually
        running on this machine right now."""
        from loom.core.model_router import resolve

        cfg = self.settings.models
        fallbacks = getattr(self.bundle, "fallbacks", None) or {}
        roles = {
            "orchestrator": cfg.orchestrator,
            "advisor": cfg.advisor,
            "escalation": cfg.escalation_model,
            **cfg.subagents,
        }
        tags: list[str] = []
        for role, model in roles.items():
            if role in fallbacks:
                continue
            # A substituted role runs on a local model too — just not the one
            # the config names, so report the tag that's actually loaded.
            model = self._substituted_model(role) or model
            if not cfg.is_local(model):
                continue
            tag = resolve(model).name
            if tag not in tags:
                tags.append(tag)
        return tags

    # Args worth showing bare — the one that says *what* the call is about.
    # Everything else reads better as "key: value".
    _HEADLINE_ARGS = ("path", "file_path", "command", "pattern", "query", "url", "subagent_type")

    @staticmethod
    def _call_args_brief(call) -> str:
        """A one-line gist of a tool call. The interesting argument — the path
        being read, the command being run — is shown bare, since prefixing it
        with ``path:`` costs width and says nothing."""
        args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {}) or {}
        if not isinstance(args, dict):
            return str(args)[:80]
        for key in Session._HEADLINE_ARGS:
            if args.get(key):
                return str(args[key]).splitlines()[0][:100]
        rest = [k for k in args if args[k] not in (None, "", [], {})]
        if not rest:
            return ""
        return f"{rest[0]}: {str(args[rest[0]]).splitlines()[0][:70]}"

    def _thread_for(self, source: str | None, node: str = "agent") -> Thread:
        """The weave thread a piece of output belongs to.

        Source labels arrive as ``"role · model (⌂ local)"`` (or just
        ``"model (☁ cloud)"`` when no role matched); the orchestrator is
        ``None``/``"orchestrator"``. Anything that isn't the orchestrator is a
        delegated thread and gets its own indented rail."""
        if source in (None, "orchestrator") and node in ("agent", "model", ""):
            origin = self.model_origin("orchestrator")
            model, is_local = origin if origin else ("", False)
            return self.weave.thread("orchestrator", model, is_local, depth=0)
        label = source or node
        role, _, rest = label.partition(" · ")
        model = rest
        if "(" in model:
            model = model[: model.index("(")].strip()
        if not rest:  # bare node name, e.g. a nested graph with no metadata
            role, model = label, ""
            origin = self.model_origin(label)
            if origin:
                model, _ = origin
        return self.weave.thread(role or label, model, "cloud" not in label, depth=1)

    def _print_tool_call(self, call, node: str, source: str | None = None) -> None:
        name = call.get("name", "?") if isinstance(call, dict) else getattr(call, "name", "?")
        args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {}) or {}
        thread = self._thread_for(source, node)
        # Delegation calls: name the model that will do the work and where it
        # runs, so a billed hand-off is visible at the moment it happens.
        target = None
        if name == "task" and isinstance(args, dict):
            target = args.get("subagent_type") or "general-purpose"
        elif name == "consult":
            target = "advisor"
        origin = self.model_origin(target) if target else None
        badge = render.model_badge(self.console, origin[0], origin[1]) if origin else None
        self.weave.tool_call(thread, name, self._call_args_brief(call), target=badge)
        # Inline diff for file edits — unless the approval prompt is about to
        # render the same diff anyway.
        if name in ("write_file", "edit_file") and isinstance(args, dict) and not self._will_prompt(name, args):
            diff = self._diff_for(name, args)
            if diff:
                self.weave.block(thread, diff)

    def _will_prompt(self, tool_name: str, tool_input: dict) -> bool:
        """True if this call is about to trigger the interactive approval
        prompt (which shows its own diff preview)."""
        from loom.core import permissions as perm_engine
        from loom.core.permissions import Decision

        if self.yolo or tool_name in self.session_allowed:
            return False
        if self.accept_edits and tool_name in ("write_file", "edit_file"):
            return False
        return perm_engine.check(tool_name, tool_input, self.settings.permissions) is Decision.ask

    def _print_tool_result(self, msg, source: str | None = None, node: str = "agent", nested: bool = False) -> None:
        """A tool result belongs to the thread that *called* the tool, not to
        the graph node that produced it — the node is always "tools". Depth
        comes from the stream namespace: a nested namespace means a subagent
        ran the tool inside its own graph."""
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            return
        lines = content.splitlines()
        bad = str(getattr(msg, "status", "")) == "error" or lines[0].lower().startswith("error")
        stack = self.weave._stack
        depth = 1 if nested else 0
        thread = stack[depth] if depth < len(stack) else self._thread_for(source, node)
        self.weave.tool_result(thread, lines[0][:120], extra_lines=len(lines) - 1, bad=bad)

    def _node_label(self, node: str) -> str:
        """Node name plus its ⌂ local / ☁ cloud badge when the model is known."""
        origin = self.model_origin(node)
        return f"{node} · {self._where_badge(origin[1])}" if origin else node

    def _print_assistant(self, text: str, node: str, source: str | None = None) -> None:
        thread = self._thread_for(source, node)
        self.weave.open(thread)
        self.weave.markdown(text)
        self.weave.end_block()

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

    @staticmethod
    def _chunk_parts(chunk) -> tuple[str, str]:
        """(text, reasoning) in a streamed chunk. Reasoning arrives as
        Anthropic-style "thinking" content blocks or as Ollama/OpenAI-compat
        ``additional_kwargs["reasoning_content"]``."""
        text: list[str] = []
        thinking: list[str] = []
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            text.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = part.get("type")
                if kind == "text":
                    text.append(part.get("text", ""))
                elif kind in ("thinking", "reasoning"):
                    thinking.append(str(part.get("thinking") or part.get("reasoning") or part.get("text") or ""))
        reasoning = (getattr(chunk, "additional_kwargs", None) or {}).get("reasoning_content")
        if reasoning:
            thinking.append(str(reasoning))
        return "".join(text), "".join(thinking)

    def _source_label(self, model: str, provider: str = "") -> str | None:
        """Label for an emitting model: None for the orchestrator itself,
        else "role · model (⌂ local / ☁ cloud)" — so subagent/advisor output
        is attributed, and billed cloud calls are distinguishable from free
        local ones."""
        from loom.core.model_router import resolve

        if not model:
            return None
        cfg = self.settings.models
        if model == resolve(cfg.orchestrator).name:
            return None
        # A role on Ollama fallback streams under the cloud-fallback model's
        # name, so it simply won't match here and shows as "<model> (☁ cloud)".
        roles = {"advisor": cfg.advisor, "escalation": cfg.escalation_model, **cfg.subagents}
        role = next((r for r, ms in roles.items() if resolve(ms).name == model), None)
        provider = (provider or "").lower()
        if provider:
            is_local = provider in ("ollama", "chatollama")
        elif role is not None:
            is_local = cfg.is_local(roles[role])
        else:
            # Ollama tags look like "qwen3:14b"; cloud names never carry a colon.
            is_local = ":" in model and not model.startswith("claude")
        where = "⌂ local" if is_local else "☁ cloud"
        return f"{role} · {model} ({where})" if role else f"{model} ({where})"

    def _note_task(self, call) -> None:
        """Record a ``task`` delegation so the subagent it spawns can be
        attributed to its declared ``subagent_type`` when it streams."""
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
        if name != "task":
            return
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
        self._pending_subagents.append((args or {}).get("subagent_type") or "general-purpose")

    def _attribute_ns(self, ns: tuple) -> str | None:
        """Map a nested delegation namespace to the subagent that owns it.

        A subagent streams under namespace ``("tools:<id>", ...)``; ``<id>`` is
        LangGraph's internal task id, not the ``task`` tool-call id, so it can't
        be matched back to the call directly. Instead we bind each namespace, on
        first sight, to the next un-bound ``subagent_type`` from ``_note_task``,
        in delegation order. This is authoritative even when the running model
        is ambiguous — e.g. a local role that fell back to the same cloud model
        another role (``reviewer``) already uses, which model-name matching
        would mislabel."""
        if not ns:
            return None
        key = ns[-1]
        if key not in self._ns_role and self._pending_subagents:
            self._ns_role[key] = self._pending_subagents.pop(0)
        return self._ns_role.get(key)

    def _role_label(self, role: str, model: str = "", provider: str = "") -> str:
        """``role · model (⌂ local / ☁ cloud)`` for a subagent identified by its
        namespace. Prefers the actually-running model name from the stream;
        falls back to the role's configured model (fallback-aware) when the
        update carries none."""
        from loom.core.model_router import resolve

        is_local: bool | None = None
        provider = (provider or "").lower()
        if provider:
            is_local = provider in ("ollama", "chatollama")
        if not model:
            origin = self.model_origin(role)
            if origin:
                model = resolve(origin[0]).name
                if is_local is None:
                    is_local = origin[1]
        if is_local is None:
            # Ollama tags look like "qwen3:14b"; cloud names never carry a colon.
            is_local = ":" in model and not model.startswith("claude")
        where = "⌂ local" if is_local else "☁ cloud"
        return f"{role} · {model or '?'} ({where})"

    def _stream_source(self, meta: dict | None) -> str | None:
        """Source label for a token chunk, from its callback metadata."""
        meta = meta or {}
        return self._source_label(str(meta.get("ls_model_name") or ""), str(meta.get("ls_provider") or ""))

    def _msg_source(self, msg, nested: bool) -> str:
        """Who produced an updates-mode AI message: "orchestrator", a
        "role · model (badge)" label, or "subagent" when a nested-graph
        message doesn't say which model it came from."""
        rmeta = getattr(msg, "response_metadata", None) or {}
        model = str(rmeta.get("model_name") or rmeta.get("model") or "")
        label = self._source_label(model)
        if label:
            return label
        if model:  # it IS the orchestrator model
            return "orchestrator"
        return "subagent" if nested else "orchestrator"

    def _stream(self, agent, inputs, run_config) -> str | None:
        """Token-level streaming; falls back to per-update rendering when the
        installed langgraph doesn't support multi-mode streams. Returns the
        final assistant text. ``subgraphs=True`` surfaces intermediate steps
        from nested graphs; messages-mode token streaming reaches every model
        call in the run tree (subagents and the advisor included) either way."""
        if not self.settings.ui.streaming:
            return self._stream_updates(agent.stream(inputs, config=run_config, stream_mode="updates"))
        for kwargs in ({"subgraphs": True}, {}):
            try:
                stream = agent.stream(inputs, config=run_config, stream_mode=["updates", "messages"], **kwargs)
                return self._stream_multi(stream)
            except (TypeError, ValueError):
                continue
        return self._stream_updates(agent.stream(inputs, config=run_config, stream_mode="updates"))

    def _stream_multi(self, stream) -> str | None:
        ui = self.settings.ui
        self._pending_subagents = []  # per-turn subagent attribution state
        self._ns_role = {}
        streamed: set[str] = set()  # finalized token-streamed texts (dedup vs updates)
        buf: list[str] = []
        open_key: tuple[str, str | None] | None = None  # (kind, source) of the open token block
        final_text = None

        def finish_block() -> None:
            nonlocal buf, open_key
            if open_key is not None:
                if open_key[0] == "text" and buf:
                    streamed.add("".join(buf).strip())
                buf = []
                open_key = None
            self.weave.end_block()

        def emit(kind: str, source: str | None, piece: str) -> None:
            """Append tokens to the current block, opening a new rail whenever
            the kind (text vs thinking) or the emitting model changes."""
            nonlocal open_key, buf
            if open_key != (kind, source):
                if open_key is not None and open_key[0] == "text" and buf:
                    streamed.add("".join(buf).strip())
                    buf = []
                open_key = (kind, source)
                self.weave.open(self._thread_for(source), kind)
            if kind == "text":
                buf.append(piece)
            self.weave.text(piece)

        for item in stream:
            if not isinstance(item, tuple):
                continue
            ns: tuple = ()
            if len(item) == 3:  # subgraphs=True → (namespace, mode, payload)
                ns, mode, payload = item
            elif len(item) == 2:
                mode, payload = item
            else:
                continue
            nested = bool(ns)
            if mode == "messages":
                chunk, meta = payload
                if type(chunk).__name__ != "AIMessageChunk":
                    continue
                text, thinking = self._chunk_parts(chunk)
                if not text and not thinking:
                    continue
                role = self._attribute_ns(ns)
                source = (
                    self._role_label(role, str(meta.get("ls_model_name") or ""), str(meta.get("ls_provider") or ""))
                    if role
                    else self._stream_source(meta)
                )
                if thinking and ui.show_thinking:
                    emit("thinking", source, thinking)
                if text:
                    emit("text", source, text)
                continue

            # updates mode — structure: tool calls, results, non-streamed text
            finish_block()
            for node, update in (payload or {}).items():
                msgs = (update or {}).get("messages") if isinstance(update, dict) else None
                if not msgs:
                    continue
                msg = msgs[-1]
                if getattr(msg, "type", "") == "tool":
                    # A `task` result is the delegated thread handing its
                    # summary back and dropping its context. Tie the rail off
                    # first, so the summary reads as arriving on the caller's
                    # rail after the subagent is done — which is what happened.
                    if getattr(msg, "name", "") == "task":
                        self._close_delegated_thread()
                    if ui.show_tool_calls:
                        role = self._attribute_ns(ns) if ns else None
                        self._print_tool_result(
                            msg,
                            source=self._role_label(role) if role else None,
                            node=node,
                            nested=nested,
                        )
                    continue
                calls = getattr(msg, "tool_calls", []) or []
                # Learn delegations before attributing, so a subagent's own
                # nested output binds to the right task in order.
                for call in calls:
                    self._note_task(call)
                role = self._attribute_ns(ns)
                if role:
                    rmeta = getattr(msg, "response_metadata", None) or {}
                    source = self._role_label(role, str(rmeta.get("model_name") or rmeta.get("model") or ""))
                else:
                    source = self._msg_source(msg, nested)
                for call in calls:
                    if ui.show_tool_calls:
                        self._print_tool_call(call, node, source=source)
                text = getattr(msg, "content", "")
                if text:
                    text = str(text) if isinstance(text, str) else self._chunk_text(msg)
                    if text.strip() and text.strip() not in streamed:
                        self._print_assistant(text, node, source=source)
                    # Nested-graph text is a subagent's answer, not the turn's.
                    if not nested:
                        final_text = text
        finish_block()
        if final_text is not None and not (self.bundle and self.bundle.persistent):
            self.messages.append(("assistant", final_text))
        return final_text

    def _close_delegated_thread(self) -> None:
        """Tie off the deepest open subagent rail, if one is open."""
        stack = self.weave._stack
        if len(stack) > 1:
            self.weave.close_thread(stack[-1], "context dropped, summary returned")

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


def _roster(session: Session):
    """The welcome card's fleet summary: who is driving, who advises, and what
    the subagents are running on.

    The subagent roles collapse into one ``fleet`` row — a session has seven of
    them sharing three model tags, and listing all seven turns the card into a
    config dump. Fallback-aware, so a role the cloud is covering because Ollama
    is down is counted as cloud, not quietly listed as free."""
    from loom.core.model_router import resolve

    cfg = session.settings.models
    rows = []
    for role in ("orchestrator", "advisor"):
        origin = session.model_origin(role)
        if origin:
            rows.append((role, resolve(origin[0]).name, origin[1], ""))

    tags, billed = [], 0
    for role in sorted(cfg.subagents):
        origin = session.model_origin(role)
        if origin is None:
            continue
        model, is_local = resolve(origin[0]).name, origin[1]
        if not is_local:
            billed += 1
        elif model not in tags:
            tags.append(model)
    if tags:
        note = f"{len(cfg.subagents)} roles" + (f", {billed} on cloud" if billed else "")
        rows.append(("fleet", " · ".join(tags), True, note))
    elif cfg.subagents:
        rows.append(("fleet", cfg.cloud_fallback, False, f"{len(cfg.subagents)} roles, no local models"))
    return rows


def _banner(session: Session):
    from loom import __version__

    ui = session.settings.ui
    home = str(Path.home())
    cwd = str(session.cwd)
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home) :]
    body = banner_mod.welcome(
        session.console,
        version=__version__,
        roles=[] if ui.compact else _roster(session),
        cwd=cwd,
        tagline="hybrid local/cloud agent fleet",
    )
    return render.card(session.console, body, title="loom")


# The status line lives under the input box: what is running, in what mode,
# and what it has cost. prompt_toolkit renders it, so it carries its own
# style names (see loom.ui.prompt).
def _toolbar_state(session: Session) -> dict:
    from loom.core.model_router import resolve

    model, is_local = session.model_origin("orchestrator")
    cfg = session.settings.models
    local = [t for t in session.local_model_tags() if not (is_local and t == resolve(cfg.orchestrator).name)]
    modes = [session.approval_mode] if session.approval_mode != "default" else []
    for name, on in (("plan", session.plan), ("local", session.local_only), ("airgap", session.airgap), ("vim", session.vim)):
        if on:
            modes.append(name)
    return {
        "model": model,
        "is_local": is_local,
        "local_tags": local,
        "modes": modes or ["default"],
        "cost": session.tracker.session.cloud_cost,
        "estimated": session.tracker.session.has_estimates(),
        "local_share": session.tracker.session.local_share(),
        "turns": session.tracker.turns,
    }


def _toolbar(session: Session):
    """The status line as prompt_toolkit fragments."""
    from loom.ui import prompt as prompt_mod
    from loom.ui.theme import active_theme_name, theme_of

    name = active_theme_name(session.settings.ui)
    if not theme_of(session.console).unicode:
        name = "ascii"
    return prompt_mod.status_line(_toolbar_state(session), theme_name=name)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _setup_hint(session: Session) -> None:
    """First-run guidance when the configured roles have no way to run.

    Judged against what this config actually routes to, and against the keys
    Loom can really see — the wizard writes them to settings.json's env block,
    so checking os.environ alone calls a working setup broken.
    """
    import os

    from loom.core import providers

    config = session.settings.models
    routed = providers.routed_providers(
        [*config.all_models().values(), config.escalation_model, config.cloud_fallback]
    )

    def have(key: str) -> bool:
        return bool(os.environ.get(key) or session.settings.env.get(key))

    missing = []
    for provider_id in routed:
        keys = providers.credential_keys(provider_id)
        if not keys:  # needs no key at all (e.g. Vertex ADC)
            continue
        if any(have(k) for k in keys):
            return  # something can run — not a first-run dead end
        missing.append((providers.get(provider_id).label, keys[0]))

    try:
        from loom.core import ollama

        if ollama.status(config).running:
            return
    except Exception:
        pass
    if not routed:  # all-local config, daemon down
        render.note(session.console, "no Ollama daemon and no cloud roles — tasks will fail", kind="warn")
        rows = [("local", "[loom.muted]install Ollama (https://ollama.com), then `loom models pull`[/loom.muted]")]
    elif not missing:
        return  # every routed provider is credentialed
    else:
        render.note(session.console, "no key for the models you have configured — tasks will fail", kind="warn")
        rows = [(label, f"[loom.muted]export {key}=…[/loom.muted]") for label, key in missing]
    session.console.print(render.kv(rows + [("check", "[loom.muted]/doctor[/loom.muted]")]))


def _maybe_run_onboarding(session: Session) -> None:
    """True first run (no settings.json anywhere yet): offer the setup wizard
    instead of silently falling back to packaged defaults. Falls back to the
    passive `_setup_hint` if the user declines or it's not a first run."""
    from loom.ui import onboarding

    if not onboarding.needs_onboarding(session.cwd):
        _setup_hint(session)
        _maybe_ask_privacy(session)
        return
    render.note(
        session.console,
        "no settings.json yet — setup picks your models and asks what Loom may share "
        "([loom.warp]/setup[/loom.warp] to redo this later)",
    )
    # Asked, not assumed. Someone who just wants to try one prompt against the
    # packaged defaults should be able to say no and get a prompt, and the
    # answer isn't recorded — the offer returns on the next start, which is
    # what "runs on first start after installing" has to mean for anyone who
    # skipped it the first time.
    try:
        if not render.confirm(session.console, "  run setup now?", default=True):
            render.note(session.console, "skipped — /setup any time")
            _setup_hint(session)
            return
        settings = onboarding.run(session.console, root=session.cwd)
        onboarding.maybe_setup_playwright(session.console, settings)
    except (KeyboardInterrupt, EOFError):
        render.note(session.console, "setup skipped — /setup any time to configure models")
        return
    session.reload_settings()
    session.rebuild()


def _maybe_ask_privacy(session: Session) -> None:
    """Catch the two cases the setup wizard doesn't cover.

    An install that predates privacy modes has a settings.json, so it is not a
    "first run" by any measure the wizard uses — but it has never been asked
    this question, and defaulting someone into a data-sharing answer they were
    never shown is exactly the thing this feature exists to avoid. Separately,
    a machine that has answered globally still has to be asked about each new
    project it opens.
    """
    from loom.core import telemetry as tel
    from loom.ui import privacy as privacy_mod

    try:
        if tel.needs_decision():
            privacy_mod.run(session.console, session.cwd)
        else:
            privacy_mod.maybe_ask_project(session.console, session.cwd)
    except (KeyboardInterrupt, EOFError):
        render.note(session.console, "skipped — nothing is shared until you run /privacy")
    except Exception as exc:
        # Never let a consent prompt stop Loom from starting — but never let it
        # fail silently either. A swallowed failure here leaves the user at
        # "none" forever while they believe they answered, which is exactly how
        # a machine ends up sending nothing and nobody knows why.
        render.note(
            session.console,
            f"couldn't ask about privacy ({type(exc).__name__}) — run [loom.warp]/privacy[/loom.warp] to set it",
            kind="warn",
        )


def _activate_telemetry(session: Session) -> None:
    """Switch on whatever this project consented to, and say so once.

    The line matters: a tool that is uploading your prompts should say it is
    uploading your prompts, every session, not once at setup and never again.
    """
    try:
        mode = telemetry.activate(session.cwd)
    except Exception as exc:
        render.note(
            session.console,
            f"telemetry failed to start ({type(exc).__name__}) — nothing will be reported this session",
            kind="warn",
        )
        return
    if mode == "none":
        return
    info = telemetry.mode_info(mode)
    render.note(
        session.console,
        f"privacy [loom.warp]{info.label}[/loom.warp] {ink(session.console).dot} {info.blurb} "
        f"[loom.muted](/privacy)[/loom.muted]",
        kind="warn" if mode == "full" else "muted",
    )


def run(settings: Settings, cwd: str = ".", *, plan=False, local_only=False, yolo=False, airgap=False) -> None:
    session = Session(settings, cwd, plan=plan, local_only=local_only, yolo=yolo, airgap=airgap)
    if settings.ui.banner:
        session.console.print(_banner(session))
        session.console.print(
            banner_mod.hint_line(session.console, ["/help", "/status", "shift+tab modes", "ctrl+c interrupt"])
        )
    _maybe_run_onboarding(session)
    # After onboarding, never before: the wizard is where consent is given, and
    # activating a reporter that the user is one prompt away from declining
    # would be the one bug this feature cannot afford.
    _activate_telemetry(session)

    prompt_session = _make_prompt_session(session)
    session._prompt_session = prompt_session

    while True:
        try:
            session.console.print()
            line = _read_line(prompt_session, session)
        except (EOFError, KeyboardInterrupt):
            render.note(session.console, "bye")
            break

        line = (line or "").strip()
        if not line:
            continue
        if line.startswith("/"):
            try:
                if not slash.dispatch(session, line):
                    break
            except KeyboardInterrupt:
                render.note(session.console, "interrupted", kind="warn")
            except Exception as exc:
                _report_crash(session, exc, f"/{line[1:].split()[0] if len(line) > 1 else ''}")
            continue

        try:
            reply = session.run_turn(line)
        except KeyboardInterrupt:
            render.note(session.console, "interrupted", kind="warn")
            continue
        except Exception as exc:
            _report_crash(session, exc, "turn")
            continue
        if session.plan and reply and not session._interrupted:
            session.offer_plan_execution()

    telemetry.flush()


def _report_crash(session: Session, exc: BaseException, where: str) -> None:
    """Show a bug the way a user needs to see it, and report it if allowed.

    A crash inside one command should not end the session — the transcript is
    usually the most valuable thing in the room when something breaks.
    """
    render.note(session.console, f"{where} failed: {type(exc).__name__}: {exc}", kind="bad")
    telemetry.capture_exception(exc, where)
    if telemetry.current_mode() == "none":
        from loom.core import update as update_mod

        render.note(
            session.console,
            f"report it at https://github.com/{update_mod.REPO}/issues "
            "[loom.muted](or /privacy to send crashes automatically)[/loom.muted]",
            kind="tip",
        )


def _make_prompt_session(session: Session | None = None):
    from loom.ui import prompt as prompt_mod

    try:
        return prompt_mod.make_prompt_session(session)
    except Exception:
        return None  # fall back to builtin input()


def _read_line(prompt_session, session: Session) -> str:
    from loom.ui import prompt as prompt_mod
    from loom.ui.theme import active_theme_name, theme_of

    symbol = session.settings.ui.prompt_symbol
    if prompt_session is None:
        return input(f"{symbol or ink(session.console).prompt} ")
    name = active_theme_name(session.settings.ui)
    if not theme_of(session.console).unicode:
        name = "ascii"
    return prompt_session.prompt(
        prompt_mod.caret(name, symbol),
        bottom_toolbar=lambda: _toolbar(session),
    )
