"""Hooks engine — run user shell commands around tool events.

Mirrors Claude Code hooks: a matched command runs at an event
(``pre_tool_use`` / ``post_tool_use`` / ``user_prompt_submit`` / ``stop``),
receives a JSON event on stdin plus ``LOOM_*`` env vars, and can BLOCK by
exiting non-zero — the tool for ``pre_tool_use``, the whole turn for
``user_prompt_submit``.

The two tool events are run by :class:`loom.middleware.policy.PolicyMiddleware`
around every tool call; the two turn events by ``Session.run_turn``, once per
turn. A turn event has no tool, so only hooks with a ``*`` (or empty) matcher
run for it.
"""

from __future__ import annotations

import fnmatch
import json
import os
import subprocess
from dataclasses import dataclass, field

from loom.core.settings import Hook, Hooks


def _matcher_matches(matcher: str, tool_name: str) -> bool:
    # Support "a|b|c" alternation of globs, plus plain globs and "*".
    for part in matcher.split("|"):
        if fnmatch.fnmatch(tool_name, part.strip()):
            return True
    return False


# Events where a non-zero exit stops what was about to happen.
_BLOCKING_EVENTS = frozenset({"pre_tool_use", "user_prompt_submit"})


@dataclass
class HookOutcome:
    blocked: bool = False
    messages: list[str] = field(default_factory=list)  # stdout/notes to surface
    block_reason: str = ""


def _run_one(hook: Hook, event: str, payload: dict, cwd: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["LOOM_EVENT"] = event
    env["LOOM_TOOL_NAME"] = str(payload.get("tool", ""))
    env["LOOM_TOOL_INPUT"] = json.dumps(payload.get("input", {}))
    try:
        proc = subprocess.run(
            hook.command,
            shell=True,
            cwd=cwd,
            input=json.dumps({"event": event, **payload}),
            capture_output=True,
            text=True,
            timeout=hook.timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return 1, f"hook timed out after {hook.timeout}s: {hook.command}"
    except OSError as exc:
        # The shell could not be started at all (cwd gone, fd limit). Reported
        # as a failed hook rather than raised: a pre-hook that cannot run must
        # block, not crash the tool call it was guarding.
        return 1, f"hook could not run ({exc.strerror or exc}): {hook.command}"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    return proc.returncode, "\n".join(filter(None, [out, err]))


def run_event(
    hooks_for_event: list[Hook],
    event: str,
    *,
    tool_name: str = "",
    tool_input: dict | None = None,
    cwd: str = ".",
    extra: dict | None = None,
) -> HookOutcome:
    """Run every hook whose matcher matches ``tool_name`` for ``event``.

    ``extra`` is merged into the JSON event on stdin — the prompt for
    ``user_prompt_submit``, the turn's status for ``stop``.
    """
    outcome = HookOutcome()
    payload = {"tool": tool_name, "input": tool_input or {}, "cwd": cwd, **(extra or {})}
    for hook in hooks_for_event:
        if tool_name and not _matcher_matches(hook.matcher, tool_name):
            continue
        if not tool_name and hook.matcher not in ("*", ""):
            # Non-tool events (user_prompt_submit/stop) only run "*" matchers.
            continue
        code, text = _run_one(hook, event, payload, cwd)
        if text:
            outcome.messages.append(text)
        if code != 0 and event in _BLOCKING_EVENTS:
            outcome.blocked = True
            outcome.block_reason = text or f"blocked by hook: {hook.command}"
            break
    return outcome


def pre_tool_use(hooks: Hooks, tool_name: str, tool_input: dict, cwd: str = ".") -> HookOutcome:
    return run_event(hooks.pre_tool_use, "pre_tool_use", tool_name=tool_name, tool_input=tool_input, cwd=cwd)


def post_tool_use(hooks: Hooks, tool_name: str, tool_input: dict, cwd: str = ".") -> HookOutcome:
    return run_event(hooks.post_tool_use, "post_tool_use", tool_name=tool_name, tool_input=tool_input, cwd=cwd)


def user_prompt_submit(hooks: Hooks, prompt: str, cwd: str = ".") -> HookOutcome:
    """Before a turn. A hook exiting non-zero refuses the prompt."""
    return run_event(hooks.user_prompt_submit, "user_prompt_submit", cwd=cwd, extra={"prompt": prompt})


def stop(hooks: Hooks, status: str, cwd: str = ".") -> HookOutcome:
    """After a turn; ``status`` is ``complete``, ``failed`` or ``interrupted``.
    Informational — it cannot block, the turn is already over."""
    return run_event(hooks.stop, "stop", cwd=cwd, extra={"status": status})
