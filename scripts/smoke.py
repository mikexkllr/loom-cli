#!/usr/bin/env python3
"""Loom wiring smoke test — NO API calls, NO Ollama needed.

Exercises the parts that touch the deepagents/LangChain runtime surfaces
(PolicyMiddleware.wrap_tool_call, PromptSizeGuard escalation decision, the REPL
stream/turn loop, permission + hook enforcement) using a STUB agent, so you can
confirm the integration is sound right after `pip install -e .` without spending
tokens or standing up local models.

    python scripts/smoke.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_results: list[bool] = []


def check(name: str, ok: bool) -> None:
    _results.append(ok)
    print(f"  {PASS if ok else FAIL}  {name}")


# ---------------------------------------------------------------------------
# 1. Settings load + merge
# ---------------------------------------------------------------------------
def test_settings() -> None:
    print("settings:")
    from loom.core import settings as st

    s = st.load_settings()
    check("defaults load", bool(s.models.orchestrator))
    check("permissions present", "read_file" in s.permissions.allow)
    check("ui defaults", bool(s.ui.prompt_symbol))


# ---------------------------------------------------------------------------
# 2. Permission engine + PolicyMiddleware gating a stub tool call
# ---------------------------------------------------------------------------
def test_policy_middleware() -> None:
    print("policy middleware:")
    from loom.core.settings import Hook, Hooks, Permissions, Settings
    from loom.middleware import policy
    from loom.middleware.policy import PolicyMiddleware

    settings = Settings(
        permissions=Permissions(default_mode="ask", allow=["read_file"], deny=["execute(rm -rf*)"]),
        hooks=Hooks(pre_tool_use=[Hook(matcher="write_file", command="exit 3")]),
    )
    mw = PolicyMiddleware(settings, cwd=".")

    def handler(_req):  # would run the real tool; must NOT be reached when blocked
        return SimpleNamespace(content="TOOL RAN", executed=True)

    def req(name, args):
        return SimpleNamespace(call={"name": name, "args": args, "id": f"id-{name}"})

    # allow -> handler runs
    allowed = mw.wrap_tool_call(req("read_file", {"path": "x"}), handler)
    check("allow lets tool run", getattr(allowed, "executed", False) is True)

    # deny -> short-circuit ToolMessage, handler NOT run
    denied = mw.wrap_tool_call(req("execute", {"command": "rm -rf /tmp/x"}), handler)
    check("deny short-circuits", "policy" in str(getattr(denied, "content", denied)).lower())

    # ask + auto_approve on -> runs; auto_approve off + confirm False -> blocked
    policy.auto_approve.set(True)
    asked = mw.wrap_tool_call(req("edit_file", {"path": "a"}), handler)
    check("ask + yolo runs", getattr(asked, "executed", False) is True)
    policy.auto_approve.set(False)
    policy.confirm_callback.set(lambda n, i, r: False)
    declined = mw.wrap_tool_call(req("edit_file", {"path": "a"}), handler)
    check("ask + decline blocks", getattr(declined, "executed", False) is not True)

    # pre_tool_use hook exiting non-zero blocks write_file (allowed by default ask+yolo)
    policy.auto_approve.set(True)
    hooked = mw.wrap_tool_call(req("write_file", {"path": "a", "content": "x"}), handler)
    check("pre-hook non-zero blocks", "hook" in str(getattr(hooked, "content", hooked)).lower())
    policy.auto_approve.set(False)


# ---------------------------------------------------------------------------
# 3. PromptSizeGuard escalation decision
# ---------------------------------------------------------------------------
def test_prompt_size_guard() -> None:
    print("prompt-size guard:")
    from loom.core import settings as st
    from loom.middleware.prompt_size_guard import PromptSizeGuard

    s = st.load_settings()
    guard = PromptSizeGuard("ollama/qwen3:4b", s.models)
    window = s.models.context_window_for("ollama/qwen3:4b")

    small = SimpleNamespace(messages=[SimpleNamespace(content="hi")], system_prompt="")
    big_text = "x" * (window * 5)  # ~window*1.25 tokens, over the escalate threshold
    big = SimpleNamespace(messages=[SimpleNamespace(content=big_text)], system_prompt="")

    # _maybe_escalate should leave small request's model untouched (no .override called)
    guard._maybe_escalate(small)
    check("small prompt: no escalation", guard.escalation_count == 0)

    # For the big one, build_model may fail (no API key) — that's fine; we only
    # assert the escalation was *attempted* (counter increments before build).
    try:
        guard._maybe_escalate(big)
    except Exception:
        pass
    check("large prompt: escalation attempted", guard.escalation_count == 1)


# ---------------------------------------------------------------------------
# 4. REPL turn loop against a stub agent (no model)
# ---------------------------------------------------------------------------
def test_repl_turn() -> None:
    print("repl turn loop:")
    from loom.core import settings as st
    from loom.ui.repl import Session

    session = Session(st.load_settings(), cwd=".")

    class StubAgent:
        def stream(self, inputs, **kw):
            # Mimic deepagents `updates` stream: {node: {"messages": [msg]}}
            yield {"explorer": {"messages": [SimpleNamespace(content="looked around", tool_calls=[{"name": "ls"}])]}}
            yield {"agent": {"messages": [SimpleNamespace(content="Done: added the endpoint.", tool_calls=[])]}}

    # Inject a stub bundle so ensure_bundle() doesn't build a real orchestrator.
    session.bundle = SimpleNamespace(agent=StubAgent(), persistent=False, mode="normal")
    session.run_turn("add a health endpoint")

    transcript = "".join(str(m) for m in session.messages)
    check("assistant reply captured", "added the endpoint" in transcript)
    check("slash dispatch works", _slash_ok(session))


def _slash_ok(session) -> bool:
    from loom.ui import slash

    cont = slash.dispatch(session, "/help")  # should render and return True
    exit_signal = slash.dispatch(session, "/exit")  # should return False
    return cont is True and exit_signal is False


# ---------------------------------------------------------------------------
def main() -> int:
    print("\nLoom smoke test (no API calls)\n" + "-" * 32)
    for test in (test_settings, test_policy_middleware, test_prompt_size_guard, test_repl_turn):
        try:
            test()
        except Exception as exc:  # a crash is itself a failure
            import traceback

            traceback.print_exc()
            check(f"{test.__name__} raised {type(exc).__name__}", False)
    ok = all(_results)
    print("-" * 32)
    print(f"{sum(_results)}/{len(_results)} checks passed — {'ALL GOOD' if ok else 'FAILURES ABOVE'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
