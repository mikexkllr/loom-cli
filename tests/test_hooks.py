"""Hook matching + execution (pre_tool_use can block)."""

import pytest

pytest.importorskip("pydantic")

from loom.core.hooks import post_tool_use, pre_tool_use
from loom.core.settings import Hook, Hooks


def test_matcher_matches_and_runs(tmp_path):
    hooks = Hooks(post_tool_use=[Hook(matcher="write_file", command="echo hooked")])
    outcome = post_tool_use(hooks, "write_file", {"path": "a"}, cwd=str(tmp_path))
    assert not outcome.blocked
    assert any("hooked" in m for m in outcome.messages)


def test_matcher_skips_non_matching(tmp_path):
    hooks = Hooks(post_tool_use=[Hook(matcher="execute", command="echo nope")])
    outcome = post_tool_use(hooks, "write_file", {"path": "a"}, cwd=str(tmp_path))
    assert outcome.messages == []


def test_alternation_matcher(tmp_path):
    hooks = Hooks(post_tool_use=[Hook(matcher="write_file|edit_file", command="echo ok")])
    assert post_tool_use(hooks, "edit_file", {}, cwd=str(tmp_path)).messages


def test_pre_tool_use_nonzero_blocks(tmp_path):
    hooks = Hooks(pre_tool_use=[Hook(matcher="*", command="exit 2")])
    outcome = pre_tool_use(hooks, "execute", {"command": "danger"}, cwd=str(tmp_path))
    assert outcome.blocked is True


def test_pre_tool_use_zero_allows(tmp_path):
    hooks = Hooks(pre_tool_use=[Hook(matcher="*", command="true")])
    outcome = pre_tool_use(hooks, "execute", {"command": "ok"}, cwd=str(tmp_path))
    assert outcome.blocked is False


def test_hook_receives_tool_name_in_env(tmp_path):
    hooks = Hooks(post_tool_use=[Hook(matcher="*", command="echo $LOOM_TOOL_NAME")])
    outcome = post_tool_use(hooks, "write_file", {"path": "a"}, cwd=str(tmp_path))
    assert any("write_file" in m for m in outcome.messages)


# ------------------------------------------------------------ turn events


def test_user_prompt_submit_gets_the_prompt_and_can_refuse_it(tmp_path):
    """Configurable, listed by /hooks, documented — and never run by anything."""
    from loom.core.hooks import user_prompt_submit

    seen = tmp_path / "seen.json"
    hooks = Hooks(user_prompt_submit=[Hook(command=f"cat > {seen}")])
    outcome = user_prompt_submit(hooks, "refactor the auth module", cwd=str(tmp_path))
    assert not outcome.blocked
    import json

    event = json.loads(seen.read_text())
    assert event["event"] == "user_prompt_submit"
    assert event["prompt"] == "refactor the auth module"

    refusing = Hooks(user_prompt_submit=[Hook(command="echo 'no deploys on friday' >&2; exit 2")])
    outcome = user_prompt_submit(refusing, "deploy", cwd=str(tmp_path))
    assert outcome.blocked and "no deploys on friday" in outcome.block_reason


def test_stop_reports_the_status_and_never_blocks(tmp_path):
    from loom.core.hooks import stop

    hooks = Hooks(stop=[Hook(command="echo status=$(python3 -c 'import json,sys; print(json.load(sys.stdin)[\"status\"])'); exit 1")])
    outcome = stop(hooks, "interrupted", cwd=str(tmp_path))
    assert not outcome.blocked
    assert any("status=interrupted" in m for m in outcome.messages)


def test_a_turn_event_only_runs_hooks_that_match_everything(tmp_path):
    from loom.core.hooks import user_prompt_submit

    hooks = Hooks(user_prompt_submit=[Hook(matcher="execute", command="exit 1")])
    assert not user_prompt_submit(hooks, "hi", cwd=str(tmp_path)).blocked


def test_a_hook_that_cannot_be_started_blocks_instead_of_raising(tmp_path):
    hooks = Hooks(pre_tool_use=[Hook(matcher="*", command="true")])
    outcome = pre_tool_use(hooks, "execute", {"command": "ls"}, cwd=str(tmp_path / "gone"))
    assert outcome.blocked and "could not run" in outcome.block_reason
