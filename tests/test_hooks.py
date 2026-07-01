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
