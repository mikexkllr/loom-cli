"""LOOM_USAGE_LOG: the per-call record a receipt is summed from."""

import json
import uuid

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langchain_core")

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from loom.core.config import LoomConfig
from loom.core.usage import USAGE_LOG_ENV, UsageTracker


class _Msg:
    def __init__(self, inp, out, model, cache_read=0):
        self.usage_metadata = {
            "input_tokens": inp,
            "output_tokens": out,
            "input_token_details": {"cache_read": cache_read},
        }
        self.response_metadata = {"model_name": model}


class _Result:
    def __init__(self, message):
        self.generations = [[type("G", (), {"message": message})()]]


def _tracker():
    return UsageTracker(LoomConfig(orchestrator="claude-sonnet-4-6", subagents={}))


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_no_log_unless_the_variable_is_set(tmp_path, monkeypatch):
    monkeypatch.delenv(USAGE_LOG_ENV, raising=False)
    t = _tracker()
    t.start_turn()
    t.on_llm_end(_Result(_Msg(100, 10, "claude-sonnet-4-6")), run_id=uuid.uuid4())
    assert list(tmp_path.iterdir()) == []


def test_one_line_per_call_with_role_thread_and_cache(tmp_path, monkeypatch):
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv(USAGE_LOG_ENV, str(log))
    t = _tracker()
    t.start_turn()

    t.on_llm_end(_Result(_Msg(1_000, 50, "claude-sonnet-4-6", cache_read=800)), run_id=uuid.uuid4())
    task, llm = uuid.uuid4(), uuid.uuid4()
    t.on_tool_start({"name": "task"}, "", run_id=task, parent_run_id=None,
                    inputs={"subagent_type": "explorer"})
    t.on_chat_model_start({}, [], run_id=llm, parent_run_id=task, metadata={})
    t.on_llm_end(_Result(_Msg(9_000, 300, "claude-sonnet-4-6")), run_id=llm)
    summary = ToolMessage(content="retry.py:88, max 5", tool_call_id="1")
    t.on_tool_end(Command(update={"messages": [summary]}), run_id=task)

    turn, main_call, tool, sub_call, result = _lines(log)
    assert turn["event"] == "turn" and turn["turn"] == 1
    assert main_call == {**main_call, "event": "llm", "role": "orchestrator", "thread": "main",
                         "input": 1_000, "cache_read": 800, "output": 50}
    assert tool == {**tool, "event": "tool", "name": "task", "role": "orchestrator"}
    assert sub_call["role"] == "explorer" and sub_call["thread"] == str(task)
    assert result == {**result, "event": "result", "role": "explorer", "thread": str(task),
                      "chars": len("retry.py:88, max 5")}


def test_a_subagents_own_tool_calls_name_the_subagent(tmp_path, monkeypatch):
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv(USAGE_LOG_ENV, str(log))
    t = _tracker()
    t.start_turn()
    task, grep = uuid.uuid4(), uuid.uuid4()
    t.on_tool_start({"name": "task"}, "", run_id=task, parent_run_id=None,
                    inputs={"subagent_type": "bash"})
    t.on_tool_start({"name": "grep"}, "", run_id=grep, parent_run_id=task)
    t.on_tool_end("3 matches", run_id=grep)  # not a delegation: nothing logged

    events = [(r["event"], r.get("name"), r.get("role")) for r in _lines(log)]
    assert events == [("turn", None, None), ("tool", "task", "orchestrator"), ("tool", "grep", "bash")]


def test_an_unwritable_log_never_breaks_accounting(tmp_path, monkeypatch):
    monkeypatch.setenv(USAGE_LOG_ENV, str(tmp_path / "missing-dir" / "usage.jsonl"))
    t = _tracker()
    t.start_turn()
    t.on_llm_end(_Result(_Msg(100, 10, "claude-sonnet-4-6")), run_id=uuid.uuid4())
    assert t.turn.orchestrator_tokens() == 110
