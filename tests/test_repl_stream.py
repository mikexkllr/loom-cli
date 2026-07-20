"""REPL streaming internals: reasoning extraction, per-model attribution,
inline edit diffs, and the Claude Code-style approval selector."""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")
pytest.importorskip("rich")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessageChunk

from loom.core import settings as st
from loom.core.model_router import resolve
from loom.ui.repl import Session


def _session(tmp_path):
    return Session(st.load_settings(tmp_path), cwd=str(tmp_path))


# ------------------------------------------------------------ chunk parsing


def test_chunk_parts_plain_text():
    text, thinking = Session._chunk_parts(AIMessageChunk(content="hello"))
    assert text == "hello"
    assert thinking == ""


def test_chunk_parts_anthropic_thinking_blocks():
    chunk = AIMessageChunk(
        content=[
            {"type": "thinking", "thinking": "let me reason"},
            {"type": "text", "text": "the answer"},
        ]
    )
    text, thinking = Session._chunk_parts(chunk)
    assert text == "the answer"
    assert thinking == "let me reason"


def test_chunk_parts_ollama_reasoning_content():
    chunk = AIMessageChunk(content="out", additional_kwargs={"reasoning_content": "hmm"})
    text, thinking = Session._chunk_parts(chunk)
    assert text == "out"
    assert thinking == "hmm"


# ------------------------------------------------------------ attribution


def test_stream_source_orchestrator_is_unlabeled(tmp_path):
    s = _session(tmp_path)
    name = resolve(s.settings.models.orchestrator).name
    assert s._stream_source({"ls_model_name": name}) is None
    assert s._stream_source(None) is None
    assert s._stream_source({}) is None


def test_stream_source_labels_subagent_with_badge(tmp_path):
    s = _session(tmp_path)
    cfg = s.settings.models
    role, model_string = next(iter(cfg.subagents.items()))
    name = resolve(model_string).name
    provider = "ollama" if cfg.is_local(model_string) else "anthropic"
    label = s._stream_source({"ls_model_name": name, "ls_provider": provider})
    assert label is not None
    assert name in label
    assert ("⌂ local" in label) == cfg.is_local(model_string)


def test_stream_source_unknown_model_shows_model_and_cloud(tmp_path):
    s = _session(tmp_path)
    label = s._stream_source({"ls_model_name": "mystery-9000", "ls_provider": "anthropic"})
    assert label == "mystery-9000 (☁ cloud)"


# ------------------------------------------------------------ inline diffs


def test_will_prompt_follows_modes_and_session_allow(tmp_path):
    s = _session(tmp_path)
    args = {"path": "a.txt", "old_string": "x", "new_string": "y"}
    assert s._will_prompt("edit_file", args) is True  # default mode asks
    s.accept_edits = True
    assert s._will_prompt("edit_file", args) is False
    s.accept_edits = False
    s.yolo = True
    assert s._will_prompt("edit_file", args) is False
    s.yolo = False
    s.session_allowed.add("edit_file")
    assert s._will_prompt("edit_file", args) is False


def test_print_tool_call_renders_diff_when_not_prompting(tmp_path, capsys):
    (tmp_path / "a.txt").write_text("old line\n")
    s = _session(tmp_path)
    s.yolo = True  # no approval prompt → diff renders inline
    call = {"name": "edit_file", "args": {"path": "a.txt", "old_string": "old line", "new_string": "new line"}}
    s._print_tool_call(call, "model")
    out = capsys.readouterr().out
    assert "edit_file" in out
    assert "+new line" in out and "-old line" in out


# ------------------------------------------------------------ approval selector


def _scripted_prompt(monkeypatch, answers):
    seq = list(answers)
    monkeypatch.setattr("rich.prompt.Prompt.ask", staticmethod(lambda *a, **k: seq.pop(0)))


def test_confirm_yes(tmp_path, monkeypatch):
    s = _session(tmp_path)
    _scripted_prompt(monkeypatch, ["1"])
    assert s._confirm("execute", {"command": "ls"}, "requires approval") is True


def test_confirm_dont_ask_again_persists_for_session(tmp_path, monkeypatch):
    s = _session(tmp_path)
    _scripted_prompt(monkeypatch, ["2"])
    assert s._confirm("execute", {"command": "ls"}, "requires approval") is True
    assert "execute" in s.session_allowed
    # Second call short-circuits without prompting (no scripted answers left).
    assert s._confirm("execute", {"command": "rm x"}, "requires approval") is True


def test_confirm_decline_with_feedback(tmp_path, monkeypatch):
    s = _session(tmp_path)
    _scripted_prompt(monkeypatch, ["3", "use pathlib instead"])
    result = s._confirm("execute", {"command": "sed -i"}, "requires approval")
    assert result == (False, "use pathlib instead")
