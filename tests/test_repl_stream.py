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


# ------------------------------------------------------------ local models in banner/toolbar


def test_local_model_tags_lists_subagent_models(tmp_path):
    s = _session(tmp_path)
    cfg = s.settings.models
    expected = {resolve(m).name for m in cfg.subagents.values() if cfg.is_local(m)}
    assert set(s.local_model_tags()) == expected
    # A role on Ollama fallback isn't actually running locally — it drops out.
    from types import SimpleNamespace

    local_role = next(r for r, m in cfg.subagents.items() if cfg.is_local(m))
    s.bundle = SimpleNamespace(fallbacks={local_role: cfg.subagents[local_role]}, persistent=False)
    remaining = {
        resolve(m).name for r, m in cfg.subagents.items() if cfg.is_local(m) and r != local_role
    }
    assert set(s.local_model_tags()) == remaining


def test_banner_and_toolbar_show_local_models(tmp_path):
    from loom.ui.repl import _banner, _toolbar

    s = _session(tmp_path)
    cfg = s.settings.models
    tags = s.local_model_tags()
    assert tags, "default config should assign local models to subagents"
    banner_out = _banner(s).renderable.plain
    toolbar_out = _toolbar(s)
    for out in (banner_out, toolbar_out):
        assert "⌂" in out
        assert tags[0] in out
        assert cfg.orchestrator in out


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


# ------------------------------------------------------------ caller attribution


def _ai_msg(model_name=None, tool_calls=None, content=""):
    from langchain_core.messages import AIMessage

    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    if model_name:
        msg.response_metadata = {"model_name": model_name}
    return msg


def test_msg_source_attribution(tmp_path):
    s = _session(tmp_path)
    cfg = s.settings.models
    orch = resolve(cfg.orchestrator).name
    assert s._msg_source(_ai_msg(orch), nested=False) == "orchestrator"
    assert s._msg_source(_ai_msg(), nested=False) == "orchestrator"
    assert s._msg_source(_ai_msg(), nested=True) == "subagent"
    role, model_string = next(iter(cfg.subagents.items()))
    label = s._msg_source(_ai_msg(resolve(model_string).name), nested=True)
    assert role in label
    assert ("⌂ local" in label) == cfg.is_local(model_string)


def test_tool_calls_are_always_attributed(tmp_path, capsys):
    s = _session(tmp_path)
    s._print_tool_call({"name": "read_file", "args": {"path": "x"}}, "model")
    assert "[orchestrator]" in capsys.readouterr().out
    s._print_tool_call({"name": "read_file", "args": {"path": "x"}}, "model", source="editor · q (⌂ local)")
    assert "editor · q (⌂ local)" in capsys.readouterr().out


def test_turn_complete_marker_prints(tmp_path, capsys):
    from types import SimpleNamespace

    s = _session(tmp_path)

    class QuietAgent:
        def stream(self, *a, **k):
            return iter([])

    s.bundle = SimpleNamespace(agent=QuietAgent(), persistent=False, fallbacks={})
    s.run_turn("hello")
    assert "✔ turn complete" in capsys.readouterr().out


# ------------------------------------------------------------ approvals cross threads


def test_confirm_callback_survives_worker_threads(tmp_path):
    """LangGraph runs tools in worker threads; the confirm callback (a plain
    Slot, not a contextvar) must be visible there or approvals silently
    auto-deny without ever prompting."""
    import threading

    from loom.middleware import policy

    seen = []
    policy.confirm_callback.set(lambda n, i, r: (seen.append(n) or True))
    try:
        result = []
        t = threading.Thread(target=lambda: result.append(policy.confirm_callback.get()("execute", {}, "ask")))
        t.start()
        t.join()
        assert result == [True]
        assert seen == ["execute"]
    finally:
        policy.confirm_callback.set(lambda n, i, r: False)


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
