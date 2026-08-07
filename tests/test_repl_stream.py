"""REPL streaming internals: reasoning extraction, per-model attribution,
inline edit diffs, and the Claude Code-style approval selector."""

import time

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


def _plain(console, renderable) -> str:
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


def test_banner_and_toolbar_show_local_models(tmp_path):
    from loom.ui.repl import _banner, _toolbar

    s = _session(tmp_path)
    cfg = s.settings.models
    tags = s.local_model_tags()
    assert tags, "default config should assign local models to subagents"
    s.console.width = 200  # the welcome card lists models; don't let it wrap
    banner_out = _plain(s.console, _banner(s))
    # The status line is prompt_toolkit fragments, not a string.
    toolbar_out = "".join(text for _, text in _toolbar(s))
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


def test_subagent_attributed_by_delegation_not_model_name(tmp_path):
    """A local role that fell back to the cloud model another role uses must
    still be attributed to the role actually delegated to — model-name matching
    would mislabel it (e.g. explorer-on-fallback shown as reviewer, when both
    run claude-haiku-4-5)."""
    s = _session(tmp_path)
    s._pending_subagents = []
    s._ns_role = {}
    s._note_task({"name": "task", "args": {"subagent_type": "explorer"}})
    ns = ("tools:abc123",)
    assert s._attribute_ns(ns) == "explorer"
    label = s._role_label("explorer", "claude-haiku-4-5", "anthropic")
    assert label.startswith("explorer · ")
    assert "reviewer" not in label
    assert "☁ cloud" in label
    # Namespace stays bound; the delegation isn't re-consumed on later chunks.
    assert s._attribute_ns(ns) == "explorer"
    assert s._pending_subagents == []


def test_attribute_ns_binds_in_delegation_order(tmp_path):
    s = _session(tmp_path)
    s._pending_subagents = []
    s._ns_role = {}
    s._note_task({"name": "task", "args": {"subagent_type": "explorer"}})
    s._note_task({"name": "task", "args": {"subagent_type": "searcher"}})
    assert s._attribute_ns(("tools:1",)) == "explorer"
    assert s._attribute_ns(("tools:2",)) == "searcher"
    # The orchestrator's own (top-level) namespace is never a subagent.
    assert s._attribute_ns(()) is None
    # A non-task call doesn't consume a delegation slot.
    s._note_task({"name": "execute", "args": {"command": "ls"}})
    assert s._pending_subagents == []


def test_role_label_locality(tmp_path):
    s = _session(tmp_path)
    assert "⌂ local" in s._role_label("explorer", "qwen3.5:4b", "ollama")
    assert "☁ cloud" in s._role_label("reviewer", "claude-haiku-4-5", "anthropic")
    # No provider → infer local/cloud from the model-name shape.
    assert "⌂ local" in s._role_label("explorer", "qwen3.5:4b")
    assert "☁ cloud" in s._role_label("reviewer", "claude-haiku-4-5")


def test_tool_calls_are_always_attributed(tmp_path, capsys):
    """A call names its caller whenever the caller changed — the rail carries
    the attribution in between, but it must never be silent about a switch."""
    s = _session(tmp_path)
    s._print_tool_call({"name": "read_file", "args": {"path": "x"}}, "model")
    out = capsys.readouterr().out
    assert "orchestrator" in out and "read_file" in out

    s._print_tool_call({"name": "read_file", "args": {"path": "x"}}, "model", source="editor · q (⌂ local)")
    out = capsys.readouterr().out
    assert "editor" in out and "q" in out
    # A delegated caller is indented under the thread that spawned it.
    assert out.splitlines()[-1].startswith(("╎", ":"))


def test_repeated_calls_from_one_caller_do_not_repeat_the_header(tmp_path, capsys):
    s = _session(tmp_path)
    for _ in range(3):
        s._print_tool_call({"name": "read_file", "args": {"path": "x"}}, "model")
    assert capsys.readouterr().out.count("orchestrator") == 1


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


# ------------------------------------------------------------ prompts vs. the stream
#
# LangGraph runs tool calls in worker threads, so an approval prompt is drawn
# from a different thread than the one rendering the stream. Nothing used to
# stand between them, and the two ways that broke are exactly what a user
# sees as "it crashed": the transcript stops mid-sentence, or the question is
# on screen but scrolled away under live output.


def test_stream_resumes_after_a_prompt_tears_the_block_down(tmp_path):
    """A confirm ends the open block from its own thread. The render loop
    caches what it thinks is open, so without consulting the weave every
    remaining token of the message goes nowhere — silently."""
    import io
    import threading

    s = _session(tmp_path)
    s.console.file = io.StringIO()
    meta = {"ls_model_name": "orch", "ls_provider": "anthropic"}

    def stream():
        yield ("messages", (AIMessageChunk(content="the answer is "), meta))
        # what _confirm_locked() does first, on a tool worker thread
        t = threading.Thread(target=s.weave.end_block)
        t.start()
        t.join()
        yield ("messages", (AIMessageChunk(content="forty-two exactly. "), meta))

    s.weave.reset()
    s._stream_multi(stream())
    assert "forty-two exactly." in s.console.file.getvalue()


def test_a_waiting_prompt_is_the_last_thing_on_screen(tmp_path, monkeypatch):
    """While a prompt waits for an answer the render loop must stop drawing.
    Otherwise a sibling subagent streams straight over the question and the
    user is looking at live output with no idea anything wants input."""
    import io
    import threading

    s = _session(tmp_path)
    s.console.file = io.StringIO()
    answer = threading.Event()

    def blocking_ask(*a, **k):
        s.console.print("  > [1/2/3] (1): ", end="")
        assert answer.wait(5), "prompt was never released"
        return "1"

    monkeypatch.setattr("rich.prompt.Prompt.ask", staticmethod(blocking_ask))
    worker = threading.Thread(
        target=lambda: s._confirm("write_file", {"path": "a.py", "content": "x"}, "requires approval")
    )

    def stream():
        yield ("messages", (AIMessageChunk(content="editing two files. "), {"ls_model_name": "orch"}))
        worker.start()
        while not s.console.file.getvalue().rstrip().endswith("(1):"):
            time.sleep(0.01)  # the prompt is up and holding the screen
        yield (("tools:1",), "messages", (AIMessageChunk(content="scanning the repo… "), {}))

    s.weave.reset()
    renderer = threading.Thread(target=lambda: s._stream_multi(stream()))
    renderer.start()
    time.sleep(0.4)  # ample time for the stream to run over the prompt, if it can

    waiting = s.console.file.getvalue()
    answer.set()
    worker.join(5)
    renderer.join(5)

    assert "scanning the repo" not in waiting, "stream drew over a prompt that was waiting for input"
    assert waiting.rstrip().endswith("(1):"), "the question was not the last thing on screen"
    assert "scanning the repo" in s.console.file.getvalue(), "stream did not resume after the answer"


def test_mid_stream_error_is_not_read_as_an_unsupported_stream_mode(tmp_path):
    """``agent.stream()`` is lazy, so kwargs errors surface on the first
    ``next()``. Treating a later ValueError the same way re-ran the whole turn
    — tool calls and approval prompts included — behind the user's back."""
    s = _session(tmp_path)
    attempts = []

    class Agent:
        def stream(self, inputs, config=None, stream_mode=None, **kwargs):
            attempts.append(kwargs)

            def gen():
                yield ("messages", (AIMessageChunk(content="hi "), {"ls_model_name": "orch"}))
                raise ValueError("provider hiccup")

            return gen()

    with pytest.raises(ValueError, match="provider hiccup"):
        s._stream(Agent(), {}, {})
    assert attempts == [{"subgraphs": True}], "the turn was silently restarted"


def test_unsupported_stream_kwargs_still_fall_back(tmp_path):
    """The fallback the probe exists to preserve: a langgraph that rejects
    ``subgraphs`` raises on the first next(), and the retry must still run."""
    s = _session(tmp_path)
    attempts = []

    class Agent:
        def stream(self, inputs, config=None, stream_mode=None, **kwargs):
            attempts.append(kwargs)

            def gen():
                if kwargs.get("subgraphs"):
                    raise TypeError("unexpected keyword argument 'subgraphs'")
                yield ("messages", (AIMessageChunk(content="hello"), {"ls_model_name": "orch"}))

            return gen()

    s._stream(Agent(), {}, {})
    assert attempts == [{"subgraphs": True}, {}]
