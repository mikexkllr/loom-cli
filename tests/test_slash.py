"""Slash-command registry and handlers against a live (offline) Session."""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")
pytest.importorskip("rich")
pytest.importorskip("langchain_core")

from loom.core import settings as st
from loom.ui import slash
from loom.ui.repl import Session

EXPECTED_COMMANDS = {
    # Claude Code-compatible set
    "help", "exit", "clear", "compact", "cost", "doctor", "init", "mcp",
    "memory", "model", "permissions", "status", "export", "hooks", "vim",
    "resume", "undo",
    # Loom-specific
    "plan", "local", "yolo", "agents", "models", "settings", "theme", "cwd",
    "airgap", "setup",
}


def _session(tmp_path):
    return Session(st.load_settings(tmp_path), cwd=str(tmp_path))


def test_expected_commands_registered():
    assert EXPECTED_COMMANDS <= set(slash._REGISTRY)


def test_config_aliases_to_settings(tmp_path, capsys):
    s = _session(tmp_path)
    assert slash.dispatch(s, "/config") is True
    assert "settings.json" in capsys.readouterr().out


def test_status_and_mcp_and_cost_render(tmp_path, capsys):
    s = _session(tmp_path)
    for cmd in ("/status", "/mcp", "/cost", "/hooks", "/permissions"):
        assert slash.dispatch(s, cmd) is True
    out = capsys.readouterr().out
    assert "playwright" in out  # /mcp and /status list the default server
    assert "0" in out  # /cost shows zeroed usage


def test_model_origin_badges(tmp_path):
    """The UI can tell cloud from local per role, following live fallbacks."""
    s = _session(tmp_path)
    cfg = s.settings.models
    model, is_local = s.model_origin("orchestrator")
    assert model == cfg.orchestrator
    assert is_local == cfg.is_local(cfg.orchestrator)
    # Stream-node aliases resolve to the orchestrator.
    assert s.model_origin("agent") == s.model_origin("model") == s.model_origin("orchestrator")
    assert s.model_origin("no-such-role") is None
    # A role on Ollama fallback reports the billed cloud model, not the config one.
    local_role = next((r for r, m in cfg.subagents.items() if cfg.is_local(m)), None)
    if local_role:
        s.bundle = type("B", (), {"fallbacks": {local_role: cfg.subagents[local_role]}, "persistent": False})()
        assert s.model_origin(local_role) == (cfg.cloud_fallback, False)


def test_status_shows_cloud_or_local(tmp_path, capsys):
    s = _session(tmp_path)
    slash.dispatch(s, "/status")
    out = capsys.readouterr().out
    assert "cloud" in out or "local" in out


def test_memory_missing_then_present(tmp_path, capsys):
    s = _session(tmp_path)
    slash.dispatch(s, "/memory")
    assert "/init" in capsys.readouterr().out
    (tmp_path / "LOOM.md").write_text("# My project\nUse tabs.")
    slash.dispatch(s, "/memory")
    assert "My project" in capsys.readouterr().out


def test_memory_file_prepended_once(tmp_path):
    (tmp_path / "LOOM.md").write_text("Always use tabs.")
    s = _session(tmp_path)
    first = s._prepare_text("do the thing")
    assert "Always use tabs." in first and "do the thing" in first
    second = s._prepare_text("next")
    assert "Always use tabs." not in second


def test_compact_summary_carried_into_next_message(tmp_path):
    s = _session(tmp_path)
    s.pending_context = "We built a health endpoint."
    text = s._prepare_text("continue")
    assert "We built a health endpoint." in text
    assert s.pending_context is None


def test_export_writes_transcript(tmp_path, capsys):
    s = _session(tmp_path)
    s.messages = [("user", "hi"), ("assistant", "hello there")]
    target = tmp_path / "out.md"
    slash.dispatch(s, f"/export {target}")
    content = target.read_text()
    assert "hello there" in content and "## You" in content


def test_vim_toggles(tmp_path, capsys):
    s = _session(tmp_path)
    slash.dispatch(s, "/vim")
    assert s.vim is True
    slash.dispatch(s, "/vim")
    assert s.vim is False


def test_unknown_command_is_handled(tmp_path, capsys):
    s = _session(tmp_path)
    assert slash.dispatch(s, "/definitely-not-a-command") is True
    assert "unknown command" in capsys.readouterr().out


def test_default_prompt_symbol_is_claude_code_style():
    assert st.UISettings().prompt_symbol == ">"


def test_model_no_args_shows_roles(tmp_path, capsys):
    s = _session(tmp_path)
    slash.dispatch(s, "/model")
    out = capsys.readouterr().out
    assert "orchestrator" in out and "tester" in out


def test_model_role_set_routes_to_settings(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(st, "set_value", lambda key, value, *a, **k: calls.append((key, value)))
    s = _session(tmp_path)
    monkeypatch.setattr(s, "reload_settings", lambda: None)
    slash.dispatch(s, "/model editor ollama/qwen3:14b")
    slash.dispatch(s, "/model advisor claude-opus-4-8")
    slash.dispatch(s, "/model claude-sonnet-4-6")  # bare model → orchestrator
    assert calls == [
        ("models.subagents.editor", "ollama/qwen3:14b"),
        ("models.advisor", "claude-opus-4-8"),
        ("models.orchestrator", "claude-sonnet-4-6"),
    ]


def test_setup_dispatches_to_onboarding_and_reloads(tmp_path, monkeypatch):
    from loom.ui import onboarding

    calls = []
    monkeypatch.setattr(onboarding, "run", lambda console, **kw: calls.append(kw) or None)
    s = _session(tmp_path)
    reloaded = []
    monkeypatch.setattr(s, "reload_settings", lambda: reloaded.append(True))
    monkeypatch.setattr(s, "rebuild", lambda: reloaded.append(True))

    assert slash.dispatch(s, "/setup") is True
    assert calls and calls[0]["root"] == s.cwd
    assert reloaded == [True, True]


def test_setup_with_role_args_filters_roles(tmp_path, monkeypatch):
    from loom.ui import onboarding

    calls = []
    monkeypatch.setattr(onboarding, "run", lambda console, **kw: calls.append(kw) or None)
    s = _session(tmp_path)
    monkeypatch.setattr(s, "reload_settings", lambda: None)
    monkeypatch.setattr(s, "rebuild", lambda: None)

    slash.dispatch(s, "/setup orchestrator advisor")
    assert calls[0]["roles"] == ("orchestrator", "advisor")


def test_setup_cancelled_does_not_crash(tmp_path, monkeypatch, capsys):
    from loom.ui import onboarding

    def _cancel(console, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(onboarding, "run", _cancel)
    s = _session(tmp_path)
    assert slash.dispatch(s, "/setup") is True
    assert "cancelled" in capsys.readouterr().out


def test_airgap_toggle(tmp_path, capsys):
    s = _session(tmp_path)
    slash.dispatch(s, "/airgap")
    assert s.airgap is True
    slash.dispatch(s, "/airgap")
    assert s.airgap is False


def test_resume_lists_and_switches_thread(tmp_path, capsys):
    from loom.core import sessions as sessions_mod

    sessions_mod.record(tmp_path, "loom-20260101-000000", "fix the login bug")
    s = _session(tmp_path)
    slash.dispatch(s, "/resume")
    assert "fix the login bug" in capsys.readouterr().out
    slash.dispatch(s, "/resume 1")
    assert s.thread_id == "loom-20260101-000000"
    slash.dispatch(s, "/resume nope")
    assert "no such session" in capsys.readouterr().out


def test_undo_via_slash(tmp_path, capsys):
    from loom.core import undo as undo_mod

    target = tmp_path / "x.txt"
    target.write_text("before")
    token = undo_mod.current_turn_id.set("t1")
    try:
        undo_mod.snapshot(tmp_path, "x.txt")
        target.write_text("after")
    finally:
        undo_mod.current_turn_id.reset(token)

    s = _session(tmp_path)
    slash.dispatch(s, "/undo")
    assert target.read_text() == "before"
    slash.dispatch(s, "/undo")
    assert "nothing to undo" in capsys.readouterr().out


def test_diff_preview_for_write_and_edit(tmp_path):
    s = _session(tmp_path)
    (tmp_path / "a.py").write_text("old line\n")
    diff = s._diff_for("write_file", {"path": "a.py", "content": "new line\n"})
    text = diff.plain
    assert "-old line" in text and "+new line" in text
    diff2 = s._diff_for("edit_file", {"path": "b.py", "old_string": "foo", "new_string": "bar"})
    assert "-foo" in diff2.plain and "+bar" in diff2.plain
    assert s._diff_for("execute", {"command": "ls"}) is None


def test_first_turn_injects_repo_map_and_mentions(tmp_path):
    (tmp_path / "main.py").write_text("print('hi')")
    s = _session(tmp_path)
    text = s._prepare_text("check @main.py")
    assert "[Repo map]" in text and "print('hi')" in text
    # second turn: no repo map again
    assert "[Repo map]" not in s._prepare_text("next")
