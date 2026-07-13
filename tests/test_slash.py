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
    # Loom-specific
    "plan", "local", "yolo", "agents", "models", "settings", "theme", "cwd",
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
