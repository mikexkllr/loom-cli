"""CLI routing: subcommands must not be swallowed by the [PROMPT] argument."""

import sys
import types

import pytest

pytest.importorskip("typer")
pytest.importorskip("pydantic")

from typer.testing import CliRunner

import loom.cli.main as main_mod
from loom.cli.main import app
from loom.core import update as update_mod

runner = CliRunner()


def test_doctor_is_a_subcommand_not_a_task():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "loom doctor" in result.output
    assert "python" in result.output


def test_nested_subcommands_resolve():
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert "config.yaml" in result.output


def test_models_subcommand_resolves():
    result = runner.invoke(app, ["models", "status"])
    # Exit code depends on whether ollama is installed; either way it must hit
    # the models command, not the task runner.
    assert "ollama" in result.output.lower()
    assert "Missing dependency" not in result.output


def test_free_form_prompt_still_reaches_task_runner():
    result = runner.invoke(app, ["explain this codebase"])
    # In a minimal env this fails on the heavy deps — but it must reach the
    # task path, not be treated as an unknown command.
    assert "No such command" not in result.output


def test_update_subcommand_resolves():
    result = runner.invoke(app, ["update"])
    assert "No such command" not in result.output


def test_update_from_source_install_prints_git_hint():
    # The test process isn't a frozen PyInstaller binary, so `loom update`
    # must fall back to the source-install hint rather than trying to
    # self-replace a nonexistent binary.
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "git pull && uv sync" in result.output


# ---------------------------------------------------------------------------
# Startup update check (_maybe_offer_update) — reached at the top of every
# REPL / one-shot task launch, skipped by subcommands. The safety property
# that matters most: a non-interactive/piped stdin must never hang waiting
# on a Confirm prompt, it just prints a notice and moves on.
# ---------------------------------------------------------------------------


def _fake_check():
    return update_mod.UpdateCheck(asset="loom-macos-arm64", current_sha256="old", latest_sha256="new")


def test_startup_check_noop_when_up_to_date(monkeypatch, capsys):
    monkeypatch.setattr(update_mod, "check_for_startup", lambda: None)
    main_mod._maybe_offer_update()
    assert capsys.readouterr().out == ""


def test_startup_check_non_interactive_only_notifies(monkeypatch, capsys):
    monkeypatch.setattr(update_mod, "check_for_startup", _fake_check)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    relaunched = []
    monkeypatch.setattr(update_mod, "apply_and_relaunch", lambda *a, **k: relaunched.append(True))

    main_mod._maybe_offer_update()

    assert relaunched == []
    out = capsys.readouterr().out
    assert "update available" in out
    assert "loom update" in out


def test_startup_check_interactive_decline_keeps_running(monkeypatch, capsys):
    monkeypatch.setattr(update_mod, "check_for_startup", _fake_check)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    from rich.prompt import Confirm

    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: False))
    relaunched = []
    monkeypatch.setattr(update_mod, "apply_and_relaunch", lambda *a, **k: relaunched.append(True))

    main_mod._maybe_offer_update()

    assert relaunched == []
    assert "continuing with the current version" in capsys.readouterr().out


def test_startup_check_interactive_accept_triggers_relaunch(monkeypatch, capsys):
    monkeypatch.setattr(update_mod, "check_for_startup", _fake_check)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    from rich.prompt import Confirm

    monkeypatch.setattr(Confirm, "ask", staticmethod(lambda *a, **k: True))
    calls = []
    monkeypatch.setattr(update_mod, "apply_and_relaunch", lambda result, **k: calls.append((result, k)))

    main_mod._maybe_offer_update()

    assert len(calls) == 1
    result, kwargs = calls[0]
    assert result.asset == "loom-macos-arm64"
    assert kwargs["argv"] == sys.argv[1:]
