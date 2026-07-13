"""CLI routing: subcommands must not be swallowed by the [PROMPT] argument."""

import pytest

pytest.importorskip("typer")
pytest.importorskip("pydantic")

from typer.testing import CliRunner

from loom.cli.main import app

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
