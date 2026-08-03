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
    assert "doctor" in result.output
    # The health rows, not a model reply — proof it never reached the agent.
    assert "python" in result.output and "ollama" in result.output


def test_nested_subcommands_resolve():
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert "config.yaml" in result.output


def test_config_set_writes_winning_settings_layer(tmp_path, monkeypatch):
    # `loom config set` must land in settings.json (the layer that overrides
    # config.yaml), not the shadowed config.yaml — same trap /model had.
    import json

    from loom.core import settings as st

    us = tmp_path / "settings.json"
    monkeypatch.setattr(st, "USER_SETTINGS_PATH", us)
    proj = tmp_path / "proj"
    (proj / ".loom").mkdir(parents=True)

    result = runner.invoke(app, ["config", "set", "orchestrator", "gpt-4o", "--root", str(proj)])
    assert result.exit_code == 0
    assert json.loads(us.read_text())["models"]["orchestrator"] == "gpt-4o"
    assert st.load_settings(root=proj).models.orchestrator == "gpt-4o"


_SUBAGENTS = ("explorer", "editor", "bash", "searcher", "general-purpose", "tester", "reviewer")


def _all_cloud(model: str = "go:glm-5") -> dict:
    """Every role pinned to ``model`` — including the subagents, which would
    otherwise inherit config.yaml's local defaults through the merge."""
    return {
        "orchestrator": model,
        "advisor": model,
        "escalation_model": model,
        "cloud_fallback": model,
        "subagents": dict.fromkeys(_SUBAGENTS, model),
    }


def _doctor_on(tmp_path, monkeypatch, models: dict) -> str:
    """Run `loom doctor` against a throwaway project pinned to ``models``."""
    import json

    from loom.core import ollama as ollama_mod
    from loom.core import settings as st

    monkeypatch.setattr(st, "USER_SETTINGS_PATH", tmp_path / "user-settings.json")
    monkeypatch.setattr(
        ollama_mod, "status", lambda cfg: ollama_mod.OllamaStatus(False, False, [], cfg.ollama_endpoint)
    )
    proj = tmp_path / "proj"
    (proj / ".loom").mkdir(parents=True, exist_ok=True)  # a test may call this twice
    (proj / ".loom" / "settings.json").write_text(json.dumps({"models": models}))
    result = runner.invoke(app, ["doctor", "--root", str(proj)])
    assert result.exit_code == 0
    return result.output


def test_doctor_only_reports_providers_the_config_uses(tmp_path, monkeypatch):
    # An all-cloud config must not be told its Anthropic key is missing — no
    # role asks for one. Reporting fixed infrastructure instead of the actual
    # routing reads as "your setup is broken" when nothing is.
    out = _doctor_on(tmp_path, monkeypatch, _all_cloud())
    assert "ANTHROPIC_API_KEY" not in out
    assert "OpenCode Go" in out


def test_doctor_reports_anthropic_when_a_role_routes_there(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = _doctor_on(tmp_path, monkeypatch, {"orchestrator": "claude-sonnet-5"})
    assert "ANTHROPIC_API_KEY" in out


def test_doctor_downgrades_ollama_when_nothing_runs_locally(tmp_path, monkeypatch):
    # A stopped daemon is a non-event with no local roles, but a real failure
    # the moment one subagent needs it.
    all_cloud = _doctor_on(tmp_path, monkeypatch, _all_cloud())
    assert "no local roles configured" in all_cloud

    one_local = _all_cloud() | {"subagents": dict.fromkeys(_SUBAGENTS, "go:glm-5") | {"explorer": "ollama/qwen3.5:2b"}}
    assert "not reachable" in _doctor_on(tmp_path, monkeypatch, one_local)


def test_opencode_go_defaults_are_not_region_locked():
    # deepseek-v4-* 403 with a RegionError until the account opts in, so the
    # wizard must never hand them to a new user as a default.
    from loom.core import providers

    go = providers.get("opencode_go")
    for tier in ("main", "flagship", "light"):
        assert not go.model_for_tier(tier).startswith("deepseek")


def test_models_subcommand_resolves():
    result = runner.invoke(app, ["models", "status"])
    # Exit code depends on whether ollama is installed; either way it must hit
    # the models command, not the task runner.
    assert "ollama" in result.output.lower()
    assert "Missing dependency" not in result.output


def test_playwright_subcommand_resolves(monkeypatch, tmp_path):
    from loom.core import playwright_setup as pw_mod

    monkeypatch.setattr(pw_mod, "status", lambda: pw_mod.PlaywrightStatus(True, True, tmp_path))
    result = runner.invoke(app, ["playwright", "status"])
    assert result.exit_code == 0
    assert "installed" in result.output.lower()


def test_playwright_install_subcommand_resolves(monkeypatch):
    from loom.core import playwright_setup as pw_mod

    monkeypatch.setattr(pw_mod, "install_browsers", lambda console, browser="chromium": 0)
    result = runner.invoke(app, ["playwright", "install"])
    assert result.exit_code == 0
    assert "installed" in result.output.lower()


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


# ---------------------------------------------------------------------------
# Options after the prompt
#
# Click turns off interspersed args for groups so a subcommand keeps its own
# flags (`loom models pull --all`). That also swallowed the natural
# `loom "fix the tests" --yolo`, which failed with "No such command '--yolo'" —
# a confusing error for the most obvious way to type the command.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["fix the tests", "--yolo"],
        ["fix the tests", "--plan"],
        ["fix the tests", "--local-only"],
        ["fix the tests", "--accept-edits"],
        ["fix the tests", "--advisor-threshold", "high"],
        ["fix the tests", "--yolo", "--plan"],
    ],
)
def test_options_may_follow_the_prompt(argv):
    result = runner.invoke(app, argv)
    assert "No such command" not in result.output, result.output
    assert "Got unexpected extra argument" not in result.output, result.output


def test_options_may_still_precede_the_prompt():
    """The form the module docstring documents must keep working."""
    result = runner.invoke(app, ["--plan", "fix the tests"])
    assert "No such command" not in result.output


def test_a_root_option_after_the_prompt_is_honoured(tmp_path, monkeypatch):
    """Not just parsed — the value has to reach the run path."""
    seen = {}
    monkeypatch.setattr(main_mod, "_run_task", lambda *a, **kw: seen.update(kw))
    monkeypatch.setattr(main_mod, "_maybe_offer_update", lambda: None)
    result = runner.invoke(app, ["fix the tests", "--root", str(tmp_path), "--yolo"])
    assert result.exit_code == 0, result.output
    assert seen.get("root") == str(tmp_path)
    assert seen.get("yolo") is True


def test_a_subcommands_own_flags_are_left_for_it(tmp_path, monkeypatch):
    """The reason interspersed args are off by default: a flag after a subcommand
    belongs to the subcommand, and must not be eaten by the root group."""
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    result = runner.invoke(app, ["config", "set", "orchestrator", "gpt-4o", "--root", str(proj)])
    assert result.exit_code == 0, result.output
    assert "No such command" not in result.output


def test_an_unknown_subcommand_still_reports_itself_as_a_prompt():
    """With no matching command, the token is a task — that is the whole design."""
    result = runner.invoke(app, ["defintely-not-a-command"])
    assert "No such command" not in result.output
