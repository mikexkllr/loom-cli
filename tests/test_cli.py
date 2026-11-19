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


# ------------------------------------------------------------------ privacy


def test_privacy_is_a_subcommand_not_a_task(tmp_path, monkeypatch):
    from loom.core import telemetry as tel

    monkeypatch.setattr(tel, "store_path", lambda: tmp_path / "telemetry.json")
    result = runner.invoke(app, ["privacy", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert "nothing leaves this machine" in result.output


def test_privacy_set_round_trip(tmp_path, monkeypatch):
    from loom.core import telemetry as tel

    monkeypatch.setattr(tel, "store_path", lambda: tmp_path / "telemetry.json")
    result = runner.invoke(app, ["privacy", "set", "errors", "--root", str(tmp_path)])
    assert result.exit_code == 0
    consent = tel.load()
    assert consent.mode == "errors" and consent.decided is True

    result = runner.invoke(app, ["privacy", "set", "bogus", "--root", str(tmp_path)])
    assert result.exit_code == 1
    assert tel.load().mode == "errors"  # unchanged


def test_doctor_reports_the_privacy_mode(tmp_path, monkeypatch):
    from loom.core import telemetry as tel

    monkeypatch.setattr(tel, "store_path", lambda: tmp_path / "telemetry.json")
    tel.save(tel.Consent(mode="none", decided=True))
    out = _doctor_on(tmp_path, monkeypatch, _all_cloud())
    assert "privacy" in out and "nothing leaves this machine" in out


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


def test_a_missing_optional_provider_fails_cleanly_not_as_a_traceback(monkeypatch, tmp_path):
    """`_build_cached` raises bare ImportError (not ModuleNotFoundError) when
    an optional provider package is absent. Catching only the subclass let it
    escape PyInstaller's entry point as a raw traceback ending in
    'Failed to execute script' — the worst failure mode a shipped binary has.
    """
    from loom.ui import repl as repl_mod

    def boom(self):
        raise ImportError("langchain-aws isn't installed. Install it with `uv sync --extra bedrock`.")

    monkeypatch.setattr(repl_mod.Session, "ensure_bundle", boom)
    result = runner.invoke(app, ["--root", str(tmp_path), "do a thing"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "uv sync --extra bedrock" in result.output
    assert "Traceback" not in result.output


def test_a_subcommand_after_a_global_option_is_not_billed_as_a_task(monkeypatch):
    """`loom --root /x doctor` fills the callback's positional `prompt` with
    "doctor" before Click looks for a subcommand — but only when the option
    uses a separated value (`--root .`, not `--root=.`), which is how people
    naturally type it. It used to run silently as a task, billing a model call
    for the word "doctor"."""
    ran = []
    monkeypatch.setattr(main_mod, "_run_task", lambda *a, **k: ran.append(a))
    monkeypatch.setattr(main_mod, "_maybe_offer_update", lambda: None)

    result = runner.invoke(app, ["--root", ".", "doctor"])
    assert ran == [], "a command name must never reach the task runner"
    assert result.exit_code == 2
    assert "is a command, not a task" in result.output


def test_the_guard_does_not_swallow_real_tasks(monkeypatch):
    # Only an exact, whole-prompt match is a misordered command. A task that
    # merely mentions one is still a task.
    prompts = []
    monkeypatch.setattr(main_mod, "_run_task", lambda *a, **k: prompts.append(a[1]))
    monkeypatch.setattr(main_mod, "_maybe_offer_update", lambda: None)

    for prompt in ("explain this codebase", "fix the doctor module", "update the README"):
        assert runner.invoke(app, ["--root", ".", prompt]).exit_code == 0
    assert prompts == ["explain this codebase", "fix the doctor module", "update the README"]


def test_guard_covers_every_registered_command(monkeypatch):
    """Derived from the live Click group, so a command added later is covered
    without anyone remembering to update a list."""
    ran = []
    monkeypatch.setattr(main_mod, "_run_task", lambda *a, **k: ran.append(a))
    monkeypatch.setattr(main_mod, "_maybe_offer_update", lambda: None)

    import typer.main

    names = typer.main.get_command(app).list_commands(None)
    assert {"doctor", "chat", "config", "setup"} <= set(names)
    for name in names:
        assert runner.invoke(app, ["--root", ".", name]).exit_code == 2, name
    assert ran == []


# --------------------------------------------------- the top-level crash net


def test_crash_label_never_leaks_the_users_prompt(monkeypatch):
    """The tag is matched against the app's own command names, so a task —
    which is the user's own words — can only ever be labelled "task"."""
    from loom.cli import main as cli

    cases = {
        ("loom",): "repl",
        ("loom", "refactor the auth module"): "task",
        ("loom", "--plan", "add pagination to /users"): "task",
        ("loom", "doctor"): "doctor",
        # A separated option value occupies a positional slot without being
        # one — the same trap `main()` guards against for prompts.
        ("loom", "--root", ".", "doctor"): "doctor",
        ("loom", "--root", ".", "models", "pull"): "models:pull",
        ("loom", "agents", "list"): "agents:list",
        ("loom", "models", "nonsense"): "models:?",
    }
    for argv, expected in cases.items():
        monkeypatch.setattr(sys, "argv", list(argv))
        assert cli._crashed_command() == expected, argv


def test_every_command_has_a_crash_net(monkeypatch, capsys):
    """Click only handles its own exceptions. Before this, anything escaping a
    subcommand printed a raw traceback and reached no reporter — `loom doctor`,
    `loom setup` and `loom models pull` had no handler at all."""
    from loom.cli import main as cli
    from loom.core import telemetry

    sent = []
    monkeypatch.setattr(telemetry, "report", lambda where, exc, **t: sent.append((where, exc)))
    monkeypatch.setattr(telemetry, "activate", lambda *a, **k: "errors")
    monkeypatch.setattr(telemetry, "flush", lambda *a, **k: None)
    monkeypatch.setattr(telemetry, "current_mode", lambda: "errors")

    boom = RuntimeError("simulated bug")
    real_app = cli.app

    class _Exploding:
        """Keeps the real app's command registry — only invocation fails, so
        the label lookup runs against the same data it does in production."""

        registered_commands = real_app.registered_commands
        registered_groups = real_app.registered_groups

        def __call__(self):
            raise boom

    monkeypatch.setattr(cli, "app", _Exploding())
    monkeypatch.setattr(sys, "argv", ["loom", "doctor"])

    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 1
    assert sent == [("cli:doctor", boom)]
    assert "simulated bug" in capsys.readouterr().out


def test_a_chosen_exit_code_is_not_a_crash(monkeypatch):
    """typer.Exit is how commands report their own failure; reporting those as
    crashes would bury the real ones."""
    from loom.cli import main as cli
    from loom.core import telemetry

    sent = []
    monkeypatch.setattr(telemetry, "report", lambda *a, **k: sent.append(a))

    def clean_exit():
        raise SystemExit(2)

    monkeypatch.setattr(cli, "app", clean_exit)
    monkeypatch.setattr(sys, "argv", ["loom", "doctor"])
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 2
    assert sent == []


def test_ctrl_c_is_not_a_crash(monkeypatch):
    from loom.cli import main as cli
    from loom.core import telemetry

    sent = []
    monkeypatch.setattr(telemetry, "report", lambda *a, **k: sent.append(a))

    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "app", interrupted)
    monkeypatch.setattr(sys, "argv", ["loom"])
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 130
    assert sent == []


def test_a_broken_label_lookup_still_reports_the_crash(monkeypatch):
    """The label is computed inside the crash handler. If it throws, the
    original exception must still get out — losing a bug to the code that was
    supposed to describe it is the worst possible trade."""
    from loom.cli import main as cli

    monkeypatch.setattr(cli, "_command_label", lambda: 1 / 0)
    assert cli._crashed_command() == "unknown"


# ------------------------------------------- the config the commands read


def _project_with_models(tmp_path, monkeypatch, models: dict):
    import json

    from loom.core import ollama as ollama_mod
    from loom.core import settings as st

    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr(st, "USER_SETTINGS_PATH", tmp_path / "user-settings.json")
    monkeypatch.setattr(
        ollama_mod,
        "status",
        lambda cfg: ollama_mod.OllamaStatus(True, True, ["llama-custom:8b"], cfg.ollama_endpoint),
    )
    proj = tmp_path / "proj"
    (proj / ".loom").mkdir(parents=True, exist_ok=True)
    (proj / ".loom" / "settings.json").write_text(json.dumps({"models": models}))
    return proj


def test_agents_and_models_commands_read_the_settings_that_are_in_force(tmp_path, monkeypatch):
    """`/setup`, `/model` and `loom config set` all write settings.json. These
    commands read config.yaml on its own, the layer underneath — so after any
    of them they listed, checked and offered to pull the packaged defaults."""
    proj = _project_with_models(
        tmp_path, monkeypatch, {"subagents": dict.fromkeys(_SUBAGENTS, "ollama/llama-custom:8b")}
    )

    listed = runner.invoke(app, ["agents", "list", "--root", str(proj)])
    assert listed.exit_code == 0
    assert "llama-custom:8b" in listed.output and "qwen3.5" not in listed.output

    models = runner.invoke(app, ["models", "list", "--root", str(proj)])
    assert models.exit_code == 0
    assert "llama-custom:8b" in models.output and "qwen3.5" not in models.output
    assert "not pulled" not in models.output

    status = runner.invoke(app, ["models", "status", "--root", str(proj)])
    assert status.exit_code == 0
    assert "all required local models are installed" in status.output


def test_models_pull_pulls_what_is_configured(tmp_path, monkeypatch):
    from loom.core import ollama as ollama_mod

    proj = _project_with_models(
        tmp_path, monkeypatch, {"subagents": dict.fromkeys(_SUBAGENTS, "ollama/not-here-yet:4b")}
    )
    pulled = []
    monkeypatch.setattr(ollama_mod, "pull", lambda tag, endpoint, console=None, **k: pulled.append(tag) or 0)

    assert runner.invoke(app, ["models", "pull", "--root", str(proj)]).exit_code == 0
    assert pulled == ["not-here-yet:4b"]
    # The config spelling works as an argument too.
    assert runner.invoke(app, ["models", "pull", "ollama/other:2b", "--root", str(proj)]).exit_code == 0
    assert pulled[-1] == "other:2b"


# ---------------------------------------------------- flags that went nowhere


def _capture_repl(monkeypatch):
    from loom.ui import repl

    seen = {}
    monkeypatch.setattr(repl, "run", lambda settings, **kw: seen.update(kw))
    monkeypatch.setattr(main_mod, "_maybe_offer_update", lambda: None)
    return seen


def test_the_repl_is_started_with_every_flag_it_was_given(tmp_path, monkeypatch):
    """`loom --accept-edits` opened the REPL in default mode, and
    `--advisor-threshold` was dropped on the floor the same way."""
    seen = _capture_repl(monkeypatch)
    result = runner.invoke(app, ["--accept-edits", "--advisor-threshold", "HIGH", "--airgap", f"--root={tmp_path}"])
    assert result.exit_code == 0
    assert seen["accept_edits"] is True and seen["advisor_threshold"] == "high" and seen["airgap"] is True

    seen.clear()
    assert runner.invoke(app, ["chat", "--accept-edits", f"--root={tmp_path}"]).exit_code == 0
    assert seen["accept_edits"] is True


def test_an_unknown_advisor_threshold_is_refused_before_anything_starts(tmp_path, monkeypatch):
    seen = _capture_repl(monkeypatch)
    result = runner.invoke(app, ["--advisor-threshold", "sometimes", f"--root={tmp_path}"])
    assert result.exit_code == 2
    assert "low, medium or high" in result.output
    assert seen == {}


def test_loop_flags_without_a_task_say_so(tmp_path, monkeypatch):
    seen = _capture_repl(monkeypatch)
    for flags in (["--loop", "5"], ["--until", "pytest -q"]):
        result = runner.invoke(app, [*flags, f"--root={tmp_path}"])
        assert result.exit_code == 2, flags
        assert "need a task" in result.output
    assert seen == {}


def test_the_headless_run_honours_the_advisor_threshold(tmp_path, monkeypatch):
    from loom.ui import repl

    built = {}

    class _Session:
        def __init__(self, settings, **kw):
            built.update(kw)
            self.console = main_mod.console
            self.last_turn_failed = False
            self._interrupted = False

        def ensure_bundle(self):
            raise RuntimeError("stop here")

    monkeypatch.setattr(repl, "Session", _Session)
    monkeypatch.setattr(main_mod, "_maybe_offer_update", lambda: None)
    result = runner.invoke(app, ["do the thing", "--advisor-threshold", "low", "--accept-edits", f"--root={tmp_path}"])
    assert result.exit_code == 1  # the stubbed build failure
    assert built["advisor_threshold"] == "low" and built["accept_edits"] is True


def test_an_interrupted_headless_run_exits_130(tmp_path, monkeypatch):
    """Ctrl-C inside a turn is caught there, so the run ended "interrupted"
    and exited 0 — indistinguishable, to a script, from a finished one."""
    from types import SimpleNamespace

    from loom.ui import repl

    class _Session:
        def __init__(self, settings, **kw):
            self.console = main_mod.console
            self.last_turn_failed = False
            self._interrupted = False
            self.bundle = SimpleNamespace(
                mode="normal", subagent_names=[], fallbacks={}, substitutions={}, has_advisor=False
            )

        def ensure_bundle(self):
            return self.bundle

        def model_origin(self, role):
            return ("zen/gpt-5.5", False)

        def run_turn(self, prompt):
            self._interrupted = True  # what Session.run_turn does on KeyboardInterrupt

    monkeypatch.setattr(repl, "Session", _Session)
    monkeypatch.setattr(main_mod, "_maybe_offer_update", lambda: None)
    assert runner.invoke(app, ["do the thing", f"--root={tmp_path}"]).exit_code == 130


def test_a_broken_settings_file_is_named_not_reported_as_a_crash(tmp_path, monkeypatch):
    from loom.core import settings as st

    monkeypatch.setattr(st, "USER_SETTINGS_PATH", tmp_path / "user-settings.json")
    monkeypatch.setattr(main_mod, "_maybe_offer_update", lambda: None)
    proj = tmp_path / "proj"
    (proj / ".loom").mkdir(parents=True)
    broken = proj / ".loom" / "settings.json"
    broken.write_text('{"ui": {"theme": "mono",}}')  # the trailing comma everyone types

    result = runner.invoke(app, ["doctor", "--root", str(proj)])
    assert result.exit_code == 2
    assert "settings could not be loaded" in result.output
    assert str(broken) in result.output.replace("\n", "")


def test_the_headless_roster_shows_what_each_role_runs_on(tmp_path, monkeypatch):
    """It read the settings map: `(inherit)` for the reviewer, and the
    configured local model for a role that was about to be billed to the
    cloud fallback."""
    from types import SimpleNamespace

    from loom.core import settings as st
    from loom.ui.repl import Session

    s = Session(st.load_settings(tmp_path), cwd=str(tmp_path))
    cfg = s.settings.models
    s.bundle = SimpleNamespace(
        model_string=cfg.orchestrator,
        subagent_names=["explorer", "reviewer"],
        subagent_models={"explorer": cfg.cloud_fallback, "reviewer": cfg.advisor},
        fallbacks={"explorer": cfg.subagents["explorer"]},
        substitutions={},
        has_advisor=True,
        persistent=False,
    )
    rows = {role: (model, is_local, note) for role, model, is_local, note in main_mod._fleet_rows(s)}
    assert rows["explorer"] == (cfg.cloud_fallback, False, "cloud fallback")
    assert rows["reviewer"][0] == cfg.advisor
    assert "advisor" in rows

    s.bundle.has_advisor = False  # local-only / airgap
    assert "advisor" not in {role for role, *_ in main_mod._fleet_rows(s)}
