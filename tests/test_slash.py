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
    "plan", "local", "yolo", "agents", "ollama", "playwright", "settings", "theme", "cwd",
    "airgap", "setup", "privacy", "about",
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


def test_prompt_symbol_defaults_to_the_theme_caret():
    """Empty means "use the theme's caret" (❯, or > on an ASCII terminal);
    setting it explicitly still wins."""
    from loom.ui import prompt as prompt_mod

    assert st.UISettings().prompt_symbol == ""
    assert prompt_mod.caret("loom", "")[0][1] == "❯ "
    assert prompt_mod.caret("ascii", "")[0][1] == "> "
    assert prompt_mod.caret("loom", "$")[0][1] == "$ "


def test_model_no_args_shows_roles(tmp_path, capsys):
    s = _session(tmp_path)
    slash.dispatch(s, "/model")
    out = capsys.readouterr().out
    assert "orchestrator" in out and "tester" in out


def test_models_plural_aliases_to_model(tmp_path, capsys):
    # `/models` (plural) is the config command, not the Ollama status check —
    # it must render the role table, same as `/model`.
    s = _session(tmp_path)
    assert slash.dispatch(s, "/models") is True
    out = capsys.readouterr().out
    assert "orchestrator" in out and "tester" in out


def test_ollama_status_command_registered(tmp_path):
    # The Ollama daemon/model health check lives at `/ollama` now.
    assert "ollama" in slash._REGISTRY
    assert "models" not in slash._REGISTRY  # freed up; it's an alias


def test_playwright_command_reports_missing_browser(tmp_path, monkeypatch, capsys):
    from loom.core import playwright_setup as pw_mod

    monkeypatch.setattr(pw_mod, "status", lambda: pw_mod.PlaywrightStatus(True, False, tmp_path / "ms-playwright"))
    s = _session(tmp_path)
    assert slash.dispatch(s, "/playwright") is True
    assert "no browser installed" in capsys.readouterr().out


def test_playwright_install_arg_invokes_installer(tmp_path, monkeypatch):
    from loom.core import playwright_setup as pw_mod

    calls = []
    monkeypatch.setattr(pw_mod, "install_browsers", lambda console, browser="chromium": calls.append(browser) or 0)
    s = _session(tmp_path)
    assert slash.dispatch(s, "/playwright install") is True
    assert calls == ["chromium"]


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


def test_model_no_args_never_hits_the_network(tmp_path, monkeypatch, capsys):
    """The bare /model summary only needs local Ollama status — it must not
    call the (network-touching) full candidate picker."""
    from loom.ui import slash as slash_mod

    def _boom(session):
        raise AssertionError("bare /model must not call _model_candidates")

    monkeypatch.setattr(slash_mod, "_model_candidates", _boom)
    s = _session(tmp_path)
    assert slash.dispatch(s, "/model") is True


def test_model_candidates_lists_full_local_catalog_and_cloud_models(tmp_path, monkeypatch):
    from loom.core import model_catalog as catalog
    from loom.core import ollama
    from loom.core import providers as prov
    from loom.core import recommendations as rec

    monkeypatch.setattr(ollama, "status", lambda cfg: ollama.OllamaStatus(True, True, ["qwen3:14b"], "http://x"))
    monkeypatch.setattr(rec, "detect_hardware", lambda: rec.Hardware("Darwin", 32, "apple", 32))
    monkeypatch.setattr(catalog, "available_models", lambda p, env: (list(p.example_models), False))

    s = _session(tmp_path)
    candidates = slash._model_candidates(s)

    assert ("ollama/qwen3:14b", "local · installed") in candidates
    # Every curated local tier not already installed shows up, needing a pull.
    local_tags = {m for m, where in candidates if where.startswith("local ·") and m != "ollama/qwen3:14b"}
    assert local_tags == {f"ollama/{m.tag}" for m in rec.all_local_models() if m.tag != "qwen3:14b"}
    # Every cloud provider's (mocked) example models show up, labeled non-live.
    anthropic = prov.get("anthropic")
    for m in anthropic.example_models:
        assert (anthropic.model_string(m), "cloud · example") in candidates


def test_model_candidates_labels_live_catalog_results(tmp_path, monkeypatch):
    from loom.core import model_catalog as catalog
    from loom.core import ollama
    from loom.core import providers as prov
    from loom.core import recommendations as rec

    monkeypatch.setattr(ollama, "status", lambda cfg: ollama.OllamaStatus(True, True, [], "http://x"))
    monkeypatch.setattr(rec, "detect_hardware", lambda: rec.Hardware("Darwin", 32, "apple", 32))

    def fake_available(p, env):
        if p.id == "opencode_zen":
            return (["gpt-5.5", "grok-4.5"], True)
        return (list(p.example_models), False)

    monkeypatch.setattr(catalog, "available_models", fake_available)

    s = _session(tmp_path)
    candidates = slash._model_candidates(s)
    zen = prov.get("opencode_zen")
    assert (zen.model_string("gpt-5.5"), "cloud · live") in candidates
    assert (zen.model_string("grok-4.5"), "cloud · live") in candidates


def test_setup_dispatches_to_onboarding_and_reloads(tmp_path, monkeypatch):
    from loom.ui import onboarding

    calls = []
    s = _session(tmp_path)
    monkeypatch.setattr(onboarding, "run", lambda console, **kw: calls.append(kw) or s.settings)
    monkeypatch.setattr(onboarding, "maybe_setup_playwright", lambda console, settings: None)
    reloaded = []
    monkeypatch.setattr(s, "reload_settings", lambda: reloaded.append(True))
    monkeypatch.setattr(s, "rebuild", lambda: reloaded.append(True))

    assert slash.dispatch(s, "/setup") is True
    assert calls and calls[0]["root"] == s.cwd
    assert reloaded == [True, True]


def test_setup_with_role_args_filters_roles(tmp_path, monkeypatch):
    from loom.ui import onboarding

    calls = []
    s = _session(tmp_path)
    monkeypatch.setattr(onboarding, "run", lambda console, **kw: calls.append(kw) or s.settings)
    monkeypatch.setattr(onboarding, "maybe_setup_playwright", lambda console, settings: None)
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


# ------------------------------------------------------------------ /privacy


@pytest.fixture
def _isolated_telemetry(tmp_path, monkeypatch):
    from loom.core import telemetry as tel

    monkeypatch.setattr(tel, "store_path", lambda: tmp_path / "telemetry.json")
    tel._reset_for_tests()
    yield tel
    tel._reset_for_tests()


def test_privacy_describes_the_default_state(tmp_path, capsys, _isolated_telemetry):
    s = _session(tmp_path)
    assert slash.dispatch(s, "/privacy") is True
    out = capsys.readouterr().out
    assert "nothing leaves this machine" in out


def test_privacy_set_persists_and_reactivates(tmp_path, capsys, _isolated_telemetry):
    tel = _isolated_telemetry
    s = _session(tmp_path)
    assert slash.dispatch(s, "/privacy set none") is True
    consent = tel.load()
    assert consent.mode == "none" and consent.decided is True
    assert "in effect now" in capsys.readouterr().out

    assert slash.dispatch(s, "/privacy set bogus") is True
    assert tel.load().mode == "none"  # unchanged


def test_privacy_here_requires_a_mode_first(tmp_path, capsys, _isolated_telemetry):
    s = _session(tmp_path)
    assert slash.dispatch(s, "/privacy here") is True
    assert "set a mode first" in capsys.readouterr().out
    assert _isolated_telemetry.project_share(tmp_path) is None  # nothing recorded


# ---------------------------------------------------------------------------
# One doctor, not two
# ---------------------------------------------------------------------------


def _zen_only_session(tmp_path, monkeypatch):
    """A project whose every role routes through OpenCode Zen, with its key."""
    import json

    from loom.core import ollama as ollama_mod

    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr(
        ollama_mod, "status", lambda cfg: ollama_mod.OllamaStatus(False, False, [], cfg.ollama_endpoint)
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_API_KEY", "zen-key")
    roles = ("explorer", "editor", "bash", "searcher", "reviewer", "general-purpose", "tester")
    (tmp_path / ".loom").mkdir()
    (tmp_path / ".loom" / "settings.json").write_text(
        json.dumps(
            {
                "models": {
                    "orchestrator": "zen/gpt-5.5",
                    "advisor": "zen/gpt-5.5",
                    "escalation_model": "zen/gpt-5.5",
                    "cloud_fallback": "zen/gpt-5.5",
                    "subagents": dict.fromkeys(roles, "zen/gpt-5.5"),
                }
            }
        )
    )
    return _session(tmp_path)


def test_doctor_in_the_repl_judges_the_setup_by_its_routing(tmp_path, monkeypatch, capsys):
    """`loom doctor` was rewritten to report on the providers a config actually
    uses. `/doctor` kept its own copy of the old logic, so the same Zen-only
    machine got a red "anthropic_api_key not set" and a red Ollama from one and
    a clean bill from the other."""
    s = _zen_only_session(tmp_path, monkeypatch)
    assert slash.dispatch(s, "/doctor") is True
    out = capsys.readouterr().out
    assert "anthropic" not in out.lower()
    assert "OpenCode Zen" in out and "OPENCODE_API_KEY set" in out
    assert "no local roles configured" in out  # a stopped daemon is a non-event here
    assert "not reachable" not in out


def test_both_doctors_print_the_same_checkup(tmp_path, monkeypatch, capsys):
    from typer.testing import CliRunner

    from loom.cli.main import app

    s = _zen_only_session(tmp_path, monkeypatch)
    slash.dispatch(s, "/doctor")
    in_repl = capsys.readouterr().out
    from_shell = CliRunner().invoke(app, ["doctor", "--root", str(tmp_path)]).output

    def rows(text):
        return [" ".join(line.split()) for line in text.splitlines() if ("✔" in line or "✘" in line or "◌" in line)]

    assert rows(in_repl) and rows(in_repl) == rows(from_shell)


def test_the_probe_calls_the_cloud_fallback_when_local_roles_exist():
    from loom.core.config import LoomConfig
    from loom.ui import doctor

    hybrid = LoomConfig(
        orchestrator="zen/gpt-5.5",
        advisor="zen/gpt-5.5",
        escalation_model="zen/gpt-5.5",
        cloud_fallback="zen/claude-haiku-4-5",
        subagents={"explorer": "ollama/qwen3.5:4b"},
    )
    assert doctor.probe_plan(hybrid)["cloud_fallback"] == "zen/claude-haiku-4-5"
    all_cloud = hybrid.model_copy(update={"subagents": {"explorer": "zen/gpt-5.5"}})
    assert "cloud_fallback" not in doctor.probe_plan(all_cloud)  # nothing would ever fall back


# ---------------------------------------------------------------------------
# Surfaces that listed `models.subagents` instead of the fleet
# ---------------------------------------------------------------------------


def test_model_lists_inherited_roles_and_the_fallback(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "200")
    s = _session(tmp_path)
    cfg = s.settings.models
    assert "reviewer" not in cfg.subagents  # the default config leaves it to inherit
    slash.dispatch(s, "/model")
    out = capsys.readouterr().out
    reviewer_row = next(line for line in out.splitlines() if line.strip().startswith("reviewer"))
    assert cfg.advisor in reviewer_row and "inherits advisor" in reviewer_row
    assert "cloud_fallback" in out and cfg.cloud_fallback in out


def test_model_can_set_the_cloud_fallback(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(st, "set_value", lambda key, value, *a, **k: calls.append((key, value)))
    s = _session(tmp_path)
    monkeypatch.setattr(s, "reload_settings", lambda: None)
    slash.dispatch(s, "/model cloud_fallback zen/claude-haiku-4-5")
    assert calls == [("models.cloud_fallback", "zen/claude-haiku-4-5")]


def test_ollama_counts_the_models_the_config_asks_for(tmp_path, monkeypatch, capsys):
    """It printed `len(installed)`: "all 14 configured model(s) downloaded" to a
    setup that configures three."""
    from loom.core import ollama as ollama_mod

    s = _session(tmp_path)
    required = ollama_mod.required_local_models(s.settings.models)
    installed = [*required, *(f"unrelated-{i}:latest" for i in range(11))]
    monkeypatch.setattr(
        ollama_mod, "status", lambda cfg: ollama_mod.OllamaStatus(True, True, installed, cfg.ollama_endpoint)
    )
    slash.dispatch(s, "/ollama")
    out = capsys.readouterr().out
    assert f"all {len(required)} configured model(s) downloaded" in out
    assert f"all {len(installed)} " not in out


# ---------------------------------------------------------------------------
# /settings: a typo is not a crash
# ---------------------------------------------------------------------------


def test_settings_with_a_bad_key_or_value_says_so_and_changes_nothing(tmp_path, monkeypatch, capsys):
    us = tmp_path / "home" / "settings.json"
    monkeypatch.setattr(st, "USER_SETTINGS_PATH", us)
    s = _session(tmp_path)
    for line in ("/settings ui.them light", "/settings ui.theme neon", "/settings permissions.default_mode maybe"):
        assert slash.dispatch(s, line) is True  # handled here, not by the crash net
    out = capsys.readouterr().out
    assert out.count("not set") == 3
    assert not us.exists()
    assert s.settings.ui.theme != "neon"


def test_settings_writes_against_the_sessions_project_not_the_shell_cwd(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(st, "set_value", lambda key, value, root=".": seen.append(str(root)))
    s = _session(tmp_path)
    monkeypatch.setattr(s, "reload_settings", lambda: None)
    slash.dispatch(s, "/settings ui.compact true")
    slash.dispatch(s, "/theme mono")
    assert seen == [str(s.cwd), str(s.cwd)]


# ---------------------------------------------------------------------------
# Transcripts whose messages are block lists
# ---------------------------------------------------------------------------


def _claude_message(text):
    from langchain_core.messages import AIMessage

    return AIMessage(content=[{"type": "text", "text": text}, {"type": "tool_use", "id": "t", "name": "task", "input": {}}])


def test_export_writes_text_for_block_content(tmp_path):
    s = _session(tmp_path)
    s.messages = [("user", "hi"), _claude_message("Mapped the repo.")]
    slash.dispatch(s, "/export out.md")
    content = (tmp_path / "out.md").read_text()  # relative to the project
    assert "Mapped the repo." in content
    assert "tool_use" not in content and "{'type'" not in content


def test_export_with_nothing_to_say_writes_no_file(tmp_path, capsys):
    s = _session(tmp_path)
    slash.dispatch(s, "/export empty.md")
    assert not (tmp_path / "empty.md").exists()
    assert "nothing to export" in capsys.readouterr().out


def test_compact_summarises_text_and_keeps_the_session_if_the_summary_is_empty(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    prompts = []

    class Model:
        def __init__(self, reply):
            self.reply = reply

        def invoke(self, prompt, config=None):
            prompts.append(prompt)
            return SimpleNamespace(content=self.reply)

    s = _session(tmp_path)
    s.messages = [("user", "map the repo"), _claude_message("Mapped the repo.")]
    thread = s.thread_id

    # An empty summary must not replace the conversation with nothing.
    monkeypatch.setattr("loom.core.model_router.build_model", lambda *a, **k: Model([{"type": "tool_use", "id": "x"}]))
    slash.dispatch(s, "/compact")
    assert "compact failed" in capsys.readouterr().out
    assert s.thread_id == thread and s.pending_context is None and len(s.messages) == 2

    monkeypatch.setattr(
        "loom.core.model_router.build_model",
        lambda *a, **k: Model([{"type": "text", "text": "Goal: map the repo. Done."}]),
    )
    slash.dispatch(s, "/compact")
    assert s.pending_context == "Goal: map the repo. Done."
    assert s.thread_id != thread
    assert "ai: Mapped the repo." in prompts[-1] and "tool_use" not in prompts[-1]
