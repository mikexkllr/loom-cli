"""Onboarding wizard's interactive glue (prompt_* / run), exercised with
scripted Prompt.ask / Confirm.ask answers rather than a real terminal — the
same approach `test_onboarding.py` avoids needing, but this file specifically
covers the branching logic inside the prompt_* helpers and `run()` itself.
"""

import io

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")

from rich.console import Console

from loom.core import model_catalog as catalog_mod
from loom.core import ollama as ollama_mod
from loom.core import providers as prov
from loom.core import recommendations as rec
from loom.core import settings as settings_mod
from loom.ui import onboarding as ob

HW = rec.Hardware(os_name="Darwin", ram_gb=32, gpu_vendor="apple", vram_gb=32)


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False)


def _as_confirm(scripted):
    """Adapt a _Scripted queue to render.confirm's signature.

    The wizard's yes/no questions go through render.confirm now — it swallows
    EOF so a piped or CI run declines instead of crashing — so the scripts
    drive that rather than rich's Confirm directly.
    """

    def _confirm(_console, _question, **_kwargs):
        return scripted()

    return _confirm


class _Scripted:
    """Feeds a fixed sequence of answers to Prompt.ask/Confirm.ask, in call
    order, regardless of the prompt text — mirrors piping fixed input lines
    into a real terminal, but deterministic and fast."""

    def __init__(self, answers: list):
        self.answers = list(answers)

    def __call__(self, *args, **kwargs):
        if not self.answers:
            raise EOFError("scripted answers exhausted")
        return self.answers.pop(0)


@pytest.fixture(autouse=True)
def _isolated_user_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod, "USER_SETTINGS_PATH", tmp_path / "user-settings.json")
    return tmp_path


@pytest.fixture(autouse=True)
def _no_provider_env(monkeypatch):
    """Strip real provider credentials from the environment — a developer's
    exported ANTHROPIC_API_KEY would otherwise trigger the keep/overwrite
    prompt and desync the scripted answers."""
    for p in prov.PROVIDERS:
        for v in p.env_vars:
            monkeypatch.delenv(v.key, raising=False)


@pytest.fixture(autouse=True)
def _no_real_ollama(monkeypatch):
    """Every test gets a fixed, fake Ollama status/pull — no real daemon or
    network calls, and no accidental `ollama pull` of a multi-GB model."""
    status = ollama_mod.OllamaStatus(installed=True, running=True, models=["qwen3:14b"], endpoint="http://x")
    monkeypatch.setattr(ob.ollama_mod, "status", lambda config: status)
    pulled = []

    def fake_pull(tag, endpoint=ollama_mod.DEFAULT_ENDPOINT, console=None):
        pulled.append(tag)
        return 0

    monkeypatch.setattr(ob.ollama_mod, "pull", fake_pull)
    return pulled


@pytest.fixture(autouse=True)
def _no_real_privacy_step(monkeypatch):
    """Keep the privacy step out of the model-wizard scripts.

    The wizard's privacy prompt is one more Prompt.ask in the sequence; every
    scripted answer queue in this file was written for the model questions, so
    the step is stubbed to a recorder here and covered on its own terms in
    test_privacy.py. The calls are returned so a test can assert the step did
    (or did not) run."""
    from loom.ui import privacy as privacy_mod

    calls = []
    monkeypatch.setattr(privacy_mod, "run", lambda console, root=".", **kw: calls.append((console, root, kw)))
    return calls


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """model_catalog's live listing must never touch a real network in
    tests — force every attempt to fail closed so available_models() always
    falls back to the provider's hardcoded example models."""

    def raise_connect(*a, **k):
        raise catalog_mod.httpx.ConnectError("no network in tests")

    monkeypatch.setattr(catalog_mod.httpx, "get", raise_connect)


def test_prompt_local_model_picks_installed_by_number(monkeypatch):
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["1"]))}))
    tag = ob.prompt_local_model(_console(), HW)
    assert tag == "qwen3:14b"


def test_prompt_local_model_custom_tag_offers_pull(monkeypatch, _no_real_ollama):
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["qwen2.5-coder:32b"]))}))
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([True])))
    tag = ob.prompt_local_model(_console(), HW)
    assert tag == "qwen2.5-coder:32b"
    assert _no_real_ollama == ["qwen2.5-coder:32b"]


def test_prompt_local_model_declines_pull(monkeypatch, _no_real_ollama):
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["qwen2.5-coder:32b"]))}))
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([False])))
    tag = ob.prompt_local_model(_console(), HW)
    assert tag == "qwen2.5-coder:32b"
    assert _no_real_ollama == []  # declined — nothing pulled


def test_prompt_provider_picks_by_number(monkeypatch):
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["2"]))}))
    candidates = prov.cloud_providers()
    picked = ob.prompt_provider(_console(), candidates)
    assert picked is candidates[1]


def test_prompt_credentials_reuses_known_env_without_prompting(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("should not prompt when already known")

    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_boom)}))
    env = ob.prompt_credentials(_console(), prov.get("anthropic"), {"ANTHROPIC_API_KEY": "already-set"})
    assert env == {"ANTHROPIC_API_KEY": "already-set"}


def test_prompt_credentials_prompts_for_missing_vars(monkeypatch):
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["new-key"]))}))
    env = ob.prompt_credentials(_console(), prov.get("anthropic"), {})
    assert env == {"ANTHROPIC_API_KEY": "new-key"}


def test_prompt_credentials_keeps_preexisting_value_on_confirm(monkeypatch):
    """A key from a previous setup (settings.json env block) is offered for
    keep/overwrite — keeping reuses it without prompting for a new value."""

    def _boom(*a, **k):
        raise AssertionError("kept the existing value — must not prompt for a new one")

    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_boom)}))
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([True])))
    env = ob.prompt_credentials(_console(), prov.get("anthropic"), {}, {"ANTHROPIC_API_KEY": "old-key"})
    assert env == {"ANTHROPIC_API_KEY": "old-key"}


def test_prompt_credentials_overwrites_preexisting_value_on_decline(monkeypatch):
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["rotated-key"]))}))
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([False])))
    env = ob.prompt_credentials(_console(), prov.get("anthropic"), {}, {"ANTHROPIC_API_KEY": "old-key"})
    assert env == {"ANTHROPIC_API_KEY": "rotated-key"}


def test_prompt_credentials_blank_after_decline_keeps_old_value(monkeypatch):
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted([""]))}))
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([False])))
    env = ob.prompt_credentials(_console(), prov.get("anthropic"), {}, {"ANTHROPIC_API_KEY": "old-key"})
    assert env == {"ANTHROPIC_API_KEY": "old-key"}


def test_rerun_offers_overwrite_of_saved_key(monkeypatch, tmp_path):
    """Second run of the wizard: the previously saved API key is surfaced and
    can be rotated — the old value must not be silently reused."""
    monkeypatch.setattr(ob.rec, "detect_hardware", lambda: HW)
    # First run: quick setup with cloud, key "first-key".
    # Confirms in order: use cloud provider? yes · customize models per tier? no.
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["quick", "1", "1", "first-key", "user"]))}))
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([True, False])))
    ob.run(_console(), root=tmp_path, verify=False)

    # Second run: same flow, but decline the keep and enter a rotated key.
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["quick", "1", "1", "second-key", "user"]))}))
    # Confirms in order: use cloud provider? yes · keep existing key? no · customize models per tier? no.
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([True, False, False])))
    settings = ob.run(_console(), root=tmp_path, verify=False)
    assert settings.env["ANTHROPIC_API_KEY"] == "second-key"


def test_prompt_cloud_model_defaults_to_tier(monkeypatch):
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(lambda *a, default=None, **k: default)}))
    model_id = ob.prompt_cloud_model(_console(), prov.get("anthropic"), "flagship")
    assert model_id == "claude-opus-4-8"


class _FakeModelsResponse:
    def __init__(self, ids):
        self._ids = ids

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": [{"id": i} for i in self._ids]}


def test_prompt_cloud_model_picks_from_live_catalog_by_number(monkeypatch):
    monkeypatch.setattr(
        catalog_mod.httpx, "get", lambda url, **k: _FakeModelsResponse(["gpt-5.6-terra", "gpt-5.6-sol"])
    )
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["2"]))}))
    model_id = ob.prompt_cloud_model(_console(), prov.get("openai"), "main", {"OPENAI_API_KEY": "sk-x"})
    # available_models() sorts ids: gpt-5.6-sol (1), gpt-5.6-terra (2).
    assert model_id == "gpt-5.6-terra"


def test_prompt_cloud_model_typed_id_overrides_live_catalog(monkeypatch):
    monkeypatch.setattr(catalog_mod.httpx, "get", lambda url, **k: _FakeModelsResponse(["gpt-5.6-terra"]))
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["gpt-5.6-custom"]))}))
    model_id = ob.prompt_cloud_model(_console(), prov.get("openai"), "main", {"OPENAI_API_KEY": "sk-x"})
    assert model_id == "gpt-5.6-custom"


def test_run_quick_setup_local_only(monkeypatch, tmp_path):
    monkeypatch.setattr(ob.rec, "detect_hardware", lambda: HW)
    answers = ["quick", "1", "user"]  # mode, local model#, scope (Confirm handles the cloud toggle)
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(answers))}))
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([False])))

    settings = ob.run(_console(), root=tmp_path, verify=False)
    assert settings.models.orchestrator == "ollama/qwen3:14b"
    assert settings.models.subagents["editor"] == "ollama/qwen3:14b"


def test_run_quick_setup_with_cloud_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(ob.rec, "detect_hardware", lambda: HW)
    answers = ["quick", "1", "1", "test-key", "user"]  # mode, local#, provider#, key, scope
    # use cloud provider? yes · customize models per tier? no (recommended defaults).
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(answers))}))
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([True, False])))

    settings = ob.run(_console(), root=tmp_path, verify=False)
    assert settings.models.orchestrator == "anthropic:claude-sonnet-5"
    assert settings.models.advisor == "anthropic:claude-opus-4-8"
    # reviewer reads diffs, so quick setup keeps it local like the rest of the fleet
    assert settings.models.subagents["reviewer"] == "ollama/qwen3:14b"
    assert settings.models.subagents["editor"] == "ollama/qwen3:14b"
    assert settings.env["ANTHROPIC_API_KEY"] == "test-key"


def test_run_quick_setup_with_per_tier_model_customization(monkeypatch, tmp_path):
    """Opting into per-tier customization shows the catalog picker for each
    tier and lets the user override the provider's built-in default."""
    monkeypatch.setattr(ob.rec, "detect_hardware", lambda: HW)
    answers = [
        "quick", "1",  # mode, local#
        "1", "test-key",  # provider#, key
        "claude-haiku-4-5",  # main tier override (orchestrator + escalation) — default would be sonnet
        "",  # flagship tier — accept default (opus)
        "user",  # scope
    ]
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(answers))}))
    # use cloud provider? yes · customize models per tier? yes.
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([True, True])))

    settings = ob.run(_console(), root=tmp_path, verify=False)
    assert settings.models.orchestrator == "anthropic:claude-haiku-4-5"
    assert settings.models.escalation_model == "anthropic:claude-haiku-4-5"
    assert settings.models.advisor == "anthropic:claude-opus-4-8"  # accepted default
    # Only tiers a cloud role actually uses are asked about — no "light" prompt
    # now that reviewer is local, which the answer queue above depends on.
    assert settings.models.subagents["reviewer"] == "ollama/qwen3:14b"


def test_run_respects_explicit_scope_skips_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(ob.rec, "detect_hardware", lambda: HW)
    # No scope answer in the queue at all — if `run` still prompted for scope
    # despite `scope="project"`, this would raise EOFError.
    answers = ["quick", "1", "n"]
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(answers))}))
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([False])))

    settings = ob.run(_console(), root=tmp_path, verify=False, scope="project")
    project_file = settings_mod.project_settings_paths(tmp_path)[0]
    assert project_file.exists()
    assert settings.models.orchestrator == "ollama/qwen3:14b"


def test_run_advanced_setup_per_role(monkeypatch, tmp_path):
    monkeypatch.setattr(ob.rec, "detect_hardware", lambda: HW)
    # mode; then per role in `roles=("orchestrator", "editor")`:
    #   orchestrator: cloud, provider#1 (anthropic), key, model id (default)
    #   editor: local, model#
    # then scope.
    answers = [
        "advanced",
        "cloud", "1", "test-key", "claude-sonnet-4-6",
        "local", "1",
        "user",
    ]
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(answers))}))
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([])))

    settings = ob.run(_console(), root=tmp_path, verify=False, roles=("orchestrator", "editor"))
    assert settings.models.orchestrator == "anthropic:claude-sonnet-4-6"
    assert settings.models.subagents["editor"] == "ollama/qwen3:14b"
    assert settings.env["ANTHROPIC_API_KEY"] == "test-key"


# --------------------------------------------------------------------------
# The privacy step — stubbed above so the model scripts stay in sync; these
# pin *when* run() invokes it.
# --------------------------------------------------------------------------


def test_quick_setup_includes_the_privacy_step(monkeypatch, tmp_path, _no_real_privacy_step):
    """Quick mode exists to be fast, not to silently decide what goes home —
    the privacy question is part of it."""
    monkeypatch.setattr(ob.rec, "detect_hardware", lambda: HW)
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["quick", "1", "user"]))}))
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([False])))

    ob.run(_console(), root=tmp_path, verify=False)

    assert len(_no_real_privacy_step) == 1


def test_advanced_setup_includes_the_privacy_step(monkeypatch, tmp_path, _no_real_privacy_step):
    monkeypatch.setattr(ob.rec, "detect_hardware", lambda: HW)
    answers = ["advanced", "local", "1", "user"]
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(answers))}))
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([])))

    ob.run(_console(), root=tmp_path, verify=False, roles=("editor",))

    assert len(_no_real_privacy_step) == 1


def test_privacy_false_skips_the_step(monkeypatch, tmp_path, _no_real_privacy_step):
    """`/setup <role>` re-picks one model — not the moment to re-ask consent."""
    monkeypatch.setattr(ob.rec, "detect_hardware", lambda: HW)
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["advanced", "local", "1", "user"]))}))
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([])))

    ob.run(_console(), root=tmp_path, verify=False, roles=("editor",), privacy=False)

    assert _no_real_privacy_step == []


# --------------------------------------------------------------------------
# maybe_setup_playwright — kept out of run() itself (see its docstring) so it
# has its own scripted-prompt tests rather than threading through every
# run() answer sequence above.
# --------------------------------------------------------------------------


def test_maybe_setup_playwright_skips_when_server_disabled(tmp_path, monkeypatch):
    from loom.core import playwright_setup as pw_mod

    def boom(*a, **k):
        raise AssertionError("must not check status when the server is disabled")

    monkeypatch.setattr(pw_mod, "status", boom)
    settings = settings_mod.load_settings(tmp_path)
    settings.mcp_servers["playwright"].enabled = False
    ob.maybe_setup_playwright(_console(), settings)  # no-op, no exception


def test_maybe_setup_playwright_hints_when_npx_missing(monkeypatch, tmp_path):
    from loom.core import playwright_setup as pw_mod

    monkeypatch.setattr(pw_mod, "status", lambda: pw_mod.PlaywrightStatus(False, False, tmp_path))

    def boom(*a, **k):
        raise AssertionError("must not prompt Confirm when npx is missing")

    monkeypatch.setattr(ob.render, "confirm", boom)
    settings = settings_mod.load_settings(tmp_path)
    ob.maybe_setup_playwright(_console(), settings)


def test_maybe_setup_playwright_noop_when_already_installed(monkeypatch, tmp_path):
    from loom.core import playwright_setup as pw_mod

    monkeypatch.setattr(pw_mod, "status", lambda: pw_mod.PlaywrightStatus(True, True, tmp_path))

    def boom(*a, **k):
        raise AssertionError("must not prompt Confirm when browsers are already installed")

    monkeypatch.setattr(ob.render, "confirm", boom)
    settings = settings_mod.load_settings(tmp_path)
    ob.maybe_setup_playwright(_console(), settings)


def test_maybe_setup_playwright_installs_on_confirm(monkeypatch, tmp_path):
    from loom.core import playwright_setup as pw_mod

    monkeypatch.setattr(pw_mod, "status", lambda: pw_mod.PlaywrightStatus(True, False, tmp_path))
    installed = []
    monkeypatch.setattr(pw_mod, "install_browsers", lambda console: installed.append(True) or 0)
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([True])))
    settings = settings_mod.load_settings(tmp_path)
    ob.maybe_setup_playwright(_console(), settings)
    assert installed == [True]


def test_maybe_setup_playwright_declines_confirm(monkeypatch, tmp_path):
    from loom.core import playwright_setup as pw_mod

    monkeypatch.setattr(pw_mod, "status", lambda: pw_mod.PlaywrightStatus(True, False, tmp_path))

    def boom(console):
        raise AssertionError("must not install when declined")

    monkeypatch.setattr(pw_mod, "install_browsers", boom)
    monkeypatch.setattr(ob.render, "confirm", _as_confirm(_Scripted([False])))
    settings = settings_mod.load_settings(tmp_path)
    ob.maybe_setup_playwright(_console(), settings)
