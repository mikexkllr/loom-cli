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

from loom.core import ollama as ollama_mod
from loom.core import providers as prov
from loom.core import recommendations as rec
from loom.core import settings as settings_mod
from loom.ui import onboarding as ob

HW = rec.Hardware(os_name="Darwin", ram_gb=32, gpu_vendor="apple", vram_gb=32)


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False)


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


def test_prompt_local_model_picks_installed_by_number(monkeypatch):
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["1"]))}))
    tag = ob.prompt_local_model(_console(), HW)
    assert tag == "qwen3:14b"


def test_prompt_local_model_custom_tag_offers_pull(monkeypatch, _no_real_ollama):
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["qwen2.5-coder:32b"]))}))
    monkeypatch.setattr(ob, "Confirm", type("C", (), {"ask": staticmethod(_Scripted([True]))}))
    tag = ob.prompt_local_model(_console(), HW)
    assert tag == "qwen2.5-coder:32b"
    assert _no_real_ollama == ["qwen2.5-coder:32b"]


def test_prompt_local_model_declines_pull(monkeypatch, _no_real_ollama):
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(["qwen2.5-coder:32b"]))}))
    monkeypatch.setattr(ob, "Confirm", type("C", (), {"ask": staticmethod(_Scripted([False]))}))
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


def test_prompt_cloud_model_defaults_to_tier(monkeypatch):
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(lambda *a, default=None, **k: default)}))
    model_id = ob.prompt_cloud_model(_console(), prov.get("anthropic"), "flagship")
    assert model_id == "claude-opus-4-8"


def test_run_quick_setup_local_only(monkeypatch, tmp_path):
    monkeypatch.setattr(ob.rec, "detect_hardware", lambda: HW)
    answers = ["quick", "1", "user"]  # mode, local model#, scope (Confirm handles the cloud toggle)
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(answers))}))
    monkeypatch.setattr(ob, "Confirm", type("C", (), {"ask": staticmethod(_Scripted([False]))}))

    settings = ob.run(_console(), root=tmp_path)
    assert settings.models.orchestrator == "ollama/qwen3:14b"
    assert settings.models.subagents["editor"] == "ollama/qwen3:14b"


def test_run_quick_setup_with_cloud_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(ob.rec, "detect_hardware", lambda: HW)
    answers = ["quick", "1", "1", "test-key", "user"]  # mode, local#, provider#, key, scope
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(answers))}))
    monkeypatch.setattr(ob, "Confirm", type("C", (), {"ask": staticmethod(_Scripted([True]))}))

    settings = ob.run(_console(), root=tmp_path)
    assert settings.models.orchestrator == "anthropic:claude-sonnet-5"
    assert settings.models.advisor == "anthropic:claude-opus-4-8"
    assert settings.models.subagents["reviewer"] == "anthropic:claude-haiku-4-5"
    assert settings.models.subagents["editor"] == "ollama/qwen3:14b"
    assert settings.env["ANTHROPIC_API_KEY"] == "test-key"


def test_run_respects_explicit_scope_skips_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(ob.rec, "detect_hardware", lambda: HW)
    # No scope answer in the queue at all — if `run` still prompted for scope
    # despite `scope="project"`, this would raise EOFError.
    answers = ["quick", "1", "n"]
    monkeypatch.setattr(ob, "Prompt", type("P", (), {"ask": staticmethod(_Scripted(answers))}))
    monkeypatch.setattr(ob, "Confirm", type("C", (), {"ask": staticmethod(_Scripted([False]))}))

    settings = ob.run(_console(), root=tmp_path, scope="project")
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
    monkeypatch.setattr(ob, "Confirm", type("C", (), {"ask": staticmethod(_Scripted([]))}))

    settings = ob.run(_console(), root=tmp_path, roles=("orchestrator", "editor"))
    assert settings.models.orchestrator == "anthropic:claude-sonnet-4-6"
    assert settings.models.subagents["editor"] == "ollama/qwen3:14b"
    assert settings.env["ANTHROPIC_API_KEY"] == "test-key"
