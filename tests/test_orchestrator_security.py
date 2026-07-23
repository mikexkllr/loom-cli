"""Security-model regression tests for the deepagents integration.

deepagents builds a fresh middleware stack per subagent and auto-adds an
unrestricted ``general-purpose`` subagent when none carries that name. These
tests pin Loom's countermeasures: the reserved name is always claimed, the
policy gate rides inside every subagent, read-only subagents actually lose
their write tools, and the ``delete`` tool is covered everywhere.
"""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langchain_core")

from loom.core import ollama
from loom.core.config import LoomConfig
from loom.core.ollama import OllamaStatus
from loom.core.settings import Settings
from loom.middleware.policy import PolicyMiddleware
from loom.middleware.tool_exclusion import ToolExclusionMiddleware
from loom.subagents import WRITE_TOOLS, build_all_subagents


def _config(**kw):
    defaults = dict(
        orchestrator="ollama/qwen3:14b",
        subagents={
            n: "ollama/qwen3:4b"
            for n in ("explorer", "editor", "bash", "searcher", "reviewer", "general-purpose", "tester")
        },
    )
    defaults.update(kw)
    return LoomConfig(**defaults)


def _settings(**kw):
    return Settings(models=_config(**kw))


def _exclusions(sub) -> set[str]:
    out: set[str] = set()
    for m in sub["middleware"]:
        if isinstance(m, ToolExclusionMiddleware):
            out |= set(m._excluded)
    return out


# ---------------------------------------------------------------------------
# Per-subagent middleware
# ---------------------------------------------------------------------------


def test_every_subagent_carries_the_policy_gate():
    settings = _settings()
    subs = build_all_subagents(settings.models, settings, ".")
    for sub in subs:
        assert any(isinstance(m, PolicyMiddleware) for m in sub["middleware"]), sub["name"]


def test_policy_gate_skipped_without_settings():
    subs = build_all_subagents(_config())  # bare LoomConfig back-compat path
    for sub in subs:
        assert not any(isinstance(m, PolicyMiddleware) for m in sub["middleware"])


def test_read_only_subagents_lose_write_tools():
    subs = {s["name"]: s for s in build_all_subagents(_config())}
    for name in ("explorer", "searcher", "reviewer"):
        assert WRITE_TOOLS <= _exclusions(subs[name]), name


def test_editor_and_tester_cannot_execute():
    subs = {s["name"]: s for s in build_all_subagents(_config())}
    assert "execute" in _exclusions(subs["editor"])
    assert "execute" in _exclusions(subs["tester"])
    # ...but they keep their write tools.
    assert "write_file" not in _exclusions(subs["editor"])
    assert "write_file" not in _exclusions(subs["tester"])


def test_read_only_build_strips_write_tools_everywhere():
    subs = build_all_subagents(_config(), read_only=True)
    for sub in subs:
        assert WRITE_TOOLS <= _exclusions(sub), sub["name"]


# ---------------------------------------------------------------------------
# Orchestrator cannot re-do a subagent's broad search itself
# ---------------------------------------------------------------------------


def test_orchestrator_loses_broad_search_tools():
    """glob/grep/ls must be quarantined to explorer/searcher. Previously this was
    prompt-only guidance ("delegate, don't investigate yourself"), which the
    orchestrator model could ignore — strong cloud models (e.g. gpt-5.5) map the
    tree with ls and sweep files themselves instead of routing recon to a local
    explorer, in a cloud context, after (or instead of) a subagent doing it."""
    from loom.core.orchestrator import _orchestrator_excluded_tools

    normal = _orchestrator_excluded_tools(airgap=False)
    assert {"glob", "grep", "ls", "write_file", "edit_file", "delete", "execute"} <= normal
    assert "read_file" not in normal  # targeted single-file confirmations stay allowed


def test_orchestrator_airgap_loses_every_filesystem_tool():
    from loom.core.orchestrator import _ALL_FS_TOOLS, _orchestrator_excluded_tools

    assert _orchestrator_excluded_tools(airgap=True) == set(_ALL_FS_TOOLS)


# ---------------------------------------------------------------------------
# delete-tool coverage
# ---------------------------------------------------------------------------


def test_delete_supports_path_permission_specifiers():
    from loom.core.permissions import Decision, Permissions, check

    perms = Permissions(default_mode="allow", deny=["delete(secrets/**)"])
    assert check("delete", {"file_path": "/secrets/key.pem"}, perms) is Decision.deny
    assert check("delete", {"file_path": "/src/app.py"}, perms) is Decision.allow


def test_default_settings_gate_delete():
    from loom.core import settings as settings_mod
    from loom.core.permissions import Decision, check

    defaults = settings_mod._read_json(settings_mod.DEFAULT_SETTINGS_PATH)
    perms = settings_mod.Permissions(**defaults["permissions"])
    assert check("delete", {"file_path": "/x.py"}, perms) is Decision.ask
    # Delegation itself is allowed — approval happens at the write inside.
    assert check("task", {}, perms) is Decision.allow


def test_policy_snapshots_before_delete(tmp_path, monkeypatch):
    from loom.core import undo

    target = tmp_path / "doomed.txt"
    target.write_text("precious")
    undo.current_turn_id.set("t-del")
    mw = PolicyMiddleware(_settings(), cwd=str(tmp_path))

    from types import SimpleNamespace

    from loom.middleware import policy as policy_mod

    policy_mod.auto_approve.set(True)
    try:
        req = SimpleNamespace(call={"name": "delete", "args": {"file_path": "doomed.txt"}, "id": "1"})
        mw.wrap_tool_call(req, lambda r: target.unlink() or SimpleNamespace(content="gone"))
    finally:
        policy_mod.auto_approve.set(False)
        undo.current_turn_id.set("")

    assert not target.exists()
    assert undo.undo_last(tmp_path) == ["doomed.txt"]
    assert target.read_text() == "precious"


# ---------------------------------------------------------------------------
# The reserved general-purpose name survives every run mode
# ---------------------------------------------------------------------------


def _stub_ollama(monkeypatch, models=("qwen3:4b", "qwen3:14b")):
    monkeypatch.setattr(
        ollama, "status", lambda cfg: OllamaStatus(True, True, list(models), "http://x")
    )


@pytest.mark.parametrize("mode", [{}, {"plan": True}, {"local_only": True}, {"airgap": True}])
def test_general_purpose_always_present(monkeypatch, mode):
    pytest.importorskip("deepagents")
    from loom.core.orchestrator import build_orchestrator

    _stub_ollama(monkeypatch)
    bundle = build_orchestrator(_settings(), **mode)
    assert "general-purpose" in bundle.subagent_names


def test_general_purpose_rebinds_local_when_configured_cloud(monkeypatch):
    """A cloud-assigned general-purpose must not vanish in airgap mode (that
    would resurrect deepagents' unrestricted default)."""
    pytest.importorskip("deepagents")
    from loom.core.orchestrator import build_orchestrator

    _stub_ollama(monkeypatch)
    settings = _settings()
    settings.models = settings.models.model_copy(
        update={"subagents": {**settings.models.subagents, "general-purpose": "claude-haiku-4-5"}}
    )
    bundle = build_orchestrator(settings, airgap=True)
    assert "general-purpose" in bundle.subagent_names


def test_task_tool_never_advertises_foreign_agents(monkeypatch):
    """End-to-end against real deepagents: the compiled task tool must expose
    exactly the subagents Loom passed — no auto-added extras."""
    pytest.importorskip("deepagents")
    import deepagents.graph as dg
    import deepagents.middleware.subagents as sam

    captured: list[list[str]] = []
    orig = sam.SubAgentMiddleware.__init__

    def spy(self, *args, **kwargs):
        subs = kwargs.get("subagents") or (args[1] if len(args) > 1 else [])
        captured.append([s.get("name") for s in subs if isinstance(s, dict)])
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(sam.SubAgentMiddleware, "__init__", spy)
    monkeypatch.setattr(dg, "SubAgentMiddleware", sam.SubAgentMiddleware)
    _stub_ollama(monkeypatch)

    from loom.core.orchestrator import build_orchestrator

    for mode in ({}, {"plan": True}, {"airgap": True}):
        captured.clear()
        bundle = build_orchestrator(_settings(), **mode)
        assert captured, "SubAgentMiddleware was never constructed"
        assert set(captured[0]) == set(bundle.subagent_names), mode


def test_plan_mode_general_purpose_is_read_only(monkeypatch):
    pytest.importorskip("deepagents")
    captured: list[dict] = []
    import deepagents.graph as dg
    import deepagents.middleware.subagents as sam

    orig = sam.SubAgentMiddleware.__init__

    def spy(self, *args, **kwargs):
        subs = kwargs.get("subagents") or (args[1] if len(args) > 1 else [])
        captured.extend(s for s in subs if isinstance(s, dict))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(sam.SubAgentMiddleware, "__init__", spy)
    monkeypatch.setattr(dg, "SubAgentMiddleware", sam.SubAgentMiddleware)
    _stub_ollama(monkeypatch)

    from loom.core.orchestrator import build_orchestrator

    build_orchestrator(_settings(), plan=True)
    gp = next(s for s in captured if s["name"] == "general-purpose")
    excluded: set[str] = set()
    for m in gp["middleware"]:
        if isinstance(m, ToolExclusionMiddleware):
            excluded |= set(m._excluded)
    assert WRITE_TOOLS <= excluded


# ---------------------------------------------------------------------------
# Legacy config key
# ---------------------------------------------------------------------------


def test_legacy_general_key_maps_to_general_purpose():
    config = LoomConfig(subagents={"general": "ollama/custom:7b"})
    assert "general" not in config.subagents
    assert config.subagents["general-purpose"] == "ollama/custom:7b"


def test_legacy_general_key_survives_layer_merge(tmp_path):
    """A user config.yaml written under the old name must still override the
    packaged default, which now uses the new name."""
    from loom.core import config as cfg

    user = tmp_path / "config.yaml"
    user.write_text("subagents:\n  general: ollama/custom:7b\n")
    config = cfg.load_config(path=user)
    assert config.subagents["general-purpose"] == "ollama/custom:7b"
