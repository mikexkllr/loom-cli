"""Subagent registry, specs, and config-driven model assignment."""

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("pydantic")

from loom.core import config as cfg
from loom.subagents import SPECS, describe_subagents


def _config():
    return cfg.load_config(path=cfg.DEFAULT_CONFIG_PATH)


def test_all_seven_subagents_registered():
    # "general-purpose" must keep deepagents' reserved name — it overrides the
    # unrestricted default subagent create_deep_agent would otherwise add.
    assert set(SPECS) == {"explorer", "editor", "bash", "searcher", "reviewer", "general-purpose", "tester"}


def test_modes_match_spec():
    assert SPECS["explorer"].mode == "read-only"
    assert SPECS["searcher"].mode == "read-only"
    assert SPECS["reviewer"].mode == "read-only"
    assert SPECS["editor"].mode == "write"
    assert SPECS["bash"].mode == "write"
    assert SPECS["general-purpose"].mode == "write"
    assert SPECS["tester"].mode == "write"


def test_tool_sets_match_spec():
    # deepagents' FilesystemMiddleware injects ls/read/write/edit/glob/grep/execute
    # into every subagent, so Loom's specs only list extra tools (web_search,
    # plus MCP browser tools that are appended at orchestrator build time).
    assert {t.name for t in SPECS["explorer"].tools} == set()
    assert {t.name for t in SPECS["editor"].tools} == set()
    assert {t.name for t in SPECS["bash"].tools} == set()
    assert {t.name for t in SPECS["reviewer"].tools} == set()

    assert {t.name for t in SPECS["searcher"].tools} == {"web_search"}
    assert {t.name for t in SPECS["general-purpose"].tools} == {"web_search"}

    # tester: browser_* MCP tools are injected at build time; write_file is
    # supplied by deepagents' FilesystemMiddleware.
    assert {t.name for t in SPECS["tester"].tools} == set()


def test_describe_subagents_marks_local_vs_cloud():
    rows = {r["name"]: r for r in describe_subagents(_config())}
    assert rows["explorer"]["scope"] == "local"
    assert rows["reviewer"]["scope"] == "cloud"


def test_local_subagents_get_prompt_size_guard():
    from loom.middleware.prompt_size_guard import PromptSizeGuard

    config = _config()
    sub = SPECS["explorer"].build(config)  # explorer is local
    assert any(isinstance(m, PromptSizeGuard) for m in sub["middleware"])
    assert sub["name"] == "explorer"
    assert sub["system_prompt"]


def test_reviewer_has_structured_response_format():
    from loom.subagents import build_all_subagents
    from loom.core.advisor import ReviewVerdict

    subs = {s["name"]: s for s in build_all_subagents(_config())}
    assert subs["reviewer"].get("response_format") is ReviewVerdict


# ---------------------------------------------------------------------------
# Role inheritance: reviewer trails the advisor unless pinned
# ---------------------------------------------------------------------------


def test_reviewer_inherits_the_advisor_model_when_unassigned():
    from loom.subagents import model_for

    config = cfg.LoomConfig(
        orchestrator="claude-sonnet-5",
        advisor="claude-opus-4-8",
        subagents={"explorer": "ollama/qwen3:4b", "general-purpose": "ollama/qwen3:9b"},
    )
    assert model_for(config, "reviewer") == "claude-opus-4-8"


def test_an_explicit_reviewer_assignment_wins_over_inheritance():
    from loom.subagents import model_for

    config = cfg.LoomConfig(
        orchestrator="claude-sonnet-5",
        advisor="claude-opus-4-8",
        subagents={"reviewer": "ollama/qwen3:9b"},
    )
    assert model_for(config, "reviewer") == "ollama/qwen3:9b"


def test_other_roles_still_inherit_general_purpose_not_the_advisor():
    from loom.subagents import model_for

    config = cfg.LoomConfig(
        orchestrator="claude-sonnet-5",
        advisor="claude-opus-4-8",
        subagents={"general-purpose": "ollama/qwen3:9b"},
    )
    assert model_for(config, "explorer") == "ollama/qwen3:9b"
    assert model_for(config, "editor") == "ollama/qwen3:9b"


def test_packaged_default_leaves_reviewer_to_the_advisor():
    """The shipped config must not pin reviewer, or inheritance never applies."""
    from loom.core.config import DEFAULT_CONFIG_PATH, load_config
    from loom.subagents import model_for

    config = load_config(DEFAULT_CONFIG_PATH)
    assert "reviewer" not in config.subagents
    assert model_for(config, "reviewer") == config.advisor


def test_reviewer_subagent_is_built_on_the_inherited_model():
    from loom.subagents import build_all_subagents

    config = cfg.LoomConfig(
        orchestrator="claude-sonnet-5",
        advisor="ollama/qwen3:27b",
        subagents={"general-purpose": "ollama/qwen3:9b"},
    )
    reviewer = next(s for s in build_all_subagents(config) if s["name"] == "reviewer")
    assert getattr(reviewer["model"], "model", None) == "qwen3:27b"
