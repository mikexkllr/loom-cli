"""Subagent registry, specs, and config-driven model assignment."""

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("pydantic")

from loom.core import config as cfg
from loom.subagents import SPECS, describe_subagents


def _config():
    return cfg.load_config(path=cfg.DEFAULT_CONFIG_PATH)


def test_all_seven_subagents_registered():
    assert set(SPECS) == {"explorer", "editor", "bash", "searcher", "reviewer", "general", "tester"}


def test_modes_match_spec():
    assert SPECS["explorer"].mode == "read-only"
    assert SPECS["searcher"].mode == "read-only"
    assert SPECS["reviewer"].mode == "read-only"
    assert SPECS["editor"].mode == "write"
    assert SPECS["bash"].mode == "write"
    assert SPECS["general"].mode == "write"
    assert SPECS["tester"].mode == "write"


def test_tool_sets_match_spec():
    explorer_tools = {t.name for t in SPECS["explorer"].tools}
    assert explorer_tools == {"ls", "read_file", "glob", "grep"}

    editor_tools = {t.name for t in SPECS["editor"].tools}
    assert editor_tools == {"read_file", "write_file", "edit_file"}

    bash_tools = {t.name for t in SPECS["bash"].tools}
    assert "execute" in bash_tools

    reviewer_tools = {t.name for t in SPECS["reviewer"].tools}
    assert reviewer_tools == {"read_file", "grep"}

    general_tools = {t.name for t in SPECS["general"].tools}
    assert "execute" in general_tools and "write_file" in general_tools

    # tester: write_file for evidence reports; browser_* tools come from the
    # Playwright MCP server at build time.
    assert {t.name for t in SPECS["tester"].tools} == {"write_file"}


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
