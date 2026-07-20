"""Graphify integration: module helpers, default settings, orchestrator
wiring, and the /graphify slash command."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")
pytest.importorskip("rich")
pytest.importorskip("langchain_core")

from loom.core import graphify
from loom.core import settings as st
from loom.ui import slash
from loom.ui.repl import Session


def _session(tmp_path):
    return Session(st.load_settings(tmp_path), cwd=str(tmp_path))


def _write_graph(tmp_path, nodes=3, edges=2):
    out = tmp_path / graphify.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph.json").write_text(
        json.dumps({"nodes": [{}] * nodes, "edges": [{}] * edges})
    )


# ------------------------------------------------------------ module helpers


def test_graph_file_and_stats(tmp_path):
    assert not graphify.graph_exists(tmp_path)
    assert graphify.graph_stats(tmp_path) is None
    _write_graph(tmp_path, nodes=5, edges=7)
    assert graphify.graph_exists(tmp_path)
    stats = graphify.graph_stats(tmp_path)
    assert stats["nodes"] == 5 and stats["edges"] == 7


def test_graph_stats_survives_unknown_schema(tmp_path):
    out = tmp_path / graphify.OUT_DIR
    out.mkdir(parents=True)
    (out / "graph.json").write_text('{"weird": true}')
    stats = graphify.graph_stats(tmp_path)
    assert stats is not None and "size_kb" in stats


def test_build_command_forms():
    assert graphify.build_command() == ["graphify", "."]
    assert graphify.build_command(update=True) == ["graphify", ".", "--update"]


def test_graph_tools_from_filters_by_name():
    tools = [SimpleNamespace(name=n) for n in ("query_graph", "browser_click", "get_node", "execute")]
    assert [t.name for t in graphify.graph_tools_from(tools)] == ["query_graph", "get_node"]


# ------------------------------------------------------------ settings & permissions


def test_default_settings_include_graphify_server_disabled():
    s = st.load_settings()
    assert "graphify" in s.mcp_servers
    srv = s.mcp_servers["graphify"]
    assert srv.enabled is False  # opt-in: needs the CLI installed + a built graph
    assert srv.command == "graphify"
    assert "--mcp" in srv.args


def test_graph_query_tools_are_allowed_by_default():
    from loom.core.permissions import Decision, check

    s = st.load_settings()
    for tool in sorted(graphify.GRAPH_TOOL_NAMES):
        assert check(tool, {}, s.permissions) is Decision.allow


# ------------------------------------------------------------ orchestrator prompt


def test_graph_suffix_mentions_the_tools():
    from loom.core.orchestrator import GRAPH_SUFFIX

    for tool in ("query_graph", "get_node", "shortest_path"):
        assert tool in GRAPH_SUFFIX


# ------------------------------------------------------------ slash command


def test_graphify_status_when_not_installed(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(graphify.shutil, "which", lambda _: None)
    s = _session(tmp_path)
    assert slash.dispatch(s, "/graphify") is True
    out = capsys.readouterr().out
    assert "not installed" in out
    assert "not built" in out


def test_graphify_on_requires_a_graph(tmp_path, capsys, monkeypatch):
    s = _session(tmp_path)
    slash.dispatch(s, "/graphify on")
    assert "run /graphify build first" in capsys.readouterr().out


def test_graphify_build_enables_server(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(st, "USER_SETTINGS_PATH", tmp_path / "user-settings.json")
    monkeypatch.setattr(graphify.shutil, "which", lambda _: "/usr/bin/graphify")

    def fake_build(cwd, update=False):
        _write_graph(tmp_path)
        return 0

    monkeypatch.setattr(graphify, "build", fake_build)
    s = _session(tmp_path)
    slash.dispatch(s, "/graphify build")
    out = capsys.readouterr().out
    assert "graph ready" in out
    assert s.settings.mcp_servers["graphify"].enabled is True
    assert s.bundle is None  # rebuilt so the MCP server connects next turn


def test_graphify_query_runs_cli(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(graphify.shutil, "which", lambda _: "/usr/bin/graphify")
    _write_graph(tmp_path)
    calls = []

    def fake_run_cli(cwd, *args, timeout=120):
        calls.append(list(args))
        return 0, "auth -> db via SessionStore"

    monkeypatch.setattr(graphify, "run_cli", fake_run_cli)
    s = _session(tmp_path)
    slash.dispatch(s, '/graphify query what connects auth to the database?')
    assert calls == [["query", "what connects auth to the database?"]]
    assert "SessionStore" in capsys.readouterr().out
