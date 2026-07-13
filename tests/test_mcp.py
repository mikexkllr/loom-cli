"""MCP settings schema, permission globs, and the sync tool wrapper."""

import asyncio
import json
import threading

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")

from loom.core import permissions as perm
from loom.core import settings as st
from loom.core.mcp import MCPManager, wrap_async_tool
from loom.core.settings import MCPServer


# ----------------------------------------------------------------------- schema


def test_default_settings_include_playwright():
    s = st.load_settings()
    assert "playwright" in s.mcp_servers
    pw = s.mcp_servers["playwright"]
    assert pw.transport == "stdio"
    assert pw.command == "npx"
    assert "@playwright/mcp@latest" in pw.args
    assert pw.enabled


def test_http_server_requires_url():
    with pytest.raises(Exception):
        MCPServer(transport="streamable_http")
    ok = MCPServer(transport="streamable_http", url="http://localhost:8931/mcp")
    assert ok.url


def test_project_layer_can_disable_playwright(tmp_path):
    proj = tmp_path / "proj"
    (proj / ".loom").mkdir(parents=True)
    (proj / ".loom" / "settings.json").write_text(
        json.dumps({"mcp_servers": {"playwright": {"enabled": False}}})
    )
    s = st.load_settings(root=proj)
    assert s.mcp_servers["playwright"].enabled is False
    # Disabled servers are filtered out by the manager.
    mgr = MCPManager(s.mcp_servers)
    assert "playwright" not in mgr.servers


# ------------------------------------------------------------------ permissions


def test_browser_tools_allowed_by_default_glob():
    s = st.load_settings()
    assert perm.check("browser_navigate", {"url": "http://localhost:3000"}, s.permissions) == perm.Decision.allow
    assert perm.check("browser_click", {}, s.permissions) == perm.Decision.allow


def test_bare_glob_does_not_overmatch():
    p = st.Permissions(default_mode="ask", allow=["browser_*"])
    assert perm.check("browser_snapshot", {}, p) == perm.Decision.allow
    assert perm.check("execute", {"command": "rm -rf ."}, p) == perm.Decision.ask


# ---------------------------------------------------------------------- manager


def test_manager_no_servers_is_noop():
    mgr = MCPManager({})
    assert mgr.start() == []
    mgr.stop()  # idempotent, no loop running


def test_manager_stdio_without_command_records_error():
    mgr = MCPManager({"broken": MCPServer(transport="stdio", command="")})
    conns = mgr._connections()
    assert conns == {}
    assert "broken" in mgr.errors


# ---------------------------------------------------------------------- wrapper


def test_wrap_async_tool_dispatches_to_manager_loop():
    pytest.importorskip("langchain_core")
    from langchain_core.tools import StructuredTool

    calls: list[str] = []

    async def _shout(text: str) -> str:
        calls.append(threading.current_thread().name)
        return text.upper()

    async_tool = StructuredTool.from_function(coroutine=_shout, name="shout", description="uppercase")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="test-mcp-loop", daemon=True)
    thread.start()
    try:
        wrapped = wrap_async_tool(async_tool, loop, timeout=10)
        assert wrapped.name == "shout"
        # Sync path from the main thread runs on the manager loop's thread.
        assert wrapped.invoke({"text": "hi"}) == "HI"
        assert calls == ["test-mcp-loop"]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
