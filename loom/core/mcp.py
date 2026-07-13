"""MCP client integration — persistent sessions on a background event loop.

Loom's agent loop is synchronous, but MCP servers speak an async protocol and
stateful servers (Playwright's browser above all) must keep ONE session alive
across tool calls — a per-call session would open a fresh browser every click.

So :class:`MCPManager` runs a dedicated daemon thread with its own asyncio
loop. A single long-lived task on that loop opens every configured server
session inside one ``AsyncExitStack`` and holds it until shutdown (open and
close happen in the same task, which anyio's cancel scopes require). Each
loaded MCP tool is wrapped in a sync ``StructuredTool`` that dispatches to the
manager loop via ``run_coroutine_threadsafe``, so subagents can call browser
tools from plain sync code.

Failures degrade gracefully: a server that won't start (or a missing
``langchain-mcp-adapters`` install) is recorded in ``errors`` and simply
contributes no tools.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from typing import Any

from loom.core.settings import MCPServer, Settings

# How long to wait for all servers to come up (first `npx @playwright/mcp`
# run downloads the package) and for a single tool call to finish.
_STARTUP_TIMEOUT = 120.0
_CALL_TIMEOUT = 180.0


def wrap_async_tool(tool: Any, loop: asyncio.AbstractEventLoop, timeout: float = _CALL_TIMEOUT) -> Any:
    """Wrap an async-only LangChain tool so it is callable from sync code.

    All invocations — sync or async — are marshalled onto ``loop``, where the
    underlying MCP session lives; touching the session from another loop or
    thread is unsafe.
    """
    from langchain_core.tools import StructuredTool

    def _call(**kwargs: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(tool.ainvoke(kwargs), loop)
        return future.result(timeout=timeout)

    async def _acall(**kwargs: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(tool.ainvoke(kwargs), loop)
        return await asyncio.wrap_future(future)

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.args_schema,
        func=_call,
        coroutine=_acall,
    )


class MCPManager:
    """Owns the MCP background loop, sessions, and sync-wrapped tools."""

    def __init__(self, servers: dict[str, MCPServer]):
        self.servers = {name: s for name, s in servers.items() if s.enabled}
        self.tools: list[Any] = []
        self.tools_by_server: dict[str, list[str]] = {}
        self.errors: dict[str, str] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: asyncio.Event | None = None
        self._runner_future: Any = None
        self._started = False

    # ------------------------------------------------------------------ start

    def start(self) -> list[Any]:
        """Connect to every enabled server; return the sync-wrapped tools."""
        if self._started:
            return self.tools
        self._started = True
        if not self.servers:
            return []

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: F401
        except ImportError:
            self.errors["*"] = "langchain-mcp-adapters is not installed; MCP servers disabled"
            return []

        connections = self._connections()
        if not connections:
            return []

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="loom-mcp", daemon=True)
        self._thread.start()

        ready = threading.Event()
        async_tools: list[Any] = []
        self._runner_future = asyncio.run_coroutine_threadsafe(
            self._runner(connections, async_tools, ready), self._loop
        )
        if not ready.wait(timeout=_STARTUP_TIMEOUT):
            self.errors["*"] = f"MCP servers did not come up within {_STARTUP_TIMEOUT:.0f}s"
            return []

        self.tools = [wrap_async_tool(t, self._loop) for t in async_tools]
        atexit.register(self.stop)
        return self.tools

    def _connections(self) -> dict[str, dict[str, Any]]:
        connections: dict[str, dict[str, Any]] = {}
        for name, server in self.servers.items():
            if server.transport == "stdio":
                if not server.command:
                    self.errors[name] = "stdio server has no command"
                    continue
                connections[name] = {
                    "transport": "stdio",
                    "command": server.command,
                    "args": list(server.args),
                    "env": dict(server.env) or None,
                }
            else:
                connections[name] = {"transport": server.transport, "url": server.url}
        return connections

    async def _runner(
        self,
        connections: dict[str, dict[str, Any]],
        async_tools: list[Any],
        ready: threading.Event,
    ) -> None:
        """Single task that opens all sessions, serves until shutdown, closes."""
        from contextlib import AsyncExitStack

        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        self._shutdown = asyncio.Event()
        client = MultiServerMCPClient(connections)
        try:
            async with AsyncExitStack() as stack:
                for name in connections:
                    try:
                        session = await stack.enter_async_context(client.session(name))
                        loaded = await load_mcp_tools(session)
                        self.tools_by_server[name] = [t.name for t in loaded]
                        async_tools.extend(loaded)
                    except Exception as exc:  # server-level failure: skip it
                        self.errors[name] = f"{type(exc).__name__}: {exc}"
                ready.set()
                await self._shutdown.wait()
        finally:
            ready.set()  # never leave start() hanging on a crash

    # ------------------------------------------------------------------- stop

    def stop(self) -> None:
        """Close sessions and stop the background loop. Idempotent."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        if self._shutdown is not None:
            loop.call_soon_threadsafe(self._shutdown.set)
            try:
                self._runner_future.result(timeout=10)
            except Exception:
                pass
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None


# ----------------------------------------------------------------------------
# Process-wide singleton — sessions persist across orchestrator rebuilds
# (e.g. REPL mode switches) so the browser keeps its state.
# ----------------------------------------------------------------------------

_manager: MCPManager | None = None


def get_mcp_tools(settings: Settings) -> list[Any]:
    """Start (once) and return sync-wrapped tools for all configured servers."""
    global _manager
    if _manager is None:
        _manager = MCPManager(settings.mcp_servers)
        _manager.start()
    return _manager.tools


def mcp_errors() -> dict[str, str]:
    """Connection errors from the active manager (empty if none/unstarted)."""
    return dict(_manager.errors) if _manager is not None else {}


def mcp_status(settings: Settings) -> list[dict[str, Any]]:
    """Per-server status rows for the /mcp and /doctor commands.

    Reflects the live manager when started; otherwise just the configuration.
    """
    rows: list[dict[str, Any]] = []
    for name, server in settings.mcp_servers.items():
        row: dict[str, Any] = {
            "name": name,
            "transport": server.transport,
            "target": server.url or f"{server.command} {' '.join(server.args)}".strip(),
            "enabled": server.enabled,
            "state": "not connected",
            "tools": [],
        }
        if _manager is not None and _manager._started:
            if name in _manager.tools_by_server:
                row["state"] = "connected"
                row["tools"] = list(_manager.tools_by_server[name])
            elif name in _manager.errors:
                row["state"] = f"failed: {_manager.errors[name]}"
            elif not server.enabled:
                row["state"] = "disabled"
        elif not server.enabled:
            row["state"] = "disabled"
        rows.append(row)
    return rows


def shutdown_mcp() -> None:
    global _manager
    if _manager is not None:
        _manager.stop()
        _manager = None
