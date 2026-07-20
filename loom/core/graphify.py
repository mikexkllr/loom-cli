"""Graphify integration — a code knowledge graph + GraphRAG for the repo.

`Graphify <https://github.com/safishamsi/graphify>`_ (``uv tool install
graphifyy``) parses the codebase with tree-sitter into an explicit graph of
entities and relationships (calls, imports, defines) stored at
``graphify-out/graph.json``. Queries traverse the graph instead of re-reading
files, so "where is X / what connects A to B / what depends on Y" costs a
subgraph's worth of tokens rather than a glob+grep+read sweep.

Loom mounts it two ways:

- ``graphify . --mcp`` runs as a stdio MCP server (see ``mcp_servers`` in
  settings.json, entry ``graphify``, disabled until a graph exists). Its
  read-only tools (:data:`GRAPH_TOOL_NAMES`) are handed to the orchestrator
  and the explorer/searcher subagents.
- The ``/graphify`` REPL command builds/updates the graph and toggles the
  server; ``/graphify query|path|explain`` runs one-off CLI queries.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

# Read-only tools exposed by `graphify --mcp`. Also allow-listed in
# default_settings.json so they never stall on an approval prompt.
GRAPH_TOOL_NAMES = frozenset({"query_graph", "get_node", "shortest_path", "list_prs"})

OUT_DIR = "graphify-out"
PYPI_NAME = "graphifyy"  # yes, double-y — the CLI it installs is `graphify`
INSTALL_HINT = (
    f"install: `uv tool install {PYPI_NAME}` (or `pipx install {PYPI_NAME}`) — "
    "https://github.com/safishamsi/graphify"
)


def binary() -> str | None:
    """Path to the graphify CLI: PATH first, then the uv/pipx tool-bin dir
    (~/.local/bin), which may not be on PATH right after a fresh install."""
    found = shutil.which("graphify")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "graphify"
    return str(local) if local.exists() else None


def installed() -> bool:
    return binary() is not None


def install() -> tuple[bool, str]:
    """Install the graphify CLI via `uv tool install` (pipx as fallback),
    streaming installer output to the terminal. Returns (ok, how)."""
    for runner, cmd in (
        ("uv", ["uv", "tool", "install", PYPI_NAME]),
        ("pipx", ["pipx", "install", PYPI_NAME]),
    ):
        if shutil.which(runner) is None:
            continue
        if subprocess.run(cmd).returncode == 0 and installed():
            return True, " ".join(cmd)
        return False, " ".join(cmd)
    return False, "neither `uv` nor `pipx` found"


def graph_file(cwd: str | Path = ".") -> Path:
    return Path(cwd).resolve() / OUT_DIR / "graph.json"


def graph_exists(cwd: str | Path = ".") -> bool:
    return graph_file(cwd).exists()


def graph_stats(cwd: str | Path = ".") -> dict[str, Any] | None:
    """Best-effort {nodes, edges, size_kb} from graph.json; None if absent."""
    path = graph_file(cwd)
    if not path.exists():
        return None
    stats: dict[str, Any] = {"size_kb": path.stat().st_size // 1024}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        for nodes_key in ("nodes", "entities"):
            if isinstance(data.get(nodes_key), list):
                stats["nodes"] = len(data[nodes_key])
                break
        for edges_key in ("edges", "links", "relationships"):
            if isinstance(data.get(edges_key), list):
                stats["edges"] = len(data[edges_key])
                break
    except Exception:
        pass  # stats stay size-only for unknown schema versions
    return stats


def format_stats(stats: dict[str, Any] | None) -> str:
    """Human line for :func:`graph_stats`, e.g. ``1,204 nodes · 3,880 edges · 412 KB``."""
    if not stats:
        return ""
    parts = []
    if "nodes" in stats:
        parts.append(f"{stats['nodes']:,} nodes")
    if "edges" in stats:
        parts.append(f"{stats['edges']:,} edges")
    if "size_kb" in stats:
        parts.append(f"{stats['size_kb']:,} KB")
    return " · ".join(parts)


def graph_tools_from(tools: list[Any]) -> list[Any]:
    """Filter a flat MCP tool list down to Graphify's graph-query tools."""
    return [t for t in tools if getattr(t, "name", "") in GRAPH_TOOL_NAMES]


def build_command(update: bool = False) -> list[str]:
    """The CLI invocation that (re)builds the graph for the current repo."""
    cmd = [binary() or "graphify", "."]
    if update:
        cmd.append("--update")
    return cmd


def build(cwd: str | Path, update: bool = False) -> int:
    """Run the graph build in ``cwd``, streaming output to the terminal."""
    return subprocess.run(build_command(update), cwd=str(cwd)).returncode


def run_cli(cwd: str | Path, *args: str, timeout: int = 120) -> tuple[int, str]:
    """Run a one-off `graphify <args>` query; returns (exit code, output)."""
    proc = subprocess.run(
        [binary() or "graphify", *args], cwd=str(cwd), capture_output=True, text=True, timeout=timeout
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()
