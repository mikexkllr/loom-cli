"""Optional web search tool for the ``searcher`` subagent.

Kept dependency-light: if no ``LOOM_SEARCH_API_KEY`` (Tavily-compatible) is set,
the tool degrades to a clear no-op message rather than crashing, so local-only
runs still work.
"""

from __future__ import annotations

import json
import os

import httpx
from langchain_core.tools import tool

_TAVILY_URL = "https://api.tavily.com/search"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for ``query``. Returns titles, URLs, and snippets.

    Requires ``LOOM_SEARCH_API_KEY`` (a Tavily API key). Without it, returns a
    message explaining web search is disabled.
    """
    api_key = os.environ.get("LOOM_SEARCH_API_KEY")
    if not api_key:
        return (
            "web_search is disabled (no LOOM_SEARCH_API_KEY set). "
            "Proceed using local tools, or ask the user to enable web search."
        )
    try:
        resp = httpx.post(
            _TAVILY_URL,
            json={"api_key": api_key, "query": query, "max_results": max_results},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return f"error: web_search failed: {exc}"

    lines = []
    for item in data.get("results", [])[:max_results]:
        lines.append(f"- {item.get('title', '?')} ({item.get('url', '')})\n  {item.get('content', '')[:300]}")
    return "\n".join(lines) if lines else "(no results)"
