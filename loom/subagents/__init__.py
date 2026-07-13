"""Subagent registry.

Exposes the seven specialized subagents and a builder that resolves them against
config into deepagents subagent dicts.
"""

from __future__ import annotations

from typing import Any

from loom.core.config import LoomConfig
from loom.subagents import bash, editor, explorer, general, reviewer, searcher, tester

# Ordered registry of specs. Names match config keys and the spec table.
SPECS = {
    "explorer": explorer.SPEC,
    "editor": editor.SPEC,
    "bash": bash.SPEC,
    "searcher": searcher.SPEC,
    "reviewer": reviewer.SPEC,
    "general": general.SPEC,
    "tester": tester.SPEC,
}


def build_all_subagents(config: LoomConfig) -> list[dict[str, Any]]:
    """Resolve every registered subagent into a deepagents subagent dict."""
    out: list[dict[str, Any]] = []
    for name, spec in SPECS.items():
        sub = spec.build(config)
        if name == "reviewer":
            # Reviewer returns a structured verdict the orchestrator can gate on.
            sub["response_format"] = reviewer.RESPONSE_FORMAT
        out.append(sub)
    return out


def describe_subagents(config: LoomConfig) -> list[dict[str, str]]:
    """Lightweight view for ``loom agents list`` — no model construction."""
    rows = []
    for name, spec in SPECS.items():
        model = config.subagents.get(name, "(inherit)")
        rows.append(
            {
                "name": name,
                "model": model,
                "scope": "local" if config.is_local(model) else "cloud",
                "mode": spec.mode,
                "tools": ", ".join(t.name for t in spec.tools) if spec.tools else "(inherit)",
                "description": spec.description,
            }
        )
    return rows


__all__ = ["SPECS", "build_all_subagents", "describe_subagents"]
