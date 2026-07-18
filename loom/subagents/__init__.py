"""Subagent registry.

Exposes the seven specialized subagents and a builder that resolves them against
config into deepagents subagent dicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loom.core.config import LoomConfig
from loom.subagents import bash, editor, explorer, general, reviewer, searcher, tester
from loom.subagents.base import WRITE_TOOLS

if TYPE_CHECKING:
    from loom.core.settings import Settings

# Ordered registry of specs. Names match config keys and the spec table.
# "general-purpose" MUST keep that exact name: it overrides the subagent
# deepagents would otherwise auto-add with the orchestrator's model and an
# unrestricted toolset (see loom/subagents/general.py).
SPECS = {
    "explorer": explorer.SPEC,
    "editor": editor.SPEC,
    "bash": bash.SPEC,
    "searcher": searcher.SPEC,
    "reviewer": reviewer.SPEC,
    "general-purpose": general.SPEC,
    "tester": tester.SPEC,
}


def build_all_subagents(
    config: LoomConfig,
    settings: "Settings | None" = None,
    cwd: str = ".",
    *,
    read_only: bool = False,
) -> list[dict[str, Any]]:
    """Resolve every registered subagent into a deepagents subagent dict.

    ``settings`` attaches the per-subagent policy gate (permissions, hooks,
    /undo snapshots). ``read_only=True`` (plan mode) strips the write/execute
    tools from every subagent, not just the read-only ones.
    """
    extra = WRITE_TOOLS if read_only else frozenset()
    out: list[dict[str, Any]] = []
    for name, spec in SPECS.items():
        sub = spec.build(config, settings, cwd, extra_excluded=extra)
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


__all__ = ["SPECS", "WRITE_TOOLS", "build_all_subagents", "describe_subagents"]
