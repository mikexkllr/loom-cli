"""Subagent registry.

Exposes the seven specialized subagents and a builder that resolves them against
config into deepagents subagent dicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loom.core.config import LoomConfig
from loom.subagents import bash, editor, explorer, general, reviewer, searcher, tester
from loom.subagents.base import ALL_FS_TOOLS, READ_FS_TOOLS, WRITE_TOOLS

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
    ladder: tuple[tuple[str, int], ...] = (),
    backend: Any | None = None,
    local_only: bool = False,
) -> list[dict[str, Any]]:
    """Resolve every registered subagent into a deepagents subagent dict.

    ``settings`` attaches the per-subagent policy gate (permissions, hooks,
    /undo snapshots). ``read_only=True`` (plan mode) strips the write/execute
    tools from every subagent, not just the read-only ones. ``ladder`` is the
    served-local-model ladder each local subagent's prompt-size guard climbs
    before escalating to the cloud. ``backend`` is the orchestrator's storage
    backend — passing it lets each spec install its own tool-allowlisted
    ``FilesystemMiddleware`` in place of deepagents' unrestricted default
    (see :meth:`loom.subagents.base.SubagentSpec.build`).

    ``local_only=True`` (``--local-only`` and ``--airgap``) drops cloud-backed
    roles here rather than filtering them out of the returned list, which is a
    correctness difference and not a tidiness one: constructing a cloud model
    validates its credentials, so a fleet containing one cloud role — the default
    config's ``reviewer`` — made both no-cloud modes fail to start on a machine
    with no cloud key, which is the machine most likely to be asking for them.
    """
    extra = WRITE_TOOLS if read_only else frozenset()
    out: list[dict[str, Any]] = []
    for name, spec in SPECS.items():
        if local_only and not config.is_local(model_for(config, name)):
            continue
        sub = spec.build(
            config, settings, cwd, extra_excluded=extra, ladder=ladder, backend=backend
        )
        if name == "reviewer":
            # Reviewer returns a structured verdict the orchestrator can gate on.
            sub["response_format"] = reviewer.RESPONSE_FORMAT
        out.append(sub)
    return out


def model_for(config: LoomConfig, name: str) -> str:
    """The model a registered subagent actually runs on, honouring the spec's
    inheritance (the reviewer trails the advisor) and any explicit config
    assignment. Use this instead of ``config.subagents[name]`` — an unassigned
    role has a real model, it just isn't spelled out in the file."""
    spec = SPECS.get(name)
    return config.model_for(name, spec.inherits if spec else "general-purpose")


def describe_subagents(config: LoomConfig) -> list[dict[str, str]]:
    """Lightweight view for ``loom agents list`` — no model construction."""
    rows = []
    for name, spec in SPECS.items():
        model = model_for(config, name)
        rows.append(
            {
                "name": name,
                "model": model,
                "scope": "local" if config.is_local(model) else "cloud",
                "mode": spec.mode,
                "tools": ", ".join(sorted(spec.fs_tools) + [t.name for t in spec.tools]),
                "description": spec.description,
            }
        )
    return rows


__all__ = [
    "ALL_FS_TOOLS",
    "READ_FS_TOOLS",
    "SPECS",
    "WRITE_TOOLS",
    "build_all_subagents",
    "describe_subagents",
    "model_for",
]
