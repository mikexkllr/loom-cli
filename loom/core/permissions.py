"""Permission engine — matches tool calls against allow/ask/deny rules.

Rule syntax (Claude Code-flavored):

    "read_file"            -> matches any call to read_file
    "browser_*"            -> bare names are globs: any tool starting browser_
    "execute(git *)"       -> execute whose command matches the glob "git *"
    "write_file(src/**)"   -> write_file whose path matches "src/**"
    "*"                    -> matches every tool call

Evaluation precedence: deny > ALWAYS_ALLOWED > allow > ask > default_mode. The
first of those (in that priority order) that yields a match wins — so an
explicit ``deny`` still overrides everything, including the always-allowed set.
"""

from __future__ import annotations

import fnmatch
from enum import Enum

from loom.core.settings import Permissions

# Which input field a "(specifier)" glob is tested against, per tool.
_SPECIFIER_FIELD = {
    "execute": "command",
    "write_file": "path",
    "edit_file": "path",
    "delete": "path",
    "read_file": "path",
    "ls": "path",
    "glob_tool": "pattern",
    "glob": "pattern",
    "grep_tool": "pattern",
    "grep": "pattern",
    "web_search": "query",
}


# Tools that never stall on an approval prompt.
#
# Two groups, one rule. The coordination tools (spawning subagents, tracking
# the plan, consulting the advisor) only orchestrate — the real side effects
# they set up are gated per-call by these same rules when they happen. The
# read-only filesystem tools navigate and read and mutate nothing: `ls`,
# `glob` and `grep` cannot change a byte on disk, cannot run a command, and
# cannot leave the sandbox root, so an approval prompt for one buys no safety
# and costs the thing approval prompts are for — a user who reads them.
# Prompting on recon is also what trains someone to hit "yes, don't ask again"
# on the shell prompt that actually mattered.
#
# `read_file` is deliberately *not* here: it is the one read that a `deny`
# rule is routinely written for (`read_file(.env)`, `read_file(secrets/**)`),
# and that has to keep working. Deny is checked before this set, so a deny
# rule still wins over everything below — this only removes `ask`.
#
# The packaged defaults already allow all of these. This set is what makes it
# hold when they are *not* the active defaults: a user or project
# settings.json that sets `permissions.allow` replaces the packaged list
# wholesale (lists don't merge on load), which silently re-gated the
# coordination tools once already and would do the same to recon.
_COORDINATION_TOOLS = frozenset({"task", "write_todos", "consult"})
READ_ONLY_TOOLS = frozenset({"ls", "glob", "grep"})
ALWAYS_ALLOWED = _COORDINATION_TOOLS | READ_ONLY_TOOLS


class Decision(str, Enum):
    allow = "allow"
    ask = "ask"
    deny = "deny"


def _tool_input_value(tool_name: str, field: str, tool_input: dict) -> str:
    """Return the value for a specifier field, aliasing ``file_path`` to ``path``.

    deepagents' FilesystemMiddleware tools use ``file_path`` for the target path.
    We also strip a leading ``/`` so that virtual absolute paths produced by
    those tools (``/src/foo.py``) match relative globs the user wrote
    (``src/**``).
    """
    if field == "path":
        value = tool_input.get("path") or tool_input.get("file_path")
    else:
        value = tool_input.get(field)
    if value is None:
        return ""
    value = str(value)
    # Normalize virtual absolute paths to relative-looking strings for glob
    # matching. The actual sandbox resolution happens later in the tool.
    if value.startswith("/"):
        value = value.lstrip("/")
    return value


def _rule_matches(rule: str, tool_name: str, tool_input: dict) -> bool:
    rule = rule.strip()
    if rule in ("*", tool_name):
        return True
    if "(" not in rule:
        # Bare rule: glob over the tool name (covers MCP tool families like
        # "browser_*" without enumerating every tool).
        return fnmatch.fnmatch(tool_name, rule)
    if "(" in rule and rule.endswith(")"):
        name, spec = rule[:-1].split("(", 1)
        name = name.strip()
        if name not in ("*", tool_name):
            return False
        field = _SPECIFIER_FIELD.get(tool_name)
        value = _tool_input_value(tool_name, field, tool_input) if field else " ".join(map(str, tool_input.values()))
        return fnmatch.fnmatch(value, spec.strip())
    return False


def _any(rules: list[str], tool_name: str, tool_input: dict) -> bool:
    return any(_rule_matches(r, tool_name, tool_input) for r in rules)


def check(tool_name: str, tool_input: dict | None, permissions: Permissions) -> Decision:
    """Return the permission :class:`Decision` for a tool call."""
    tool_input = tool_input or {}
    if _any(permissions.deny, tool_name, tool_input):
        return Decision.deny
    if tool_name in ALWAYS_ALLOWED:
        return Decision.allow
    if _any(permissions.allow, tool_name, tool_input):
        return Decision.allow
    if _any(permissions.ask, tool_name, tool_input):
        return Decision.ask
    return Decision(permissions.default_mode)
