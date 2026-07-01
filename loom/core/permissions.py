"""Permission engine — matches tool calls against allow/ask/deny rules.

Rule syntax (Claude Code-flavored):

    "read_file"            -> matches any call to read_file
    "execute(git *)"       -> execute whose command matches the glob "git *"
    "write_file(src/**)"   -> write_file whose path matches "src/**"
    "*"                    -> matches every tool call

Evaluation precedence: deny > allow > ask > default_mode. The first list (in
that priority order) that yields a match wins.
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
    "read_file": "path",
    "ls": "path",
    "glob_tool": "pattern",
    "grep_tool": "pattern",
    "web_search": "query",
}


class Decision(str, Enum):
    allow = "allow"
    ask = "ask"
    deny = "deny"


def _rule_matches(rule: str, tool_name: str, tool_input: dict) -> bool:
    rule = rule.strip()
    if rule in ("*", tool_name):
        return True
    if "(" in rule and rule.endswith(")"):
        name, spec = rule[:-1].split("(", 1)
        name = name.strip()
        if name not in ("*", tool_name):
            return False
        field = _SPECIFIER_FIELD.get(tool_name)
        value = str(tool_input.get(field, "")) if field else " ".join(map(str, tool_input.values()))
        return fnmatch.fnmatch(value, spec.strip())
    return False


def _any(rules: list[str], tool_name: str, tool_input: dict) -> bool:
    return any(_rule_matches(r, tool_name, tool_input) for r in rules)


def check(tool_name: str, tool_input: dict | None, permissions: Permissions) -> Decision:
    """Return the permission :class:`Decision` for a tool call."""
    tool_input = tool_input or {}
    if _any(permissions.deny, tool_name, tool_input):
        return Decision.deny
    if _any(permissions.allow, tool_name, tool_input):
        return Decision.allow
    if _any(permissions.ask, tool_name, tool_input):
        return Decision.ask
    return Decision(permissions.default_mode)
