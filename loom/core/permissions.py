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

Shell commands are matched per command, not per string. A glob over the whole
line cannot tell ``git status`` from ``git status && curl … | sh``: both start
with ``git ``, so ``allow: ["execute(git *)"]`` approved the second one too,
and ``deny: ["execute(sudo *)"]`` never saw a ``sudo`` that came after a
``cd``. So a compound line is split on ``&&``, ``||``, ``;``, ``|``, ``&`` and
newlines (quotes respected), and:

* ``deny`` and ``ask`` fire if the line **or any command in it** matches;
* ``allow`` holds only if **every** command in it matches an allow rule, and
  never for a line containing command substitution (``$(…)``, backticks) —
  what runs there is not on the line to be judged.

A rule with no specifier (``execute``, ``*``, ``execute(*)``) still means what
it says: that tool, whatever it runs.

This is a guard against a model wandering past a rule, not a sandbox. A
determined command can always be spelled some way no glob anticipates.
"""

from __future__ import annotations

import fnmatch
import re
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


# ---------------------------------------------------------------------------
# Shell commands: judge each command on the line, not the line as one string
# ---------------------------------------------------------------------------

# Tools whose specifier is a shell command line.
_SHELL_TOOLS = frozenset({"execute"})

_SUBSTITUTION = ("$(", "`", "<(", ">(")
# Quote-unaware on purpose: it only ever feeds deny/ask, where matching too
# much is the safe direction.
_SUBSTITUTION_SPLIT = re.compile(r"\$\(|[<>]\(|\)|`|&&|\|\||[;|\n]")
_ENV_PREFIX = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)+")


def command_parts(command: str) -> list[str]:
    """The individual commands on a shell line.

    Splits on ``&&``, ``||``, ``;``, ``|``, a backgrounding ``&`` and newlines,
    leaving quoted text and redirections (``2>&1``, ``&>log``) alone.
    """
    parts: list[str] = []
    buf: list[str] = []
    quote = ""
    i, n = 0, len(command)

    def flush() -> None:
        text = "".join(buf).strip()
        if text:
            parts.append(text)
        buf.clear()

    while i < n:
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            buf.extend((ch, command[i + 1]))
            i += 2
            continue
        if command[i : i + 2] in ("&&", "||"):
            flush()
            i += 2
            continue
        if ch in ";|\n":
            flush()
            i += 1
            continue
        if ch == "&":
            before = command[i - 1] if i else ""
            after = command[i + 1] if i + 1 < n else ""
            if before not in "<>" and after != ">":  # not part of a redirection
                flush()
                i += 1
                continue
        buf.append(ch)
        i += 1
    flush()
    return parts


def _bare(part: str) -> str:
    """``part`` without the grouping and ``VAR=value`` prefixes that would hide
    the command word from a rule like ``sudo *``."""
    return _ENV_PREFIX.sub("", part.lstrip("({! \t"))


def _has_substitution(command: str) -> bool:
    return any(marker in command for marker in _SUBSTITUTION)


def _fragments(command: str) -> list[str]:
    """Every command a deny/ask rule should be tried against on this line."""
    out: list[str] = []
    for part in command_parts(command):
        for candidate in (part, _bare(part)):
            if candidate and candidate not in out:
                out.append(candidate)
    if _has_substitution(command):
        for piece in _SUBSTITUTION_SPLIT.split(command):
            candidate = _bare(piece.strip())
            if candidate and candidate not in out:
                out.append(candidate)
    return out


def _command(tool_name: str, tool_input: dict) -> str | None:
    if tool_name not in _SHELL_TOOLS:
        return None
    command = tool_input.get("command")
    return command if isinstance(command, str) and command.strip() else None


def _matches_any_command(rules: list[str], tool_name: str, tool_input: dict) -> bool:
    """deny/ask: the whole line, or any single command on it."""
    if _any(rules, tool_name, tool_input):
        return True
    command = _command(tool_name, tool_input)
    if command is None:
        return False
    return any(
        _any(rules, tool_name, {**tool_input, "command": fragment}) for fragment in _fragments(command)
    )


def _is_blanket(rule: str, tool_name: str) -> bool:
    """A rule that allows the tool whatever its arguments are."""
    rule = rule.strip()
    if "(" not in rule:
        return rule == "*" or fnmatch.fnmatch(tool_name, rule)
    if not rule.endswith(")"):
        return False
    name, spec = rule[:-1].split("(", 1)
    return name.strip() in ("*", tool_name) and spec.strip() == "*"


def _allowed(rules: list[str], tool_name: str, tool_input: dict) -> bool:
    """allow: every command on the line has to be allowed."""
    command = _command(tool_name, tool_input)
    if command is None:
        return _any(rules, tool_name, tool_input)
    parts = command_parts(command)
    substitution = _has_substitution(command)
    if len(parts) <= 1 and not substitution:
        return _any(rules, tool_name, tool_input)
    if any(_is_blanket(rule, tool_name) for rule in rules):
        return True
    if substitution:
        return False
    return all(_any(rules, tool_name, {**tool_input, "command": part}) for part in parts)


def check(tool_name: str, tool_input: dict | None, permissions: Permissions) -> Decision:
    """Return the permission :class:`Decision` for a tool call."""
    tool_input = tool_input or {}
    if _matches_any_command(permissions.deny, tool_name, tool_input):
        return Decision.deny
    if tool_name in ALWAYS_ALLOWED:
        return Decision.allow
    if _allowed(permissions.allow, tool_name, tool_input):
        return Decision.allow
    if _matches_any_command(permissions.ask, tool_name, tool_input):
        return Decision.ask
    return Decision(permissions.default_mode)
