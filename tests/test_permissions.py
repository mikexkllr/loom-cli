"""Permission rule matching + decision precedence."""

import pytest

pytest.importorskip("pydantic")

from loom.core.permissions import Decision, check
from loom.core.settings import Permissions


def test_allow_bare_tool_name():
    p = Permissions(allow=["read_file"], default_mode="ask")
    assert check("read_file", {"path": "x"}, p) is Decision.allow
    assert check("write_file", {"path": "x"}, p) is Decision.ask  # falls to default


def test_deny_beats_allow():
    p = Permissions(allow=["*"], deny=["execute(rm -rf*)"])
    assert check("execute", {"command": "rm -rf /tmp/x"}, p) is Decision.deny
    assert check("execute", {"command": "ls"}, p) is Decision.allow


def test_specifier_glob_on_command():
    p = Permissions(ask=["execute(git *)"], default_mode="deny")
    assert check("execute", {"command": "git status"}, p) is Decision.ask
    assert check("execute", {"command": "npm install"}, p) is Decision.deny


def test_specifier_glob_on_path():
    p = Permissions(allow=["write_file(src/**)"], default_mode="ask")
    assert check("write_file", {"path": "src/app/x.py"}, p) is Decision.allow
    assert check("write_file", {"path": "secret.env"}, p) is Decision.ask


def test_wildcard():
    p = Permissions(allow=["*"], default_mode="deny")
    assert check("anything", {}, p) is Decision.allow


def test_default_mode_fallback():
    p = Permissions(default_mode="deny")
    assert check("write_file", {"path": "x"}, p) is Decision.deny


def test_coordination_tools_always_allowed():
    """task/write_todos/consult never prompt, even when a user settings.json
    replaces the packaged allow list (lists override, they don't merge) or
    flips the default mode."""
    p = Permissions(allow=["read_file"], default_mode="ask")
    for tool in ("task", "write_todos", "consult"):
        assert check(tool, {}, p) is Decision.allow
    p = Permissions(default_mode="deny")
    for tool in ("task", "write_todos", "consult"):
        assert check(tool, {}, p) is Decision.allow


def test_explicit_deny_beats_always_allowed():
    p = Permissions(deny=["task"], default_mode="allow")
    assert check("task", {}, p) is Decision.deny


def test_read_only_tools_never_prompt():
    """ls/glob/grep mutate nothing, so no configuration may put them behind an
    approval prompt — not a wiped allow list, not `ask: ["*"]`, not a deny-by
    -default mode. Prompting on recon buys no safety and trains the user to
    approve the shell prompt that does matter."""
    for p in (
        Permissions(allow=[], ask=["*"], default_mode="ask"),
        Permissions(allow=["read_file"], default_mode="deny"),
        Permissions(ask=["ls", "glob", "grep"], default_mode="allow"),
    ):
        for tool in ("ls", "glob", "grep"):
            assert check(tool, {"path": "x", "pattern": "y"}, p) is Decision.allow


def test_read_file_is_still_gateable():
    """`read_file` is deliberately not in the always-allowed set: a deny rule
    for a secrets file is the whole reason that gate exists, and unlike ls/glob
    /grep it names one exact path a rule can be written against."""
    p = Permissions(allow=[], ask=["read_file"], default_mode="ask")
    assert check("read_file", {"path": ".env"}, p) is Decision.ask


def test_explicit_deny_still_beats_read_only_tools():
    """Airgap mode denies every filesystem tool by name for the orchestrator;
    that must keep winning over the always-allowed set."""
    p = Permissions(deny=["ls", "glob", "grep"], default_mode="allow")
    for tool in ("ls", "glob", "grep"):
        assert check(tool, {}, p) is Decision.deny


def test_shell_still_asks():
    """The counterpart to the rule above: making recon free must not make the
    shell free."""
    p = Permissions(allow=[], ask=["execute"], default_mode="ask")
    assert check("execute", {"command": "rm -rf build"}, p) is Decision.ask
