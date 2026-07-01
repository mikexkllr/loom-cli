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
