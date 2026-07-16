"""Sandbox path resolution used by permission/undo hooks.

Filesystem and shell tools are now provided by deepagents' FilesystemMiddleware;
Loom only keeps the path-sandboxing utility here so hooks can normalize and gate
file paths before any tool runs.
"""

import pytest

pytest.importorskip("langchain_core")

from loom.tools import sandbox


def test_sandbox_blocks_traversal(tmp_path):
    sandbox.set_root(tmp_path)
    with pytest.raises(ValueError):
        sandbox.resolve_in_sandbox("../escape.txt")


def test_sandbox_allows_absolute_under_root(tmp_path):
    sandbox.set_root(tmp_path)
    # Virtual absolute paths (e.g. deepagents file_path) resolve under the root.
    (tmp_path / "foo.txt").write_text("ok")
    assert sandbox.resolve_in_sandbox("/foo.txt") == (tmp_path / "foo.txt").resolve()
