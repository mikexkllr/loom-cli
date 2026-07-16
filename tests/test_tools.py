"""Sandboxed filesystem + shell tools."""

import pytest

pytest.importorskip("langchain_core")

from loom.tools import edit_file, execute, grep, ls, read_file, sandbox, write_file


def _invoke(tool, **kwargs):
    # LangChain tools are invoked via .invoke(dict); .func calls the raw callable.
    return tool.invoke(kwargs)


def test_write_read_edit_roundtrip(tmp_path):
    sandbox.set_root(tmp_path)
    assert "wrote" in _invoke(write_file, file_path="a.txt", content="hello world")
    assert _invoke(read_file, file_path="a.txt") == "1\thello world"
    _invoke(edit_file, file_path="a.txt", old_string="world", new_string="loom")
    assert _invoke(read_file, file_path="a.txt") == "1\thello loom"


def test_edit_rejects_ambiguous(tmp_path):
    sandbox.set_root(tmp_path)
    _invoke(write_file, file_path="b.txt", content="x x")
    out = _invoke(edit_file, file_path="b.txt", old_string="x", new_string="y")
    assert "appears 2 times" in out


def test_edit_replace_all(tmp_path):
    sandbox.set_root(tmp_path)
    _invoke(write_file, file_path="c.txt", content="x x")
    out = _invoke(
        edit_file,
        file_path="c.txt",
        old_string="x",
        new_string="y",
        replace_all=True,
    )
    assert "edited" in out
    assert _invoke(read_file, file_path="c.txt") == "1\ty y"


def test_sandbox_blocks_traversal(tmp_path):
    sandbox.set_root(tmp_path)
    with pytest.raises(ValueError):
        sandbox.resolve_in_sandbox("../escape.txt")


def test_grep_finds_match(tmp_path):
    sandbox.set_root(tmp_path)
    _invoke(write_file, file_path="src/x.py", content="def foo():\n    return 1\n")
    out = _invoke(grep, pattern="def foo", path=".")
    assert "src/x.py" in out


def test_grep_output_modes(tmp_path):
    sandbox.set_root(tmp_path)
    _invoke(write_file, file_path="src/x.py", content="def foo():\n    return 1\n")
    assert _invoke(grep, pattern="def foo", path=".", output_mode="count") == "src/x.py: 1"
    content = _invoke(grep, pattern="def foo", path=".", output_mode="content")
    assert "src/x.py:1:def foo():" in content


def test_ls_lists(tmp_path):
    sandbox.set_root(tmp_path)
    _invoke(write_file, file_path="one.txt", content="1")
    out = _invoke(ls, path=".")
    assert "one.txt" in out


def test_execute_runs_in_root(tmp_path):
    sandbox.set_root(tmp_path)
    out = _invoke(execute, command="echo loom")
    assert "loom" in out
    assert "exit 0" in out


def test_execute_timeout(tmp_path):
    sandbox.set_root(tmp_path)
    out = _invoke(execute, command="sleep 5", timeout=1)
    assert "timed out" in out
