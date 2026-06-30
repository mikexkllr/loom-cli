"""Worktree isolation — falls back gracefully when not in a git repo."""

from loom.core import worktree


def test_non_git_falls_back_in_place(tmp_path):
    assert worktree.is_git_repo(tmp_path) is False
    with worktree.worktree_for("editor", repo=tmp_path) as wt:
        assert wt.created is False
        assert wt.path == tmp_path.resolve()


def test_git_repo_gets_isolated_worktree(tmp_path):
    import shutil
    import subprocess

    if shutil.which("git") is None:
        import pytest

        pytest.skip("git not available")

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("hi")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)

    with worktree.worktree_for("editor", repo=tmp_path) as wt:
        assert wt.created is True
        assert wt.path.exists()
        assert wt.path != tmp_path.resolve()
        assert (wt.path / "f.txt").exists()
    # cleaned up after exit
    assert not wt.path.exists()
