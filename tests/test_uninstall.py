"""Uninstall: what it finds, what it refuses to touch, and what the command
actually deletes.

Every test here pins ``installed_binaries`` and ``USER_CONFIG_DIR`` at a tmp
path. That is not politeness — an unpinned `loom uninstall --yes --purge` in a
test run would delete the developer's own install and their ~/.loom.
"""

import os

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("typer")
pytest.importorskip("yaml")

from typer.testing import CliRunner

from loom.cli.main import app
from loom.core import config as cfg
from loom.core import uninstall as un

runner = CliRunner()


@pytest.fixture(autouse=True)
def wide_console(monkeypatch):
    """Pin the render width so output assertions don't depend on where a line
    happens to fold.

    The plan prints absolute paths through a folding `render.kv` column, and
    `tmp_path` is a different length on every machine — long on a developer's
    Mac, short on a CI runner. At 80 columns that moved the break into the
    middle of the very word a test was looking for, so the suite passed
    locally and failed in CI on a difference that has nothing to do with
    uninstalling anything.
    """
    monkeypatch.setenv("COLUMNS", "200")


def _binary(path, *, frozen: bool = True):
    """A file that looks like a standalone binary (or a console script)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF\x02\x01\x01" if frozen else b"#!/usr/bin/env python\n")
    path.chmod(0o755)
    return path


def _home(monkeypatch, tmp_path):
    """Point $LOOM_HOME at a tmp dir with a plausible spread of contents."""
    home = tmp_path / "loomhome"
    (home / "skills" / "demo").mkdir(parents=True)
    (home / "settings.json").write_text('{"env": {"ANTHROPIC_API_KEY": "sk-secret"}}')
    (home / "telemetry.json").write_text('{"mode": "errors"}')
    (home / "history").write_text("hello\n")
    monkeypatch.setattr(cfg, "USER_CONFIG_DIR", home)
    return home


def _no_real_binaries(monkeypatch, paths=()):
    monkeypatch.setattr(un, "installed_binaries", lambda: list(paths))


# ------------------------------------------------------------------ discovery


def test_console_scripts_are_not_removal_candidates(tmp_path):
    """A `#!` shim belongs to a venv `uv sync` owns; deleting half a source
    install is worse than leaving it alone."""
    assert un._looks_frozen(_binary(tmp_path / "frozen", frozen=True)) is True
    assert un._looks_frozen(_binary(tmp_path / "shim", frozen=False)) is False
    assert un._looks_frozen(tmp_path / "missing") is False


def test_installed_binaries_finds_the_install_dir_override(tmp_path, monkeypatch):
    target = _binary(tmp_path / "bin" / un.BINARY_STEM)
    _binary(tmp_path / "bin" / "loom-shim", frozen=False)
    monkeypatch.setenv(un.INSTALL_DIR_ENV, str(tmp_path / "bin"))
    monkeypatch.setattr(un.shutil, "which", lambda _name: None)
    assert un.installed_binaries() == [target]


def test_installed_binaries_skips_a_source_shim_on_path(tmp_path, monkeypatch):
    shim = _binary(tmp_path / "venv" / "bin" / un.BINARY_STEM, frozen=False)
    monkeypatch.setenv(un.INSTALL_DIR_ENV, str(tmp_path / "nowhere"))
    monkeypatch.setattr(un.shutil, "which", lambda name: str(shim) if name == un.BINARY_STEM else None)
    assert un.installed_binaries() == []


def test_running_binary_is_none_from_source():
    assert un.running_binary() is None  # pytest is never a frozen build


# ----------------------------------------------------------------------- plan


def test_plan_names_what_lives_in_the_user_dir(tmp_path, monkeypatch):
    """Consent needs specifics: "delete ~/.loom" isn't informed, "delete your
    provider keys and your privacy choice" is."""
    home = _home(monkeypatch, tmp_path)
    _no_real_binaries(monkeypatch)
    p = un.plan(tmp_path)
    assert p.data == home
    assert p.data_size > 0
    named = dict(p.data_contents)
    assert "settings.json" in named and "provider API keys" in named["settings.json"]
    assert "telemetry.json" in named and "privacy" in named["telemetry.json"]
    assert "skills" in named


def test_plan_reports_project_data_but_never_removes_it(tmp_path, monkeypatch):
    _home(monkeypatch, tmp_path)
    _no_real_binaries(monkeypatch)
    project = tmp_path / "repo"
    (project / ".loom" / "sessions").mkdir(parents=True)
    p = un.plan(project)
    assert p.project == (project / ".loom")
    # It is reported, and it is not one of the things an uninstall deletes.
    assert p.project not in p.binaries and p.project != p.data


def test_plan_is_empty_when_there_is_nothing_to_remove(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "USER_CONFIG_DIR", tmp_path / "never-created")
    _no_real_binaries(monkeypatch)
    assert un.plan(tmp_path).empty is True


def test_unknown_entries_are_announced_not_deleted_silently(tmp_path, monkeypatch):
    home = _home(monkeypatch, tmp_path)
    (home / "surprise.db").write_text("x")
    assert "surprise.db" in dict(un.data_contents())


# -------------------------------------------------------------------- removal


def test_remove_deletes_trees_and_reports_failure_without_raising(tmp_path):
    tree = tmp_path / "tree"
    (tree / "nested").mkdir(parents=True)
    (tree / "nested" / "f").write_text("x")
    ok, detail = un.remove(tree)
    assert ok and not tree.exists() and str(tree) in detail

    ok, _ = un.remove(tmp_path / "already-gone")
    assert ok is True  # missing_ok: nothing to do is not a failure


def test_human_bytes_scales_down_to_the_json_sized_things():
    assert un.human_bytes(512) == "512 B"
    assert un.human_bytes(2_048) == "2.0 KB"
    assert un.human_bytes(5_000_000) == "5.0 MB"
    assert un.human_bytes(3_200_000_000) == "3.2 GB"


def test_windows_defers_the_delete_to_a_detached_helper(tmp_path, monkeypatch):
    """A running .exe is locked on Windows, so it can only go after we exit."""
    target = _binary(tmp_path / "loom-win")
    spawned = {}
    monkeypatch.setattr(un, "is_windows", lambda: True)
    monkeypatch.setattr(un.subprocess, "Popen", lambda cmd, **kw: spawned.update(cmd=cmd, kw=kw))
    un.delete_running_and_exit(target)  # must return, not os._exit
    assert target.exists()  # deleted later, by the helper
    bat = target.with_suffix(".uninstall.bat")
    script = bat.read_text()
    assert str(target) in script  # the file it will delete
    assert str(os.getpid()) in script  # the PID it waits on first
    assert spawned["cmd"][0] == "cmd"


def test_posix_unlinks_the_running_binary_and_leaves_immediately(tmp_path, monkeypatch):
    """Nothing may run between the unlink and the exit — a onefile build reads
    its own module archive out of sys.executable."""
    target = _binary(tmp_path / "loom-posix")
    order = []
    monkeypatch.setattr(un, "is_windows", lambda: False)
    monkeypatch.setattr(un.os, "unlink", lambda p: order.append(("unlink", p)))
    monkeypatch.setattr(un.os, "_exit", lambda code: order.append(("exit", code)))
    un.delete_running_and_exit(target)
    assert order == [("unlink", str(target)), ("exit", 0)]

    # The exit code has to be carried in: nothing can run after the unlink, so
    # a failure earlier in the uninstall can't be raised once we get here.
    order.clear()
    un.delete_running_and_exit(target, code=1)
    assert order[-1] == ("exit", 1)


def test_reinstall_hint_matches_the_platform(monkeypatch):
    monkeypatch.setattr(un, "is_windows", lambda: False)
    assert "install.sh" in un.reinstall_hint()
    monkeypatch.setattr(un, "is_windows", lambda: True)
    assert "install.ps1" in un.reinstall_hint()


# ------------------------------------------------------------------ the command


def test_dry_run_removes_nothing(tmp_path, monkeypatch):
    home = _home(monkeypatch, tmp_path)
    binary = _binary(tmp_path / "bin" / un.BINARY_STEM)
    _no_real_binaries(monkeypatch, [binary])

    result = runner.invoke(app, ["uninstall", "--dry-run", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert home.exists() and binary.exists()
    assert "dry run" in result.output


def test_plan_is_printed_before_anything_is_asked(tmp_path, monkeypatch):
    home = _home(monkeypatch, tmp_path)
    _no_real_binaries(monkeypatch, [_binary(tmp_path / "bin" / un.BINARY_STEM)])
    out = runner.invoke(app, ["uninstall", "--dry-run", "--root", str(tmp_path)]).output
    assert str(home) in out  # the directory it would delete
    assert "provider API keys" in out  # and what is in it, not just its name
    assert "privacy choice" in out


def test_yes_keeps_your_keys_unless_purge_says_otherwise(tmp_path, monkeypatch):
    """Nobody was asked, so the destructive half doesn't happen by default."""
    home = _home(monkeypatch, tmp_path)
    binary = _binary(tmp_path / "bin" / un.BINARY_STEM)
    _no_real_binaries(monkeypatch, [binary])

    result = runner.invoke(app, ["uninstall", "--yes", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert not binary.exists()
    assert home.exists()
    assert "kept" in result.output


def test_purge_removes_the_user_directory(tmp_path, monkeypatch):
    home = _home(monkeypatch, tmp_path)
    binary = _binary(tmp_path / "bin" / un.BINARY_STEM)
    _no_real_binaries(monkeypatch, [binary])

    result = runner.invoke(app, ["uninstall", "--yes", "--purge", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert not home.exists() and not binary.exists()
    assert "reinstall anytime" in result.output


def test_purge_leaves_the_project_alone(tmp_path, monkeypatch):
    _home(monkeypatch, tmp_path)
    project = tmp_path / "repo"
    (project / ".loom").mkdir(parents=True)
    (project / ".loom" / "sessions.db").write_text("data")
    _no_real_binaries(monkeypatch, [_binary(tmp_path / "bin" / un.BINARY_STEM)])

    runner.invoke(app, ["uninstall", "--yes", "--purge", "--root", str(project)])
    assert (project / ".loom" / "sessions.db").exists()


def test_contradictory_flags_are_refused(tmp_path, monkeypatch):
    _home(monkeypatch, tmp_path)
    _no_real_binaries(monkeypatch)
    result = runner.invoke(app, ["uninstall", "--purge", "--keep-data", "--root", str(tmp_path)])
    assert result.exit_code == 2


def test_declining_at_the_prompt_removes_nothing(tmp_path, monkeypatch):
    """CliRunner has no stdin, so every confirm hits EOF — which must read as
    "no", not as consent."""
    home = _home(monkeypatch, tmp_path)
    binary = _binary(tmp_path / "bin" / un.BINARY_STEM)
    _no_real_binaries(monkeypatch, [binary])

    result = runner.invoke(app, ["uninstall", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert home.exists() and binary.exists()
    assert "nothing removed" in result.output


def test_nothing_installed_says_so_and_points_at_the_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "USER_CONFIG_DIR", tmp_path / "absent")
    _no_real_binaries(monkeypatch)
    result = runner.invoke(app, ["uninstall", "--yes", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert "nothing to uninstall" in result.output
    assert "source checkout" in result.output


def test_uninstall_is_a_subcommand_not_a_task(tmp_path, monkeypatch):
    """`loom --root . uninstall` must never be billed to a model as the task
    "uninstall" — the trap the callback already guards `doctor` against. The
    guard reads the live command list, so a new subcommand is covered by it,
    and this proves that for the riskiest possible word."""
    import loom.cli.main as main_mod

    ran = []
    monkeypatch.setattr(main_mod, "_run_task", lambda *a, **k: ran.append(a))
    monkeypatch.setattr(main_mod, "_maybe_offer_update", lambda: None)

    result = runner.invoke(app, ["--root", str(tmp_path), "uninstall"])
    assert ran == [], "a command name must never reach the task runner"
    assert result.exit_code == 2
    assert "is a command, not a task" in result.output
