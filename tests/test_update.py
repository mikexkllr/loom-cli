"""Self-update helpers: frozen-vs-source detection, asset naming, the
throttled startup check, and the checksum cache. The download itself needs a
published release and isn't covered here, but the *ordering* around the swap
is — that ordering is what made "yes, update me" end in a traceback."""

import sys

import pytest

pytest.importorskip("httpx")

import httpx

from loom.core import update


def test_is_frozen_false_under_pytest():
    assert update.is_frozen() is False


def test_asset_name_matches_current_platform():
    name = update.asset_name()
    assert name.startswith(("loom-macos-", "loom-linux-", "loom-windows-"))


def test_asset_name_rejects_unknown_platform(monkeypatch):
    monkeypatch.setattr(update.platform, "system", lambda: "PlayStation")
    with pytest.raises(RuntimeError, match="unsupported platform"):
        update.asset_name()


def test_up_to_date_compares_checksums():
    same = update.UpdateCheck(asset="loom-linux-x64", current_sha256="a", latest_sha256="a")
    diff = update.UpdateCheck(asset="loom-linux-x64", current_sha256="a", latest_sha256="b")
    assert same.up_to_date is True
    assert diff.up_to_date is False


def test_check_for_startup_short_circuits_when_not_frozen(monkeypatch):
    # Source (uv sync) installs have no binary to replace, so the startup
    # check must never even try the network.
    def boom(*a, **k):
        raise AssertionError("network hit despite not being a frozen install")

    monkeypatch.setattr(update, "is_frozen", lambda: False)
    monkeypatch.setattr(update, "_fetch_latest_sha", boom)
    assert update.check_for_startup() is None


def test_check_for_startup_never_raises_on_network_failure(monkeypatch):
    monkeypatch.setattr(update, "is_frozen", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(update, "_fetch_latest_sha", boom)
    assert update.check_for_startup() is None


def test_check_for_startup_rechecks_instead_of_trusting_a_stale_answer(tmp_path, monkeypatch):
    """"You are up to date" is a claim about what GitHub has published, and it
    expires the moment the next release lands. Caching it for six hours meant a
    user who started Loom once was told nothing for six hours while several
    builds shipped — the exact symptom of "it never offers me an update"."""
    monkeypatch.setattr(update, "is_frozen", lambda: True)
    monkeypatch.setattr(update, "CACHE_PATH", tmp_path / "update_check.json")
    asset = update.asset_name()

    latest = "sha-v1"
    calls = []

    def fake_fetch(_asset, *, timeout):
        calls.append(_asset)
        return latest

    monkeypatch.setattr(update, "_fetch_latest_sha", fake_fetch)
    monkeypatch.setattr(update, "_sha256", lambda _path: "sha-v1")  # the running binary
    monkeypatch.setattr(sys, "executable", str(tmp_path / "loom"))
    (tmp_path / "loom").write_bytes(b"binary")

    # First start: up to date, nothing to report.
    assert update.check_for_startup() is None
    assert calls == [asset]

    # A new release lands. The very next start must notice it.
    latest = "sha-v2"
    result = update.check_for_startup()
    assert calls == [asset, asset], "a successful check must not be cached"
    assert result is not None and result.latest_sha256 == "sha-v2"


def test_check_for_startup_backs_off_after_a_failure(tmp_path, monkeypatch):
    """The expensive case is an *offline* start, which pays the full timeout.
    That, and only that, is worth caching."""
    monkeypatch.setattr(update, "is_frozen", lambda: True)
    monkeypatch.setattr(update, "CACHE_PATH", tmp_path / "update_check.json")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "loom"))
    (tmp_path / "loom").write_bytes(b"binary")

    calls = []

    def failing_fetch(_asset, *, timeout):
        calls.append(_asset)
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(update, "_fetch_latest_sha", failing_fetch)

    assert update.check_for_startup() is None
    assert len(calls) == 1
    assert update.check_for_startup() is None
    assert len(calls) == 1, "a recent failure must not be retried on every launch"

    # Once the backoff expires, it tries again.
    assert update.check_for_startup(retry_after_failure_minutes=0) is None
    assert len(calls) == 2


def test_check_for_startup_reports_stale_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "is_frozen", lambda: True)
    monkeypatch.setattr(update, "CACHE_PATH", tmp_path / "update_check.json")
    monkeypatch.setattr(update, "_fetch_latest_sha", lambda *a, **k: "new-sha")
    monkeypatch.setattr(update, "_sha256", lambda _path: "old-sha")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "loom"))
    (tmp_path / "loom").write_bytes(b"binary")

    result = update.check_for_startup()
    assert result is not None
    assert result.up_to_date is False
    assert result.current_sha256 == "old-sha"
    assert result.latest_sha256 == "new-sha"


# ------------------------------------------------------- the swap is terminal


class _RecordingConsole:
    """A console that logs its prints into a shared event list."""

    def __init__(self, events):
        self.events = events
        self.file = self

    def print(self, *args, **_kw):
        self.events.append(("print", " ".join(str(a) for a in args)))

    def flush(self):
        pass


@pytest.fixture
def _swap_spy(monkeypatch):
    """Record replace/exec/exit order without touching the filesystem."""
    events = []
    monkeypatch.setattr(update.os, "replace", lambda *a: events.append(("replace", None)))
    monkeypatch.setattr(update.os, "execve", lambda *a: events.append(("execve", a[0], a[2])))
    monkeypatch.setattr(update.os, "_exit", lambda code: events.append(("_exit", code)))
    monkeypatch.setattr(update.platform, "system", lambda: "Darwin")
    return events


def test_nothing_runs_between_the_replace_and_leaving(_swap_spy, tmp_path, monkeypatch):
    """The whole bug in one assertion.

    A PyInstaller onefile binary reads its module archive out of
    `sys.executable` on demand. The moment that path holds a different build,
    the next lazy import decompresses the wrong bytes and dies with
    `zlib.error: incorrect header check`. So `os.replace` must be immediately
    followed by exec/exit, with no Python — no print, no f-string touching an
    unimported module — in between.
    """
    monkeypatch_env = update.os.environ
    update._swap_and_leave(tmp_path / "new", tmp_path / "running", ["--root", "."])
    assert [e[0] for e in _swap_spy] == ["replace", "execve"]
    assert _swap_spy[1][1] == str(tmp_path / "running")

    # A onefile launch tells its child where its bundle lives through _PYI_*.
    # Inherited across the exec, they make the *new* build skip unpacking and
    # then fail to find a Python it never extracted.
    handed_over = _swap_spy[1][2]
    assert not [k for k in handed_over if k.startswith("_PYI_") or k == "_MEIPASS2"]
    assert "PATH" in handed_over or "PATH" not in monkeypatch_env

    _swap_spy.clear()
    update._swap_and_leave(tmp_path / "new", tmp_path / "running", None)
    assert _swap_spy == [("replace", None), ("_exit", 0)]


def test_pyinstaller_env_is_not_inherited_by_the_new_build(_swap_spy, tmp_path, monkeypatch):
    monkeypatch.setenv("_PYI_APPLICATION_HOME_DIR", "/tmp/_MEIstale")
    monkeypatch.setenv("_PYI_ARCHIVE_FILE", "/tmp/old-loom")
    monkeypatch.setenv("_MEIPASS2", "/tmp/_MEIstale")
    monkeypatch.setenv("LOOM_HOME", "/tmp/loomhome")

    update._swap_and_leave(tmp_path / "new", tmp_path / "running", [])
    env = _swap_spy[1][2]
    assert "_PYI_APPLICATION_HOME_DIR" not in env
    assert "_PYI_ARCHIVE_FILE" not in env
    assert "_MEIPASS2" not in env
    assert env["LOOM_HOME"] == "/tmp/loomhome", "unrelated env must survive"


def test_apply_says_what_it_did_before_it_swaps(_swap_spy, tmp_path, monkeypatch):
    """The success line has to be printed *before* the replace: afterwards this
    process can no longer render anything it hasn't already imported."""
    events = _swap_spy
    monkeypatch.setattr(update, "_download_to_tmp", lambda *a, **k: tmp_path / "new")
    monkeypatch.setattr(update.Path, "chmod", lambda self, mode: None)

    update.apply(
        update.UpdateCheck(asset="loom-macos-arm64", current_sha256="a", latest_sha256="b"),
        console=_RecordingConsole(events),
    )
    kinds = [event[0] for event in events]
    assert kinds.index("print") < kinds.index("replace")
    assert kinds[-1] == "_exit"


def test_relaunch_says_what_it_did_before_it_swaps(_swap_spy, tmp_path, monkeypatch):
    events = _swap_spy
    monkeypatch.setattr(update, "_download_to_tmp", lambda *a, **k: tmp_path / "new")
    monkeypatch.setattr(update.Path, "chmod", lambda self, mode: None)

    update.apply_and_relaunch(
        update.UpdateCheck(asset="loom-macos-arm64", current_sha256="a", latest_sha256="b"),
        console=_RecordingConsole(events),
        argv=["--root", "."],
    )
    kinds = [event[0] for event in events]
    assert kinds.index("print") < kinds.index("replace")
    assert kinds[-1] == "execve"
