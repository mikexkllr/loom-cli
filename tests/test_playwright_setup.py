"""Playwright MCP setup helpers: browser-cache detection and the
`npx playwright install` wrapper."""

import io

import pytest

pytest.importorskip("rich")

from loom.core import playwright_setup as pw


def _quiet_console():
    from rich.console import Console

    return Console(file=io.StringIO(), force_terminal=False)


# --------------------------------------------------------------------- browsers_dir


def test_browsers_dir_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "custom"))
    assert pw.browsers_dir() == tmp_path / "custom"


def test_browsers_dir_platform_default_macos(monkeypatch):
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
    monkeypatch.setattr(pw.sys, "platform", "darwin")
    assert str(pw.browsers_dir()).endswith("Library/Caches/ms-playwright")


def test_browsers_dir_platform_default_linux(monkeypatch, tmp_path):
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(pw.sys, "platform", "linux")
    assert pw.browsers_dir() == tmp_path / "ms-playwright"


# --------------------------------------------------------------------------- status


def test_status_reports_missing_npx(monkeypatch, tmp_path):
    monkeypatch.setattr(pw.shutil, "which", lambda name: None)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "nope"))
    st = pw.status()
    assert st.npx_available is False
    assert st.browsers_installed is False


def test_status_detects_installed_browser(monkeypatch, tmp_path):
    cache = tmp_path / "ms-playwright"
    (cache / "chromium-1234").mkdir(parents=True)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(cache))
    monkeypatch.setattr(pw.shutil, "which", lambda name: "/usr/bin/npx")
    st = pw.status()
    assert st.npx_available is True
    assert st.browsers_installed is True
    assert st.browsers_dir == cache


def test_status_empty_cache_dir_is_not_installed(monkeypatch, tmp_path):
    cache = tmp_path / "ms-playwright"
    cache.mkdir()
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(cache))
    monkeypatch.setattr(pw.shutil, "which", lambda name: "/usr/bin/npx")
    assert pw.status().browsers_installed is False


# --------------------------------------------------------------------- install_browsers


def test_install_browsers_without_npx_prints_hint_and_fails(monkeypatch):
    monkeypatch.setattr(pw.shutil, "which", lambda name: None)
    console = _quiet_console()
    assert pw.install_browsers(console) == 1


class _FakeProc:
    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self._returncode = returncode

    def wait(self):
        return self._returncode


def test_install_browsers_streams_and_succeeds(monkeypatch):
    monkeypatch.setattr(pw.shutil, "which", lambda name: "/usr/bin/npx")
    seen = {}

    def fake_popen(cmd, **kwargs):
        seen["cmd"] = cmd
        return _FakeProc(["Downloading chromium…\n", "Chromium 120.0 downloaded\n"], returncode=0)

    monkeypatch.setattr(pw.subprocess, "Popen", fake_popen)
    console = _quiet_console()
    assert pw.install_browsers(console, "chromium") == 0
    assert seen["cmd"] == ["npx", "-y", "playwright", "install", "chromium"]


def test_install_browsers_reports_failure(monkeypatch):
    monkeypatch.setattr(pw.shutil, "which", lambda name: "/usr/bin/npx")
    monkeypatch.setattr(
        pw.subprocess, "Popen", lambda cmd, **k: _FakeProc(["error: boom\n"], returncode=1)
    )
    assert pw.install_browsers(_quiet_console()) == 1


def test_install_browsers_handles_launch_failure(monkeypatch):
    monkeypatch.setattr(pw.shutil, "which", lambda name: "/usr/bin/npx")

    def raise_oserror(cmd, **kwargs):
        raise OSError("no such file")

    monkeypatch.setattr(pw.subprocess, "Popen", raise_oserror)
    assert pw.install_browsers(_quiet_console()) == 1
