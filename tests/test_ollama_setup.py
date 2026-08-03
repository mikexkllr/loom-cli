"""Installing Ollama on the user's machine.

This module runs installers and launches a background service, so most of what
matters here is what it *refuses* to do: no silent sudo, no install against a
remote endpoint, no command that wasn't shown first.
"""

from __future__ import annotations

import subprocess

import pytest

from loom.core import ollama_setup


def _no_tools(monkeypatch, present=()):
    monkeypatch.setattr(ollama_setup.shutil, "which", lambda name: f"/usr/bin/{name}" if name in present else None)


# ------------------------------------------------------------- method choice


def test_macos_prefers_homebrew(monkeypatch):
    _no_tools(monkeypatch, present=("brew",))
    methods = ollama_setup.install_methods("Darwin")
    assert methods[0].id == "brew"
    assert methods[0].command == ("brew", "install", "ollama")
    assert not methods[0].needs_root


def test_windows_prefers_winget(monkeypatch):
    _no_tools(monkeypatch, present=("winget",))
    assert ollama_setup.install_methods("Windows")[0].id == "winget"


def test_linux_offers_the_vendor_script_and_says_it_needs_root(monkeypatch):
    _no_tools(monkeypatch, present=("curl", "sh"))
    method = ollama_setup.install_methods("Linux")[0]
    assert method.id == "script"
    # The user must be told a password prompt is coming.
    assert method.needs_root is True
    assert "password" in method.note


def test_manual_is_always_the_last_resort(monkeypatch):
    _no_tools(monkeypatch)  # nothing installed
    for system in ("Darwin", "Linux", "Windows", "Plan9"):
        methods = ollama_setup.install_methods(system)
        assert methods[-1].id == "manual"
        assert not methods[-1].automatic
        assert ollama_setup.DOWNLOAD_URL in methods[-1].display


def test_loom_never_builds_a_sudo_command(monkeypatch):
    """Loom must not run sudo itself — where root is needed, the vendor's own
    installer prompts for it, in the user's terminal."""
    _no_tools(monkeypatch, present=("brew", "winget", "curl", "sh"))
    for system in ("Darwin", "Linux", "Windows"):
        for method in ollama_setup.install_methods(system):
            assert "sudo" not in method.command, method
            assert not method.display.startswith("sudo"), method


def test_every_automatic_method_can_be_shown_before_it_runs(monkeypatch):
    _no_tools(monkeypatch, present=("brew", "winget", "curl", "sh"))
    for system in ("Darwin", "Linux", "Windows"):
        for method in ollama_setup.install_methods(system):
            if method.automatic:
                assert method.display.strip(), f"{method.id} has no displayable command"
                assert method.label.strip()


# ------------------------------------------------------------------ endpoint


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("http://localhost:11434", True),
        ("http://127.0.0.1:11434", True),
        ("http://[::1]:11434", True),
        ("http://0.0.0.0:11434", True),
        ("http://gpu-box.lan:11434", False),
        ("https://ollama.example.com", False),
        ("http://192.168.1.50:11434", False),
    ],
)
def test_local_endpoint_detection(endpoint, expected):
    assert ollama_setup.is_local_endpoint(endpoint) is expected


def test_serve_refuses_a_remote_endpoint(monkeypatch):
    """Starting a daemon here cannot help one configured to live elsewhere —
    and silently starting a local one would quietly ignore the config."""
    monkeypatch.setattr(
        ollama_setup.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned against a remote endpoint")
    )
    assert ollama_setup.serve("http://gpu-box.lan:11434") is False


def test_serve_is_a_no_op_when_already_up(monkeypatch):
    monkeypatch.setattr(ollama_setup, "is_up", lambda e, timeout=2.0: True)
    monkeypatch.setattr(ollama_setup.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned needlessly"))
    assert ollama_setup.serve("http://localhost:11434") is True


def test_serve_without_the_binary_fails_rather_than_hanging(monkeypatch):
    monkeypatch.setattr(ollama_setup, "is_up", lambda e, timeout=2.0: False)
    monkeypatch.setattr(ollama_setup.shutil, "which", lambda name: None)
    assert ollama_setup.serve("http://localhost:11434") is False


def test_serve_detaches_so_the_daemon_outlives_this_process(monkeypatch):
    """A daemon killed by our own Ctrl-C would have to be restarted on every
    Loom command."""
    seen = {}
    monkeypatch.setattr(ollama_setup.shutil, "which", lambda name: "/usr/local/bin/ollama")
    monkeypatch.setattr(ollama_setup, "is_up", lambda e, timeout=2.0: bool(seen))

    def _popen(argv, **kwargs):
        seen.update(kwargs)
        seen["argv"] = argv
        return object()

    monkeypatch.setattr(ollama_setup.subprocess, "Popen", _popen)
    assert ollama_setup.serve("http://localhost:11434", timeout=5) is True
    assert seen["argv"] == ["/usr/local/bin/ollama", "serve"]
    assert seen["start_new_session"] is True
    assert seen["stdin"] is subprocess.DEVNULL
    # Its output must not interleave with the UI on this terminal.
    assert seen["stdout"] is not None and seen["stdout"] is not subprocess.PIPE


def test_serve_gives_up_instead_of_waiting_forever(monkeypatch):
    monkeypatch.setattr(ollama_setup.shutil, "which", lambda name: "/usr/local/bin/ollama")
    monkeypatch.setattr(ollama_setup, "is_up", lambda e, timeout=2.0: False)
    monkeypatch.setattr(ollama_setup.subprocess, "Popen", lambda *a, **k: object())
    monkeypatch.setattr(ollama_setup.time, "sleep", lambda s: None)
    assert ollama_setup.serve("http://localhost:11434", timeout=0.2) is False


# ------------------------------------------------------------------- install


def test_install_returns_the_installer_exit_code(monkeypatch):
    monkeypatch.setattr(
        ollama_setup.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 3})()
    )
    method = ollama_setup.InstallMethod("brew", "Homebrew", ("brew", "install", "ollama"), "brew install ollama")
    assert ollama_setup.install(_console(), method) == 3


def test_install_keeps_the_terminal_so_password_prompts_reach_the_user(monkeypatch):
    """Capturing the installer's stdio would deadlock a sudo prompt."""
    seen = {}
    monkeypatch.setattr(
        ollama_setup.subprocess,
        "run",
        lambda argv, **k: seen.update(argv=argv, kwargs=k) or type("R", (), {"returncode": 0})(),
    )
    method = ollama_setup.InstallMethod("brew", "Homebrew", ("brew", "install", "ollama"), "brew install ollama")
    ollama_setup.install(_console(), method)
    assert "capture_output" not in seen["kwargs"]
    assert "stdout" not in seen["kwargs"]


def test_install_refuses_the_manual_method():
    assert ollama_setup.install(_console(), ollama_setup.MANUAL) != 0


def test_a_missing_installer_binary_is_reported_not_raised(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(ollama_setup.subprocess, "run", _boom)
    method = ollama_setup.InstallMethod("brew", "Homebrew", ("brew", "install", "ollama"), "brew install ollama")
    assert ollama_setup.install(_console(), method) == 127


# -------------------------------------------------------------------- wizard


def test_ensure_ollama_is_a_no_op_when_the_daemon_answers(monkeypatch):
    from loom.core import config as cfg
    from loom.ui import onboarding as ob

    monkeypatch.setattr(ollama_setup, "is_up", lambda e, timeout=2.0: True)
    monkeypatch.setattr(
        ollama_setup, "install", lambda *a, **k: pytest.fail("installed over a working daemon")
    )
    assert ob.ensure_ollama(_console(), cfg.LoomConfig()) is True


def test_ensure_ollama_does_not_install_for_a_remote_endpoint(monkeypatch):
    from loom.core import config as cfg
    from loom.ui import onboarding as ob

    monkeypatch.setattr(ollama_setup, "is_up", lambda e, timeout=2.0: False)
    monkeypatch.setattr(ollama_setup, "install", lambda *a, **k: pytest.fail("installed for a remote daemon"))
    config = cfg.LoomConfig(ollama_endpoint="http://gpu-box.lan:11434")
    assert ob.ensure_ollama(_console(), config) is False


def test_ensure_ollama_respects_a_declined_install(monkeypatch):
    from loom.core import config as cfg
    from loom.ui import onboarding as ob

    monkeypatch.setattr(ollama_setup, "is_up", lambda e, timeout=2.0: False)
    monkeypatch.setattr(ollama_setup.shutil, "which", lambda n: None)
    monkeypatch.setattr(ob.shutil, "which", lambda n: None)
    monkeypatch.setattr(ollama_setup, "install", lambda *a, **k: pytest.fail("installed after being declined"))
    monkeypatch.setattr(ob.render, "confirm", lambda *a, **k: False)
    assert ob.ensure_ollama(_console(), cfg.LoomConfig()) is False


def _console():
    from loom.core.settings import UISettings
    from loom.ui.theme import make_console

    return make_console(UISettings())
