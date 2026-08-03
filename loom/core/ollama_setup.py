"""Install and start Ollama, so "get local models working" is one answer.

Everything Loom does locally depends on a running Ollama daemon, and until now
the setup wizard could only point at a web page and hope. This closes that gap:
it finds the best installer for the platform, shows the exact command, and —
once the binary exists — starts the daemon and waits for it to answer.

Three rules, because this installs software on someone's machine:

* **Package managers first.** Homebrew and winget are verifiable and undo
  cleanly (``brew uninstall ollama``). The vendor's ``curl | sh`` script is
  offered only where there is no package manager, and never silently.
* **The exact command is always shown before it runs**, and always confirmed.
* **Loom never runs ``sudo`` itself.** Where the installer needs root it says
  so and lets the installer do its own prompting, in the user's terminal.

A non-local ``ollama_endpoint`` is left alone entirely — installing a daemon
here does nothing for one configured to live somewhere else.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

if TYPE_CHECKING:
    from rich.console import Console

DOWNLOAD_URL = "https://ollama.com/download"
SERVE_TIMEOUT = 30.0


@dataclass(frozen=True)
class InstallMethod:
    """One way to get Ollama onto this machine."""

    id: str  # brew | winget | script | manual
    label: str
    command: tuple[str, ...]  # argv; empty for `manual`
    display: str  # the command as a human would type it
    needs_root: bool = False
    note: str = ""

    @property
    def automatic(self) -> bool:
        return bool(self.command)


MANUAL = InstallMethod(
    id="manual",
    label="Download it yourself",
    command=(),
    display=DOWNLOAD_URL,
    note="no supported package manager found on this machine",
)


def is_local_endpoint(endpoint: str) -> bool:
    """True if ``endpoint`` names this machine.

    Installing a daemon locally cannot help an endpoint pointed at another
    host, so every install path checks this first.
    """
    try:
        host = (urlparse(endpoint).hostname or "").lower()
    except ValueError:
        return False
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "")


def install_methods(system: str | None = None) -> list[InstallMethod]:
    """Available installers for this platform, best first."""
    system = system or platform.system()
    methods: list[InstallMethod] = []

    if system == "Darwin":
        if shutil.which("brew"):
            methods.append(
                InstallMethod(
                    "brew",
                    "Homebrew",
                    ("brew", "install", "ollama"),
                    "brew install ollama",
                    note="uninstall later with `brew uninstall ollama`",
                )
            )
    elif system == "Windows":
        if shutil.which("winget"):
            methods.append(
                InstallMethod(
                    "winget",
                    "winget",
                    ("winget", "install", "--id", "Ollama.Ollama", "-e", "--source", "winget"),
                    "winget install --id Ollama.Ollama -e",
                    note="uninstall later with `winget uninstall Ollama.Ollama`",
                )
            )
    elif system == "Linux":
        if shutil.which("curl") and shutil.which("sh"):
            methods.append(
                InstallMethod(
                    "script",
                    "Ollama's official install script",
                    ("/bin/sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"),
                    "curl -fsSL https://ollama.com/install.sh | sh",
                    needs_root=True,
                    note="the vendor's script; it will ask for your password to install a systemd service",
                )
            )

    methods.append(MANUAL)
    return methods


def install(console: "Console", method: InstallMethod) -> int:
    """Run ``method``, streaming its output. Returns the exit code.

    The installer keeps this terminal, so a password prompt or a licence
    question reaches the user rather than deadlocking on a pipe we own.
    """
    if not method.automatic:
        return 1
    try:
        return subprocess.run(method.command, check=False).returncode
    except FileNotFoundError:
        console.print(f"[loom.bad.b]{method.command[0]} not found[/loom.bad.b]")
        return 127
    except KeyboardInterrupt:
        console.print("[loom.warn]install cancelled[/loom.warn]")
        return 130


def is_up(endpoint: str, timeout: float = 2.0) -> bool:
    try:
        return httpx.get(f"{endpoint}/api/tags", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def serve(endpoint: str, *, timeout: float = SERVE_TIMEOUT) -> bool:
    """Start ``ollama serve`` detached and wait for the endpoint to answer.

    Detached on purpose: the daemon has to outlive this process, or every
    subsequent Loom command would have to start it again. Its output goes to a
    log file rather than this terminal, where it would interleave with the UI.
    """
    if not is_local_endpoint(endpoint):
        return False
    if is_up(endpoint):
        return True
    binary = shutil.which("ollama")
    if binary is None:
        return False

    from loom.core import config as cfg

    log_dir = cfg.USER_CONFIG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / "ollama-serve.log").open("a", encoding="utf-8")
    try:
        subprocess.Popen(  # noqa: S603 — fixed argv, resolved binary
            [binary, "serve"],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # survives this process; not killed by our Ctrl-C
            env={**os.environ},
        )
    except OSError:
        log.close()
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_up(endpoint, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def log_path() -> str:
    from loom.core import config as cfg

    return str(cfg.USER_CONFIG_DIR / "ollama-serve.log")
