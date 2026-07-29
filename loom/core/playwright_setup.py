"""Playwright MCP helpers — get the bundled browser tester working with one
command.

The default ``playwright`` MCP server (see ``default_settings.json``) runs
``npx @playwright/mcp@latest``, which needs two things npx alone doesn't
guarantee: Node/npx itself on PATH, and Playwright's browser binaries
downloaded to its cache dir. Without the second, the server connects fine but
every ``browser_*`` call fails at runtime with an "executable doesn't exist"
error — a confusing failure mode for a fresh install. These helpers detect
that state up front and fix it with the same one-command-install pattern
``loom/core/ollama.py`` uses for local models.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console


def browsers_dir() -> Path:
    """Where Playwright caches downloaded browser binaries — respects
    ``PLAYWRIGHT_BROWSERS_PATH`` (the same env var Playwright itself reads),
    else the OS-standard default cache location."""
    override = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ms-playwright"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "ms-playwright"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "ms-playwright"


@dataclass
class PlaywrightStatus:
    npx_available: bool
    browsers_installed: bool  # best-effort: a browser-named dir exists in the cache
    browsers_dir: Path


def status() -> PlaywrightStatus:
    d = browsers_dir()
    installed = False
    if d.is_dir():
        installed = any(p.name.startswith(("chromium", "firefox", "webkit")) for p in d.iterdir())
    return PlaywrightStatus(
        npx_available=shutil.which("npx") is not None, browsers_installed=installed, browsers_dir=d
    )


def install_browsers(console: "Console | None" = None, browser: str = "chromium") -> int:
    """Download Playwright's ``browser`` binary via ``npx playwright install``,
    streaming output live. Returns the subprocess exit code (0 = success)."""
    from rich.console import Console as RichConsole

    console = console or RichConsole()
    if shutil.which("npx") is None:
        console.print(f"[red]{INSTALL_HINT}[/red]")
        return 1

    console.print(
        f"[cyan]installing[/cyan] Playwright's {browser} browser (this can take a minute) …"
    )
    try:
        proc = subprocess.Popen(
            ["npx", "-y", "playwright", "install", browser],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        console.print(f"[red]install failed:[/red] {exc}")
        return 1

    for line in proc.stdout or ():
        console.print(f"[dim]{line.rstrip()}[/dim]")
    code = proc.wait()
    if code != 0 and sys.platform.startswith("linux"):
        console.print(
            "[yellow]hint:[/yellow] on Linux this often means missing system libraries — "
            f"try `npx -y playwright install --with-deps {browser}` (installs apt packages, needs sudo)."
        )
    return code


INSTALL_HINT = (
    "npx not found — the Playwright MCP server needs Node.js. Install it from "
    "https://nodejs.org (macOS: `brew install node`), then `loom playwright install` "
    "downloads the browser it drives."
)
