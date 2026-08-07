"""Uninstall: take Loom off this machine, and say exactly what that means.

The counterpart to :mod:`loom.core.update`. Update swaps the frozen binary in
place; this removes it, and — only when asked — the user-level directory Loom
writes to (``$LOOM_HOME``, default ``~/.loom``: config, settings, telemetry
consent, input history, user skills).

Two things are deliberately never deleted, because they are not part of the
install:

* **A project's ``.loom/``** — sessions, undo snapshots, artifacts and a
  ``settings.json`` that is meant to be committed belong to that repo. The
  plan reports the one in the current project so you can delete it yourself.
* **Ollama and its models** — Loom can install the daemon and pull gigabytes
  of weights, but it is a separate program other tools use, installed through
  the system package manager. The plan reports the disk it is holding and the
  command that frees it.

Only *standalone* binaries are removal candidates. A console script (one
starting ``#!``) belongs to a virtualenv that ``uv sync`` owns, and deleting
half of a source install is worse than leaving it alone.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from loom.core import config as cfg
from loom.core import update

# Where the install scripts put the binary (scripts/install.sh, install.ps1),
# checked in addition to whatever is on PATH — an install that never made it
# onto PATH is exactly the one a user wants help removing.
INSTALL_DIR_ENV = "LOOM_INSTALL_DIR"

# Spelt as a stem plus a suffix rather than a literal: a bare "loom.<word>"
# string literal anywhere under loom/ is read as a theme style name by the
# style-resolution guard in tests/test_ui_render.py.
BINARY_STEM = "loom"
BINARY_NAMES = (BINARY_STEM, f"{BINARY_STEM}.exe")


def _default_install_dirs() -> list[Path]:
    out = [Path.home() / ".local" / "bin"]
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        out.append(Path(local_appdata) / "loom" / "bin")
    override = os.environ.get(INSTALL_DIR_ENV)
    if override:
        out.insert(0, Path(override))
    return out


def _looks_frozen(path: Path) -> bool:
    """True for a standalone executable, False for a ``#!`` console script.

    A source install's ``loom`` is a two-line shim into a venv; removing it
    leaves the venv claiming to have Loom installed while the command is
    gone. The frozen binary is the only thing this module owns.
    """
    try:
        with path.open("rb") as f:
            return f.read(2) != b"#!"
    except OSError:
        return False


def running_binary() -> Path | None:
    """The frozen binary executing right now, if this is a binary install."""
    if not update.is_frozen():
        return None
    try:
        return Path(sys.executable).resolve()
    except OSError:
        return None


def installed_binaries() -> list[Path]:
    """Every standalone ``loom`` this machine can find, running one first."""
    found: list[Path] = []

    def add(path: Path | None) -> None:
        if path is None:
            return
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in found or not resolved.is_file() or not _looks_frozen(resolved):
            return
        found.append(resolved)

    add(running_binary())
    for name in BINARY_NAMES:
        which = shutil.which(name)
        add(Path(which) if which else None)
        for directory in _default_install_dirs():
            add(directory / name)
    return found


def data_dir() -> Path:
    """``$LOOM_HOME`` (default ``~/.loom``) — everything Loom wrote for *you*,
    as opposed to for a project."""
    return cfg.USER_CONFIG_DIR


# What lives in there, in the order a person would want to hear about it. The
# point of naming these is consent: "delete ~/.loom" is not informed, "delete
# your API keys, your model routing and your privacy choice" is.
DATA_CONTENTS: tuple[tuple[str, str], ...] = (
    ("settings.json", "model routing, permissions, provider API keys"),
    ("config.yaml", "legacy model-routing defaults"),
    ("telemetry.json", "your privacy choice"),
    ("skills", "skills you wrote for every project"),
    ("history", "REPL input history"),
    ("update_check.json", "update-check cache"),
    ("ollama-serve.log", "Ollama daemon log"),
)


def data_contents() -> list[tuple[str, str]]:
    """The ``(name, what it is)`` rows of :func:`data_dir` that actually exist,
    plus anything unrecognised, so nothing is deleted unannounced."""
    base = data_dir()
    if not base.is_dir():
        return []
    known = {name for name, _ in DATA_CONTENTS}
    rows = [(name, blurb) for name, blurb in DATA_CONTENTS if (base / name).exists()]
    rows += sorted((entry.name, "") for entry in base.iterdir() if entry.name not in known)
    return rows


def dir_size(path: Path) -> int:
    """Bytes on disk under ``path``. Unreadable entries count as zero — a size
    is a courtesy, and no uninstall should fail over one."""
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def human_bytes(size: float) -> str:
    """Sizes here run from a few kilobytes of JSON to a skills tree, so this
    scales down further than ``ollama.human_size``, which only ever describes
    model weights."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{int(size)} B"


def project_data(root: str | Path) -> Path | None:
    """This project's ``.loom/`` — reported, never removed."""
    path = Path(root).resolve() / ".loom"
    return path if path.is_dir() else None


@dataclass
class Plan:
    """What an uninstall would touch, resolved before anything is deleted."""

    binaries: list[Path] = field(default_factory=list)
    data: Path | None = None  # None when there is nothing to purge
    data_size: int = 0
    data_contents: list[tuple[str, str]] = field(default_factory=list)
    project: Path | None = None
    frozen: bool = False

    @property
    def running(self) -> Path | None:
        """The binary that is executing this uninstall, if it is in the plan."""
        current = running_binary()
        return current if current in self.binaries else None

    @property
    def empty(self) -> bool:
        return not self.binaries and self.data is None


def plan(root: str | Path = ".") -> Plan:
    """Resolve what is on this machine. Reads only — nothing is removed."""
    base = data_dir()
    has_data = base.is_dir()
    return Plan(
        binaries=installed_binaries(),
        data=base if has_data else None,
        data_size=dir_size(base) if has_data else 0,
        data_contents=data_contents(),
        project=project_data(root),
        frozen=update.is_frozen(),
    )


def ollama_disk(config) -> tuple[int, int]:
    """``(model count, bytes)`` the local Ollama daemon is holding, or
    ``(0, 0)`` if it isn't reachable. Reported so the number is visible; Loom
    doesn't delete another program's data."""
    from loom.core import ollama

    sizes = ollama.installed_sizes(config)
    return len(sizes), sum(sizes.values())


def remove(path: Path) -> tuple[bool, str]:
    """Delete a file or directory tree. Returns ``(ok, message)`` — never
    raises, because a half-finished uninstall should report what it managed
    rather than end in a traceback."""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError as exc:
        return False, f"{path} — {exc.strerror or exc}"
    return True, str(path)


def _schedule_windows_delete(target: Path) -> None:
    """Windows holds an exclusive lock on a running ``.exe``, so it can only
    be deleted once this process is gone. Spawn a detached helper that waits
    for our PID to disappear and then removes it — the same shape as
    :func:`loom.core.update._schedule_windows_swap`."""
    bat = target.with_suffix(".uninstall.bat")
    pid = os.getpid()
    bat.write_text(
        "@echo off\r\n"
        ":wait\r\n"
        f'tasklist /FI "PID eq {pid}" 2>NUL | find "{pid}" >NUL\r\n'
        "if not errorlevel 1 (\r\n"
        "  timeout /t 1 /nobreak >NUL\r\n"
        "  goto wait\r\n"
        ")\r\n"
        f'del /F /Q "{target}" >NUL 2>&1\r\n'
        'del "%~f0"\r\n',
        encoding="utf-8",
    )
    # Looked up rather than named directly: these constants only exist on
    # Windows, and reaching for them by attribute makes the helper impossible
    # to exercise anywhere else — including in a test of this exact branch.
    detached = getattr(subprocess, "DETACHED_PROCESS", 0)
    new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(["cmd", "/c", str(bat)], creationflags=detached | new_group, close_fds=True)


def is_windows() -> bool:
    return platform.system() == "Windows"


def delete_running_and_exit(target: Path, *, code: int = 0) -> None:
    """Remove the binary this process is running from, then leave immediately.

    **Nothing may run between the unlink and the exit** — not a print, not an
    f-string that touches a lazily-imported module. A PyInstaller onefile
    build reads its module archive out of ``sys.executable`` on demand, so
    once that path is gone the next not-yet-imported module has nowhere to
    come from. ``loom.core.update._swap_and_leave`` documents the same trap
    from the other side, where the bytes changed instead of vanishing; it
    turned "yes, update me" into a traceback, and this would turn "yes,
    uninstall" into one.

    ``os._exit`` is deliberate: a normal return runs interpreter shutdown,
    which can still import — so ``code`` has to be carried in rather than
    raised afterwards, or an uninstall that failed to purge would still report
    success. On Windows the file is locked and can't be unlinked at all, so
    the deletion is handed to a detached helper and this returns normally,
    leaving the caller to exit however it likes.
    """
    if is_windows():
        _schedule_windows_delete(target)
        return
    path = str(target)
    os.unlink(path)
    os._exit(code)


def reinstall_hint() -> str:
    """How to get Loom back — the last line of a good uninstall."""
    if is_windows():
        return f"irm https://raw.githubusercontent.com/{update.REPO}/main/scripts/install.ps1 | iex"
    return f"curl -LsSf https://raw.githubusercontent.com/{update.REPO}/main/scripts/install.sh | sh"


def path_hint(binaries: list[Path]) -> str | None:
    """The leftover a file deletion can't clean up: the PATH entry the
    Windows installer added, or a shell profile line the user added by hand."""
    if not binaries:
        return None
    directory = binaries[0].parent
    if is_windows():
        return f"remove {directory} from your user PATH in System Settings → Environment Variables"
    return f"if you added `export PATH=\"{directory}:$PATH\"` to your shell profile, drop that line too"
