"""Self-update: check the GitHub "latest" release for a newer build of the
frozen `loom` binary and swap it in for the running executable.

Only meaningful for PyInstaller-frozen installs (see scripts/install.sh and
.github/workflows/release.yml, which publish assets named by `asset_name()`
plus a `checksums.txt`) — a `uv sync` source install has no binary to
replace, so callers should check `is_frozen()` first and fall back to
`git pull && uv sync` otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from loom.core import config as cfg

REPO = "mikexkllr/loom-cli"
RELEASES_BASE = f"https://github.com/{REPO}/releases/latest/download"
TIMEOUT = 30.0
STARTUP_CHECK_TIMEOUT = 3.0
CACHE_PATH = cfg.USER_CONFIG_DIR / "update_check.json"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def asset_name() -> str:
    """Must match the names release.yml publishes assets under."""
    system = platform.system()
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    if system == "Darwin":
        return f"loom-macos-{arch}"
    if system == "Linux":
        return f"loom-linux-{arch}"
    if system == "Windows":
        return "loom-windows-x64.exe"
    raise RuntimeError(f"unsupported platform: {system} {machine}")


@dataclass
class UpdateCheck:
    asset: str
    current_sha256: str
    latest_sha256: str

    @property
    def up_to_date(self) -> bool:
        return self.current_sha256 == self.latest_sha256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_latest_sha(asset: str, *, timeout: float) -> str | None:
    resp = httpx.get(f"{RELEASES_BASE}/checksums.txt", timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    for line in resp.text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == asset:
            return parts[0]
    return None


def check() -> UpdateCheck:
    """Compare the running binary's checksum against the latest release's
    published checksum for this OS/arch — no version numbers to parse or
    drift out of sync, just "is this the same bytes GitHub has right now"."""
    asset = asset_name()
    running = Path(sys.executable).resolve()
    latest_sha = _fetch_latest_sha(asset, timeout=TIMEOUT)
    if latest_sha is None:
        raise RuntimeError(f"no checksum published for {asset} in the latest release")
    return UpdateCheck(asset=asset, current_sha256=_sha256(running), latest_sha256=latest_sha)


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text())
    except (OSError, ValueError):
        return {}


def _save_cache(data: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(data))
    except OSError:
        pass  # best-effort — a failed cache write just means we re-check sooner


def check_for_startup(min_interval_hours: float = 6.0) -> UpdateCheck | None:
    """Best-effort, throttled update check for use at app startup.

    Never raises and never blocks long: a broken network, an unpublished
    release, or a slow GitHub response must not get in the way of starting
    the app. Returns None if up to date, unreachable, or checked recently
    (cached in `CACHE_PATH`) — the caller shouldn't distinguish those cases.
    """
    if not is_frozen():
        return None
    try:
        asset = asset_name()
        cache = _load_cache()
        now = time.time()
        cache_fresh = (
            cache.get("asset") == asset
            and cache.get("latest_sha256")
            and now - cache.get("checked_at", 0) < min_interval_hours * 3600
        )
        if cache_fresh:
            latest_sha = cache["latest_sha256"]
        else:
            latest_sha = _fetch_latest_sha(asset, timeout=STARTUP_CHECK_TIMEOUT)
            if latest_sha is None:
                return None
            _save_cache({"checked_at": now, "asset": asset, "latest_sha256": latest_sha})

        running = Path(sys.executable).resolve()
        result = UpdateCheck(asset=asset, current_sha256=_sha256(running), latest_sha256=latest_sha)
        return None if result.up_to_date else result
    except Exception:
        return None


def _download_to_tmp(result: UpdateCheck, running: Path, *, console) -> Path:
    """Download `result.asset` next to `running`, verify its checksum, and
    return the verified temp file's path. Caller owns cleanup on failure."""
    url = f"{RELEASES_BASE}/{result.asset}"
    console.print(f"[cyan]downloading[/cyan] {url}")

    try:
        fd, tmp_name = tempfile.mkstemp(dir=running.parent, prefix=".loom-update-")
    except PermissionError as exc:
        raise RuntimeError(
            f"can't write to {running.parent} ({exc}) — reinstall to a user-writable "
            "directory, or re-run with elevated permissions"
        ) from exc
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as tmp, httpx.stream("GET", url, timeout=TIMEOUT, follow_redirects=True) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes(1024 * 1024):
                tmp.write(chunk)

        downloaded_sha = _sha256(tmp_path)
        if downloaded_sha != result.latest_sha256:
            raise RuntimeError(
                f"checksum mismatch on downloaded update (got {downloaded_sha[:12]}…, "
                f"expected {result.latest_sha256[:12]}…) — aborted, nothing replaced"
            )
        return tmp_path
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def apply(result: UpdateCheck, *, console) -> None:
    """Download and swap in the new build. Takes effect on the *next*
    launch — see `apply_and_relaunch` to resume the current session too."""
    running = Path(sys.executable).resolve()
    tmp_path = _download_to_tmp(result, running, console=console)
    try:
        tmp_path.chmod(0o755)
        if platform.system() == "Windows":
            _schedule_windows_swap(tmp_path, running)
            console.print(
                "[green]update downloaded[/green] — it finishes applying a moment "
                "after loom exits; run loom again shortly"
            )
        else:
            os.replace(tmp_path, running)
            console.print("[green]updated[/green] — restart loom to run the new build")
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def apply_and_relaunch(result: UpdateCheck, *, console, argv: list[str]) -> None:
    """Like `apply`, but resumes the current invocation on the new build
    instead of requiring a second manual launch. Never returns on success —
    either execs into the new binary (Unix) or exits after a child run of it
    (Windows), so callers should treat this as the last thing they do."""
    running = Path(sys.executable).resolve()
    tmp_path = _download_to_tmp(result, running, console=console)
    try:
        tmp_path.chmod(0o755)
        if platform.system() == "Windows":
            # Can't swap our own locked .exe yet; run the verified download
            # directly and let the detached helper move it into place once
            # both this process and the child below have exited.
            _schedule_windows_swap(tmp_path, running)
            console.print("[cyan]relaunching on the new build…[/cyan]")
            proc = subprocess.Popen([str(tmp_path), *argv])
            code = proc.wait()
            raise SystemExit(code)
        else:
            os.replace(tmp_path, running)
            console.print("[cyan]relaunching on the new build…[/cyan]")
            os.execv(str(running), [str(running), *argv])
    except SystemExit:
        raise
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _schedule_windows_swap(new_path: Path, running: Path) -> None:
    """Windows keeps an exclusive lock on a running .exe, so the swap has to
    happen after this process (and, for a relaunch, its child running from
    new_path) exits. Spawn a detached helper that waits for our PID to
    disappear, then moves the new file into place."""
    bat = running.with_suffix(".update.bat")
    pid = os.getpid()
    bat.write_text(
        "@echo off\r\n"
        ":wait\r\n"
        f'tasklist /FI "PID eq {pid}" 2>NUL | find "{pid}" >NUL\r\n'
        "if not errorlevel 1 (\r\n"
        "  timeout /t 1 /nobreak >NUL\r\n"
        "  goto wait\r\n"
        ")\r\n"
        f'move /Y "{new_path}" "{running}" >NUL\r\n'
        'del "%~f0"\r\n',
        encoding="utf-8",
    )
    subprocess.Popen(
        ["cmd", "/c", str(bat)],
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
