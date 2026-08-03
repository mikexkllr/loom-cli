"""Bridges to the `sentry` and `langfuse` CLIs, used only during setup.

Both tools are already installed on plenty of developer machines and both are
already authenticated there. When they are, the privacy step can offer to fetch
a DSN or verify a key pair instead of asking someone to go copy strings out of
a web dashboard — which is the difference between an opt-in people complete and
one they abandon.

Everything degrades: no binary, not logged in, no network, or a changed JSON
shape all just mean "type it in yourself". Nothing here is on a hot path, and
nothing here is required for telemetry to work — the credentials can always be
pasted directly.

Kept apart from :mod:`loom.core.telemetry` the same way ``ollama_setup`` is kept
apart from ``ollama``: one module decides what is allowed, the other shells out.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

# `sentry` installs to ~/.local/bin, which is on PATH for most shells but not
# for a GUI-launched process. Checking the canonical location too costs one
# stat and saves a "not installed" that is simply wrong.
_EXTRA_BIN_DIRS = (Path.home() / ".local" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin"))

TIMEOUT = 30.0

SENTRY_DOCS = "https://docs.sentry.io/product/sentry-basics/dsn-explainer/"
LANGFUSE_DOCS = "https://langfuse.com/faq/all/where-are-langfuse-api-keys"
LANGFUSE_DEFAULT_HOST = "https://cloud.langfuse.com"


def _which(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for directory in _EXTRA_BIN_DIRS:
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _run(argv: list[str], *, timeout: float = TIMEOUT) -> tuple[int, str, str]:
    """Run a CLI and return ``(code, stdout, stderr)``. Never raises."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            # A setup helper must never inherit a half-written terminal or sit
            # waiting on a password prompt nobody can see.
            stdin=subprocess.DEVNULL,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except (OSError, subprocess.SubprocessError):
        return 1, "", "could not run"


def _json(out: str) -> Any:
    try:
        return json.loads(out)
    except (ValueError, TypeError):
        return None


# ----------------------------------------------------------------------------
# Sentry
# ----------------------------------------------------------------------------


def sentry_cli() -> str | None:
    """Path to the `sentry` CLI, or None.

    This is the newer ``sentry`` command (``sentry auth``/``sentry project``),
    not the older ``sentry-cli`` release-management tool — the two are separate
    binaries with different verbs, and only this one can hand back a DSN.
    """
    return _which("sentry")


def sentry_logged_in() -> bool:
    if not sentry_cli():
        return False
    code, out, _ = _run([str(sentry_cli()), "auth", "status"])
    return code == 0 and "Authenticated" in out


def sentry_orgs() -> list[dict[str, str]]:
    """``[{"slug": ..., "name": ...}]`` for the orgs this login can see."""
    binary = sentry_cli()
    if not binary:
        return []
    code, out, _ = _run([binary, "org", "list", "--json"])
    data = _json(out) if code == 0 else None
    if not isinstance(data, list):
        return []
    return [
        {"slug": str(o.get("slug") or ""), "name": str(o.get("name") or o.get("slug") or "")}
        for o in data
        if isinstance(o, dict) and o.get("slug")
    ]


def sentry_projects(org: str = "") -> list[dict[str, str]]:
    """``[{"slug", "name", "org"}]``. ``org`` empty means every visible org."""
    binary = sentry_cli()
    if not binary:
        return []
    code, out, _ = _run([binary, "project", "list", "--json"])
    data = _json(out) if code == 0 else None
    # The CLI wraps project lists in {"data": [...]} but returns orgs bare, so
    # accept either rather than depending on which one this version does.
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    out_rows = []
    for p in rows:
        if not isinstance(p, dict) or not p.get("slug"):
            continue
        slug_org = str((p.get("organization") or {}).get("slug") or "") if isinstance(p.get("organization"), dict) else ""
        if org and slug_org and slug_org != org:
            continue
        out_rows.append(
            {"slug": str(p["slug"]), "name": str(p.get("name") or p["slug"]), "org": slug_org or org}
        )
    return out_rows


def sentry_create_project(org: str, name: str, platform: str = "python") -> str | None:
    """Create a Sentry project and return its slug.

    The only call in this module with a side effect on someone's Sentry
    account, so every caller confirms first — see
    :func:`loom.ui.privacy.prompt_sentry_dsn`.
    """
    binary = sentry_cli()
    if not binary or not name:
        return None
    target = f"{org}/{name}" if org else name
    code, out, _ = _run([binary, "project", "create", target, platform, "--json"])
    if code != 0:
        return None
    data = _json(out)
    if isinstance(data, dict):
        return str(data.get("slug") or "") or None
    return None


def sentry_dsn(org: str, project: str) -> str | None:
    """The public DSN for ``org/project``.

    Reads the project-keys endpoint and takes the first key's *public* DSN —
    the client-side one. The ``secret`` variant in the same payload is a legacy
    form that embeds a secret key, and must never be written to a config file.
    """
    binary = sentry_cli()
    if not binary or not org or not project:
        return None
    code, out, _ = _run([binary, "api", f"/projects/{org}/{project}/keys/", "--json"])
    if code != 0:
        return None
    data = _json(out)
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return None
    for key in rows:
        if not isinstance(key, dict) or key.get("isActive") is False:
            continue
        dsn = key.get("dsn")
        if isinstance(dsn, dict) and dsn.get("public"):
            return str(dsn["public"])
    return None


# ----------------------------------------------------------------------------
# Langfuse
# ----------------------------------------------------------------------------


def langfuse_cli() -> str | None:
    return _which("langfuse")


def langfuse_verify(public_key: str, secret_key: str, host: str = "") -> tuple[bool, str]:
    """Check a Langfuse key pair against the server.

    Returns ``(ok, detail)``. Falls back to a plain HTTP call when the CLI
    isn't installed, so a verified key doesn't depend on an optional npm
    package. Never raises; a network failure reports as unverified, not as an
    invalid key, because those two want different advice.
    """
    if not (public_key and secret_key):
        return False, "both keys are needed"
    host = host or LANGFUSE_DEFAULT_HOST

    binary = langfuse_cli()
    if binary:
        _code, out, err = _run(
            [binary, "api", "projects", "get-public", "--public-key", public_key,
             "--secret-key", secret_key, "--host", host]
        )
        # A rejected key is reported as a JSON body on *stderr* while a good
        # one comes back on stdout, so both streams have to be read. Judging by
        # exit code alone would call every failure "CLI unavailable" and judging
        # by stdout alone would miss the one message that says what went wrong.
        data = _json(out) or _json(err)
        if isinstance(data, dict):
            if data.get("message"):
                return False, str(data["message"])
            if data.get("data") or data.get("id") or data.get("name"):
                return True, "key verified"
        # Anything else (usage error, changed output shape, missing npm deps)
        # is not a verdict about the key — fall through and ask the API directly.

    return _langfuse_verify_http(public_key, secret_key, host)


def _langfuse_verify_http(public_key: str, secret_key: str, host: str) -> tuple[bool, str]:
    """Same check over plain HTTP basic auth — httpx is already a hard
    dependency, so this path always exists."""
    try:
        import httpx

        resp = httpx.get(
            f"{host.rstrip('/')}/api/public/projects",
            auth=(public_key, secret_key),
            timeout=10.0,
        )
    except Exception as exc:  # network down, bad host, TLS — not a bad key
        return False, f"could not reach {host} ({type(exc).__name__})"
    if resp.status_code == 200:
        return True, "key verified"
    if resp.status_code in (401, 403):
        return False, "credentials rejected"
    return False, f"{host} answered {resp.status_code}"


def langfuse_health(host: str = "") -> bool:
    """True if a Langfuse server is reachable at ``host`` (no auth needed)."""
    try:
        import httpx

        return httpx.get(f"{(host or LANGFUSE_DEFAULT_HOST).rstrip('/')}/api/public/health", timeout=8.0).status_code < 500
    except Exception:
        return False
