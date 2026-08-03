"""Ollama backend helpers — make local models easy to install and run.

Ollama is the cross-platform backend: it transparently uses Metal on macOS and
CUDA on Linux/Windows, so Loom needs no per-platform model code. These helpers
let the CLI check the daemon, list installed models, and pull models through
the daemon's HTTP API — which works the same whether the daemon is local or a
remote host named in ``ollama_endpoint``, and doesn't need the ``ollama``
CLI binary at all.
"""

from __future__ import annotations

import functools
import json
import shutil
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from loom.core.config import LoomConfig
from loom.core.model_router import resolve

if TYPE_CHECKING:
    from rich.console import Console

DEFAULT_ENDPOINT = "http://localhost:11434"


@dataclass
class OllamaStatus:
    installed: bool  # ollama binary on PATH (informational — HTTP API needs none)
    running: bool  # daemon answering on the endpoint
    models: list[str]  # installed model tags
    endpoint: str


def status(config: LoomConfig) -> OllamaStatus:
    installed = shutil.which("ollama") is not None
    running = False
    models: list[str] = []
    try:
        resp = httpx.get(f"{config.ollama_endpoint}/api/tags", timeout=3)
        resp.raise_for_status()
        running = True
        models = [m["name"] for m in resp.json().get("models", [])]
    except (httpx.HTTPError, KeyError):
        pass
    return OllamaStatus(installed, running, models, config.ollama_endpoint)


@functools.lru_cache(maxsize=64)
def context_length(tag: str, endpoint: str = DEFAULT_ENDPOINT) -> int | None:
    """The model's real trained context length, per the daemon's ``/api/show``.

    Loom otherwise has to assume a window for any local model missing a
    ``context_windows`` entry, and that assumption has to be conservative —
    which makes the prompt-size guard escalate to the cloud on prompts the
    model could have held comfortably. Ollama already knows the answer.

    The key is architecture-prefixed (``qwen3.context_length``,
    ``llama.context_length``, ...), so match on the suffix. Returns None if the
    daemon is unreachable or reports nothing usable — callers keep their
    default.
    """
    try:
        resp = httpx.post(f"{endpoint}/api/show", json={"model": tag}, timeout=5)
        resp.raise_for_status()
        info = resp.json().get("model_info") or {}
    except (httpx.HTTPError, ValueError, TypeError):
        return None
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int) and value > 0:
            return value
    return None


def is_served(tag: str, available: list[str] | set[str]) -> bool:
    """True if ``tag`` is satisfied by an installed model, treating a missing
    ``:latest`` suffix as equivalent on either side (``qwen3`` matches
    ``qwen3:latest`` and vice versa)."""
    have = set(available)
    if tag in have:
        return True
    if ":" not in tag:
        return f"{tag}:latest" in have
    if tag.endswith(":latest"):
        return tag.rsplit(":", 1)[0] in have
    return f"{tag}:latest" in have


def required_local_models(config: LoomConfig) -> list[str]:
    """Distinct Ollama model tags Loom needs, derived from config."""
    tags: list[str] = []
    for model in config.all_models().values():
        rm = resolve(model)
        if rm.is_local and rm.name not in tags:
            tags.append(rm.name)
    return tags


def missing_models(config: LoomConfig) -> list[str]:
    have = set(status(config).models)
    return [m for m in required_local_models(config) if not is_served(m, have)]


PULL_ATTEMPTS = 4
RETRYABLE = 75  # EX_TEMPFAIL: the transfer dropped, but trying again can work

# Ollama surfaces its own exhausted retries as an `error` event whose text is
# the underlying network failure. Only these are worth another attempt — a
# missing manifest or a bad tag will fail identically forever.
_TRANSIENT = (
    "connection reset",
    "max retries exceeded",
    "unexpected eof",
    "timeout",
    "timed out",
    "connection refused by peer",
    "broken pipe",
    "temporary failure",
)


def _is_transient(message: str) -> bool:
    text = message.lower()
    return any(marker in text for marker in _TRANSIENT)


def pull(
    model_tag: str,
    endpoint: str = DEFAULT_ENDPOINT,
    console: "Console | None" = None,
    *,
    attempts: int = PULL_ATTEMPTS,
) -> int:
    """Download ``model_tag`` through the Ollama daemon, with retries.

    Model weights run to several gigabytes, so a single dropped connection is
    an ordinary event rather than an exceptional one — and giving up on it
    threw away everything downloaded so far from the user's point of view.
    Ollama keeps completed blobs, so a retry resumes rather than restarting;
    what looks like lost progress is only the layer that was in flight.

    Talks to ``endpoint`` (the configured ``ollama_endpoint``), so pulls land
    on the daemon Loom actually uses — local or remote — and the ``ollama``
    CLI binary is never required. Returns 0 on success, non-zero on failure.
    """
    from loom.ui.theme import theme_of

    last = 1
    for attempt in range(1, max(1, attempts) + 1):
        last = _pull_once(model_tag, endpoint, console)
        if last == 0:
            return 0
        # Only a dropped transfer is worth repeating. An unreachable daemon or
        # a tag that doesn't exist will fail the same way every time, and
        # retrying it just makes the user wait to hear the same thing.
        if last != RETRYABLE or attempt == attempts:
            return 1 if last == RETRYABLE else last
        if console is not None:
            from loom.ui.glyphs import glyphs

            g = glyphs(theme_of(console).unicode)
            console.print(
                f"[loom.warn]{g.warn}[/loom.warn] [loom.muted]connection dropped — "
                f"retrying ({attempt + 1}/{attempts}); finished layers are kept[/loom.muted]"
            )
        time.sleep(min(2**attempt, 8))
    return 1


def _pull_once(model_tag: str, endpoint: str, console: "Console | None") -> int:
    """One download attempt. Returns 0 on success, 130 if interrupted."""
    from rich.console import Console as RichConsole
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TransferSpeedColumn,
    )

    console = console or RichConsole()
    # A caller-supplied console may be a plain Rich one; theme_of adopts it so
    # the `loom.*` names below resolve instead of raising MissingStyle.
    from loom.ui.theme import theme_of

    theme_of(console)
    try:
        with httpx.stream(
            "POST",
            f"{endpoint}/api/pull",
            json={"model": model_tag},
            timeout=httpx.Timeout(None, connect=10),
        ) as resp:
            resp.raise_for_status()
            with Progress(
                TextColumn("[loom.muted]{task.description}"),
                BarColumn(complete_style="loom.local", finished_style="loom.good", pulse_style="loom.line"),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=console,
                transient=True,
            ) as progress:
                tasks: dict[str, object] = {}
                last_status = ""
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get("error"):
                        console.print(f"[loom.bad.b]pull failed:[/loom.bad.b] {event['error']}")
                        return RETRYABLE if _is_transient(str(event["error"])) else 1
                    state = event.get("status", "")
                    digest = event.get("digest")
                    if digest and event.get("total"):
                        task_id = tasks.get(digest)
                        if task_id is None:
                            label = f"{model_tag} · {digest.split(':')[-1][:12]}"
                            task_id = progress.add_task(label, total=event["total"])
                            tasks[digest] = task_id
                        progress.update(task_id, completed=event.get("completed", 0))
                    elif state and state != last_status:
                        console.print(f"[loom.muted]{state}[/loom.muted]")
                        last_status = state
                    if state == "success":
                        return 0
    except httpx.ConnectError as exc:
        console.print(f"[loom.bad.b]pull failed:[/loom.bad.b] {daemon_hint(endpoint)} ({exc})")
        return 1
    except httpx.HTTPError as exc:
        # The stream had started, so the daemon is there — the transfer broke.
        console.print(f"[loom.bad.b]pull failed:[/loom.bad.b] {' '.join(str(exc).split()) or type(exc).__name__}")
        return RETRYABLE
    except KeyboardInterrupt:
        console.print("[loom.warn]pull interrupted — finished layers are kept, re-run to resume[/loom.warn]")
        return 130
    return 1  # stream ended without a success event


def remove(model_tag: str, endpoint: str = DEFAULT_ENDPOINT) -> tuple[bool, str]:
    """Delete ``model_tag`` from the daemon. Returns (ok, message).

    Through the HTTP API like :func:`pull`, so it works against a remote
    ``ollama_endpoint`` and needs no ``ollama`` binary.
    """
    try:
        resp = httpx.request(
            "DELETE", f"{endpoint}/api/delete", json={"model": model_tag}, timeout=30
        )
    except httpx.HTTPError as exc:
        return False, f"{daemon_hint(endpoint)} ({exc})"
    if resp.status_code == 404:
        return False, f"`{model_tag}` isn't installed"
    if resp.status_code >= 400:
        return False, " ".join(resp.text.split())[:200] or f"HTTP {resp.status_code}"
    return True, f"removed {model_tag}"


def installed_sizes(config: LoomConfig) -> dict[str, int]:
    """Installed tag -> size in bytes, as the daemon reports it.

    The catalogue's RAM figures are hand-maintained estimates; this is ground
    truth, and it is free — the tag listing already carries it.
    """
    try:
        resp = httpx.get(f"{config.ollama_endpoint}/api/tags", timeout=3)
        resp.raise_for_status()
        return {m["name"]: int(m.get("size") or 0) for m in resp.json().get("models", [])}
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        return {}


def human_size(size_bytes: float) -> str:
    return f"{size_bytes / 1e9:.1f} GB" if size_bytes >= 1e8 else f"{size_bytes / 1e6:.0f} MB"


def daemon_hint(endpoint: str) -> str:
    """One-line remedy for an unreachable daemon at ``endpoint``."""
    return (
        f"the Ollama daemon isn't reachable at {endpoint} — `loom models serve` "
        "starts it for you (or run `ollama serve` yourself), or point "
        "`ollama_endpoint` at a reachable host"
    )


INSTALL_HINT = (
    "Ollama is not installed and no daemon is reachable. `loom models install` "
    "does the whole thing — installs it with your package manager, starts the "
    "daemon, and pulls your configured models. To do it by hand instead: "
    "https://ollama.com/download (macOS: `brew install ollama`; Linux: "
    "`curl -fsSL https://ollama.com/install.sh | sh`), or set "
    "`ollama_endpoint` to a remote daemon."
)
