"""Ollama backend helpers — make local models easy to install and run.

Ollama is the cross-platform backend: it transparently uses Metal on macOS and
CUDA on Linux/Windows, so Loom needs no per-platform model code. These helpers
let the CLI check the daemon, list installed models, and pull models through
the daemon's HTTP API — which works the same whether the daemon is local or a
remote host named in ``ollama_endpoint``, and doesn't need the ``ollama``
CLI binary at all.
"""

from __future__ import annotations

import json
import shutil
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


def pull(model_tag: str, endpoint: str = DEFAULT_ENDPOINT, console: "Console | None" = None) -> int:
    """Download ``model_tag`` through the Ollama daemon's HTTP API, streaming
    per-layer progress bars to ``console``.

    Talks to ``endpoint`` (the configured ``ollama_endpoint``), so pulls land
    on the daemon Loom actually uses — local or remote — and the ``ollama``
    CLI binary is never required. Returns 0 on success, non-zero on failure.
    """
    from rich.console import Console as RichConsole
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TransferSpeedColumn,
    )

    console = console or RichConsole()
    try:
        with httpx.stream(
            "POST",
            f"{endpoint}/api/pull",
            json={"model": model_tag},
            timeout=httpx.Timeout(None, connect=10),
        ) as resp:
            resp.raise_for_status()
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
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
                        console.print(f"[red]pull failed:[/red] {event['error']}")
                        return 1
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
                        console.print(f"[dim]{state}[/dim]")
                        last_status = state
                    if state == "success":
                        return 0
    except httpx.HTTPError as exc:
        console.print(f"[red]pull failed:[/red] {daemon_hint(endpoint)} ({exc})")
        return 1
    return 1  # stream ended without a success event


def daemon_hint(endpoint: str) -> str:
    """One-line remedy for an unreachable daemon at ``endpoint``."""
    return (
        f"the Ollama daemon isn't reachable at {endpoint} — start it "
        "(`ollama serve`, or open the Ollama app) or point `ollama_endpoint` "
        "at a reachable host"
    )


INSTALL_HINT = (
    "Ollama is not installed and no daemon is reachable. Install it from "
    "https://ollama.com/download (macOS: `brew install ollama`; Linux: "
    "`curl -fsSL https://ollama.com/install.sh | sh`), or set "
    "`ollama_endpoint` to a remote daemon. Then `loom models pull` fetches "
    "the local models from your config."
)
