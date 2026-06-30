"""Ollama backend helpers — make local models easy to install and run.

Ollama is the cross-platform backend: it transparently uses Metal on macOS and
CUDA on Linux/Windows, so Loom needs no per-platform model code. These helpers
let the CLI check the daemon, list installed models, and one-shot pull every
local model named in the config.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import httpx

from loom.core.config import LoomConfig
from loom.core.model_router import resolve


@dataclass
class OllamaStatus:
    installed: bool  # ollama binary on PATH
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
    return [m for m in required_local_models(config) if m not in have]


def pull(model_tag: str) -> int:
    """Run ``ollama pull <tag>``, streaming progress to the terminal.

    Returns the process exit code. Raises ``FileNotFoundError`` if ollama isn't
    installed (the caller surfaces install instructions).
    """
    proc = subprocess.run(["ollama", "pull", model_tag])
    return proc.returncode


INSTALL_HINT = (
    "Ollama is not installed. Install it from https://ollama.com/download "
    "(macOS: `brew install ollama`; Linux: `curl -fsSL https://ollama.com/install.sh | sh`). "
    "Then run `loom models pull` to fetch the local models from your config."
)
