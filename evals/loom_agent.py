"""Harbor installed-agent adapter: run Loom inside a Terminal-Bench task container.

This is how Terminal-Bench scores every other harness on its leaderboard
(Claude Code, Codex CLI, Gemini CLI, OpenCode …): the agent CLI is installed
into the task's own container, handed the task instruction, and left to work
until it exits or the task's agent timeout fires. Hidden tests then grade the
container.

Loom is installed from the *local checkout* (a wheel built by
``evals/tbench.py``) with dependency versions pinned to ``uv.lock``, so a run
measures exactly the code in this repo.

Local models: Ollama keeps running on the host; the container reaches it at
``host.docker.internal`` (Docker Desktop). Cloud keys are passed per-exec as
environment variables and never written into the container.

Usage (normally via ``evals/tbench.py``)::

    harbor run -d terminal-bench/terminal-bench-2-1 \\
        -a evals.loom_agent:LoomAgent \\
        -m google/gemini-4-argon --ak subagent_model=ollama/qwen3.8:27b-mlx \\
        --ak wheel_dir=evals/.build
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import override

from pydantic import Field

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

PREFIX = "/installed-agent/loom"
VENV = f"{PREFIX}/venv"
LOOM_HOME = f"{PREFIX}/home"
LOG_DIR = "/logs/agent"

# Credentials Loom may need for a cloud role. Forwarded only if set on the host.
_KEY_VARS = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENCODE_ZEN_API_KEY",
    "OPENCODE_GO_API_KEY",
)

_ROLES = ("explorer", "editor", "bash", "searcher", "general-purpose", "tester", "reviewer")


class LoomOptions(InstalledAgentOptions):
    subagent_model: str = Field(
        default="ollama/qwen3.8:27b-mlx",
        description="Model for every subagent role (Loom model string).",
    )
    cloud_fallback: str = Field(
        default="google:gemini-3.5-flash",
        description="Cloud model local roles fall back to when Ollama is unreachable.",
    )
    ollama_endpoint: str = Field(default="http://host.docker.internal:11434")
    local_context: int = Field(
        default=131072,
        description=(
            "num_ctx for local models. Pinned because auto-detection inside the "
            "container would size it from the container's 2 GB, not the host GPU."
        ),
    )
    wheel_dir: str = Field(
        default="evals/.build",
        description="Host dir holding loom-*.whl and requirements.txt (built by tbench.py).",
    )
    mode: str = Field(default="", description="'' | airgap | local-only")


def _loom_model(harbor_model: str) -> str:
    """Harbor's ``provider/model`` → Loom's model string.

    ``google/gemini-4-argon`` → ``google:gemini-4-argon``; ``ollama/x`` stays.
    """
    if not harbor_model:
        raise ValueError("pass the orchestrator model with -m, e.g. -m google/gemini-4-argon")
    provider, _, name = harbor_model.partition("/")
    if provider == "ollama" or not name:
        return harbor_model
    if provider == "anthropic":
        return name
    return f"{provider}:{name}"


class LoomAgent(BaseInstalledAgent):
    options_model = LoomOptions

    @staticmethod
    @override
    def name() -> str:
        return "loom"

    @override
    def get_version_command(self) -> str | None:
        return f"{VENV}/bin/python -c 'import importlib.metadata as m; print(m.version(\"loom-cli\"))'"

    # ------------------------------------------------------------------ install

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        opts: LoomOptions = self.options  # type: ignore[assignment]
        wheel_dir = Path(opts.wheel_dir).resolve()
        wheels = sorted(wheel_dir.glob("loom_cli-*.whl"))
        reqs = wheel_dir / "requirements.txt"
        if not wheels or not reqs.exists():
            raise FileNotFoundError(f"no wheel/requirements in {wheel_dir} — run evals/tbench.py build")
        wheel = wheels[-1]

        # uv is uploaded from the host (static musl build, see tbench.py build)
        # rather than fetched with curl: installing curl means the image's
        # package manager, and a stale Debian mirror 404 once failed a trial
        # before Loom ever started. uv then needs only its own HTTPS client.
        arch = (await self.exec_as_root(environment, "uname -m")).stdout.strip()
        uv_bin = wheel_dir / f"uv-{arch}"
        if not uv_bin.exists():
            raise FileNotFoundError(f"no uv binary for {arch} in {wheel_dir} — rerun tbench.py build")
        await self.exec_as_root(environment, f"mkdir -p {PREFIX}/dist {PREFIX}/bin {LOOM_HOME}")
        await environment.upload_file(uv_bin, f"{PREFIX}/bin/uv")
        await environment.upload_file(wheel, f"{PREFIX}/dist/{wheel.name}")
        await environment.upload_file(reqs, f"{PREFIX}/dist/requirements.txt")
        # uv brings its own CPython, so the task image's Python (or lack of one)
        # is irrelevant and untouched; Loom lives in a private venv that is
        # never put on PATH, so the task's own `python` stays the task's.
        await self.exec_as_root(
            environment,
            command=(
                "set -euo pipefail; "
                f"export UV_PYTHON_INSTALL_DIR={PREFIX}/python UV_CACHE_DIR=/tmp/loom-uv-cache; "
                f"chmod +x {PREFIX}/bin/uv; "
                f"{PREFIX}/bin/uv venv -q --python 3.12 {VENV}; "
                f"{PREFIX}/bin/uv pip install -q --python {VENV}/bin/python "
                f"-r {PREFIX}/dist/requirements.txt {PREFIX}/dist/{wheel.name}; "
                "rm -rf /tmp/loom-uv-cache; "
                f"chmod -R a+rX {PREFIX}; chmod -R a+rwX {LOOM_HOME}"
            ),
            timeout_sec=900,
        )

    @override
    async def configure(self, environment: BaseEnvironment) -> None:
        opts: LoomOptions = self.options  # type: ignore[assignment]
        orchestrator = _loom_model(self.model_name or "")
        sub = opts.subagent_model
        settings = {
            "models": {
                "orchestrator": orchestrator,
                "advisor": orchestrator,
                "escalation_model": orchestrator,
                "cloud_fallback": opts.cloud_fallback,
                "subagents": {role: sub for role in _ROLES},
                "ollama_endpoint": opts.ollama_endpoint,
                "context_windows": {sub: opts.local_context} if sub.startswith("ollama/") else {},
            },
            "permissions": {"default_mode": "allow", "allow": ["*"], "ask": [], "deny": []},
            # No npx/Chromium in task images; the browser MCP would only stall.
            "mcp_servers": {},
            "ui": {"theme": "dark", "streaming": False, "show_fleet_panel": True, "banner": False},
        }
        await self._upload_config_text(
            environment,
            content=json.dumps(settings, indent=2),
            remote_path=f"{LOOM_HOME}/settings.json",
            filename="settings.json",
        )
        # Privacy choice answered up front: telemetry off, never prompt.
        await self._upload_config_text(
            environment,
            content=json.dumps({"version": 1, "mode": "none", "decided": True}),
            remote_path=f"{LOOM_HOME}/telemetry.json",
            filename="telemetry.json",
        )
        await self.exec_as_root(environment, f"chmod -R a+rwX {LOOM_HOME}")

    # ---------------------------------------------------------------------- run

    @override
    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        opts: LoomOptions = self.options  # type: ignore[assignment]
        env = {k: os.environ[k] for k in _KEY_VARS if os.environ.get(k)}
        env.update(
            LOOM_HOME=LOOM_HOME,
            LOOM_USAGE_LOG=f"{LOG_DIR}/usage.jsonl",
            COLUMNS="160",
            NO_COLOR="1",
            TERM="dumb",
            PYTHONUNBUFFERED="1",
        )
        flags = "--yolo"
        if opts.mode == "airgap":
            flags += " --airgap"
        elif opts.mode == "local-only":
            flags += " --local-only"
        # The task's WORKDIR (usually /app) is the project root, the way a user
        # would start Loom in their repo. Exit status is recorded, not raised,
        # so a crashed turn is still graded (as a fail) rather than discarded.
        await self.exec_as_agent(
            environment,
            command=(
                f"{VENV}/bin/loom {flags} --root \"$(pwd)\" {shlex.quote(instruction)} "
                f"</dev/null 2>&1 | tee {LOG_DIR}/loom.txt; "
                f"echo $? > {LOG_DIR}/exit_code"
            ),
            env=env,
        )

    # ------------------------------------------------------------------ results

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        usage = self.logs_dir / "usage.jsonl"
        cloud_in = cloud_out = cache = local_in = local_out = 0
        llm_calls: dict[str, int] = {}
        tools: dict[str, int] = {}
        if usage.exists():
            for line in usage.read_text(errors="replace").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("event") == "llm":
                    role = rec.get("role") or "?"
                    llm_calls[role] = llm_calls.get(role, 0) + 1
                    if rec.get("local"):
                        local_in += rec.get("input") or 0
                        local_out += rec.get("output") or 0
                    else:
                        cloud_in += rec.get("input") or 0
                        cloud_out += rec.get("output") or 0
                        cache += rec.get("cache_read") or 0
                elif rec.get("event") == "tool":
                    key = f"{rec.get('role')}:{rec.get('name')}"
                    tools[key] = tools.get(key, 0) + 1
        context.n_input_tokens = cloud_in
        context.n_output_tokens = cloud_out
        context.n_cache_tokens = cache

        # Loom's own receipt: the last "~$X cloud" figure it printed.
        transcript = self.logs_dir / "loom.txt"
        cost = None
        if transcript.exists():
            found = re.findall(r"session ~\$([0-9.]+)", transcript.read_text(errors="replace"))
            found = found or re.findall(r"~\$([0-9.]+) cloud", transcript.read_text(errors="replace"))
            if found:
                cost = float(found[-1])
        context.cost_usd = cost
        exit_code = (self.logs_dir / "exit_code")
        context.metadata = {
            "local_input_tokens": local_in,
            "local_output_tokens": local_out,
            "llm_calls_by_role": llm_calls,
            "tool_calls": tools,
            "loom_exit_code": exit_code.read_text().strip() if exit_code.exists() else None,
        }
