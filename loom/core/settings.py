"""Layered settings — a Claude Code-style ``settings.json`` for Loom.

Precedence (lowest to highest), deep-merged:

    packaged defaults  (loom/config/default_settings.json + default_config.yaml)
    ~/.loom/config.yaml            (legacy model routing — back-compat)
    ~/.loom/settings.json          (user)
    <project>/.loom/settings.json  (project — commit this)
    <project>/.loom/settings.local.json  (local — gitignore this)

The ``models`` section is a :class:`~loom.core.config.LoomConfig` (routing +
thresholds); the rest (``permissions``, ``hooks``, ``env``, ``ui``) is new.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from loom.core import config as cfg

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

DEFAULT_SETTINGS_PATH = cfg.PACKAGE_ROOT / "config" / "default_settings.json"
USER_SETTINGS_PATH = cfg.USER_CONFIG_DIR / "settings.json"


def project_settings_paths(root: str | Path = ".") -> list[Path]:
    base = Path(root).resolve() / ".loom"
    return [base / "settings.json", base / "settings.local.json"]


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------


class Permissions(BaseModel):
    """Allow/ask/deny rules, evaluated deny > allow > ask > default_mode."""

    default_mode: Literal["allow", "ask", "deny"] = "ask"
    allow: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class Hook(BaseModel):
    """A shell command run around a tool event.

    ``matcher`` is a glob (or ``a|b`` alternation) over the tool name, e.g.
    ``"write_file|edit_file"`` or ``"execute"`` or ``"*"``. The command receives
    a JSON event on stdin and ``LOOM_TOOL_NAME`` / ``LOOM_TOOL_INPUT`` in env.
    A pre_tool_use hook exiting non-zero blocks the tool.
    """

    matcher: str = "*"
    command: str
    timeout: int = 30


class Hooks(BaseModel):
    pre_tool_use: list[Hook] = Field(default_factory=list)
    post_tool_use: list[Hook] = Field(default_factory=list)
    user_prompt_submit: list[Hook] = Field(default_factory=list)
    stop: list[Hook] = Field(default_factory=list)


class MCPServer(BaseModel):
    """One MCP server the agent may connect to.

    stdio servers are launched as a subprocess (``command`` + ``args``); http
    servers (``transport: "streamable_http"`` or ``"sse"``) need a ``url``.
    Sessions are persistent for the process lifetime so stateful servers
    (e.g. Playwright's browser) keep their state across tool calls.
    """

    transport: Literal["stdio", "streamable_http", "sse"] = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""
    enabled: bool = True

    @model_validator(mode="after")
    def _url_or_command(self) -> "MCPServer":
        if self.transport != "stdio" and not self.url:
            raise ValueError(f"mcp server with transport {self.transport!r} requires a url")
        return self


class UISettings(BaseModel):
    theme: str = "auto"  # auto | dark | light | mono
    streaming: bool = True
    show_tool_calls: bool = True
    show_thinking: bool = True
    show_fleet_panel: bool = True
    prompt_symbol: str = ">"
    banner: bool = True

    @field_validator("theme")
    @classmethod
    def _valid_theme(cls, v: str) -> str:
        allowed = {"auto", "dark", "light", "mono"}
        if v not in allowed:
            raise ValueError(f"ui.theme must be one of {allowed}")
        return v


# Loom once reused Claude Code's CLAUDE_CODE_USE_BEDROCK flag; it now has its
# own name so a Loom env block can never reconfigure a Claude Code running in
# a subshell. Legacy settings.json env blocks are translated on the fly and
# the Claude Code name is never exported into the process environment.
_LEGACY_ENV_KEYS = {"CLAUDE_CODE_USE_BEDROCK": "LOOM_USE_BEDROCK"}


class Settings(BaseModel):
    """The fully-merged Loom settings object."""

    models: cfg.LoomConfig = Field(default_factory=cfg.LoomConfig)
    permissions: Permissions = Field(default_factory=Permissions)
    hooks: Hooks = Field(default_factory=Hooks)
    env: dict[str, str] = Field(default_factory=dict)
    ui: UISettings = Field(default_factory=UISettings)
    mcp_servers: dict[str, MCPServer] = Field(default_factory=dict)

    # Convenience passthroughs so callers can keep using `.models`-style access.
    @property
    def config(self) -> cfg.LoomConfig:
        return self.models

    def apply_env(self) -> None:
        """Inject configured env vars into the process (does not overwrite
        variables already set in the real environment)."""
        for key, value in self.env.items():
            os.environ.setdefault(_LEGACY_ENV_KEYS.get(key, key), str(value))


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh) or {}


def load_settings(root: str | Path = ".", *, extra_path: Path | None = None) -> Settings:
    """Load and deep-merge every settings layer into a validated Settings."""
    # 1. Packaged JSON defaults (permissions / hooks / env / ui).
    merged: dict[str, Any] = _read_json(DEFAULT_SETTINGS_PATH)

    # 2. Model routing defaults + legacy ~/.loom/config.yaml, folded into `models`.
    base_models = cfg._read_yaml(cfg.DEFAULT_CONFIG_PATH)
    if cfg.USER_CONFIG_PATH.exists():
        base_models = cfg._deep_merge(
            base_models, cfg._normalize_legacy_roles(cfg._read_yaml(cfg.USER_CONFIG_PATH))
        )
    merged["models"] = cfg._deep_merge(
        base_models, cfg._normalize_legacy_roles(merged.get("models", {}))
    )

    # 3. Layered settings.json files.
    layers = [USER_SETTINGS_PATH, *project_settings_paths(root)]
    if extra_path is not None:
        layers.append(extra_path)
    for layer in layers:
        data = _read_json(layer)
        if isinstance(data.get("models"), dict):
            data = {**data, "models": cfg._normalize_legacy_roles(data["models"])}
        merged = cfg._deep_merge(merged, data)

    return Settings(**merged)


def save_user_settings(settings: Settings) -> Path:
    """Persist the non-model sections to ~/.loom/settings.json.

    Model routing continues to live in config.yaml; settings.json owns
    permissions / hooks / env / ui so the two concerns stay separable.
    """
    USER_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = settings.model_dump(exclude={"models"})
    with USER_SETTINGS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return USER_SETTINGS_PATH


def _set_user_model_value(model_key: str, value: str) -> None:
    """Write a ``models.*`` override into ``~/.loom/settings.json`` — the same
    layer the setup wizard writes (see :func:`loom.ui.onboarding.apply_plan`)
    and the one that actually wins on load.

    Model routing used to be written to ``config.yaml`` here, but settings.json
    deep-merges *over* config.yaml (see :func:`load_settings`), so a ``models``
    block written by ``/setup`` silently shadowed every later ``/model`` change
    — the change persisted to config.yaml but never took effect. Writing both
    to the winning layer keeps ``/model`` and ``/setup`` in agreement.
    """
    USER_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _read_json(USER_SETTINGS_PATH)
    models: dict[str, Any] = dict(data.get("models") or {})

    parts = model_key.split(".")
    cursor: Any = models
    for part in parts[:-1]:
        nxt = dict(cursor.get(part) or {})
        cursor[part] = nxt
        cursor = nxt
    # Model ids are strings; other model keys (thresholds, depths, flags) coerce.
    cursor[parts[-1]] = cfg._coerce(value)

    # Validate the merged result before writing so a bad value never corrupts
    # the file — mirrors onboarding.apply_plan's pre-write check.
    cfg.LoomConfig(**cfg._deep_merge(cfg._read_yaml(cfg.DEFAULT_CONFIG_PATH), models))

    data["models"] = models
    with USER_SETTINGS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def set_value(dotted_key: str, value: str, root: str | Path = ".") -> Settings:
    """Set a settings value by dotted path, e.g.
    ``permissions.default_mode`` or ``ui.theme``. Model keys (``models.*``)
    are written into the user settings.json ``models`` block — the layer that
    wins over config.yaml, so ``/model`` and ``/setup`` never disagree."""
    if dotted_key.startswith("models."):
        _set_user_model_value(dotted_key[len("models.") :], value)
        return load_settings(root)

    settings = load_settings(root)
    data = settings.model_dump(exclude={"models"})
    parts = dotted_key.split(".")
    cursor: Any = data
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    # `env` values are always strings (they end up as process env vars) —
    # skip bool/int/float coercion so e.g. `env.FOO 1` doesn't become int 1.
    cursor[parts[-1]] = value if parts[0] == "env" else cfg._coerce(value)

    # Re-validate the whole object (models re-loaded from disk).
    merged = {**data, "models": settings.models.model_dump()}
    updated = Settings(**merged)
    save_user_settings(updated)
    return load_settings(root)
