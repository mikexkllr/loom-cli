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

from pydantic import BaseModel, Field, field_validator

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


class UISettings(BaseModel):
    theme: str = "auto"  # auto | dark | light | mono
    streaming: bool = True
    show_tool_calls: bool = True
    show_thinking: bool = False
    show_fleet_panel: bool = True
    prompt_symbol: str = "loom ›"
    banner: bool = True

    @field_validator("theme")
    @classmethod
    def _valid_theme(cls, v: str) -> str:
        allowed = {"auto", "dark", "light", "mono"}
        if v not in allowed:
            raise ValueError(f"ui.theme must be one of {allowed}")
        return v


class Settings(BaseModel):
    """The fully-merged Loom settings object."""

    models: cfg.LoomConfig = Field(default_factory=cfg.LoomConfig)
    permissions: Permissions = Field(default_factory=Permissions)
    hooks: Hooks = Field(default_factory=Hooks)
    env: dict[str, str] = Field(default_factory=dict)
    ui: UISettings = Field(default_factory=UISettings)

    # Convenience passthroughs so callers can keep using `.models`-style access.
    @property
    def config(self) -> cfg.LoomConfig:
        return self.models

    def apply_env(self) -> None:
        """Inject configured env vars into the process (does not overwrite
        variables already set in the real environment)."""
        for key, value in self.env.items():
            os.environ.setdefault(key, str(value))


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
        base_models = cfg._deep_merge(base_models, cfg._read_yaml(cfg.USER_CONFIG_PATH))
    merged["models"] = cfg._deep_merge(base_models, merged.get("models", {}))

    # 3. Layered settings.json files.
    layers = [USER_SETTINGS_PATH, *project_settings_paths(root)]
    if extra_path is not None:
        layers.append(extra_path)
    for layer in layers:
        merged = cfg._deep_merge(merged, _read_json(layer))

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


def set_value(dotted_key: str, value: str, root: str | Path = ".") -> Settings:
    """Set a settings value by dotted path, e.g.
    ``permissions.default_mode`` or ``ui.theme``. Model keys (``models.*``)
    are delegated to the config.yaml writer for back-compat."""
    if dotted_key.startswith("models."):
        cfg.set_value(dotted_key[len("models.") :], value)
        return load_settings(root)

    settings = load_settings(root)
    data = settings.model_dump(exclude={"models"})
    parts = dotted_key.split(".")
    cursor: Any = data
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = cfg._coerce(value)

    # Re-validate the whole object (models re-loaded from disk).
    merged = {**data, "models": settings.models.model_dump()}
    updated = Settings(**merged)
    save_user_settings(updated)
    return load_settings(root)
