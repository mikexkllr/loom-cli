"""Configuration loading, validation, and persistence for Loom.

The user-facing config lives at ``~/.loom/config.yaml``. On first run we copy the
packaged ``config/default_config.yaml`` there. Everything in the app reads a
validated :class:`LoomConfig` object, never the raw YAML.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "default_config.yaml"

USER_CONFIG_DIR = Path(os.environ.get("LOOM_HOME", Path.home() / ".loom"))
USER_CONFIG_PATH = USER_CONFIG_DIR / "config.yaml"


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------


class LoomConfig(BaseModel):
    """Validated Loom configuration."""

    orchestrator: str = "claude-sonnet-4-6"
    subagents: dict[str, str] = Field(default_factory=dict)
    advisor: str = "claude-opus-4-8"
    ollama_endpoint: str = "http://localhost:11434"

    escalation_model: str = "claude-sonnet-4-6"
    # When Ollama is unavailable (daemon down / model not pulled), local roles
    # temporarily run on this cheap cloud model instead of failing mid-run.
    cloud_fallback: str = "claude-haiku-4-5"
    context_windows: dict[str, int] = Field(default_factory=dict)

    compaction_threshold: float = 0.70
    escalation_threshold: float = 0.85
    artifact_offload_tokens: int = 2000

    advisor_threshold: str = "medium"
    max_nesting_depth: int = 2
    worktree_isolation: bool = True

    # ----- validation -----

    @model_validator(mode="before")
    @classmethod
    def _rename_general_role(cls, data: Any) -> Any:
        """Back-compat: the fallback subagent was renamed ``general`` →
        ``general-purpose`` (it must carry deepagents' reserved name to
        override the auto-added default). Old configs keep working."""
        if isinstance(data, dict):
            subagents = data.get("subagents")
            if isinstance(subagents, dict) and "general" in subagents:
                subagents = dict(subagents)
                # An explicit general-purpose entry wins over the legacy key.
                subagents.setdefault("general-purpose", subagents.pop("general"))
                data = {**data, "subagents": subagents}
        return data

    @field_validator("advisor_threshold")
    @classmethod
    def _valid_threshold(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        if v not in allowed:
            raise ValueError(f"advisor_threshold must be one of {allowed}, got {v!r}")
        return v

    @field_validator("compaction_threshold", "escalation_threshold")
    @classmethod
    def _valid_fraction(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {v}")
        return v

    @field_validator("max_nesting_depth")
    @classmethod
    def _valid_depth(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_nesting_depth must be >= 1")
        return v

    # ----- helpers -----

    def is_local(self, model: str) -> bool:
        """True if ``model`` is served by Ollama (local)."""
        return model.startswith(("ollama/", "ollama:"))

    def context_window_for(self, model: str, default: int = 32768) -> int:
        return self.context_windows.get(model, default)

    def all_models(self) -> dict[str, str]:
        """role -> model string, including orchestrator and advisor."""
        models = {"orchestrator": self.orchestrator, "advisor": self.advisor}
        models.update(self.subagents)
        return models


# ----------------------------------------------------------------------------
# Load / save
# ----------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _normalize_legacy_roles(data: dict[str, Any]) -> dict[str, Any]:
    """Rename the legacy ``subagents.general`` key to ``general-purpose`` in a
    single config layer. Applied per layer BEFORE merging so a user override
    written under the old name still beats packaged defaults under the new
    one. Within one layer, an explicit ``general-purpose`` wins."""
    subagents = data.get("subagents")
    if isinstance(subagents, dict) and "general" in subagents:
        subagents = dict(subagents)
        subagents.setdefault("general-purpose", subagents.pop("general"))
        data = {**data, "subagents": subagents}
    return data


def ensure_user_config() -> Path:
    """Create ``~/.loom/config.yaml`` from the packaged default if missing."""
    if not USER_CONFIG_PATH.exists():
        USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEFAULT_CONFIG_PATH, USER_CONFIG_PATH)
    return USER_CONFIG_PATH


def load_config(path: Path | None = None) -> LoomConfig:
    """Load and validate the Loom config.

    Precedence: explicit ``path`` > ``~/.loom/config.yaml`` > packaged default.
    The packaged default always provides fallback values for missing keys.
    """
    merged = _read_yaml(DEFAULT_CONFIG_PATH)

    if path is not None:
        merged = _deep_merge(merged, _normalize_legacy_roles(_read_yaml(path)))
    else:
        ensure_user_config()
        merged = _deep_merge(merged, _normalize_legacy_roles(_read_yaml(USER_CONFIG_PATH)))

    return LoomConfig(**merged)


def save_config(config: LoomConfig, path: Path | None = None) -> Path:
    target = path or USER_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.model_dump(), fh, sort_keys=False, default_flow_style=False)
    return target


def set_value(key: str, value: str, path: Path | None = None) -> LoomConfig:
    """Implements ``loom config set <key> <value>``.

    Supports dotted keys for nested maps, e.g. ``subagents.editor``.
    """
    config = load_config(path)
    data = config.model_dump()

    parts = key.split(".")
    cursor: Any = data
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = _coerce(value)

    updated = LoomConfig(**data)  # re-validate
    save_config(updated, path)
    return updated


def _coerce(value: str) -> Any:
    """Best-effort scalar coercion for CLI-provided values."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
