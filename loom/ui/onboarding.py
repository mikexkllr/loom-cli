"""Interactive setup wizard — configure every model role and provider
credentials from the UI, writing the result to a ``settings.json`` layer and
reloading it live. Reachable via ``/setup`` in the REPL, ``loom setup`` from
the shell, and auto-launched on a true first run (see ``needs_onboarding``).

Split for testability: :func:`apply_plan` and :func:`missing_credentials` are
pure and covered directly by tests; the ``prompt_*``/:func:`run` functions are
thin interactive glue around them and are exercised manually / via smoke
tests, matching the rest of the REPL's slash commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from loom.core import config as cfg
from loom.core import ollama as ollama_mod
from loom.core import providers as prov
from loom.core import recommendations as rec
from loom.core import settings as settings_mod
from loom.core.settings import Settings

# LoomConfig fields configured directly (not under `subagents`).
TOP_LEVEL_ROLES = ("orchestrator", "advisor", "escalation")
# Subagent roles, in the order they run day to day.
SUBAGENT_ROLES = ("explorer", "editor", "bash", "searcher", "reviewer", "general-purpose", "tester")
ALL_ROLES = TOP_LEVEL_ROLES + SUBAGENT_ROLES

# Quick-setup grouping: which roles default to cloud vs local, and which
# model "tier" (see ProviderInfo.model_for_tier) each cloud role gets.
_ROLE_TIER = {"orchestrator": "main", "advisor": "flagship", "escalation": "main", "reviewer": "light"}
_DEFAULT_CLOUD_ROLES = tuple(_ROLE_TIER)
_DEFAULT_LOCAL_ROLES = tuple(r for r in ALL_ROLES if r not in _DEFAULT_CLOUD_ROLES)


def _settings_key(role: str) -> str:
    """Dotted key under ``models`` for ``role`` (matches LoomConfig fields)."""
    if role == "escalation":
        return "escalation_model"
    if role in TOP_LEVEL_ROLES:
        return role
    return f"subagents.{role}"


# ----------------------------------------------------------------------------
# Pure logic — settings I/O and credential bookkeeping
# ----------------------------------------------------------------------------


def apply_plan(
    root: str | Path,
    scope: str,
    models: dict[str, str],
    env: dict[str, str],
) -> Settings:
    """Write role -> model-string assignments and env vars into one
    ``settings.json`` layer, then return the freshly reloaded, merged
    :class:`Settings`.

    ``scope`` is ``"user"`` (``~/.loom/settings.json``) or ``"project"``
    (``<root>/.loom/settings.json``). Both layers are deep-merged by
    :func:`loom.core.settings.load_settings` — a top-level ``"models"`` key in
    settings.json overlays ``config.yaml``'s defaults regardless of scope, so
    role assignments work the same way at either layer.
    """
    if scope == "project":
        target = settings_mod.project_settings_paths(root)[0]
    elif scope == "user":
        target = settings_mod.USER_SETTINGS_PATH
    else:
        raise ValueError(f"scope must be 'user' or 'project', got {scope!r}")
    target.parent.mkdir(parents=True, exist_ok=True)

    data = settings_mod._read_json(target)

    model_patch: dict[str, Any] = {}
    subagents_patch: dict[str, str] = {}
    for role, model_string in models.items():
        if role in TOP_LEVEL_ROLES:
            model_patch[_settings_key(role)] = model_string
        else:
            subagents_patch[role] = model_string
    if subagents_patch:
        model_patch["subagents"] = subagents_patch

    data["models"] = cfg._deep_merge(data.get("models", {}), model_patch)
    if env:
        data["env"] = {**data.get("env", {}), **env}

    # Validate before writing — a bad value should never corrupt the file.
    settings_mod.Settings(**{**data, "models": cfg._deep_merge(cfg._read_yaml(cfg.DEFAULT_CONFIG_PATH), data["models"])})

    with target.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    return settings_mod.load_settings(root)


def missing_credentials(provider: prov.ProviderInfo, known_env: dict[str, str]) -> list[prov.EnvVar]:
    """Required env vars for ``provider`` not already set (real env or the
    given settings.json env block)."""
    import os

    return [
        v
        for v in provider.env_vars
        if v.required and not (os.environ.get(v.key) or known_env.get(v.key))
    ]


def _mask(value: str) -> str:
    """Show just enough of a secret to recognize it (last 4 chars)."""
    return "…" + value[-4:] if len(value) > 8 else "(set)"


def default_role_plan(hw: rec.Hardware, local_tag: str, cloud_provider: prov.ProviderInfo | None) -> dict[str, str]:
    """The "quick setup" assignment: one local tag for local-leaning roles,
    one cloud provider (tiered per role) for cloud-leaning roles — mirrors
    the shape of ``loom/config/default_config.yaml``. ``cloud_provider=None``
    means local-only: every role gets ``local_tag``."""
    ollama = prov.get("ollama")
    plan: dict[str, str] = {}
    for role in _DEFAULT_LOCAL_ROLES:
        plan[role] = ollama.model_string(local_tag)
    for role in _DEFAULT_CLOUD_ROLES:
        if cloud_provider is None:
            plan[role] = ollama.model_string(local_tag)
        else:
            plan[role] = cloud_provider.model_string(cloud_provider.model_for_tier(_ROLE_TIER[role]))
    return plan


def needs_onboarding(root: str | Path = ".") -> bool:
    """True if there's no user- or project-level settings.json yet — a
    genuine first run, worth auto-launching the wizard for."""
    if settings_mod.USER_SETTINGS_PATH.exists():
        return False
    return not any(p.exists() for p in settings_mod.project_settings_paths(root))


# ----------------------------------------------------------------------------
# Interactive wizard
# ----------------------------------------------------------------------------


def _print_hardware_and_local_recs(console: Console, hw: rec.Hardware, installed: list[str]) -> list[str]:
    console.print(f"[dim]detected hardware: {rec.hardware_summary(hw)}[/dim]")
    recs = [r.tag for r in rec.recommend_local_models(hw)]
    options: list[str] = list(installed)
    for tag in recs:
        if tag not in options:
            options.append(tag)
    table = Table(show_header=True, header_style="bold cyan")
    for col in ("#", "Model", "Status"):
        table.add_column(col)
    for i, tag in enumerate(options, 1):
        status = "installed" if tag in installed else "recommended — needs `ollama pull`"
        table.add_row(str(i), tag, status)
    console.print(table)
    return options


def prompt_local_model(console: Console, hw: rec.Hardware, config: "cfg.LoomConfig | None" = None) -> str:
    """Pick (and optionally pull) a local Ollama model tag.

    ``config`` supplies the daemon endpoint — pulls go through the HTTP API of
    the configured (possibly remote) daemon, so no ollama binary is needed.
    """
    if config is None:
        config = cfg.LoomConfig()
    st = ollama_mod.status(config)
    options = _print_hardware_and_local_recs(console, hw, st.models)
    choice = Prompt.ask("  number, or type any ollama tag", default="1" if options else "")
    if choice.isdigit() and options and 1 <= int(choice) <= len(options):
        tag = options[int(choice) - 1]
    else:
        tag = choice.strip()
    if not tag:
        tag = options[0] if options else "qwen3.5:9b"
    if not ollama_mod.is_served(tag, st.models):
        if not st.running:
            hint = ollama_mod.daemon_hint(st.endpoint) if st.installed else ollama_mod.INSTALL_HINT
            console.print(f"[yellow]{hint}[/yellow]")
        elif Confirm.ask(f"  `{tag}` isn't pulled yet — pull it now?", default=True):
            if ollama_mod.pull(tag, config.ollama_endpoint, console) != 0:
                console.print(
                    f"[yellow]pull failed — roles on `{tag}` use the cloud fallback "
                    f"until `loom models pull {tag}` succeeds.[/yellow]"
                )
    return tag


def prompt_provider(console: Console, candidates: tuple[prov.ProviderInfo, ...]) -> prov.ProviderInfo:
    table = Table(show_header=True, header_style="bold cyan")
    for col in ("#", "Provider", "Notes"):
        table.add_column(col)
    for i, p in enumerate(candidates, 1):
        table.add_row(str(i), p.label, p.notes)
    console.print(table)
    choice = Prompt.ask("  pick a provider", choices=[str(i) for i in range(1, len(candidates) + 1)], default="1")
    return candidates[int(choice) - 1]


def prompt_credentials(
    console: Console,
    provider: prov.ProviderInfo,
    known_env: dict[str, str],
    existing_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Collect env vars for ``provider``.

    Values entered earlier in this wizard run (``known_env``) are reused
    silently. Values that pre-date this run — the merged settings.json ``env``
    block (``existing_env``) or the real environment — are shown masked and
    the user picks keep vs overwrite, so re-running setup can rotate an API
    key or endpoint instead of silently reusing the old one.
    """
    import os

    existing_env = existing_env or {}
    collected: dict[str, str] = {}
    for v in provider.env_vars:
        if known_env.get(v.key):
            collected[v.key] = known_env[v.key]
            continue
        current = os.environ.get(v.key) or existing_env.get(v.key)
        if current:
            shown = _mask(current) if v.secret else current
            if Confirm.ask(f"  {v.label} ({v.key}) is already set ({shown}) — keep it?", default=True):
                collected[v.key] = current
                continue
        value = Prompt.ask(f"  {v.label} ({v.key})", password=v.secret, default=v.default or None)
        if value:
            collected[v.key] = value
        elif current:
            # Declined the keep, then entered nothing — fall back to the old
            # value rather than leaving the provider broken.
            collected[v.key] = current
        elif v.required:
            console.print(f"[yellow]{v.key} left blank — {provider.label} won't work until it's set.[/yellow]")
    if provider.pip_extra:
        console.print(f"[dim]note: needs `pip install -e '.[{provider.pip_extra}]'`[/dim]")
    if provider.docs_url:
        console.print(f"[dim]get a key: {provider.docs_url}[/dim]")
    return collected


def prompt_cloud_model(console: Console, provider: prov.ProviderInfo, tier: str = "main") -> str:
    default = provider.model_for_tier(tier)
    examples = ", ".join(provider.example_models) or default
    console.print(f"[dim]examples: {examples}[/dim]")
    return Prompt.ask(f"  model id for {provider.label}", default=default) or default


def _configure_one_role(
    console: Console,
    role: str,
    hw: rec.Hardware,
    known_env: dict[str, str],
    config: "cfg.LoomConfig | None" = None,
    existing_env: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    console.print(f"\n[bold cyan]── {role} ──[/bold cyan]")
    kind = Prompt.ask("  local or cloud?", choices=["local", "cloud"], default="cloud")
    if kind == "local":
        tag = prompt_local_model(console, hw, config)
        return prov.get("ollama").model_string(tag), {}
    provider = prompt_provider(console, prov.cloud_providers())
    env = prompt_credentials(console, provider, known_env, existing_env)
    tier = _ROLE_TIER.get(role, "main")
    model_id = prompt_cloud_model(console, provider, tier)
    return provider.model_string(model_id), env


def _print_current_setup(console: Console, settings: Settings) -> None:
    """Show the existing role → model assignments and configured credentials,
    so a re-run of the wizard starts from what's already there instead of
    pretending it's a first run."""
    import os

    models = settings.models
    current = {
        "orchestrator": models.orchestrator,
        "advisor": models.advisor,
        "escalation": models.escalation_model,
        **models.subagents,
    }
    table = Table(title="current setup", show_header=True, header_style="bold cyan")
    for col in ("Role", "Model", "Where"):
        table.add_column(col)
    for role in ALL_ROLES:
        model = current.get(role)
        if model:
            table.add_row(role, model, "⌂ local" if models.is_local(model) else "☁ cloud")
    console.print(table)

    creds = sorted(
        {
            f"{v.key} ({_mask(str(os.environ.get(v.key) or settings.env[v.key])) if v.secret else 'set'})"
            for p in prov.PROVIDERS
            for v in p.env_vars
            if os.environ.get(v.key) or settings.env.get(v.key)
        }
    )
    if creds:
        console.print(
            "[dim]credentials on file: " + ", ".join(creds) + " — you'll be asked before any are overwritten[/dim]"
        )


def run(
    console: Console,
    *,
    root: str | Path = ".",
    roles: tuple[str, ...] = ALL_ROLES,
    scope: str | None = None,
) -> Settings:
    """Run the full wizard and return the reloaded, merged Settings.

    ``scope`` skips the "user vs project" prompt when given ("user" | "project").
    """
    console.print(
        Panel(
            "Let's configure Loom's models.\n"
            "[dim]quick setup picks one local model + one cloud provider for sensible defaults; "
            "advanced lets you set every role individually.[/dim]",
            title="✻ Loom setup",
            border_style="bold cyan",
        )
    )
    hw = rec.detect_hardware()
    # The merged current settings supply the (possibly remote) Ollama endpoint
    # pulls go to, plus the existing models/env the wizard starts from.
    current_settings = settings_mod.load_settings(root)
    models_config = current_settings.models
    existing_env = dict(current_settings.env)
    if not needs_onboarding(root):
        _print_current_setup(console, current_settings)
    mode = Prompt.ask("  quick setup or advanced?", choices=["quick", "advanced"], default="quick")

    known_env: dict[str, str] = {}
    if mode == "quick":
        console.print("\n[bold cyan]── local models (explorer/editor/bash/searcher/general-purpose/tester) ──[/bold cyan]")
        local_tag = prompt_local_model(console, hw, models_config)
        console.print("\n[bold cyan]── cloud provider (orchestrator/advisor/escalation/reviewer) ──[/bold cyan]")
        use_cloud = Confirm.ask("  use a cloud provider for these roles?", default=True)
        cloud_provider = None
        if use_cloud:
            cloud_provider = prompt_provider(console, prov.cloud_providers())
            known_env = prompt_credentials(console, cloud_provider, known_env, existing_env)
        models = default_role_plan(hw, local_tag, cloud_provider)
    else:
        models = {}
        for role in roles:
            model_string, env = _configure_one_role(console, role, hw, known_env, models_config, existing_env)
            models[role] = model_string
            known_env.update(env)

    console.print(f"\n[dim]detected: {rec.hardware_summary(hw)}[/dim]")
    console.print(f"[dim]{rec.CLOUD_RECOMMENDATION}[/dim]")

    if scope not in ("user", "project"):
        scope = Prompt.ask(
            "\n  save to user settings (~/.loom) or this project (.loom)?", choices=["user", "project"], default="user"
        )
    settings = apply_plan(root, scope, models, known_env)

    table = Table(title="configured", show_header=True, header_style="bold cyan")
    for col in ("Role", "Model"):
        table.add_column(col)
    for role in ALL_ROLES:
        table.add_row(role, models.get(role, "(unchanged)"))
    console.print(table)
    console.print(f"[bold cyan]saved to {scope} settings.json — reload complete.[/bold cyan]")
    return settings
