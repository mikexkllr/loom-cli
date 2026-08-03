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
import shutil
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text

from loom.core import config as cfg
from loom.core import model_catalog as catalog
from loom.core import ollama as ollama_mod
from loom.core import providers as prov
from loom.core import recommendations as rec
from loom.core import settings as settings_mod
from loom.core.settings import Settings
from loom.ui import render
from loom.ui.render import ink

# LoomConfig fields configured directly (not under `subagents`).
TOP_LEVEL_ROLES = ("orchestrator", "advisor", "escalation")
# Subagent roles, in the order they run day to day.
SUBAGENT_ROLES = ("explorer", "editor", "bash", "searcher", "reviewer", "general-purpose", "tester")
ALL_ROLES = TOP_LEVEL_ROLES + SUBAGENT_ROLES

# Which model "tier" (see ProviderInfo.model_for_tier) a role gets *if* it ends
# up on a cloud provider. Every role has a sensible tier; that is separate from
# whether quick setup puts it there.
_ROLE_TIER = {"orchestrator": "main", "advisor": "flagship", "escalation": "main", "reviewer": "light"}

# Quick setup's shape, and Loom's whole argument: only the three roles that
# have to reason across the *whole* task go to the cloud. Everything that
# touches raw file content, shell output or test logs stays on this machine —
# that is what keeps the orchestrator's context clean and the bill small.
# Advanced mode still lets any role go anywhere.
_DEFAULT_CLOUD_ROLES = ("orchestrator", "advisor", "escalation")
_DEFAULT_LOCAL_ROLES = tuple(r for r in ALL_ROLES if r not in _DEFAULT_CLOUD_ROLES)


# Not a role anyone assigns work to, but a model string the wizard must own:
# when Ollama is down every local role runs on it, so leaving it pointing at a
# provider the user never configured turns one dead daemon into a dead session.
FALLBACK_KEY = "cloud_fallback"


def _settings_key(role: str) -> str:
    """Dotted key under ``models`` for ``role`` (matches LoomConfig fields)."""
    if role == "escalation":
        return "escalation_model"
    if role == FALLBACK_KEY or role in TOP_LEVEL_ROLES:
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
        if role == FALLBACK_KEY or role in TOP_LEVEL_ROLES:
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


def effective_model(role: str, settings: Settings) -> str:
    """What ``role`` actually resolves to in the merged config."""
    models = settings.models
    if role == FALLBACK_KEY:
        return models.cloud_fallback
    if role == "escalation":
        return models.escalation_model
    if role in TOP_LEVEL_ROLES:
        return getattr(models, role)
    return models.subagents.get(role, "")


def shadowed_roles(intended: dict[str, str], settings: Settings) -> dict[str, tuple[str, str]]:
    """``role -> (what we wrote, what Loom will actually use)`` for roles a
    higher-priority layer overrides.

    Settings merge across four layers, and a write to a layer that a later one
    also names has no effect. Rather than reason about precedence in the UI —
    and get it subtly wrong — this writes, reloads, and compares. That catches
    every cause of a silently-ignored save, including ones nobody anticipated.
    """
    return {
        role: (wanted, effective_model(role, settings))
        for role, wanted in intended.items()
        if effective_model(role, settings) != wanted
    }


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


def default_role_plan(
    hw: rec.Hardware,
    local_tag: str,
    cloud_provider: prov.ProviderInfo | None,
    tier_models: dict[str, str] | None = None,
) -> dict[str, str]:
    """The "quick setup" assignment: one local tag for local-leaning roles,
    one cloud provider (tiered per role) for cloud-leaning roles — mirrors
    the shape of ``loom/config/default_config.yaml``. ``cloud_provider=None``
    means local-only: every role gets ``local_tag``. ``tier_models`` lets the
    caller override a tier's model id (e.g. from the catalog picker) instead
    of the provider's built-in ``main``/``flagship``/``light`` default."""
    ollama = prov.get("ollama")
    plan: dict[str, str] = {}
    for role in _DEFAULT_LOCAL_ROLES:
        plan[role] = ollama.model_string(local_tag)
    for role in _DEFAULT_CLOUD_ROLES:
        if cloud_provider is None:
            plan[role] = ollama.model_string(local_tag)
        else:
            tier = _ROLE_TIER[role]
            model_id = (tier_models or {}).get(tier) or cloud_provider.model_for_tier(tier)
            plan[role] = cloud_provider.model_string(model_id)
    if cloud_provider is not None:
        # One provider, one key. Without this the fallback keeps whatever a
        # previous config left it pointing at, and the first time Ollama is
        # down every local role fails on a credential the user never set.
        plan[FALLBACK_KEY] = cloud_provider.model_string(cloud_provider.model_for_tier("light"))
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


def ensure_ollama(console: Console, config: "cfg.LoomConfig", *, ask: bool = True) -> bool:
    """Get to a reachable Ollama daemon, installing and starting it if needed.

    Returns True if the daemon answers by the end. Three cases, cheapest first:
    already up; installed but not running (just start it); not installed at all
    (offer an installer, then start it).

    Every step is confirmed and every command is shown, because this installs
    software and launches a background service on someone's machine.
    """
    from loom.core import ollama_setup

    endpoint = config.ollama_endpoint
    if ollama_setup.is_up(endpoint):
        return True

    if not ollama_setup.is_local_endpoint(endpoint):
        # Nothing installed here can fix a daemon configured to live elsewhere.
        render.note(
            console,
            f"`ollama_endpoint` points at {endpoint} — start Ollama there, or change the endpoint",
            kind="warn",
        )
        return False

    if shutil.which("ollama") is None:
        console.print()
        render.rule(console, "ollama")
        render.note(console, "Ollama runs your local models — it isn't installed yet")
        methods = [m for m in ollama_setup.install_methods() if m.automatic]
        if not methods:
            render.note(console, f"install it from {ollama_setup.DOWNLOAD_URL}, then re-run /setup", kind="warn")
            return False
        method = methods[0]
        console.print()
        console.print(
            render.card(
                console,
                render.stack(
                    Text(method.display, style="loom.bright"),
                    Text(method.note, style="loom.muted") if method.note else None,
                    Text("it will ask for your password", style="loom.warn") if method.needs_root else None,
                ),
                title=f"install via {method.label}",
            )
        )
        if ask and not render.confirm(console, "  run it now?"):
            render.note(console, f"skipped — install from {ollama_setup.DOWNLOAD_URL} any time")
            return False
        if ollama_setup.install(console, method) != 0:
            render.note(console, f"install failed — try {ollama_setup.DOWNLOAD_URL}", kind="bad")
            return False
        render.note(console, "Ollama installed", kind="good")

    # Installed but not answering: start it.
    if ask and not render.confirm(console, "  start the Ollama daemon now?"):
        render.note(console, "skipped — `ollama serve` starts it when you're ready")
        return False
    with render.Working(console, "starting the Ollama daemon"):
        started = ollama_setup.serve(endpoint)
    if started:
        render.note(console, f"daemon running at {endpoint}", kind="good")
        return True
    render.note(
        console,
        f"the daemon didn't come up within {ollama_setup.SERVE_TIMEOUT:.0f}s — "
        f"see {ollama_setup.log_path()}, or run `ollama serve` yourself",
        kind="warn",
    )
    return False


def _print_hardware_and_local_recs(
    console: Console,
    hw: rec.Hardware,
    installed: list[str],
    daemon_up: bool | None = None,
    config: "cfg.LoomConfig | None" = None,
) -> list[str]:
    """List every installed Ollama model plus Loom's full curated catalog
    (see recommendations.py — there's no stable public API for "every model
    in the Ollama library", so this is a hand-maintained snapshot, not a live
    query), annotated with whether each fits the detected hardware."""
    g = ink(console)
    render.note(console, f"detected: {rec.hardware_summary(hw)}")
    options: list[str] = list(installed)
    # Without this, every row reads "needs pull" when the daemon is simply
    # down, which looks like a catalogue of missing models rather than one
    # unreachable service.
    if daemon_up is False:
        render.note(
            console,
            "Ollama isn't reachable, so nothing shows as downloaded — you can still "
            "pick a model and pull it once the daemon is up",
            kind="warn",
        )
    elif installed:
        render.note(console, f"{len(installed)} model(s) already downloaded", kind="good")
    table = render.table(
        console,
        ("#", {"justify": "right", "no_wrap": True}),
        ("model", {"no_wrap": True}),
        ("", {"overflow": "fold"}),
    )
    sizes = ollama_mod.installed_sizes(config) if config is not None else {}
    for i, tag in enumerate(options, 1):
        on_disk = sizes.get(tag, 0)
        table.add_row(
            Text(str(i), style="loom.warp"),
            Text(tag, style="loom.local"),
            Text(
                f"{g.ok} downloaded" + (f" {g.dot} {ollama_mod.human_size(on_disk)}" if on_disk else ""),
                style="loom.good",
            ),
        )
    for m in rec.all_local_models():
        if m.tag in installed:
            continue
        options.append(m.tag)
        fits = rec.fits_hardware(hw, m)
        # The download size is the thing people are surprised by, so it goes
        # first when it's known — a 9.6 GB pull on an 8 GB laptop should be
        # obvious before it starts, not after.
        detail = Text()
        if m.size_gb:
            detail.append(f"{m.size_gb:.1f} GB", style="loom.text" if fits else "loom.warn")
            detail.append(f" {g.dot} ", style="loom.line")
        detail.append("needs pull", style="loom.muted")
        detail.append(f" {g.dot} ", style="loom.line")
        detail.append(
            "fits your hardware" if fits else f"needs a {m.min_gb:.0f}GB machine",
            style="loom.good" if fits else "loom.warn",
        )
        detail.append(f" {g.dot} {m.blurb}", style="loom.muted")
        table.add_row(Text(str(len(options)), style="loom.warp"), Text(m.tag, style="loom.text"), detail)
    console.print(table)
    return options


def prompt_local_model(console: Console, hw: rec.Hardware, config: "cfg.LoomConfig | None" = None) -> str:
    """Pick (and optionally pull) a local Ollama model tag.

    ``config`` supplies the daemon endpoint — pulls go through the HTTP API of
    the configured (possibly remote) daemon, so no ollama binary is needed.
    """
    if config is None:
        config = cfg.LoomConfig()
    if not ollama_mod.status(config).running:
        # Do this before listing models: with a live daemon the list can say
        # what's actually downloaded, and a pick can be pulled on the spot.
        ensure_ollama(console, config)
    st = ollama_mod.status(config)
    options = _print_hardware_and_local_recs(console, hw, st.models, daemon_up=st.running, config=config)
    choice = Prompt.ask("  number, or type any ollama tag", default="1" if options else "", console=console)
    if choice.isdigit() and options and 1 <= int(choice) <= len(options):
        tag = options[int(choice) - 1]
    else:
        tag = choice.strip()
    if not tag:
        tag = options[0] if options else "qwen3.5:9b"
    if not ollama_mod.is_served(tag, st.models):
        if not st.running:
            hint = ollama_mod.daemon_hint(st.endpoint) if st.installed else ollama_mod.INSTALL_HINT
            console.print(f"[loom.warn]{hint}[/loom.warn]")
        elif render.confirm(console, f"  `{tag}` isn't pulled yet — pull it now?"):
            if ollama_mod.pull(tag, config.ollama_endpoint, console) != 0:
                console.print(
                    f"[loom.warn]pull failed — roles on `{tag}` use the cloud fallback "
                    f"until `loom models pull {tag}` succeeds.[/loom.warn]"
                )
    return tag


def prompt_provider(
    console: Console,
    candidates: tuple[prov.ProviderInfo, ...],
    *,
    known_env: dict[str, str] | None = None,
    current: "prov.ProviderInfo | None" = None,
) -> prov.ProviderInfo:
    """Pick a cloud provider.

    The default is the provider you are *already* set up with — the one your
    config points at, or failing that the first one whose key is already
    present. Defaulting to the top of the list meant pressing Enter silently
    selected Anthropic and then demanded an Anthropic key, which is a poor
    thing to do to someone who has never used Anthropic and does not want to.

    Providers whose credentials are already available are marked, so it is
    obvious which ones cost you nothing to choose.
    """
    known_env = known_env or {}
    ready = {p.id: not missing_credentials(p, known_env) for p in candidates}

    usable = {p.id: prov.is_available(p.id) for p in candidates}
    default_index = next((i for i, p in enumerate(candidates, 1) if usable[p.id]), 1)
    if current is not None and current in candidates:
        default_index = candidates.index(current) + 1
    else:
        # Prefer one that is both installed and already credentialed — pressing
        # Enter should never select a provider that cannot possibly run.
        for i, p in enumerate(candidates, 1):
            if ready[p.id] and usable[p.id]:
                default_index = i
                break

    g = ink(console)
    table = render.table(
        console,
        ("#", {"justify": "right", "no_wrap": True}),
        ("provider", {"no_wrap": True}),
        ("", {"no_wrap": True}),
        ("", {"overflow": "fold"}),
    )
    for i, p in enumerate(candidates, 1):
        if not prov.is_available(p.id):
            # A key would not help: the package backing this provider isn't in
            # this install. Say so here rather than at first use.
            state, style = f"{g.warn} not installed", "loom.warn"
        elif current is not None and p is current:
            state, style = "in use", "loom.warp"
        elif ready[p.id]:
            state, style = f"{g.ok} key set", "loom.good"
        else:
            state, style = "needs a key", "loom.muted"
        table.add_row(
            Text(str(i), style="loom.warp" if i == default_index else "loom.muted"),
            Text(p.label, style="loom.bright" if i == default_index else "loom.text"),
            Text(state, style=style),
            Text(p.notes, style="loom.muted"),
        )
    console.print(table)
    console.print()
    choice = Prompt.ask(
        "  pick a provider",
        choices=[str(i) for i in range(1, len(candidates) + 1)],
        default=str(default_index),
        console=console,
    )
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
            if render.confirm(console, f"  {v.label} ({v.key}) is already set ({shown}) — keep it?", on_eof=True):
                collected[v.key] = current
                continue
        value = Prompt.ask(f"  {v.label} ({v.key})", password=v.secret, default=v.default or None, console=console)
        if value:
            collected[v.key] = value
        elif current:
            # Declined the keep, then entered nothing — fall back to the old
            # value rather than leaving the provider broken.
            collected[v.key] = current
        elif v.required:
            console.print(f"[loom.warn]{v.key} left blank — {provider.label} won't work until it's set.[/loom.warn]")
    if provider.pip_extra and not prov.is_available(provider.id):
        console.print(f"[loom.warn]{provider.label} needs a package this install doesn't have[/loom.warn]")
        console.print(f"[loom.muted]{prov.extra_install_hint(provider.pip_extra)}[/loom.muted]")
    if provider.docs_url:
        console.print(f"[loom.muted]get a key: {provider.docs_url}[/loom.muted]")
    return collected


def prompt_cloud_model(
    console: Console, provider: prov.ProviderInfo, tier: str = "main", env: dict[str, str] | None = None
) -> str:
    """Pick a model id for ``provider``. Shows a numbered picker sourced from
    the provider's live model-list endpoint when one's reachable (see
    ``model_catalog``), falling back to its hardcoded example models
    otherwise — either way you can also just type any model id directly."""
    default = provider.model_for_tier(tier)
    models, is_live = catalog.available_models(provider, env or {})
    if models:
        render.note(console, f"{'live' if is_live else 'example'} models for {provider.label}")
        table = render.table(
            console,
            ("#", {"justify": "right", "no_wrap": True}),
            ("model", {"overflow": "fold"}),
        )
        for i, m in enumerate(models, 1):
            table.add_row(Text(str(i), style="loom.warp"), Text(m, style="loom.cloud"))
        console.print(table)
        default_choice = str(models.index(default) + 1) if default in models else default
        choice = Prompt.ask("  number, or type any model id", default=default_choice, console=console)
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        return choice.strip() or default
    return Prompt.ask(f"  model id for {provider.label}", default=default, console=console) or default


def _configure_one_role(
    console: Console,
    role: str,
    hw: rec.Hardware,
    known_env: dict[str, str],
    config: "cfg.LoomConfig | None" = None,
    existing_env: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    console.print()
    render.rule(console, role)
    kind = Prompt.ask("  local or cloud?", choices=["local", "cloud"], default="cloud", console=console)
    if kind == "local":
        tag = prompt_local_model(console, hw, config)
        return prov.get("ollama").model_string(tag), {}
    provider = prompt_provider(
        console,
        prov.cloud_providers(),
        known_env={**(existing_env or {}), **known_env},
        current=_provider_of(config.orchestrator) if config is not None else None,
    )
    env = prompt_credentials(console, provider, known_env, existing_env)
    tier = _ROLE_TIER.get(role, "main")
    model_id = prompt_cloud_model(console, provider, tier, {**known_env, **env})
    return provider.model_string(model_id), env


# ----------------------------------------------------------------------------
# Preflight — prove the plan works before the user finds out mid-task
# ----------------------------------------------------------------------------


def verify_plan(console: Console, models: dict[str, str], settings: Settings) -> list:
    """Probe every distinct model in ``models`` and print the verdicts.

    Returns the failing ``(roles, Check)`` pairs. A key being valid does not
    mean a model is callable — wrong region, not opted in, not on your plan,
    retired, misspelt — and the whole point of doing this here is that the
    alternative is finding out four tool calls into a real task.
    """
    from loom.core import preflight

    console.print()
    render.rule(console, "verifying")
    render.note(console, "one tiny prompt per distinct model — a few tokens each")

    table = render.table(
        console,
        ("", {"no_wrap": True}),
        ("roles", {"no_wrap": True}),
        ("", {"justify": "center", "no_wrap": True}),
        ("model", {"no_wrap": True, "overflow": "ellipsis"}),
        ("", {"overflow": "fold"}),
    )
    g = ink(console)
    failures: list = []
    with render.Working(console, "probing models"):
        results = list(preflight.check_plan(models, settings.models))
    for roles, check in results:
        if not check.ok:
            failures.append((roles, check))
        mark, style = (g.ok, "loom.good") if check.ok else (
            (g.pending, "loom.warn") if check.state in ("not-pulled", "offline") else (g.fail, "loom.bad")
        )
        label = ", ".join(sorted(roles))
        table.add_row(
            Text(mark, style=style),
            Text(label if len(label) <= 28 else f"{sorted(roles)[0]} +{len(roles) - 1}", style="loom.text"),
            render.where(console, settings.models.is_local(check.model)),
            Text(check.model, style="loom.local" if settings.models.is_local(check.model) else "loom.cloud"),
            Text(check.detail, style="loom.muted" if check.ok else style),
        )
    console.print(table)
    for _, check in failures:
        if check.hint:
            render.note(console, f"[loom.text]{check.model}[/loom.text] — {check.hint}", kind="tip")
    return failures


def offer_pulls(console: Console, failures: list, config: "cfg.LoomConfig") -> bool:
    """Offer to download any local model that simply isn't there yet.

    Returns True if anything was pulled, so the caller can re-verify.
    """
    tags = sorted({c.model for _, c in failures if c.state == "not-pulled"})
    if not tags:
        return False
    from loom.core.model_router import resolve

    pulled = False
    for model_string in tags:
        tag = resolve(model_string).name
        if not render.confirm(console, f"  download `{tag}` now?"):
            continue
        if ollama_mod.pull(tag, config.ollama_endpoint, console) == 0:
            render.note(console, f"{tag} ready — free from here on", kind="good")
            pulled = True
        else:
            render.note(console, f"pull failed for {tag}", kind="warn")
    return pulled


def _resolve_shadowing(
    console: Console,
    models: dict[str, str],
    settings: Settings,
    *,
    root: str | Path,
    scope: str,
    env: dict[str, str],
) -> tuple[Settings, str]:
    """Catch a save that another settings layer overrides, and offer to fix it.

    Saving to ``user`` while this project has its own ``.loom/settings.json``
    writes a file that is then completely ignored — the wizard used to report
    "saved — reload complete" and the very next turn would run on the old
    models. Whatever we just wrote has to be what Loom actually resolves, or
    the user has to be told plainly that it isn't.
    """
    shadowed = shadowed_roles(models, settings)
    if not shadowed:
        return settings, scope

    console.print()
    render.rule(console, "not in effect")
    winner = settings_mod.project_settings_paths(root)[0]
    render.note(
        console,
        f"{len(shadowed)} role(s) were saved to the [loom.warp]{scope}[/loom.warp] layer but are "
        f"overridden by [loom.warp]{winner}[/loom.warp], which this project loads last",
        kind="warn",
    )
    table = render.table(
        console,
        ("role", {"no_wrap": True}),
        ("you chose", {"no_wrap": True, "overflow": "ellipsis"}),
        ("actually used", {"overflow": "ellipsis"}),
    )
    for role, (wanted, actual) in sorted(shadowed.items()):
        table.add_row(
            Text(role, style="loom.text"),
            Text(wanted, style="loom.muted"),
            Text(actual or "(unset)", style="loom.warn"),
        )
    console.print(table)
    console.print()
    render.choices(
        console,
        [
            ("1", "save to this project instead", str(winner)),
            ("2", "leave it", "the values above stay in force"),
        ],
    )
    if Prompt.ask("  choice", choices=["1", "2"], default="1", console=console) != "1":
        render.note(console, "left as-is — /setup or /model <role> to change it later", kind="warn")
        return settings, scope

    settings = apply_plan(root, "project", models, env)
    still = shadowed_roles(models, settings)
    if still:
        # settings.local.json is the only layer after this one.
        render.note(
            console,
            f"still overridden — check {settings_mod.project_settings_paths(root)[1]}",
            kind="bad",
        )
    else:
        render.note(console, f"saved to {winner} — now in effect", kind="good")
    return settings, "project"


def _verify_and_repair(
    console: Console,
    models: dict[str, str],
    settings: Settings,
    *,
    root: str | Path,
    scope: str,
    known_env: dict[str, str],
    hw: rec.Hardware,
    rounds: int = 2,
) -> tuple[dict[str, str], Settings]:
    """Probe the saved plan, then offer to fix whatever didn't answer.

    Runs against the settings that were actually written — env block included —
    so what gets verified is exactly what a real turn will use. Bounded rounds:
    a provider that is simply down should not trap anyone in a loop.
    """
    for _ in range(rounds):
        settings.apply_env()  # the key the wizard just collected has to be live
        # Probe what Loom will *use*, not what was chosen. Those differ whenever
        # another settings layer shadows the write, and verifying the choice
        # would happily green-light a config that never runs.
        live = {role: effective_model(role, settings) or wanted for role, wanted in models.items()}
        failures = verify_plan(console, live, settings)
        if not failures:
            render.note(console, "every model answered", kind="good")
            return models, settings

        if offer_pulls(console, failures, settings.models):
            continue  # something was downloaded; the picture has changed

        # A local role whose only problem is that Ollama isn't running has
        # nothing to re-pick — offering a model list there is busywork that
        # can't fix anything. Say what's wrong and move on.
        if any(check.state == "offline" for _, check in failures):
            render.note(
                console,
                "start the Ollama daemon and run [loom.warp]/doctor probe[/loom.warp] — "
                "the local roles are configured, they just have nowhere to run yet",
                kind="warn",
            )

        repairable = sorted({role for roles, check in failures if check.state != "offline" for role in roles})
        if not repairable:
            return models, settings

        console.print()
        render.choices(
            console,
            [
                ("1", f"pick another model for {len(repairable)} role(s)", ", ".join(repairable)[:60]),
                ("2", "keep it anyway", "these roles will fail at run time"),
            ],
        )
        choice = Prompt.ask("  choice", choices=["1", "2"], default="1", console=console)
        if choice != "1":
            render.note(console, "saved as-is — /setup or /model <role> to change it later", kind="warn")
            return models, settings

        state_of = {role: check.state for roles, check in failures for role in roles}
        for role in repairable:
            models[role] = _repick_role(
                console,
                role,
                models.get(role, ""),
                hw=hw,
                state=state_of.get(role, "error"),
                known_env=known_env,
                settings=settings,
            )
        settings = apply_plan(root, scope, models, known_env)

    render.note(console, "still unverified — /doctor probe re-checks any time", kind="warn")
    return models, settings


def _repick_role(
    console: Console,
    role: str,
    current: str,
    *,
    hw: rec.Hardware,
    state: str,
    known_env: dict[str, str],
    settings: Settings,
) -> str:
    """Replace one role's model, staying where it already lives.

    Deliberately does *not* re-ask local-vs-cloud. The role's placement was
    decided by the setup shape, and a model that got a 403 is a reason to try a
    different model — not a reason to re-litigate the architecture. Advanced
    mode and ``/model <role>`` are where you move a role between the two.
    """
    console.print()
    render.rule(console, role)
    if settings.models.is_local(current) or not current:
        render.note(console, "picking another local model", kind="tip")
        return prov.get("ollama").model_string(prompt_local_model(console, hw, settings.models))

    provider = _provider_of(current)
    if provider is None:  # unrecognised prefix — fall back to the full question
        model_string, env = _configure_one_role(
            console, role, hw, known_env, settings.models, dict(settings.env)
        )
        known_env.update(env)
        return model_string

    if state == "auth":
        # The key itself was refused, so a different model on the same key
        # would fail identically. Collect the credential again first.
        known_env.update(prompt_credentials(console, provider, known_env, dict(settings.env)))
    else:
        render.note(console, f"picking another {provider.label} model", kind="tip")
    tier = _ROLE_TIER.get(role, "main")
    model_id = prompt_cloud_model(console, provider, tier, {**known_env, **settings.env})
    return provider.model_string(model_id)


def _provider_of(model_string: str) -> "prov.ProviderInfo | None":
    """The provider a configured model string belongs to, by its prefix."""
    from loom.core.model_router import resolve

    prefix = resolve(model_string).raw.split(":", 1)[0].split("/", 1)[0]
    return next((p for p in prov.PROVIDERS if p.prefix == prefix), None)


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
    console.print()
    render.rule(console, "current setup")
    console.print(
        render.fleet_table(
            console,
            [(role, current[role], models.is_local(current[role]), "") for role in ALL_ROLES if current.get(role)],
            header=False,
        )
    )

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
            "[loom.muted]credentials on file: " + ", ".join(creds) + " — you'll be asked before any are overwritten[/loom.muted]"
        )


def run(
    console: Console,
    *,
    root: str | Path = ".",
    roles: tuple[str, ...] = ALL_ROLES,
    scope: str | None = None,
    verify: bool = True,
) -> Settings:
    """Run the full wizard and return the reloaded, merged Settings.

    ``scope`` skips the "user vs project" prompt when given ("user" | "project").
    ``verify=False`` skips the closing preflight, which makes real (billed)
    model calls — pass it for tests and any unattended run.
    """
    from loom.ui import banner as banner_mod

    console.print()
    console.print(
        render.card(
            console,
            render.stack(
                banner_mod.wordmark(console),
                Text(),
                Text("Let's staff your fleet.", style="loom.bright"),
                Text(
                    "quick picks one local model and one cloud provider; "
                    "advanced sets every role by hand.",
                    style="loom.muted",
                ),
            ),
            title="setup",
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
    mode = Prompt.ask("  quick setup or advanced?", choices=["quick", "advanced"], default="quick", console=console)

    known_env: dict[str, str] = {}
    if mode == "quick":
        console.print()
        render.rule(console, "local models")
        # Derived, never spelled out: a hardcoded list here silently went stale
        # the moment `reviewer` moved from cloud to local, and then the wizard
        # was telling people the opposite of what it was about to do.
        render.note(console, ", ".join(_DEFAULT_LOCAL_ROLES))
        local_tag = prompt_local_model(console, hw, models_config)
        console.print()
        render.rule(console, "cloud provider")
        render.note(console, ", ".join(_DEFAULT_CLOUD_ROLES))
        use_cloud = render.confirm(console, "  use a cloud provider for these roles?", on_eof=True)
        cloud_provider = None
        tier_models: dict[str, str] | None = None
        if use_cloud:
            console.print()
            cloud_provider = prompt_provider(
                console,
                prov.cloud_providers(),
                known_env=existing_env,
                current=_provider_of(models_config.orchestrator),
            )
            known_env = prompt_credentials(console, cloud_provider, known_env, existing_env)
            if render.confirm(
                console,
                f"  pick specific {cloud_provider.label} models per role (else use recommended defaults)?",
                default=False,
            ):
                # Derived from the roles that actually go to the cloud, so this
                # can't drift into asking about a tier nothing uses.
                tier_models = {}
                tier_roles: dict[str, list[str]] = {}
                for cloud_role in _DEFAULT_CLOUD_ROLES:
                    tier_roles.setdefault(_ROLE_TIER.get(cloud_role, "main"), []).append(cloud_role)
                for tier, tier_role_names in tier_roles.items():
                    console.print()
                    render.note(console, " + ".join(tier_role_names))
                    tier_models[tier] = prompt_cloud_model(console, cloud_provider, tier, known_env)
        models = default_role_plan(hw, local_tag, cloud_provider, tier_models)
    else:
        models = {}
        for role in roles:
            model_string, env = _configure_one_role(console, role, hw, known_env, models_config, existing_env)
            models[role] = model_string
            known_env.update(env)

    console.print(f"\n[loom.muted]detected: {rec.hardware_summary(hw)}[/loom.muted]")
    console.print(f"[loom.muted]{rec.CLOUD_RECOMMENDATION}[/loom.muted]")

    if scope not in ("user", "project"):
        scope = Prompt.ask(
            "\n  save to user settings (~/.loom) or this project (.loom)?",
            choices=["user", "project"],
            default="user",
            console=console,
        )
    settings = apply_plan(root, scope, models, known_env)
    settings, scope = _resolve_shadowing(console, models, settings, root=root, scope=scope, env=known_env)
    if verify:
        models, settings = _verify_and_repair(
            console, models, settings, root=root, scope=scope, known_env=known_env, hw=hw
        )

    console.print()
    render.rule(console, "configured")
    rows = []
    for role in ALL_ROLES:
        actual = effective_model(role, settings)
        if not actual:
            rows.append((role, "(unchanged)", True, ""))
            continue
        chose = models.get(role)
        # Anything still overridden is called out inline rather than reported
        # as if it were what the user picked.
        note = f"overrides your {chose}" if chose and chose != actual else ""
        rows.append((role, actual, settings.models.is_local(actual), note))
    console.print(render.fleet_table(console, rows, header=False))
    console.print(f"[loom.bright]saved to {scope} settings.json — reload complete.[/loom.bright]")
    return settings


def maybe_setup_playwright(console: Console, settings: Settings) -> None:
    """After the model wizard, finish Playwright MCP setup if it's needed.

    The bundled ``playwright`` MCP server ships enabled, but `npx
    @playwright/mcp` alone doesn't download the browser binaries it drives —
    without them the ``tester`` subagent connects yet fails at the first
    ``browser_*`` call. Kept out of :func:`run` itself so it doesn't disturb
    that function's scripted prompt sequence; called separately by the CLI
    and REPL ``setup`` commands once role assignment is done.
    """
    server = settings.mcp_servers.get("playwright")
    if server is None or not server.enabled:
        return

    from loom.core import playwright_setup

    st = playwright_setup.status()
    if not st.npx_available:
        console.print(f"\n[loom.muted]{playwright_setup.INSTALL_HINT}[/loom.muted]")
        return
    if st.browsers_installed:
        return

    console.print()
    render.rule(console, "playwright browser")
    render.note(console, "powers the `tester` subagent")
    if render.confirm(console, "  install it now so end-to-end browser testing works?"):
        if playwright_setup.install_browsers(console) != 0:
            console.print("[loom.warn]install failed — retry any time with `loom playwright install`.[/loom.warn]")
    else:
        console.print("[loom.muted]skipped — run `loom playwright install` any time.[/loom.muted]")
