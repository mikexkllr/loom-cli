"""The setup health check — one implementation for ``loom doctor`` and ``/doctor``.

There used to be two. The shell command was rewritten to judge a setup by what
its config actually routes to; the REPL command kept the original — a fixed
``ANTHROPIC_API_KEY`` check and an unconditional red mark for a stopped Ollama
daemon — so the same machine got a clean bill from ``loom doctor`` and a broken
one from ``/doctor`` a second later. A check that exists twice drifts; this is
the one copy both surfaces print.

Read-only, apart from :func:`probe`, which makes one small real call per
distinct model because that is the only way to learn whether a key that is
*present* is also *accepted for that model*.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.text import Text

from loom.core.config import LoomConfig
from loom.core.settings import Settings
from loom.ui import render
from loom.ui.render import ink


def routed_models(config: LoomConfig) -> list[str]:
    """Every model string this config can end up calling."""
    from loom.subagents import SPECS, model_for

    models = [
        config.orchestrator,
        config.advisor,
        *(model_for(config, name) for name in SPECS),
        *config.subagents.values(),
        config.escalation_model,
        config.cloud_fallback,
    ]
    return list(dict.fromkeys(m for m in models if m))


def probe_plan(config: LoomConfig) -> dict[str, str]:
    """role -> model string for everything worth a round-trip.

    The cloud fallback is in it whenever a local role exists: on a machine with
    no Ollama it is the model every one of those roles really runs on, and it
    was the one string the probe never called.
    """
    plan = {
        "orchestrator": config.orchestrator,
        "advisor": config.advisor,
        "escalation": config.escalation_model,
        **config.subagents,
    }
    if any(config.is_local(m) for m in plan.values() if m):
        plan["cloud_fallback"] = config.cloud_fallback
    return {role: model for role, model in plan.items() if model}


def _row(console: Console, ok: bool | None, label: str, detail: str):
    g = ink(console)
    mark, style = (
        (g.ok, "loom.good") if ok else ((g.pending, "loom.warn") if ok is None else (g.fail, "loom.bad"))
    )
    return (
        Text(f"{mark} ", style=style) + Text(label, style="loom.text"),
        Text(detail, style="loom.muted" if ok else style),
    )


def checkup(console: Console, settings: Settings, root: str | Path = ".") -> None:
    """Print the doctor table for ``settings``."""
    from loom.core import model_router, ollama, playwright_setup, providers
    from loom.core.mcp import mcp_status
    from loom.ui import privacy as privacy_mod

    config = settings.models

    def effective_env(key: str) -> str | None:
        """Value ``key`` would resolve to without mutating the real process
        env (doctor is read-only): real env wins, else settings.json's env."""
        return os.environ.get(key) or settings.env.get(key)

    def row(ok, label, detail):
        return _row(console, ok, label, detail)

    # What the config actually routes to. Doctor reports on the providers in
    # use, not on a fixed list of infrastructure — a red ✘ for an Ollama daemon
    # or an Anthropic key that no role touches reads as "your setup is broken"
    # when nothing is.
    routed = routed_models(config)
    has_local_roles = any(model_router.resolve(m).is_local for m in routed)

    lines = [row(sys.version_info >= (3, 11), "python", sys.version.split()[0])]
    st = ollama.status(config)
    if not has_local_roles and not st.running:
        # All-cloud config: the daemon being down is a non-event.
        lines.append(row(None, "ollama", "not running — no local roles configured"))
    elif st.running:
        detail = f"running @ {st.endpoint}" + ("" if st.installed else " (remote — no local binary)")
        lines.append(row(True, "ollama", detail))
        have = set(st.models)
        missing = [m for m in ollama.required_local_models(config) if not ollama.is_served(m, have)]
        lines.append(
            row(not missing, "local models", ", ".join(missing) + " missing" if missing else "all present")
        )
        if missing:
            # A missing tag only costs money if nothing local can cover it.
            from loom.core.local_pool import build_pool, plan_local_roles

            plan = plan_local_roles(config, build_pool(config, st))
            if plan.cloud:
                lines.append(
                    row(
                        None,
                        "cloud fallback",
                        f"{', '.join(sorted(plan.cloud))} will run on {config.cloud_fallback} (billed)",
                    )
                )
            if plan.substituted:
                lines.append(
                    row(
                        None,
                        "local fallback",
                        f"{', '.join(sorted(plan.substituted))} will run on another local model (free)",
                    )
                )
    else:
        lines.append(
            row(
                False,
                "ollama",
                f"not reachable @ {st.endpoint}" + ("" if st.installed else ", binary not installed"),
            )
        )
        lines.append(row(None, "cloud fallback", f"local roles will run on {config.cloud_fallback} (billed)"))

    for provider_id in providers.routed_providers(routed):
        names = providers.credential_keys(provider_id)
        if not names:  # needs no key (e.g. Vertex's ADC, a keyless custom endpoint)
            continue
        found = next((n for n in names if effective_env(n)), None)
        try:
            label = providers.get(provider_id).label
        except KeyError:
            label = provider_id
        lines.append(row(bool(found), label, f"{found} set" if found else f"{names[0]} not set"))

    pw = playwright_setup.status()
    lines.append(
        row(pw.npx_available, "npx", "found" if pw.npx_available else "not found (Playwright MCP needs Node)")
    )
    if pw.npx_available:
        lines.append(
            row(
                pw.browsers_installed,
                "playwright browsers",
                "installed" if pw.browsers_installed else "missing — run `loom playwright install`",
            )
        )
    for r in mcp_status(settings):
        ok = True if r["state"] == "connected" else (None if r["state"] in ("not connected", "disabled") else False)
        lines.append(row(ok, f"mcp:{r['name']}", r["state"]))

    lines.append(row(*privacy_mod.doctor_row(root)))
    console.print()
    render.rule(console, "doctor")
    console.print(render.kv(lines, justify="left"))


def probe(console: Console, settings: Settings) -> int:
    """Call every configured model once and print what answered.

    Returns the number of models that did not. The offline checks prove a key
    is *present*; only a real call proves it is *accepted for that model*.
    Those differ more often than you'd think — wrong region, not opted in, not
    on your plan, retired, misspelt.
    """
    from loom.core import preflight

    g = ink(console)
    models = settings.models
    settings.apply_env()
    console.print()
    render.rule(console, "probe")
    render.note(console, "one tiny prompt per distinct model — a few tokens each")
    table = render.table(
        console,
        ("", {"no_wrap": True}),
        ("roles", {"overflow": "ellipsis"}),
        ("", {"justify": "center", "no_wrap": True}),
        ("model", {"no_wrap": True, "overflow": "ellipsis"}),
        ("", {"overflow": "fold"}),
    )
    failed = []
    with render.Working(console, "probing models"):
        results = list(preflight.check_plan(probe_plan(models), models))
    for roles, check in results:
        if not check.ok:
            failed.append(check)
        mark, style = (g.ok, "loom.good") if check.ok else (
            (g.pending, "loom.warn") if check.state in ("not-pulled", "offline") else (g.fail, "loom.bad")
        )
        is_local = models.is_local(check.model)
        table.add_row(
            Text(mark, style=style),
            Text(", ".join(sorted(roles)), style="loom.text"),
            render.where(console, is_local),
            Text(check.model, style="loom.local" if is_local else "loom.cloud"),
            Text(check.detail, style="loom.muted" if check.ok else style),
        )
    console.print(table)
    console.print()
    for check in failed:
        if check.hint:
            render.note(console, f"[loom.text]{escape(check.model)}[/loom.text] — {escape(check.hint)}", kind="tip")
    return len(failed)
