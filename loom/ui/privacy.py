"""The privacy step — the one question in setup that is about the user, not
the fleet.

Two surfaces, and they answer different questions:

* :func:`run` is the wizard step. "In general, what may Loom send?" Asked once,
  during quick *and* advanced setup, and again whenever ``/privacy`` is used.
* :func:`maybe_ask_project` is the doorway question. "May it send that from
  *here*?" Asked the first time Loom starts in a given repo, because agreeing
  to share crash reports from a hobby project is not agreeing to share them
  from work.

The wording is deliberate. Every mode states what it sends *and* what it never
sends, because a consent screen that only lists benefits is not consent. Mode 3
says out loud that it uploads source code, since that is what full LLM tracing
means and burying it would make the opt-in worthless.
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text

from loom.core import telemetry as tel
from loom.core import telemetry_setup as tsetup
from loom.core.telemetry import Consent
from loom.ui import render
from loom.ui.render import ink


# ----------------------------------------------------------------------------
# Presentation
# ----------------------------------------------------------------------------


def _mode_table(console: Console, current: str) -> None:
    """The three modes side by side, with the current one marked."""
    g = ink(console)
    table = render.table(
        console,
        ("#", {"justify": "right", "no_wrap": True}),
        ("mode", {"no_wrap": True}),
        ("", {"no_wrap": True}),
        ("what leaves this machine", {"overflow": "fold"}),
    )
    for i, mode in enumerate(tel.MODES, 1):
        is_current = mode.id == current
        table.add_row(
            Text(str(i), style="loom.warp" if is_current else "loom.muted"),
            Text(mode.label, style="loom.bright" if is_current else "loom.text"),
            Text("in use" if is_current else "", style="loom.warp"),
            Text(mode.blurb, style="loom.muted"),
        )
    console.print(table)
    console.print()
    # The detail that decides it for most people is what each mode *won't*
    # send, and that does not fit in a table cell.
    for i, mode in enumerate(tel.MODES, 1):
        line = Text("  ")
        line.append(f"{i}", style="loom.warp.b")
        line.append(f"  {mode.label}", style="loom.text")
        console.print(line)
        for item in mode.sends:
            console.print(Text(f"      {g.plus} {item}", style="loom.good"))
        for item in mode.never:
            console.print(Text(f"      {g.minus} never: {item}", style="loom.muted"))
    console.print()


def prompt_mode(console: Console, current: str = tel.DEFAULT_MODE) -> str:
    """Ask which privacy mode to run in. Returns a mode id."""
    console.print()
    render.rule(console, "privacy")
    render.note(
        console,
        "Loom is off by default — nothing is sent anywhere unless you turn it on here.",
    )
    console.print()
    _mode_table(console, current)
    default = str(tel.MODE_IDS.index(current) + 1) if current in tel.MODE_IDS else "1"
    choice = Prompt.ask(
        "  how much may Loom share?",
        choices=[str(i) for i in range(1, len(tel.MODES) + 1)],
        default=default,
        console=console,
    )
    try:
        return tel.MODES[int(choice) - 1].id
    except (ValueError, IndexError):
        return tel.DEFAULT_MODE


# ----------------------------------------------------------------------------
# Credentials
# ----------------------------------------------------------------------------


def prompt_sentry_dsn(console: Console, current: str = "") -> str:
    """Get a Sentry DSN, using the `sentry` CLI when it can help.

    A DSN is a write-only ingest URL, not a secret, but it is also a 60-char
    string nobody remembers — so if the CLI is installed and logged in, this
    offers to fetch one (or make a project) rather than sending someone to a
    dashboard mid-setup.
    """
    console.print()
    render.note(console, "crash reports need a Sentry project to go to", kind="tip")
    if current:
        if render.confirm(console, f"  keep the DSN already configured ({_mask_dsn(current)})?", on_eof=True):
            return current

    dsn = _dsn_via_cli(console)
    if dsn:
        return dsn

    console.print(f"[loom.muted]find it in Sentry under Settings {ink(console).arrow} Client Keys {ink(console).dot} {tsetup.SENTRY_DOCS}[/loom.muted]")
    return (Prompt.ask("  Sentry DSN (blank to skip)", default="", console=console) or "").strip()


def _dsn_via_cli(console: Console) -> str:
    """Offer the CLI-assisted path. Returns "" to fall back to typing it in."""
    if not tsetup.sentry_cli():
        return ""
    if not tsetup.sentry_logged_in():
        render.note(console, "the `sentry` CLI is installed but not logged in — `sentry auth login`", kind="warn")
        return ""

    with render.Working(console, "asking the sentry CLI what projects you have"):
        orgs = tsetup.sentry_orgs()
        projects = tsetup.sentry_projects()
    if not orgs:
        return ""
    org = orgs[0]["slug"]
    if len(orgs) > 1:
        table = render.table(console, ("#", {"justify": "right", "no_wrap": True}), ("organization", {}))
        for i, o in enumerate(orgs, 1):
            table.add_row(Text(str(i), style="loom.warp"), Text(o["name"], style="loom.text"))
        console.print(table)
        pick = Prompt.ask("  organization", choices=[str(i) for i in range(1, len(orgs) + 1)], default="1", console=console)
        org = orgs[int(pick) - 1]["slug"]
        projects = [p for p in projects if not p.get("org") or p["org"] == org]

    options = [p["slug"] for p in projects]
    table = render.table(
        console,
        ("#", {"justify": "right", "no_wrap": True}),
        ("project", {"no_wrap": True}),
        ("", {"overflow": "fold"}),
    )
    for i, slug in enumerate(options, 1):
        table.add_row(Text(str(i), style="loom.warp"), Text(slug, style="loom.text"), Text("", style="loom.muted"))
    new_index = len(options) + 1
    table.add_row(
        Text(str(new_index), style="loom.warp"),
        Text("create one", style="loom.text"),
        Text(f"a new python project in {org}", style="loom.muted"),
    )
    console.print(table)
    choice = Prompt.ask("  number, or blank to paste a DSN instead", default=str(new_index), console=console).strip()
    if not choice.isdigit():
        return ""
    index = int(choice)

    if index == new_index:
        name = (Prompt.ask("  new project name", default="loom", console=console) or "loom").strip()
        # The only thing in this whole flow that changes someone's Sentry
        # account, so it asks in as many words.
        if not render.confirm(console, f"  create project [loom.warp]{org}/{name}[/loom.warp] in Sentry now?"):
            return ""
        with render.Working(console, f"creating {org}/{name}"):
            slug = tsetup.sentry_create_project(org, name)
        if not slug:
            render.note(console, "could not create the project — paste a DSN instead", kind="warn")
            return ""
        render.note(console, f"created {org}/{slug}", kind="good")
    elif 1 <= index <= len(options):
        slug = options[index - 1]
    else:
        return ""

    with render.Working(console, "fetching the DSN"):
        dsn = tsetup.sentry_dsn(org, slug)
    if not dsn:
        render.note(console, "the CLI didn't return a DSN — paste one instead", kind="warn")
        return ""
    render.note(console, f"DSN for {org}/{slug} ({_mask_dsn(dsn)})", kind="good")
    return dsn


def prompt_langfuse(console: Console, consent: Consent) -> tuple[str, str, str]:
    """Collect and verify a Langfuse key pair. Returns ``(public, secret, host)``."""
    console.print()
    render.note(console, "full tracing needs a Langfuse project to write traces to", kind="tip")
    console.print(f"[loom.muted]project settings {ink(console).arrow} API keys {ink(console).dot} {tsetup.LANGFUSE_DOCS}[/loom.muted]")

    host = Prompt.ask(
        "  Langfuse host",
        default=consent.langfuse_host or tsetup.LANGFUSE_DEFAULT_HOST,
        console=console,
    ).strip() or tsetup.LANGFUSE_DEFAULT_HOST

    public = consent.langfuse_public_key
    if public and render.confirm(console, f"  keep the public key on file ({_mask(public)})?", on_eof=True):
        secret = consent.langfuse_secret_key
    else:
        public = (Prompt.ask("  LANGFUSE_PUBLIC_KEY", default="", console=console) or "").strip()
        secret = (Prompt.ask("  LANGFUSE_SECRET_KEY", password=True, default="", console=console) or "").strip()
    if not (public and secret):
        render.note(console, "no keys — traces stay on this machine until you add them", kind="warn")
        return "", "", host

    with render.Working(console, "verifying the key pair"):
        ok, detail = tsetup.langfuse_verify(public, secret, host)
    # Unverified keys are still saved: the usual cause is an offline laptop,
    # and throwing away a correctly-typed key over that would be worse than a
    # warning the user can act on later with /privacy.
    render.note(console, detail, kind="good" if ok else "warn")
    return public, secret, host


def _mask(value: str) -> str:
    return "…" + value[-4:] if len(value) > 8 else "(set)"


def _mask_dsn(dsn: str) -> str:
    """A DSN is long and mostly noise; show the ingest host, which is the part
    that tells you *whose* Sentry it is."""
    try:
        return dsn.split("@", 1)[1].split("/", 1)[0]
    except (IndexError, AttributeError):
        return _mask(dsn)


# ----------------------------------------------------------------------------
# The wizard step
# ----------------------------------------------------------------------------


def run(
    console: Console,
    root: str | Path = ".",
    *,
    consent: Consent | None = None,
    force_credentials: bool = False,
) -> Consent:
    """The privacy step of setup. Always records a decision, including "none"
    — a stored ``none`` is an answer, and stops Loom asking again.

    ``force_credentials`` re-prompts for a Sentry DSN / Langfuse keys even when
    a bundled default exists — `/privacy setup` uses it so someone who wants
    crashes to go to *their* project can replace the Loom default, while the
    first-run wizard skips the question entirely (point 2/3 just work)."""
    consent = consent if consent is not None else tel.load()
    mode = prompt_mode(console, consent.mode)
    consent.mode = mode
    consent.decided = True

    if mode in ("errors", "full"):
        # A bundled DSN means point 2 just works — only ask for one when there
        # is nothing to fall back on, or when the user explicitly re-ran
        # /privacy setup to point at their own project.
        if force_credentials or not (tel.DEFAULT_SENTRY_DSN or consent.sentry_dsn):
            consent.sentry_dsn = prompt_sentry_dsn(console, consent.sentry_dsn) or consent.sentry_dsn
    if mode == "full":
        console.print()
        console.print(
            render.card(
                console,
                render.stack(
                    Text("Full tracing uploads your prompts and your code.", style="loom.warn"),
                    Text(
                        "Every file a subagent reads and every patch it writes becomes part of "
                        "a trace. That is what makes the data useful for training a local "
                        "orchestrator — and what makes it the wrong choice for anything you "
                        "are not free to share.",
                        style="loom.muted",
                    ),
                ),
                title="before you pick this",
                style="loom.warn",
            )
        )
        if not render.confirm(console, "  send full traces?", default=False):
            consent.mode = "errors" if (consent.sentry_dsn or tel.DEFAULT_SENTRY_DSN) else "none"
            render.note(console, f"kept at [loom.warp]{tel.mode_info(consent.mode).label}[/loom.warp]")
        else:
            # Skip the credential prompt only when defaults already cover it
            # and the user didn't ask to override; otherwise collect/verify.
            if force_credentials or not (
                tel.DEFAULT_LANGFUSE_PUBLIC_KEY and tel.DEFAULT_LANGFUSE_SECRET_KEY
            ):
                public, secret, host = prompt_langfuse(console, consent)
                consent.langfuse_public_key = public
                consent.langfuse_secret_key = secret
                consent.langfuse_host = host
            if not (
                (consent.langfuse_public_key and consent.langfuse_secret_key)
                or (tel.DEFAULT_LANGFUSE_PUBLIC_KEY and tel.DEFAULT_LANGFUSE_SECRET_KEY)
            ):
                consent.mode = "errors" if (consent.sentry_dsn or tel.DEFAULT_SENTRY_DSN) else "none"

    # The project Loom is being set up in has plainly consented — recording it
    # here means setup doesn't hand straight over to a second consent prompt
    # about the very directory the user just ran setup in.
    if consent.mode != "none":
        consent.projects[tel.project_key(root)] = {"share": True, "at": int(time.time())}

    tel.save(consent)
    _report(console, consent)
    return consent


def _report(console: Console, consent: Consent) -> None:
    mode = tel.mode_info(consent.mode)
    console.print()
    if consent.mode == "none":
        render.note(console, "privacy: [loom.warp]none[/loom.warp] — nothing leaves this machine", kind="good")
    else:
        render.note(
            console,
            f"privacy: [loom.warp]{mode.label}[/loom.warp] — {mode.blurb}",
            kind="good",
        )
        # Say where the data goes. A bundled default is what makes "all users"
        # work with no account; the user should still know it's Loom's project,
        # not theirs, and how to swap in their own.
        if consent.mode in ("errors", "full") and not consent.sentry_dsn and tel.DEFAULT_SENTRY_DSN:
            render.note(console, "crashes go to Loom's project by default — [loom.warp]/privacy setup[/loom.warp] to use your own", kind="tip")
        if (
            consent.mode == "full"
            and not (consent.langfuse_public_key and consent.langfuse_secret_key)
            and tel.DEFAULT_LANGFUSE_PUBLIC_KEY and tel.DEFAULT_LANGFUSE_SECRET_KEY
        ):
            render.note(console, "traces go to Loom's Langfuse project by default — [loom.warp]/privacy setup[/loom.warp] to use your own", kind="tip")
    render.note(console, f"stored in {tel.store_path()} {ink(console).dot} change it any time with [loom.warp]/privacy[/loom.warp]")


# ----------------------------------------------------------------------------
# The per-project gate
# ----------------------------------------------------------------------------


def ask_project(console: Console, root: str | Path = ".", *, consent: Consent | None = None) -> bool:
    """Ask whether this project may share, record the answer, return it."""
    consent = consent if consent is not None else tel.load()
    mode = tel.mode_info(tel.global_mode(consent))
    key = tel.project_key(root)

    console.print()
    render.rule(console, "new project")
    render.note(
        console,
        f"privacy is set to [loom.warp]{mode.label}[/loom.warp] — {mode.blurb}",
    )
    console.print(render.kv([("project", Text(key, style="loom.text"))]))
    # Declining is the safe answer, so it is the default: someone who hits
    # Enter to get past this has not agreed to send their employer's code.
    share = render.confirm(console, "  share from this project?", default=False, on_eof=False)
    tel.record_project(root, share, consent)
    render.note(
        console,
        "sharing enabled here" if share else "nothing will be sent from this project",
        kind="good",
    )
    return share


def maybe_ask_project(console: Console, root: str | Path = ".") -> None:
    """Ask the doorway question, but only when there is one to ask.

    Silent when the mode is ``none``, when this project has already answered,
    and when the user has never set a mode at all — that last one belongs to
    setup, and asking it here would be the second half of a question nobody
    heard the first half of.
    """
    try:
        if not tel.needs_project_decision(root):
            return
        ask_project(console, root)
    except (KeyboardInterrupt, EOFError):
        # No answer is not consent. Leave it unrecorded so it is asked again
        # rather than resolved by silence in either direction.
        render.note(console, "skipped — nothing shared from here until you answer", kind="muted")
    except Exception:
        pass  # a consent prompt must never be the reason Loom fails to start


# ----------------------------------------------------------------------------
# /privacy
# ----------------------------------------------------------------------------


def describe(console: Console, root: str | Path = ".") -> None:
    """Show the current privacy state: the mode, this project's answer, and
    what is actually wired up."""
    consent = tel.load()
    configured = tel.global_mode(consent)
    effective = tel.active_mode(root, consent)
    mode = tel.mode_info(configured)
    g = ink(console)

    console.print()
    render.rule(console, "privacy")
    rows = [
        ("mode", Text(f"{mode.label} {g.dot} {mode.blurb}", style="loom.bright")),
    ]
    if configured != tel.DEFAULT_MODE:
        share = tel.project_share(root, consent)
        rows.append(
            (
                "here",
                Text(
                    "sharing" if share is True else ("not sharing" if share is False else "not asked yet"),
                    style="loom.good" if share is True else "loom.muted",
                ),
            )
        )
        rows.append(("project", Text(tel.project_key(root), style="loom.muted")))
    if effective != configured:
        rows.append(("in effect", Text(tel.mode_info(effective).label, style="loom.warn")))
    if configured in ("errors", "full"):
        dsn = consent.sentry_dsn or tel.DEFAULT_SENTRY_DSN
        if dsn:
            label = "Loom's project (default)" if not consent.sentry_dsn else _mask_dsn(dsn)
            rows.append(("sentry", Text(label, style="loom.muted")))
        else:
            rows.append(("sentry", Text("no DSN — crashes go nowhere", style="loom.warn")))
    if configured == "full":
        own = consent.langfuse_public_key and consent.langfuse_secret_key
        baked = tel.DEFAULT_LANGFUSE_PUBLIC_KEY and tel.DEFAULT_LANGFUSE_SECRET_KEY
        if own:
            rows.append(("langfuse", Text(consent.langfuse_host, style="loom.muted")))
        elif baked:
            rows.append(("langfuse", Text("Loom's project (default)", style="loom.muted")))
        else:
            rows.append(("langfuse", Text("no keys — traces go nowhere", style="loom.warn")))
    rows.append(("stored", Text(str(tel.store_path()), style="loom.muted")))
    console.print(render.kv(rows))
    console.print()
    for item in mode.sends:
        console.print(Text(f"  {g.plus} {item}", style="loom.good"))
    for item in mode.never:
        console.print(Text(f"  {g.minus} never: {item}", style="loom.muted"))
    console.print()
    render.note(console, "[loom.warp]/privacy set <none|errors|full>[/loom.warp] to change it, [loom.warp]/privacy here[/loom.warp] for this project only", kind="tip")


def set_mode(console: Console, mode_id: str, root: str | Path = ".") -> bool:
    """Non-interactive mode change, for ``/privacy set x`` and ``loom privacy set x``."""
    if mode_id not in tel.MODE_IDS:
        render.note(console, f"unknown mode {mode_id!r} — one of: {', '.join(tel.MODE_IDS)}", kind="bad")
        return False
    consent = tel.load()
    consent.mode = mode_id
    consent.decided = True
    tel.save(consent)
    info = tel.mode_info(mode_id)
    render.note(console, f"privacy set to [loom.warp]{info.label}[/loom.warp] — {info.blurb}", kind="good")
    missing = [k for k in info.needs if not _have_credential(k, consent)]
    if missing:
        render.note(
            console,
            f"still needs {', '.join(missing)} — run [loom.warp]/privacy setup[/loom.warp] to fill it in",
            kind="warn",
        )
    return True


def doctor_row(root: str | Path = ".") -> tuple[bool | None, str, str]:
    """One ``(ok, label, detail)`` line for ``/doctor`` and ``loom doctor``.

    Read-only, like the rest of doctor. A configured mode that cannot work —
    consented to ``errors`` but no DSN anywhere — is a warning, because the
    user made a choice the machine isn't honouring; a project that simply
    declined is not a problem and gets a quiet ok."""
    consent = tel.load()
    mode = tel.global_mode(consent)
    info = tel.mode_info(mode)
    if mode == tel.DEFAULT_MODE:
        return True, "privacy", "none — nothing leaves this machine"
    missing = [k for k in info.needs if not _have_credential(k, consent)]
    if missing:
        return None, "privacy", f"{info.label}, but missing {', '.join(missing)} — /privacy setup"
    if tel.active_mode(root, consent) == tel.DEFAULT_MODE:
        share = tel.project_share(root, consent)
        why = "declined" if share is False else "not asked yet"
        return True, "privacy", f"{info.label} globally; this project {why} — nothing sent from here"
    return True, "privacy", f"{info.label} — sharing from this project"


def _have_credential(key: str, consent: Consent) -> bool:
    """Whether a credential is resolvable from any layer: the process env, the
    user's own consent record, or the bundle baked into this build. A mode that
    the user picked but no layer can satisfy is the one case /privacy and
    /doctor warn about."""
    return tel._baked_credential(key, consent)
