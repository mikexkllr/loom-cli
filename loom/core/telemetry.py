"""Privacy modes — what, if anything, Loom is allowed to send home.

Three modes, and the default is the private one:

``none``    nothing leaves the machine. No SDK is imported, no socket opened.
``errors``  crashes only, via Sentry: exception type, message and stack trace,
            scrubbed of paths, locals, argv and hostname. No prompts, no code.
``full``    everything ``errors`` sends, plus complete LLM traces via Langfuse:
            prompts, completions, tool calls, delegations, tokens and timings.
            That is the training corpus for a distilled Loom orchestrator, and
            it necessarily contains your source code — which is why it is a
            deliberate, separate opt-in.

Consent lives in ``$LOOM_HOME/telemetry.json`` (default ``~/.loom``), never in
a project's ``.loom/settings.json``: that file is meant to be committed, and
one developer's privacy choice is not a team-wide setting. The same file holds
the Sentry DSN and Langfuse keys, so a secret key can never be swept into a
repo by ``/setup --scope project``.

Two consent gates, both of which must pass before a single byte is sent:

1. the *global* mode above, chosen once during setup;
2. a *per-project* answer, asked the first time Loom runs in a given git repo
   or directory — because "I'll share crash reports" is not the same statement
   as "I'll share crash reports from my employer's monorepo".

Everything here degrades to a no-op rather than raising. Telemetry that can
break the tool it is supposed to be improving is worse than no telemetry.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loom.core import config as cfg

# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Mode:
    """One privacy level, and the plain-English promise attached to it."""

    id: str
    label: str
    blurb: str
    sends: tuple[str, ...]
    never: tuple[str, ...]
    needs: tuple[str, ...] = ()  # credential keys this mode can't work without


MODES: tuple[Mode, ...] = (
    Mode(
        id="none",
        label="none",
        blurb="nothing leaves this machine",
        sends=(),
        never=("crash reports", "prompts", "code", "usage counts"),
    ),
    Mode(
        id="errors",
        label="bug reports",
        blurb="crashes only — the stack trace, scrubbed",
        sends=("exception type + message", "stack trace (no local variables)", "Loom version, OS, python version"),
        never=("your prompts", "your code", "file paths outside Loom itself", "environment variables", "hostname"),
        # No credential the user has to supply: a bundled DSN ships with Loom
        # (and the wizard never asks for one when a default exists). A user who
        # wants their *own* Sentry project supplies it via /privacy setup.
        needs=(),
    ),
    Mode(
        id="full",
        label="full tracing",
        blurb="crashes plus complete LLM traces, to train a local orchestrator",
        sends=(
            "everything a bug report sends",
            "every prompt and completion, orchestrator and subagents",
            "tool calls, delegations, token counts and latencies",
        ),
        never=("anything, if you pick a different mode later — but past traces stay sent",),
        # Conceptual: the bundled DSN + bundled Langfuse keys cover a binary
        # install; a source install without the baked Langfuse secret needs the
        # user's own. `_baked_credential` is what makes the set_mode/doctor
        # "still needs" warning fire only when no layer provides the key.
        needs=("SENTRY_DSN", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"),
    ),
)

MODE_IDS: tuple[str, ...] = tuple(m.id for m in MODES)
DEFAULT_MODE = "none"

# Set LOOM_TELEMETRY=none in CI (or anywhere else) to force the private mode
# regardless of what is on disk. It can only ever tighten, never loosen: a
# stray env var must not be able to switch on tracing nobody consented to.
ENV_OVERRIDE = "LOOM_TELEMETRY"

STORE_VERSION = 1

# ----------------------------------------------------------------------------
# Bundled defaults — credentials that ship with Loom so a mode someone picks in
# setup actually works without them owning an account. Every one of them is
# write-only, and that is the whole design.
#
# A Sentry DSN is safe to publish by construction: it ingests events and cannot
# read them. Langfuse has no such credential — its API has a single auth scheme
# and a project key pair is read *and* write, so a key baked into a public
# binary would let anyone who downloads Loom read every full-tracing user's
# prompts and source code. So Loom does not ship one.
#
# Instead traces go to an ingest proxy (telemetry-proxy/, a Cloudflare Worker)
# that holds the real Langfuse key server-side and forwards. Loom sends it a
# `lct_…` client token in the secret-key position. That token is extractable
# from the binary and is *meant* to be: it grants writes only, and rotating the
# worker's LOOM_CLIENT_TOKEN retires it. A leaked write token means spam; a
# leaked Langfuse secret key means someone reads your users' code.
#
# The token is not in source. The frozen binary bakes it at build time into a
# generated loom/_built.py (see packaging/loom.spec); a source or dev build
# resolves to "" and full mode falls back to the user's own Langfuse keys via
# /privacy setup, which still work because env and the consent record both take
# precedence over these defaults.
# ----------------------------------------------------------------------------

try:  # generated at binary build time; absent in source and dev builds
    from loom._built import LOOM_CLIENT_TOKEN as _BUILT_CLIENT_TOKEN

    _BUILT_CLIENT_TOKEN = _BUILT_CLIENT_TOKEN or ""
except ImportError:
    _BUILT_CLIENT_TOKEN = ""

DEFAULT_SENTRY_DSN = "https://8d27208a70141fd3f278c6b50280322b@o4511155358334976.ingest.de.sentry.io/4511851069374544"
# Cosmetic: the proxy authenticates on the token alone, but the SDK sends basic
# auth and wants a username, and a recognisable one makes proxy logs readable.
DEFAULT_LANGFUSE_PUBLIC_KEY = "pk-lf-loom-ingest"
DEFAULT_LANGFUSE_SECRET_KEY = _BUILT_CLIENT_TOKEN
DEFAULT_LANGFUSE_HOST = "https://loom-telemetry.telemetry-proxy.workers.dev"
# Where a *user's own* key pair goes when they didn't name a host. Never a
# fallback for the bundled credentials — see `langfuse_credentials`.
LANGFUSE_CLOUD_HOST = "https://cloud.langfuse.com"


def _baked_credential(key: str, consent: "Consent | None" = None) -> bool:
    """Whether a credential is available from *any* layer — env, the user's
    consent record, or the bundle baked into this build. Used by /privacy and
    /doctor to warn only when a chosen mode genuinely cannot work."""
    if os.environ.get(key):
        return True
    if consent is not None:
        if key == "SENTRY_DSN" and consent.sentry_dsn:
            return True
        if key == "LANGFUSE_PUBLIC_KEY" and consent.langfuse_public_key:
            return True
        if key == "LANGFUSE_SECRET_KEY" and consent.langfuse_secret_key:
            return True
    if key == "SENTRY_DSN" and DEFAULT_SENTRY_DSN:
        return True
    if key == "LANGFUSE_PUBLIC_KEY" and DEFAULT_LANGFUSE_PUBLIC_KEY:
        return True
    if key == "LANGFUSE_SECRET_KEY" and DEFAULT_LANGFUSE_SECRET_KEY:
        return True
    return False


def mode_info(mode_id: str) -> Mode:
    return next((m for m in MODES if m.id == mode_id), MODES[0])


def _rank(mode_id: str) -> int:
    return MODE_IDS.index(mode_id) if mode_id in MODE_IDS else 0


# ----------------------------------------------------------------------------
# The consent store
# ----------------------------------------------------------------------------


def store_path() -> Path:
    """``$LOOM_HOME/telemetry.json``.

    Resolved on every call rather than cached at import: the test suite points
    ``LOOM_HOME`` at a temp dir, and ``/setup`` can be re-run after it changes.
    """
    return cfg.USER_CONFIG_DIR / "telemetry.json"


def _meant_host(stored: str, public_key: str, secret_key: str) -> str:
    """Migration for consent records written before the host was resolved as
    part of a credential set.

    Every such record stored ``https://cloud.langfuse.com``, whether or not
    the user had ever seen a Langfuse prompt — so on its own it says nothing.
    Keep it only when the record also carries the user's own key pair, which
    is the one case where they can have chosen it; otherwise it is a default
    masquerading as a decision, and it has to read as unset so the bundled
    credentials reach the proxy they belong to.
    """
    if stored and stored.rstrip("/") == LANGFUSE_CLOUD_HOST and not (public_key and secret_key):
        return ""
    return stored


@dataclass
class Consent:
    """The whole of what the user has agreed to, as stored on disk."""

    version: int = STORE_VERSION
    mode: str = DEFAULT_MODE
    # False until the user has actually been shown the question. Distinct from
    # `mode == "none"`, which is a real answer someone gave.
    decided: bool = False
    sentry_dsn: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    # Empty means "the user never named one", which is not the same as
    # "cloud.langfuse.com". Storing the latter for everybody is what silently
    # redirected the bundled proxy credentials at the public API.
    langfuse_host: str = ""
    # project key -> {"share": bool, "at": epoch seconds}
    projects: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "decided": self.decided,
            "sentry": {"dsn": self.sentry_dsn},
            "langfuse": {
                "public_key": self.langfuse_public_key,
                "secret_key": self.langfuse_secret_key,
                "host": self.langfuse_host,
            },
            "projects": self.projects,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Consent":
        langfuse = data.get("langfuse") or {}
        sentry = data.get("sentry") or {}
        mode = str(data.get("mode") or DEFAULT_MODE)
        return cls(
            version=int(data.get("version") or STORE_VERSION),
            # An unrecognised mode (a downgrade, a hand-edit) reads as `none`.
            # Failing closed is the only safe direction for a consent record.
            mode=mode if mode in MODE_IDS else DEFAULT_MODE,
            decided=bool(data.get("decided")),
            sentry_dsn=str(sentry.get("dsn") or ""),
            langfuse_public_key=str(langfuse.get("public_key") or ""),
            langfuse_secret_key=str(langfuse.get("secret_key") or ""),
            # A record written by an older build stored the cloud host for
            # everyone. Read it as "unset" unless the user also has their own
            # keys, which is the only case where they can have meant it.
            langfuse_host=_meant_host(
                str(langfuse.get("host") or ""),
                str(langfuse.get("public_key") or ""),
                str(langfuse.get("secret_key") or ""),
            ),
            projects=dict(data.get("projects") or {}),
        )


def load() -> Consent:
    """Read the consent store. A missing or corrupt file means "never asked"."""
    path = store_path()
    try:
        with path.open("r", encoding="utf-8") as fh:
            return Consent.from_json(json.load(fh) or {})
    except (OSError, ValueError, TypeError):
        return Consent()


def save(consent: Consent) -> Path:
    """Persist the consent store, readable only by this user.

    It holds a Langfuse secret key, so the 0600 is not decoration. The mode is
    written last-write-wins; there is no merging, because the only writer is a
    question the user just answered.
    """
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(consent.to_json(), fh, indent=2)
    try:
        path.chmod(0o600)
    except OSError:
        pass  # Windows / exotic filesystems — the content is still correct
    return path


# ----------------------------------------------------------------------------
# Effective mode
# ----------------------------------------------------------------------------


def global_mode(consent: Consent | None = None) -> str:
    """The configured mode, after the env override has had its (only
    tightening) say."""
    consent = consent if consent is not None else load()
    override = (os.environ.get(ENV_OVERRIDE) or "").strip().lower()
    if override in MODE_IDS:
        return min(consent.mode, override, key=_rank)
    return consent.mode


def needs_decision(consent: Consent | None = None) -> bool:
    """True until the user has been asked once.

    Deliberately independent of :func:`loom.ui.onboarding.needs_onboarding`:
    someone who configured Loom before privacy modes existed has settings.json
    but has never answered this, and must still be asked rather than silently
    defaulted into a choice they did not make.
    """
    consent = consent if consent is not None else load()
    return not consent.decided


# ----------------------------------------------------------------------------
# Per-project consent
# ----------------------------------------------------------------------------


def project_key(root: str | Path = ".") -> str:
    """A stable identity for "this project".

    The git work-tree root when there is one, so that ``loom`` in
    ``repo/backend`` and ``repo/frontend`` are one decision rather than two;
    the resolved directory otherwise. Walks up looking for ``.git`` instead of
    shelling out to git — this runs on every startup, and ``.git`` is a file
    (not a directory) inside a linked worktree, so both forms count.
    """
    try:
        here = Path(root).resolve()
    except OSError:
        return str(root)
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return str(candidate)
    return str(here)


def project_share(root: str | Path = ".", consent: Consent | None = None) -> bool | None:
    """Whether this project may share: True, False, or None for "never asked"."""
    consent = consent if consent is not None else load()
    entry = consent.projects.get(project_key(root))
    if not isinstance(entry, dict) or "share" not in entry:
        return None
    return bool(entry["share"])


def needs_project_decision(root: str | Path = ".", consent: Consent | None = None) -> bool:
    """True when this directory has never been asked *and* there is something
    to ask about.

    In ``none`` mode nothing is shared from anywhere, so a per-project question
    would be a prompt with no consequence — the fastest way to teach someone to
    dismiss consent dialogs without reading them.
    """
    consent = consent if consent is not None else load()
    if global_mode(consent) == "none" or needs_decision(consent):
        return False
    return project_share(root, consent) is None


def record_project(root: str | Path, share: bool, consent: Consent | None = None) -> Consent:
    """Remember this project's answer and persist it."""
    consent = consent if consent is not None else load()
    consent.projects[project_key(root)] = {"share": bool(share), "at": int(time.time())}
    save(consent)
    return consent


def active_mode(root: str | Path = ".", consent: Consent | None = None) -> str:
    """The mode actually in force *here* — both gates applied.

    A project that was never asked counts as not sharing. Consent is something
    you gave, not something you failed to refuse.
    """
    consent = consent if consent is not None else load()
    mode = global_mode(consent)
    if mode == "none":
        return "none"
    return mode if project_share(root, consent) is True else "none"


# ----------------------------------------------------------------------------
# Scrubbing — the part that has to be right
# ----------------------------------------------------------------------------

# Event fields that exist to identify a machine or replay its invocation.
# A bug report needs none of them; every one of them leaks something.
_DROP_EVENT_KEYS = ("server_name", "user", "request", "modules")
_DROP_EXTRA_KEYS = ("sys.argv",)
# Per-frame fields that carry code rather than location. `vars` is the locals;
# the three context fields are the source lines the SDK reads off disk.
_DROP_FRAME_KEYS = ("vars", "pre_context", "context_line", "post_context")

_SECRET_MARKERS = (
    "api_key", "apikey", "secret", "token", "password", "passwd",
    "authorization", "auth", "dsn", "credential", "session", "cookie",
)


def _redactions() -> list[tuple[str, str]]:
    """Literal strings that must never appear in an event, longest first.

    The home directory is in every absolute path in a stack trace and usually
    contains the user's real name. The username itself catches the cases where
    the path was already relativised. Longest-first so ``/Users/mike/.loom``
    is replaced before ``/Users/mike`` turns it into ``~/.loom`` twice.
    """
    out: list[tuple[str, str]] = []
    try:
        home = str(Path.home())
        if len(home) > 3:
            out.append((home, "~"))
    except (OSError, RuntimeError):
        pass
    for key in ("USER", "USERNAME", "LOGNAME"):
        name = os.environ.get(key) or ""
        if len(name) > 2:
            out.append((name, "<user>"))
    return sorted(set(out), key=lambda pair: -len(pair[0]))


def _redact_text(value: str, pairs: list[tuple[str, str]]) -> str:
    for needle, replacement in pairs:
        if needle in value:
            value = value.replace(needle, replacement)
    return value


def _walk(value: Any, pairs: list[tuple[str, str]], depth: int = 0) -> Any:
    """Recursively redact strings and drop anything that looks like a secret.

    Depth-capped: a Sentry event is a plain JSON tree, but a hand-built one
    could be cyclic, and a stack overflow inside the crash reporter would turn
    one bug into two.
    """
    if depth > 12:
        return "<truncated>"
    if isinstance(value, str):
        return _redact_text(value, pairs)
    if isinstance(value, dict):
        out = {}
        for key, inner in value.items():
            if isinstance(key, str) and any(m in key.lower() for m in _SECRET_MARKERS):
                out[key] = "<redacted>"
            else:
                out[key] = _walk(inner, pairs, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_walk(item, pairs, depth + 1) for item in value]
    return value


def scrub_event(event: dict[str, Any], _hint: Any = None) -> dict[str, Any] | None:
    """Sentry ``before_send``: strip everything that identifies the user or
    carries their content, then let the rest through.

    Kept pure and module-level so it is directly testable — the SDK is
    optional, but this function's behaviour is the whole basis of the promise
    made to the user in the setup wizard, so it is tested without it.
    """
    if not isinstance(event, dict):
        return None
    pairs = _redactions()

    for key in _DROP_EVENT_KEYS:
        event.pop(key, None)
    # Breadcrumbs are the log of what happened before the crash, which for a
    # coding assistant is a log of the user's work. A bug report does not need
    # it and cannot be trusted to have scrubbed it.
    event["breadcrumbs"] = []

    extra = event.get("extra")
    if isinstance(extra, dict):
        for key in _DROP_EXTRA_KEYS:
            extra.pop(key, None)

    # Frames are where a crash report turns into a code leak, in two ways.
    #
    # `vars` is the repr of every local — one frame holding `content=<the
    # whole file>` undoes everything else on this page. It is off in init too;
    # this is the belt to those braces.
    #
    # `context_line`/`pre_context`/`post_context` are the *source lines around
    # the crash*, which the SDK reads off disk and attaches by default. On a
    # source install those are real lines of real code, and on any install they
    # can come from a user-authored hook. The wizard promises "never: your
    # code", so they are dropped here as well as disabled at init — a promise
    # this specific should not rest on one flag being right.
    for exception in (event.get("exception") or {}).get("values") or []:
        for frame in (exception.get("stacktrace") or {}).get("frames") or []:
            for field_name in _DROP_FRAME_KEYS:
                frame.pop(field_name, None)

    return _walk(event, pairs, 0)


# ----------------------------------------------------------------------------
# Activation
# ----------------------------------------------------------------------------

_active_mode: str = "none"
_sentry_ready = False
_langfuse_handler: Any = None
_langfuse_failed = False


def _loom_version() -> str:
    try:
        from loom import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _init_sentry(consent: Consent) -> bool:
    """Start the Sentry SDK. False if it isn't installed or has no DSN."""
    global _sentry_ready
    if _sentry_ready:
        return True
    dsn = os.environ.get("SENTRY_DSN") or consent.sentry_dsn or DEFAULT_SENTRY_DSN
    if not dsn:
        return False
    try:
        import sentry_sdk
    except ImportError:
        return False

    import sys

    # Chosen explicitly rather than left to the defaults. The default set
    # auto-installs integrations for langchain, langgraph, mcp, httpx and more
    # — every one of which patches the exact call path Loom uses to talk to
    # models. Someone who asked for "crashes only" did not ask for their LLM
    # stack to be instrumented, and a frozen binary makes it worse: a bundled
    # integration submodule PyInstaller failed to collect raises
    # ModuleNotFoundError straight out of init(), which the except below would
    # swallow into "telemetry silently off" in the shipped build.
    from sentry_sdk.integrations.atexit import AtexitIntegration
    from sentry_sdk.integrations.dedupe import DedupeIntegration
    from sentry_sdk.integrations.excepthook import ExcepthookIntegration
    from sentry_sdk.integrations.threading import ThreadingIntegration

    try:
        sentry_sdk.init(
            dsn=dsn,
            release=f"loom@{_loom_version()}",
            environment="binary" if getattr(sys, "frozen", False) else "source",
            # Every one of these defaults would otherwise send something the
            # wizard promised it would not.
            send_default_pii=False,
            include_local_variables=False,
            # The source lines around the crash, read off disk. On a source
            # install that is the user's code; scrub_event drops them too.
            include_source_context=False,
            attach_stacktrace=False,
            max_breadcrumbs=0,
            # Crash reporting only. Performance tracing here would sample real
            # runs, and a span name is a task description.
            traces_sample_rate=0.0,
            default_integrations=False,
            auto_enabling_integrations=False,
            integrations=[
                # always_run: the default declines to fire under an interactive
                # interpreter, and Loom's REPL is close enough to one that
                # crashes there would go unreported.
                ExcepthookIntegration(always_run=True),
                # The stock atexit callback prints "Sentry is attempting to
                # send N pending events / Waiting up to 2 seconds" to stderr,
                # which lands after the shell prompt returns. Flush silently.
                AtexitIntegration(callback=lambda _pending, _timeout: None),
                DedupeIntegration(),
                # LangGraph runs tool calls — and therefore most of Loom — on
                # worker threads. sys.excepthook never fires there, so without
                # this every crash below the graph boundary was invisible.
                ThreadingIntegration(propagate_scope=True),
            ],
            before_send=scrub_event,
            before_send_transaction=lambda *_a, **_k: None,
        )
    except Exception:
        return False
    _sentry_ready = True
    return True


def langfuse_credentials(consent: Consent) -> tuple[str, str, str] | None:
    """``(public_key, secret_key, host)`` as a *set*, or None if full tracing
    has nothing to send with.

    Resolved as a unit, never field by field. A key pair and a host are one
    credential: the bundled ``pk-lf-loom-ingest`` + ``lct_…`` pair only
    authenticates against Loom's ingest proxy, and a user's own Langfuse pair
    only authenticates against their own Langfuse.

    Picking each field independently produced the one combination that cannot
    work. `Consent.langfuse_host` was *always* populated — it defaulted to
    ``https://cloud.langfuse.com`` for everyone, including users who supplied
    no keys at all — so it shadowed the bundled proxy host while the keys
    still fell through to the bundled pair. Every full-tracing user was
    therefore sending the proxy's write token straight to cloud.langfuse.com,
    which rejected it, and the traces were dropped in a background thread
    where nobody saw the 401:

        Startup: Langfuse tracer successfully initialized
        | public_key=pk-lf-loom-ingest | base_url=https://cloud.langfuse.com

    ``LANGFUSE_HOST`` from the environment still overrides in every branch —
    that is how the proxy itself gets pointed at a local server under test.
    """
    env_public = os.environ.get("LANGFUSE_PUBLIC_KEY") or ""
    env_secret = os.environ.get("LANGFUSE_SECRET_KEY") or ""
    env_host = os.environ.get("LANGFUSE_HOST") or ""

    # A complete pair from the environment, then from the user's own record.
    # Both are "their Langfuse", so both take the host they configured.
    for public, secret in ((env_public, env_secret),
                           (consent.langfuse_public_key, consent.langfuse_secret_key)):
        if public and secret:
            return public, secret, env_host or consent.langfuse_host or LANGFUSE_CLOUD_HOST

    # The bundle. Its token is only valid at the proxy, so the stored host —
    # which the user never chose — must not be allowed to redirect it.
    if DEFAULT_LANGFUSE_PUBLIC_KEY and DEFAULT_LANGFUSE_SECRET_KEY:
        return DEFAULT_LANGFUSE_PUBLIC_KEY, DEFAULT_LANGFUSE_SECRET_KEY, env_host or DEFAULT_LANGFUSE_HOST
    return None


def _init_langfuse(consent: Consent) -> Any:
    """Build the Langfuse LangChain callback handler, or None."""
    global _langfuse_handler, _langfuse_failed
    if _langfuse_handler is not None or _langfuse_failed:
        return _langfuse_handler
    resolved = langfuse_credentials(consent)
    if resolved is None:
        _langfuse_failed = True
        return None
    public, secret, host = resolved
    # Assigned, not `setdefault`-ed: the SDK reads these back out of the
    # environment, so a stray half-configured `LANGFUSE_PUBLIC_KEY` left in a
    # shell would otherwise survive and be paired with a secret from a
    # different credential set — the same mixing this function exists to stop.
    os.environ["LANGFUSE_PUBLIC_KEY"] = public
    os.environ["LANGFUSE_SECRET_KEY"] = secret
    os.environ["LANGFUSE_HOST"] = host
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        try:  # langfuse v2 kept the handler somewhere else
            from langfuse.callback import CallbackHandler  # type: ignore[no-redef]
        except ImportError:
            _langfuse_failed = True
            return None
    try:
        _langfuse_handler = CallbackHandler()
    except Exception:
        _langfuse_failed = True
        return None
    return _langfuse_handler


def activate(root: str | Path = ".") -> str:
    """Turn on whatever this project has consented to, and return that mode.

    Called once at startup. Safe to call repeatedly; safe to call when nothing
    is configured, when the SDKs are absent, and when the network is down.
    """
    global _active_mode
    consent = load()
    mode = active_mode(root, consent)
    _active_mode = mode
    if mode == "none":
        return "none"
    if not _init_sentry(consent) and mode == "errors":
        # Consented, but there is nowhere to send. Not an error worth
        # interrupting anyone over — /privacy and /doctor both report it.
        # `_active_mode` has to come back down with the return value: leaving
        # it at "errors" made current_mode() claim reporting was on while
        # every capture silently dropped on the `_sentry_ready` check.
        _active_mode = "none"
        return "none"
    if mode == "full":
        _init_langfuse(consent)
    return mode


def current_mode() -> str:
    """The mode :func:`activate` settled on for this process."""
    return _active_mode


def callbacks() -> list[Any]:
    """LangChain callbacks to add to a run config — the Langfuse tracer in
    ``full`` mode, nothing otherwise."""
    if _active_mode != "full" or _langfuse_handler is None:
        return []
    return [_langfuse_handler]


def capture_exception(exc: BaseException, where: str = "", **tags: Any) -> None:
    """Report a crash, if the user asked us to. Never raises.

    ``where`` names the call site ("turn", "tool:execute", "mcp") and lands as
    a Sentry tag, because most of what Loom catches is caught *somewhere
    specific* — an exception type alone doesn't say whether the model call
    failed, a tool blew up, or an MCP server never came up. It is a fixed
    label chosen at the call site, never user content.
    """
    if _active_mode == "none" or not _sentry_ready:
        return
    try:
        import sentry_sdk

        with sentry_sdk.new_scope() as scope:
            if where:
                scope.set_tag("where", where)
            for key, value in tags.items():
                scope.set_tag(key, str(value)[:200])
            sentry_sdk.capture_exception(exc)
    except Exception:
        pass


def report(where: str, exc: BaseException, **tags: Any) -> None:
    """``capture_exception`` with the call site first, for the many places that
    catch an exception, show it, and carry on.

    Loom deliberately swallows almost everything — a failed MCP server, a
    failed compaction, a failed stream — so that one broken thing never ends a
    session. The cost of that is that a caught error is invisible to whoever
    has to fix it, so every one of those sites reports here on the way past.
    """
    capture_exception(exc, where, **tags)


def capture_message(message: str, where: str = "", level: str = "error", **tags: Any) -> None:
    """Report a failure that never produced an exception object — a subprocess
    that exited non-zero, a provider that answered with an error body. Never
    raises.

    ``message`` must be a fixed string written here in the source, not user
    content: this bypasses the exception path but not the promise made in the
    setup wizard.
    """
    if _active_mode == "none" or not _sentry_ready:
        return
    try:
        import sentry_sdk

        with sentry_sdk.new_scope() as scope:
            if where:
                scope.set_tag("where", where)
            for key, value in tags.items():
                scope.set_tag(key, str(value)[:200])
            sentry_sdk.capture_message(message, level=level)
    except Exception:
        pass


@contextmanager
def swallowing(where: str, **tags: Any):
    """Run a best-effort block: never propagate, but never vanish either.

    Loom swallows a great deal on purpose — a failed repo map, a failed undo
    snapshot and a failed checkpointer all have to leave the session running.
    The problem was never the swallowing, it was that `except Exception: pass`
    makes "nothing went wrong" and "the guarantee you rely on is gone"
    identical from the outside.

    Use this where the failure silently removes something the user believes
    they have. Leave a plain `except: pass` where the exception *is* the
    control flow — a probe that answers "not installed", an `int()` that
    answers "not a number" — because reporting those is just noise.

    Never use it inside this module: a reporter that reports its own failures
    recurses.
    """
    try:
        yield
    except Exception as exc:
        report(where, exc, **tags)


def status() -> dict[str, Any]:
    """What activation actually achieved this process — for /doctor and
    /privacy, which until now could only report what was *configured*.

    The gap between the two is the whole bug class this exists to surface: a
    consent record saying "errors" and a process where the SDK never
    initialised look identical from the outside, and only one of them sends
    anything.
    """
    return {
        "mode": _active_mode,
        "sentry": _sentry_ready,
        "langfuse": _langfuse_handler is not None,
    }


def flush(timeout: float = 2.0) -> None:
    """Push anything queued before the process exits. Never raises, and never
    blocks for long — a slow telemetry endpoint must not hold the shell."""
    if _active_mode == "none":
        return
    if _sentry_ready:
        try:
            import sentry_sdk

            sentry_sdk.flush(timeout=timeout)
        except Exception:
            pass
    if _langfuse_handler is not None:
        try:
            from langfuse import get_client

            get_client().flush()
        except Exception:
            pass


def _reset_for_tests() -> None:
    """Drop process-level activation state. Tests only."""
    global _active_mode, _sentry_ready, _langfuse_handler, _langfuse_failed
    _active_mode = "none"
    _sentry_ready = False
    _langfuse_handler = None
    _langfuse_failed = False
