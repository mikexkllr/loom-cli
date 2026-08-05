"""The consent store and the gates that decide what, if anything, leaves the
machine. Everything here runs without the sentry/langfuse SDKs being active —
activation is exercised with fakes, and scrub_event (the function the whole
privacy promise rests on) is pure and tested directly.
"""

import json
import os
import stat

import pytest

from loom.core import telemetry as tel


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Point the consent store at a per-test file and drop activation state."""
    monkeypatch.setattr(tel, "store_path", lambda: tmp_path / "telemetry.json")
    tel._reset_for_tests()
    yield tmp_path
    tel._reset_for_tests()


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(tel.ENV_OVERRIDE, raising=False)
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)


# ------------------------------------------------------------- consent store


def test_missing_store_means_never_asked():
    consent = tel.load()
    assert consent.mode == "none"
    assert consent.decided is False
    assert tel.needs_decision(consent) is True


def test_store_round_trip(tmp_path):
    consent = tel.Consent(
        mode="full",
        decided=True,
        sentry_dsn="https://k@o/1",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        langfuse_host="https://lf.example",
        projects={"/repo": {"share": True, "at": 1}},
    )
    path = tel.save(consent)
    loaded = tel.load()
    assert loaded.mode == "full"
    assert loaded.decided is True
    assert loaded.sentry_dsn == "https://k@o/1"
    assert loaded.langfuse_secret_key == "sk"
    assert loaded.projects["/repo"]["share"] is True
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_corrupt_store_reads_as_never_asked(tmp_path):
    (tmp_path / "telemetry.json").write_text("{not json", encoding="utf-8")
    consent = tel.load()
    assert consent.decided is False
    assert consent.mode == "none"


def test_unrecognised_mode_fails_closed():
    """A hand-edit or a downgrade must not resurrect a sharing mode."""
    consent = tel.Consent.from_json({"mode": "everything", "decided": True})
    assert consent.mode == "none"
    # ...but the decision itself survives: none is the answer, not "unasked".
    assert consent.decided is True


# ------------------------------------------------------------- env override


@pytest.mark.parametrize(
    "stored,override,expected",
    [
        ("full", "none", "none"),       # tightens
        ("full", "errors", "errors"),   # tightens
        ("errors", "full", "errors"),   # can never loosen
        ("none", "full", "none"),       # can never loosen
        ("errors", "bogus", "errors"),  # junk is ignored
    ],
)
def test_env_override_only_tightens(stored, override, expected, monkeypatch):
    consent = tel.Consent(mode=stored, decided=True)
    monkeypatch.setenv(tel.ENV_OVERRIDE, override)
    assert tel.global_mode(consent) == expected


# -------------------------------------------------------------- project gate


def test_project_key_uses_git_root(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "backend" / "pkg"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    assert tel.project_key(nested) == str(repo)


def test_project_key_accepts_gitfile_worktrees(tmp_path):
    """A linked worktree has .git as a file, not a directory — still a repo."""
    repo = tmp_path / "wt"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    assert tel.project_key(repo) == str(repo)


def test_project_key_falls_back_to_the_directory(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert tel.project_key(plain) == str(plain)


def test_project_share_unknown_until_asked(tmp_path):
    consent = tel.Consent(mode="errors", decided=True)
    assert tel.project_share(tmp_path, consent) is None
    consent = tel.record_project(tmp_path, True)
    assert tel.project_share(tmp_path, consent) is True
    consent = tel.record_project(tmp_path, False)
    assert tel.project_share(tmp_path, consent) is False


def test_needs_project_decision_only_when_something_to_share(tmp_path):
    undecided = tel.Consent()
    assert tel.needs_project_decision(tmp_path, undecided) is False  # belongs to setup
    none_mode = tel.Consent(mode="none", decided=True)
    assert tel.needs_project_decision(tmp_path, none_mode) is False  # no consequence
    sharing = tel.Consent(mode="errors", decided=True)
    assert tel.needs_project_decision(tmp_path, sharing) is True
    sharing = tel.record_project(tmp_path, False, sharing)
    assert tel.needs_project_decision(tmp_path, sharing) is False  # answered


def test_active_mode_requires_both_gates(tmp_path):
    consent = tel.Consent(mode="full", decided=True)
    assert tel.active_mode(tmp_path, consent) == "none"      # never asked
    consent = tel.record_project(tmp_path, False, consent)
    assert tel.active_mode(tmp_path, consent) == "none"      # declined
    consent = tel.record_project(tmp_path, True, consent)
    assert tel.active_mode(tmp_path, consent) == "full"      # consented


# ----------------------------------------------------------------- scrubbing


def _event():
    return {
        "server_name": "mikes-macbook",
        "user": {"id": "mike"},
        "request": {"url": "http://x", "headers": {"Authorization": "Bearer abc"}},
        "modules": {"loom": "0.2.10"},
        "breadcrumbs": [{"message": "read secret.py"}],
        "extra": {"sys.argv": ["loom", "fix the auth bug"], "note": "kept"},
        "exception": {
            "values": [
                {
                    "type": "RuntimeError",
                    "value": f"boom in {os.path.expanduser('~')}/proj",
                    "stacktrace": {
                        "frames": [
                            {"filename": "loom/x.py", "vars": {"content": "the whole file"}},
                        ]
                    },
                }
            ]
        },
    }


def test_scrub_drops_machine_identity():
    event = tel.scrub_event(_event())
    for key in ("server_name", "user", "request", "modules"):
        assert key not in event
    assert event["breadcrumbs"] == []
    assert "sys.argv" not in event["extra"]
    assert event["extra"]["note"] == "kept"


def test_scrub_strips_frame_locals_and_redacts_home():
    event = tel.scrub_event(_event())
    frames = event["exception"]["values"][0]["stacktrace"]["frames"]
    assert "vars" not in frames[0]
    home = os.path.expanduser("~")
    if len(home) > 3:
        assert home not in event["exception"]["values"][0]["value"]


def test_scrub_redacts_secret_shaped_keys_anywhere():
    event = tel.scrub_event({"ctx": {"api_key": "abc", "nested": [{"session_token": "xyz"}]}})
    assert event["ctx"]["api_key"] == "<redacted>"
    assert event["ctx"]["nested"][0]["session_token"] == "<redacted>"


def test_scrub_caps_pathological_depth():
    deep = current = {}
    for _ in range(40):
        current["x"] = {}
        current = current["x"]
    out = tel.scrub_event({"deep": deep})
    assert "<truncated>" in json.dumps(out)


def test_scrub_rejects_non_dicts():
    assert tel.scrub_event(None) is None
    assert tel.scrub_event("boom") is None


# ---------------------------------------------------------------- activation


def test_activate_none_touches_no_sdks(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("an SDK was initialised in none mode")

    monkeypatch.setattr(tel, "_init_sentry", _boom)
    monkeypatch.setattr(tel, "_init_langfuse", _boom)
    assert tel.activate(".") == "none"
    assert tel.current_mode() == "none"
    assert tel.callbacks() == []


def test_activate_errors_without_dsn_degrades_to_none(tmp_path, monkeypatch):
    """A source build with no bundled DSN and no user DSN can't honour
    ``errors`` — it reads as a promise to send that nothing can keep. Pin that
    the bundled default is what makes this path activate instead."""
    monkeypatch.setattr(tel, "DEFAULT_SENTRY_DSN", "")
    tel.save(tel.Consent(mode="errors", decided=True, projects={tel.project_key(tmp_path): {"share": True, "at": 1}}))
    assert tel.activate(tmp_path) == "none"


def test_bundled_dsn_lets_errors_mode_work_without_a_user_key(tmp_path, monkeypatch):
    """Point 2 just works: a user who consented to bug reports needs no
    account — the bundled default DSN ships with Loom.

    The DSN here is a deliberate fake. conftest blanks the real bundled one for
    every test, because this call really does start a Sentry client and the
    shipped default points at Loom's production project.
    """
    monkeypatch.setattr(tel, "DEFAULT_SENTRY_DSN", "https://bundled@example.invalid/2")
    tel.save(tel.Consent(mode="errors", decided=True, projects={tel.project_key(tmp_path): {"share": True, "at": 1}}))
    pytest.importorskip("sentry_sdk")
    assert tel.activate(tmp_path) == "errors"
    assert tel.current_mode() == "errors"


def test_activate_errors_with_dsn_starts_sentry(tmp_path):
    tel.save(
        tel.Consent(
            mode="errors",
            decided=True,
            sentry_dsn="https://public@example.invalid/1",
            projects={tel.project_key(tmp_path): {"share": True, "at": 1}},
        )
    )
    pytest.importorskip("sentry_sdk")
    assert tel.activate(tmp_path) == "errors"
    assert tel.current_mode() == "errors"


def test_capture_and_flush_never_raise(tmp_path):
    """No mode, no SDK, no network — reporting must still be a no-op."""
    tel.capture_exception(RuntimeError("x"))
    tel.flush()
    tel.save(tel.Consent(mode="errors", decided=True))
    tel.capture_exception(RuntimeError("x"))  # not active for this project
    tel.flush()


def test_callbacks_only_in_full_mode_with_handler(monkeypatch):
    tel._active_mode = "errors"
    assert tel.callbacks() == []
    tel._active_mode = "full"
    assert tel.callbacks() == []  # no handler built yet
    monkeypatch.setattr(tel, "_langfuse_handler", object())
    assert len(tel.callbacks()) == 1


# ------------------------------------------------------- reporting call sites


def test_report_and_capture_message_are_no_ops_when_off():
    """The two new entry points are called from ordinary error paths all over
    Loom, so being off must never be a crash and never be a send."""
    tel._reset_for_tests()
    tel.report("tool", RuntimeError("x"), tool="grep")
    tel.capture_message("tool failed: grep", where="tool.result", tool="grep")
    assert tel.status() == {"mode": "none", "sentry": False, "langfuse": False}


def test_report_reaches_sentry_when_active(monkeypatch):
    """`report` has to actually hand the exception to the SDK — the bug this
    guards is a reporting helper that silently drops everything, which looks
    identical to "no errors happened"."""
    sent = []

    class _FakeScope:
        def __init__(self):
            self.tags = {}

        def set_tag(self, key, value):
            self.tags[key] = value

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    scope = _FakeScope()

    class _FakeSDK:
        @staticmethod
        def new_scope():
            return scope

        @staticmethod
        def capture_exception(exc):
            sent.append(exc)

    monkeypatch.setitem(__import__("sys").modules, "sentry_sdk", _FakeSDK)
    tel._active_mode = "errors"
    tel._sentry_ready = True
    exc = RuntimeError("boom")
    tel.report("tool", exc, tool="grep")
    assert sent == [exc]
    assert scope.tags == {"where": "tool", "tool": "grep"}


def test_activate_reports_none_when_sentry_cannot_start(tmp_path, monkeypatch):
    """Consented to `errors` but nothing initialised: current_mode() has to say
    "none" rather than claim reporting is on while every capture drops."""
    tel.save(
        tel.Consent(
            mode="errors",
            decided=True,
            projects={tel.project_key(tmp_path): {"share": True, "at": 1}},
        )
    )
    monkeypatch.setattr(tel, "_init_sentry", lambda _c: False)
    assert tel.activate(tmp_path) == "none"
    assert tel.current_mode() == "none"


# ---------------------------------------------- langfuse credentials are a set


def test_bundled_langfuse_credentials_go_to_the_proxy(monkeypatch):
    """The bug that silently dropped every full-tracing user's traces.

    `Consent.langfuse_host` used to default to cloud.langfuse.com for
    *everyone*, so it shadowed the bundled proxy host while the keys still
    fell through to the bundled pair — sending the proxy's write-only token
    to the public API, which rejects it, from a background thread where the
    401 was never seen.
    """
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tel, "DEFAULT_LANGFUSE_PUBLIC_KEY", "pk-lf-loom-ingest")
    monkeypatch.setattr(tel, "DEFAULT_LANGFUSE_SECRET_KEY", "lct_baked")

    consent = tel.Consent(mode="full", decided=True)
    assert tel.langfuse_credentials(consent) == (
        "pk-lf-loom-ingest", "lct_baked", tel.DEFAULT_LANGFUSE_HOST,
    )


def test_a_stored_cloud_host_does_not_redirect_bundled_credentials(monkeypatch):
    """Records written by older builds all carry the cloud host. Without keys
    beside it that is a default, not a decision, and must not win."""
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tel, "DEFAULT_LANGFUSE_PUBLIC_KEY", "pk-lf-loom-ingest")
    monkeypatch.setattr(tel, "DEFAULT_LANGFUSE_SECRET_KEY", "lct_baked")

    tel.save(tel.Consent(mode="full", decided=True))
    raw = json.loads(tel.store_path().read_text())
    raw["langfuse"]["host"] = "https://cloud.langfuse.com"  # what old builds wrote
    tel.store_path().write_text(json.dumps(raw))

    _public, _secret, host = tel.langfuse_credentials(tel.load())
    assert host == tel.DEFAULT_LANGFUSE_HOST


def test_the_users_own_keys_keep_their_own_host(monkeypatch):
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)
    consent = tel.Consent(
        mode="full", decided=True,
        langfuse_public_key="pk-lf-mine", langfuse_secret_key="sk-lf-mine",
        langfuse_host="https://langfuse.internal",
    )
    assert tel.langfuse_credentials(consent) == (
        "pk-lf-mine", "sk-lf-mine", "https://langfuse.internal",
    )
    # Own keys, no host named: their keys belong to Langfuse Cloud, never to
    # Loom's proxy — the proxy would reject them.
    consent.langfuse_host = ""
    assert tel.langfuse_credentials(consent)[2] == tel.LANGFUSE_CLOUD_HOST


def test_credentials_are_never_mixed_across_sources(monkeypatch):
    """Half a pair in the environment must not be paired with a secret from
    somewhere else — that combination authenticates nowhere."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-stray")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.setattr(tel, "DEFAULT_LANGFUSE_PUBLIC_KEY", "pk-lf-loom-ingest")
    monkeypatch.setattr(tel, "DEFAULT_LANGFUSE_SECRET_KEY", "lct_baked")

    public, secret, host = tel.langfuse_credentials(tel.Consent(mode="full", decided=True))
    assert (public, secret, host) == ("pk-lf-loom-ingest", "lct_baked", tel.DEFAULT_LANGFUSE_HOST)


def test_env_host_still_overrides_for_local_testing(monkeypatch):
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9932")
    monkeypatch.setattr(tel, "DEFAULT_LANGFUSE_PUBLIC_KEY", "pk-lf-loom-ingest")
    monkeypatch.setattr(tel, "DEFAULT_LANGFUSE_SECRET_KEY", "lct_baked")
    assert tel.langfuse_credentials(tel.Consent(mode="full", decided=True))[2] == "http://127.0.0.1:9932"


def test_no_langfuse_credentials_at_all():
    monkeypatch = pytest.MonkeyPatch()
    try:
        for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(tel, "DEFAULT_LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setattr(tel, "DEFAULT_LANGFUSE_SECRET_KEY", "")
        assert tel.langfuse_credentials(tel.Consent(mode="full", decided=True)) is None
    finally:
        monkeypatch.undo()
