"""Test-suite isolation.

``loom.core.config.USER_CONFIG_DIR`` (and everything derived from it, e.g.
``USER_SETTINGS_PATH``) defaults to the real ``~/.loom`` and is frozen at
import time. Without this, a developer's real ``~/.loom/settings.json``
(e.g. custom ``env`` vars) leaks into any test that builds models/orchestrator
in-process — notably via Typer's ``CliRunner``, which runs commands in the
same process rather than a subprocess. Set ``LOOM_HOME`` before pytest
imports any test module (conftest.py is always imported first) so every test
gets an empty, disposable home dir instead. ``setdefault`` still lets a
developer point at a specific ``LOOM_HOME`` on purpose.
"""

import os
import tempfile

import pytest

# Use /tmp directly (short, always local-disk) rather than $TMPDIR, which on
# macOS is a long per-user path that can push printed paths past the
# console's wrap width in tests that assert on CLI output.
_tmp_root = "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()
os.environ.setdefault("LOOM_HOME", os.path.join(tempfile.mkdtemp(prefix="loom-", dir=_tmp_root), ".loom"))


@pytest.fixture(autouse=True)
def _no_real_telemetry(monkeypatch):
    """Blank the credentials Loom ships with, for every test.

    Two separate hazards, one fix.

    The bundled Sentry DSN points at Loom's real project, so any test that
    reaches ``telemetry.activate()`` starts a live client against production —
    which happened: the suite opened a session there on every full run and
    printed Sentry's atexit banner over the summary.

    The Langfuse secret is baked into ``loom/_built.py`` at binary-build time
    and is gitignored, so it exists on a machine that has built a release and
    nowhere else. Tests that branch on "is a bundled key available" therefore
    passed in CI and failed locally. Deciding that from a file that may or may
    not be present is exactly the class of bug ``_isolate_project_settings``
    below exists to stop.

    Tests that are *about* the bundled defaults set them back explicitly to
    obvious fakes (see tests/test_telemetry.py).
    """
    from loom.core import telemetry as tel

    monkeypatch.setattr(tel, "DEFAULT_SENTRY_DSN", "", raising=False)
    monkeypatch.setattr(tel, "DEFAULT_LANGFUSE_PUBLIC_KEY", "", raising=False)
    monkeypatch.setattr(tel, "DEFAULT_LANGFUSE_SECRET_KEY", "", raising=False)
    tel._reset_for_tests()
    yield
    tel._reset_for_tests()


@pytest.fixture(autouse=True)
def _isolate_project_settings(monkeypatch):
    """Hide the repo's own ``.loom/settings.json`` from tests.

    ``LOOM_HOME`` above covers the *user* layer, but ``load_settings()`` with
    no root also reads ``<cwd>/.loom/settings.json`` — and pytest's cwd is this
    repo. The moment a developer runs Loom on Loom, that file appears and
    starts overriding the user layer inside tests, which then fail on the
    developer's machine and nowhere else.

    Tests that pass an explicit root (``load_settings(tmp_path)``) are
    unaffected; this only blanks the implicit cwd lookup.
    """
    from loom.core import settings as settings_mod

    real = settings_mod.project_settings_paths

    def _paths(root="."):
        if str(root) in (".", "", os.getcwd()):
            return []
        return real(root)

    monkeypatch.setattr(settings_mod, "project_settings_paths", _paths)
