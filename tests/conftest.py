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

# Use /tmp directly (short, always local-disk) rather than $TMPDIR, which on
# macOS is a long per-user path that can push printed paths past the
# console's wrap width in tests that assert on CLI output.
_tmp_root = "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()
os.environ.setdefault("LOOM_HOME", os.path.join(tempfile.mkdtemp(prefix="loom-", dir=_tmp_root), ".loom"))
