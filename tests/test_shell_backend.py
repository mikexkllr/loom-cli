"""LoomShellBackend: a backgrounded child must not hold `execute` hostage.

Found on Terminal-Bench: `python3 -m http.server &` through deepagents'
pipe-capturing execute blocked for the whole timeout, reported a timeout, and
the server was gone — so every "leave a service running" task failed.
"""

from __future__ import annotations

import os
import socket
import time

import pytest

from loom.tools.shell import LoomShellBackend

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")


def _backend(tmp_path):
    return LoomShellBackend(root_dir=tmp_path, virtual_mode=True, inherit_env=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_background_server_returns_at_once_and_keeps_running(tmp_path):
    port = _free_port()
    b = _backend(tmp_path)
    start = time.monotonic()
    r = b.execute(f"python3 -m http.server {port} --bind 127.0.0.1 & echo started", timeout=10)
    assert time.monotonic() - start < 5
    assert r.exit_code == 0 and "started" in r.output
    try:
        for _ in range(50):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        else:
            pytest.fail("background server never came up")
    finally:
        b.execute(f"pkill -f 'http.server {port}'", timeout=10)


def test_a_timeout_stops_the_whole_process_group(tmp_path):
    marker = tmp_path / "still-alive"
    b = _backend(tmp_path)
    r = b.execute(f"(sleep 3; touch {marker}) & sleep 30", timeout=1)
    assert r.exit_code == 124 and "timed out" in r.output
    time.sleep(3.5)
    assert not marker.exists()


def test_output_exit_code_and_stderr_match_the_upstream_format(tmp_path):
    r = _backend(tmp_path).execute("echo out; echo err >&2; exit 3")
    assert r.exit_code == 3
    assert r.output == "out\n\n[stderr] err\n\nExit code: 3"


def test_partial_output_survives_a_timeout(tmp_path):
    r = _backend(tmp_path).execute("echo progress; sleep 30", timeout=1)
    assert r.exit_code == 124 and "progress" in r.output


def test_runs_in_the_sandbox_root(tmp_path):
    r = _backend(tmp_path).execute("pwd")
    assert r.output.strip() == str(tmp_path.resolve())
