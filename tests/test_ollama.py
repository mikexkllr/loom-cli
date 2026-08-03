"""Ollama helpers: tag normalization and HTTP pulls against the configured
(possibly remote) daemon endpoint."""

import json

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("httpx")

import httpx

from loom.core import ollama
from loom.core.config import LoomConfig


# ---------------------------------------------------------------------------
# is_served / missing_models normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "available", "served"),
    [
        ("qwen3:4b", ["qwen3:4b"], True),
        ("qwen3", ["qwen3:latest"], True),
        ("qwen3:latest", ["qwen3"], True),
        ("qwen3:4b", ["qwen3:4b:latest"], True),  # daemon-side :latest suffix
        ("qwen3:4b", ["qwen3:14b"], False),
        ("qwen3", ["qwen3:4b"], False),
        ("qwen3:4b", [], False),
    ],
)
def test_is_served_normalizes_latest(tag, available, served):
    assert ollama.is_served(tag, available) is served


def test_missing_models_uses_normalization(monkeypatch):
    config = LoomConfig(orchestrator="ollama/qwen3", subagents={"editor": "ollama/qwen3:4b"})
    monkeypatch.setattr(
        ollama,
        "status",
        lambda cfg: ollama.OllamaStatus(True, True, ["qwen3:latest"], "http://x"),
    )
    assert ollama.missing_models(config) == ["qwen3:4b"]


# ---------------------------------------------------------------------------
# HTTP pull
# ---------------------------------------------------------------------------


class _FakeStream:
    """Stand-in for httpx.stream()'s response context manager."""

    def __init__(self, lines: list[dict], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def iter_lines(self):
        for line in self._lines:
            yield json.dumps(line)


def _quiet_console():
    import io

    from rich.console import Console

    return Console(file=io.StringIO(), force_terminal=False)


def test_pull_streams_to_the_configured_endpoint(monkeypatch):
    seen = {}

    def fake_stream(method, url, **kwargs):
        seen["method"], seen["url"], seen["json"] = method, url, kwargs.get("json")
        return _FakeStream(
            [
                {"status": "pulling manifest"},
                {"status": "pulling sha", "digest": "sha256:abc", "total": 10, "completed": 5},
                {"status": "pulling sha", "digest": "sha256:abc", "total": 10, "completed": 10},
                {"status": "success"},
            ]
        )

    monkeypatch.setattr(ollama.httpx, "stream", fake_stream)
    code = ollama.pull("qwen3:4b", "http://remote-box:11434", _quiet_console())
    assert code == 0
    assert seen["url"] == "http://remote-box:11434/api/pull"
    assert seen["json"] == {"model": "qwen3:4b"}


def test_pull_reports_daemon_errors(monkeypatch):
    monkeypatch.setattr(
        ollama.httpx,
        "stream",
        lambda *a, **k: _FakeStream([{"error": "pull model manifest: file does not exist"}]),
    )
    assert ollama.pull("nope:1b", "http://x", _quiet_console()) == 1


def test_pull_handles_unreachable_daemon(monkeypatch):
    def raise_connect(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(ollama.httpx, "stream", raise_connect)
    assert ollama.pull("qwen3:4b", "http://down:11434", _quiet_console()) == 1


# ---------------------------------------------------------------------------
# context_length: ask the daemon instead of assuming a window
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self._payload


def _post_returning(payload, monkeypatch, seen=None, status_code=200):
    def fake_post(url, **kwargs):
        if seen is not None:
            seen["url"], seen["json"] = url, kwargs.get("json")
        return _FakeResponse(payload, status_code)

    monkeypatch.setattr(ollama.httpx, "post", fake_post)


def test_context_length_reads_the_architecture_prefixed_key(monkeypatch):
    seen = {}
    _post_returning(
        {"model_info": {"general.architecture": "qwen3", "qwen3.context_length": 262144}},
        monkeypatch,
        seen,
    )
    ollama.context_length.cache_clear()
    assert ollama.context_length("qwen3.6:27b", "http://remote-box:11434") == 262144
    assert seen["url"] == "http://remote-box:11434/api/show"
    assert seen["json"] == {"model": "qwen3.6:27b"}


def test_context_length_is_none_when_the_daemon_reports_nothing_usable(monkeypatch):
    _post_returning({"model_info": {"general.architecture": "qwen3"}}, monkeypatch)
    ollama.context_length.cache_clear()
    assert ollama.context_length("qwen3:4b", "http://x") is None


def test_context_length_survives_an_unreachable_daemon(monkeypatch):
    def raise_connect(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(ollama.httpx, "post", raise_connect)
    ollama.context_length.cache_clear()
    assert ollama.context_length("qwen3:4b", "http://down:11434") is None


# ---------------------------------------------------------------------------
# pull retries: a dropped transfer is ordinary, not fatal
# ---------------------------------------------------------------------------


def test_a_dropped_transfer_is_retried(monkeypatch):
    """The real report: `max retries exceeded: read tcp ...: connection reset
    by peer` at 0.5 GB of a 9.6 GB model. Ollama keeps finished blobs, so the
    next attempt resumes — giving up threw that away."""
    attempts = []

    def _stream(*a, **k):
        attempts.append(1)
        if len(attempts) < 3:
            return _FakeStream([{"error": "max retries exceeded: read tcp 1.2.3.4:443: connection reset by peer"}])
        return _FakeStream([{"status": "success"}])

    monkeypatch.setattr(ollama.httpx, "stream", _stream)
    monkeypatch.setattr(ollama.time, "sleep", lambda s: None)
    assert ollama.pull("gemma4:e4b", "http://x", _quiet_console()) == 0
    assert len(attempts) == 3


def test_a_permanent_error_is_not_retried(monkeypatch):
    """A missing manifest fails identically forever; retrying only makes the
    user wait longer to hear the same thing."""
    attempts = []

    def _stream(*a, **k):
        attempts.append(1)
        return _FakeStream([{"error": "pull model manifest: file does not exist"}])

    monkeypatch.setattr(ollama.httpx, "stream", _stream)
    monkeypatch.setattr(ollama.time, "sleep", lambda s: pytest.fail("slept before a hopeless retry"))
    assert ollama.pull("nope:1b", "http://x", _quiet_console()) == 1
    assert len(attempts) == 1


def test_an_unreachable_daemon_is_not_retried(monkeypatch):
    attempts = []

    def _raise(*a, **k):
        attempts.append(1)
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(ollama.httpx, "stream", _raise)
    monkeypatch.setattr(ollama.time, "sleep", lambda s: pytest.fail("slept waiting for a dead daemon"))
    assert ollama.pull("qwen3:4b", "http://down:11434", _quiet_console()) == 1
    assert len(attempts) == 1


def test_retries_are_bounded(monkeypatch):
    attempts = []

    def _stream(*a, **k):
        attempts.append(1)
        return _FakeStream([{"error": "connection reset by peer"}])

    monkeypatch.setattr(ollama.httpx, "stream", _stream)
    monkeypatch.setattr(ollama.time, "sleep", lambda s: None)
    assert ollama.pull("gemma4:e4b", "http://x", _quiet_console(), attempts=3) == 1
    assert len(attempts) == 3


def test_a_broken_stream_mid_transfer_is_retryable(monkeypatch):
    """The daemon answered, so it is up — the transfer itself broke."""
    attempts = []

    def _stream(*a, **k):
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ReadError("connection reset")
        return _FakeStream([{"status": "success"}])

    monkeypatch.setattr(ollama.httpx, "stream", _stream)
    monkeypatch.setattr(ollama.time, "sleep", lambda s: None)
    assert ollama.pull("gemma4:e4b", "http://x", _quiet_console()) == 0


def test_ctrl_c_during_a_pull_stops_rather_than_retrying(monkeypatch):
    attempts = []

    def _stream(*a, **k):
        attempts.append(1)
        raise KeyboardInterrupt

    monkeypatch.setattr(ollama.httpx, "stream", _stream)
    monkeypatch.setattr(ollama.time, "sleep", lambda s: pytest.fail("retried after an interrupt"))
    assert ollama.pull("gemma4:e4b", "http://x", _quiet_console()) == 130
    assert len(attempts) == 1


def test_transient_classification():
    assert ollama._is_transient("max retries exceeded: connection reset by peer")
    assert ollama._is_transient("unexpected EOF")
    assert ollama._is_transient("i/o timeout")
    assert not ollama._is_transient("pull model manifest: file does not exist")
    assert not ollama._is_transient("model 'typo:9b' not found")
