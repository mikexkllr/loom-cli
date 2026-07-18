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
