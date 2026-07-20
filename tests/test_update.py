"""Self-update helpers: frozen-vs-source detection, asset naming, the
throttled startup check, and the checksum cache — the network/exec paths
(apply / apply_and_relaunch) aren't covered here since they require a real
frozen binary and a published release; see packaging/loom.spec's docstring
for how that's exercised manually."""

import sys

import pytest

pytest.importorskip("httpx")

from loom.core import update


def test_is_frozen_false_under_pytest():
    assert update.is_frozen() is False


def test_asset_name_matches_current_platform():
    name = update.asset_name()
    assert name.startswith(("loom-macos-", "loom-linux-", "loom-windows-"))


def test_asset_name_rejects_unknown_platform(monkeypatch):
    monkeypatch.setattr(update.platform, "system", lambda: "PlayStation")
    with pytest.raises(RuntimeError, match="unsupported platform"):
        update.asset_name()


def test_up_to_date_compares_checksums():
    same = update.UpdateCheck(asset="loom-linux-x64", current_sha256="a", latest_sha256="a")
    diff = update.UpdateCheck(asset="loom-linux-x64", current_sha256="a", latest_sha256="b")
    assert same.up_to_date is True
    assert diff.up_to_date is False


def test_check_for_startup_short_circuits_when_not_frozen(monkeypatch):
    # Source (uv sync) installs have no binary to replace, so the startup
    # check must never even try the network.
    def boom(*a, **k):
        raise AssertionError("network hit despite not being a frozen install")

    monkeypatch.setattr(update, "is_frozen", lambda: False)
    monkeypatch.setattr(update, "_fetch_latest_sha", boom)
    assert update.check_for_startup() is None


def test_check_for_startup_never_raises_on_network_failure(monkeypatch):
    monkeypatch.setattr(update, "is_frozen", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(update, "_fetch_latest_sha", boom)
    assert update.check_for_startup() is None


def test_check_for_startup_uses_cache_within_window(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "is_frozen", lambda: True)
    monkeypatch.setattr(update, "CACHE_PATH", tmp_path / "update_check.json")
    asset = update.asset_name()

    calls = []

    def fake_fetch(_asset, *, timeout):
        calls.append(_asset)
        return "same-sha"

    monkeypatch.setattr(update, "_fetch_latest_sha", fake_fetch)
    monkeypatch.setattr(update, "_sha256", lambda _path: "same-sha")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "loom"))
    (tmp_path / "loom").write_bytes(b"binary")

    # First call: cache miss, hits the network, and (same-sha) reports up to date.
    assert update.check_for_startup() is None
    assert calls == [asset]

    # Second call within the throttle window: cache hit, no second network call.
    assert update.check_for_startup() is None
    assert calls == [asset]


def test_check_for_startup_reports_stale_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "is_frozen", lambda: True)
    monkeypatch.setattr(update, "CACHE_PATH", tmp_path / "update_check.json")
    monkeypatch.setattr(update, "_fetch_latest_sha", lambda *a, **k: "new-sha")
    monkeypatch.setattr(update, "_sha256", lambda _path: "old-sha")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "loom"))
    (tmp_path / "loom").write_bytes(b"binary")

    result = update.check_for_startup()
    assert result is not None
    assert result.up_to_date is False
    assert result.current_sha256 == "old-sha"
    assert result.latest_sha256 == "new-sha"
