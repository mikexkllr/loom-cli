"""Hardware detection + local model recommendation logic."""

import pytest

from loom.core import recommendations as rec


def test_recommend_local_models_falls_back_when_undetected():
    hw = rec.Hardware(os_name="Linux", ram_gb=None, gpu_vendor=None, vram_gb=None)
    recs = rec.recommend_local_models(hw)
    assert recs == [rec._LOCAL_TIERS[0]]


def test_recommend_local_models_picks_largest_that_fits():
    hw = rec.Hardware(os_name="Darwin", ram_gb=16, gpu_vendor="apple", vram_gb=16)
    recs = rec.recommend_local_models(hw)
    assert recs[0].min_gb <= 16
    # Largest-fitting tier should be first, and nothing over budget is included.
    assert all(r.min_gb <= 16 for r in recs)
    assert recs == sorted(recs, key=lambda r: -r.min_gb)


def test_recommend_local_models_respects_top_n():
    hw = rec.Hardware(os_name="Linux", ram_gb=64, gpu_vendor="nvidia", vram_gb=48)
    assert len(rec.recommend_local_models(hw, top_n=2)) == 2


def test_recommend_local_models_tiny_hardware_gets_smallest_tier():
    hw = rec.Hardware(os_name="Linux", ram_gb=2, gpu_vendor=None, vram_gb=None)
    recs = rec.recommend_local_models(hw)
    assert recs == [rec._LOCAL_TIERS[0]]


def test_all_local_models_returns_the_full_catalog():
    assert rec.all_local_models() == rec._LOCAL_TIERS


def test_fits_hardware_true_within_budget():
    hw = rec.Hardware(os_name="Darwin", ram_gb=32, gpu_vendor="apple", vram_gb=32)
    small = next(m for m in rec._LOCAL_TIERS if m.min_gb <= 8)
    assert rec.fits_hardware(hw, small) is True


def test_fits_hardware_false_over_budget():
    hw = rec.Hardware(os_name="Darwin", ram_gb=8, gpu_vendor="apple", vram_gb=8)
    huge = rec._LOCAL_TIERS[-1]
    assert huge.min_gb > 8
    assert rec.fits_hardware(hw, huge) is False


def test_fits_hardware_false_when_hardware_undetected():
    hw = rec.Hardware(os_name="Linux", ram_gb=None, gpu_vendor=None, vram_gb=None)
    assert rec.fits_hardware(hw, rec._LOCAL_TIERS[0]) is False


@pytest.mark.parametrize(
    "hw,expected_substr",
    [
        (rec.Hardware("Darwin", 32.0, "apple", 32.0), "unified memory"),
        (rec.Hardware("Linux", 64.0, "nvidia", 24.0), "VRAM"),
        (rec.Hardware("Linux", 64.0, "amd", 20.0), "VRAM"),
        (rec.Hardware("Linux", 16.0, None, None), "RAM"),
        (rec.Hardware("Linux", None, None, None), "unknown memory"),
    ],
)
def test_hardware_summary_mentions_relevant_memory_kind(hw, expected_substr):
    assert expected_substr in rec.hardware_summary(hw)


def test_hardware_summary_labels_amd_gpu():
    hw = rec.Hardware("Linux", 32.0, "amd", 20.0)
    assert "AMD GPU" in rec.hardware_summary(hw)


def test_detect_hardware_returns_current_os():
    import platform

    hw = rec.detect_hardware()
    assert hw.os_name == platform.system()


def test_detect_amd_vram_gb_parses_rocm_smi_json(monkeypatch):
    import json

    payload = json.dumps(
        {
            "card0": {
                "VRAM Total Memory (B)": str(20 * 1024**3),
                "VRAM Total Used Memory (B)": str(1 * 1024**3),
            }
        }
    )
    monkeypatch.setattr(rec.shutil, "which", lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None)
    monkeypatch.setattr(rec, "_run", lambda cmd: payload)
    assert rec._detect_amd_vram_gb() == pytest.approx(20.0)


def test_detect_amd_vram_gb_none_when_rocm_smi_missing(monkeypatch):
    monkeypatch.setattr(rec.shutil, "which", lambda name: None)
    assert rec._detect_amd_vram_gb() is None


def test_detect_amd_vram_gb_none_on_garbage_output(monkeypatch):
    monkeypatch.setattr(rec.shutil, "which", lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None)
    monkeypatch.setattr(rec, "_run", lambda cmd: "not json")
    assert rec._detect_amd_vram_gb() is None


def test_detect_hardware_falls_back_to_amd_when_no_nvidia(monkeypatch):
    monkeypatch.setattr(rec.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rec, "_detect_ram_gb", lambda: 64.0)
    monkeypatch.setattr(rec, "_detect_nvidia_vram_gb", lambda: None)
    monkeypatch.setattr(rec, "_detect_amd_vram_gb", lambda: 20.0)
    hw = rec.detect_hardware()
    assert hw.gpu_vendor == "amd"
    assert hw.vram_gb == 20.0
