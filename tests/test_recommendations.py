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


# ---------------------------------------------------------------------------
# Context budget from GPU / unified memory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("budget_gb", "expected"),
    [
        (4, 8_192),  # tiny CPU-only laptop
        (8, 16_384),  # 8GB Apple Silicon
        (16, 32_768),  # 16GB unified / mid GPU
        (24, 65_536),  # 24GB card (4090 class)
        (48, 131_072),  # 48GB workstation card
        (128, 262_144),  # DGX Spark / big Mac Studio
    ],
)
def test_context_budget_scales_with_gpu_memory(budget_gb, expected):
    hw = rec.Hardware("Linux", float(budget_gb), "nvidia", float(budget_gb))
    assert rec.context_budget(hw) == expected


def test_context_budget_uses_system_ram_when_there_is_no_gpu():
    assert rec.context_budget(rec.Hardware("Linux", 16.0, None, None)) == 32_768


def test_context_budget_is_conservative_when_nothing_is_detectable():
    assert rec.context_budget(rec.Hardware("Linux", None, None, None)) == 16_384


def test_context_budget_treats_unified_memory_like_vram():
    """Apple Silicon and NVIDIA Grace/Jetson have no discrete VRAM — the
    unified pool is the GPU's memory, so it must size the budget the same way."""
    apple = rec.Hardware("Darwin", 64.0, "apple", 64.0, unified=True)
    grace = rec.Hardware("Linux", 64.0, "nvidia", 64.0, unified=True)
    discrete = rec.Hardware("Linux", 256.0, "nvidia", 64.0)
    assert rec.context_budget(apple) == rec.context_budget(grace) == rec.context_budget(discrete)


# ---------------------------------------------------------------------------
# NVIDIA unified-memory parts (Jetson/Tegra, Grace superchips)
# ---------------------------------------------------------------------------


def test_tegra_without_nvidia_smi_still_counts_as_a_gpu(monkeypatch):
    """Jetson ships tegrastats, not nvidia-smi. Without the unified check the
    box would look GPU-less and get the smallest possible budget."""
    monkeypatch.setattr(rec, "_detect_nvidia_vram_gb", lambda: None)
    monkeypatch.setattr(rec, "_detect_amd_vram_gb", lambda: None)
    monkeypatch.setattr(rec, "_detect_ram_gb", lambda: 128.0)
    monkeypatch.setattr(rec, "_is_nvidia_unified", lambda: True)
    monkeypatch.setattr(rec.platform, "system", lambda: "Linux")

    hw = rec.detect_hardware()
    assert hw.gpu_vendor == "nvidia" and hw.unified is True
    assert hw.vram_gb == 128.0
    assert rec.context_budget(hw) == 262_144


def test_grace_takes_the_larger_of_smi_and_system_memory(monkeypatch):
    """nvidia-smi can report less than the coherent pool on Grace parts."""
    monkeypatch.setattr(rec, "_detect_nvidia_vram_gb", lambda: 96.0)
    monkeypatch.setattr(rec, "_detect_ram_gb", lambda: 480.0)
    monkeypatch.setattr(rec, "_is_nvidia_unified", lambda: True)
    monkeypatch.setattr(rec.platform, "system", lambda: "Linux")

    hw = rec.detect_hardware()
    assert hw.vram_gb == 480.0 and hw.unified is True


def test_discrete_nvidia_is_not_marked_unified(monkeypatch):
    monkeypatch.setattr(rec, "_detect_nvidia_vram_gb", lambda: 24.0)
    monkeypatch.setattr(rec, "_detect_ram_gb", lambda: 128.0)
    monkeypatch.setattr(rec, "_is_nvidia_unified", lambda: False)
    monkeypatch.setattr(rec.platform, "system", lambda: "Linux")

    hw = rec.detect_hardware()
    assert hw.unified is False
    assert hw.vram_gb == 24.0  # system RAM must not inflate a discrete card


def test_unified_nvidia_reads_as_unified_memory_not_vram():
    hw = rec.Hardware("Linux", 128.0, "nvidia", 128.0, unified=True)
    assert "unified memory" in rec.hardware_summary(hw)
