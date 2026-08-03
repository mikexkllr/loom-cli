"""Hardware-aware local model recommendations, plus a pointer to good cloud
coding models — used by the onboarding wizard and ``/model``.

The local-model table is a best-effort, hand-curated snapshot (see
``_LOCAL_TIERS`` below) rather than a live lookup — there's no stable API for
"which Ollama model is good at coding right now". Update the table as new
models displace old ones; keep entries small enough to be pull-able without a
long wait, and prefer widely-benchmarked coding-tuned models.
"""

from __future__ import annotations

import functools
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    os_name: str  # "Darwin" | "Linux" | "Windows"
    ram_gb: float | None
    gpu_vendor: str | None  # "apple" | "nvidia" | "amd" | None
    vram_gb: float | None
    # True when the GPU has no memory of its own and shares the system pool:
    # Apple Silicon, and NVIDIA's Grace/Jetson-class parts. ``vram_gb`` is then
    # the unified pool rather than a discrete card's VRAM.
    unified: bool = False


@dataclass(frozen=True)
class LocalModelRec:
    tag: str  # ollama pull tag, e.g. "qwen2.5-coder:32b"
    min_gb: float  # minimum unified RAM (Apple) or VRAM (NVIDIA) / RAM (CPU) to run comfortably
    blurb: str
    # Approximate download size at the default quant, when known. The daemon
    # reports the true figure once a model is installed (`installed_sizes`);
    # this is only so the picker can warn *before* a multi-gigabyte download.
    size_gb: float = 0.0


# Ordered smallest -> largest. min_gb is the "runs comfortably at 4-bit quant"
# threshold; pick the largest entry whose min_gb fits the detected hardware.
#
# min_gb is the whole machine's memory, not the weights: a model has to share
# with the OS, the KV cache and everything else running. `gemma4:e4b` was
# listed at 8 GB on the strength of its "effective-4B" name and turned out to
# download 9.6 GB — larger than the entire machine it was being recommended
# for. Where a real measurement exists, size_gb records it.
_LOCAL_TIERS: tuple[LocalModelRec, ...] = (
    LocalModelRec("qwen3.5:2b", 4, "tiny — CPU-only laptops, fast but weak"),
    LocalModelRec("qwen3.5:4b", 8, "small — good recon/chat on 8GB machines"),
    LocalModelRec("gemma4:12b", 12, "Gemma 4 mid — strong all-rounder, big coding jump over Gemma 3"),
    LocalModelRec("qwen3.5:9b", 12, "current small-model sweet spot on 12-16GB"),
    LocalModelRec(
        "gemma4:e4b",
        16,
        "Gemma 4 effective-4B — 4B active params but ~9.6GB of weights, so it needs a 16GB machine",
        size_gb=9.6,
    ),
    LocalModelRec("devstral-small-2:24b", 24, "agent-first Mistral coder — 68% SWE-bench Verified, 384K ctx"),
    LocalModelRec("qwen3-coder:30b-a3b", 24, "MoE — 3B active params, fast agentic coding"),
    LocalModelRec("glm-4.7-flash", 24, "30B-A3B MoE — strongest 30B class, fast agentic tool use, 200K ctx"),
    LocalModelRec("qwen3.6:27b", 24, "current best dense local coder — 256K context"),
    LocalModelRec("gemma4:31b", 32, "Gemma 4 dense flagship — top-tier general + coding on 32GB+"),
    LocalModelRec("qwen3.6:35b", 32, "bigger qwen3.6 — top dense quality on 32GB+"),
    LocalModelRec("qwen3-coder-next", 64, "80B-A3B MoE, RL-trained for agents — strongest local coding, needs a big Mac/multi-GPU"),
)

# Short, dated pointer — cloud model rankings move fast; treat as a snapshot,
# not gospel. Loom's own default_config.yaml already ships sane defaults.
CLOUD_RECOMMENDATION = (
    "For the orchestrator/advisor roles, current strong picks are Claude Sonnet 5 / "
    "Opus 4.8 (great agentic tool use), the GPT-5.6 family (Sol/Terra), and Gemini "
    "3.5 Flash / 3.1 Pro. Loom defaults to Claude Sonnet + Opus; swap freely, this "
    "isn't a lock-in."
)


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _detect_ram_gb() -> float | None:
    """Best-effort total RAM in GB. POSIX via os.sysconf; None elsewhere."""
    import os

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024**3), 1)
    except (ValueError, AttributeError, OSError):
        return None


def _detect_nvidia_vram_gb() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    out = _run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    if not out:
        return None
    try:
        # Multiple GPUs: one line each, in MiB — take the largest.
        return max(float(line.strip()) for line in out.splitlines() if line.strip()) / 1024
    except ValueError:
        return None


def _detect_amd_vram_gb() -> float | None:
    """Best-effort VRAM for AMD GPUs via ROCm's ``rocm-smi`` (Ollama's AMD
    backend). Returns None if rocm-smi isn't installed or parsing fails —
    e.g. Windows AMD/Vulkan setups without ROCm tooling."""
    if not shutil.which("rocm-smi"):
        return None
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if not out:
        return None
    try:
        import json

        data = json.loads(out)
        totals = [
            float(v) for card in data.values() for k, v in card.items() if "Total Memory" in k
        ]
        return max(totals) / (1024**3) if totals else None
    except (ValueError, TypeError, AttributeError):
        return None


def _is_nvidia_unified() -> bool:
    """True on NVIDIA parts where the GPU shares system memory rather than
    carrying its own VRAM — Jetson/Tegra (Orin, Thor) and the Grace-based
    superchips (GH200, GB10/DGX Spark).

    These are exactly the boxes where ``nvidia-smi`` is missing (Tegra ships
    ``tegrastats`` instead) or reports only part of the coherent pool, so
    without this check a 128GB DGX Spark would look like a machine with no GPU
    at all and get the smallest possible context budget.
    """
    from pathlib import Path

    if Path("/etc/nv_tegra_release").exists():
        return True
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            model = Path(path).read_bytes().decode("utf-8", "ignore").lower()
        except OSError:
            continue
        if any(k in model for k in ("jetson", "tegra", "orin", "thor", "grace", "dgx spark")):
            return True
    return False


def detect_hardware() -> Hardware:
    os_name = platform.system()
    ram_gb = _detect_ram_gb()
    if os_name == "Darwin" and platform.machine() == "arm64":
        # Apple Silicon: unified memory *is* the GPU's memory pool.
        return Hardware(os_name, ram_gb, "apple", ram_gb, unified=True)
    unified = _is_nvidia_unified()
    vram_gb = _detect_nvidia_vram_gb()
    if vram_gb:
        # On a Grace-class part nvidia-smi may report less than the coherent
        # pool; take whichever number is larger so we don't undersize.
        if unified and ram_gb:
            vram_gb = max(vram_gb, ram_gb)
        return Hardware(os_name, ram_gb, "nvidia", vram_gb, unified=unified)
    if unified and ram_gb:
        # Tegra: no nvidia-smi, but the GPU can address system memory.
        return Hardware(os_name, ram_gb, "nvidia", ram_gb, unified=True)
    vram_gb = _detect_amd_vram_gb()
    if vram_gb:
        return Hardware(os_name, ram_gb, "amd", vram_gb)
    return Hardware(os_name, ram_gb, None, None)


def recommend_local_models(hw: Hardware, *, top_n: int = 4) -> list[LocalModelRec]:
    """Best-fit local models for ``hw``, largest-that-fits first.

    Falls back to the smallest tier if hardware couldn't be detected, so the
    wizard always has something concrete to suggest.
    """
    budget = hw.vram_gb or hw.ram_gb
    if budget is None:
        return [_LOCAL_TIERS[0]]
    fits = [t for t in _LOCAL_TIERS if t.min_gb <= budget]
    if not fits:
        fits = [_LOCAL_TIERS[0]]
    return list(reversed(fits))[:top_n]


def all_local_models() -> tuple[LocalModelRec, ...]:
    """The full hand-curated catalog, smallest first, with no hardware
    filtering — for pickers that want to show everything Loom knows about
    rather than just what fits the current machine. There's no stable public
    API for "every model in the Ollama library" (see module docstring), so
    this is the complete list this snapshot ships, not a live query."""
    return _LOCAL_TIERS


# A model never gets the whole machine. On unified memory the OS, the browser
# you left open and the KV cache all come out of the same pool, so treating
# "8GB of RAM" as "8GB for weights" recommends models that cannot load. A
# discrete card is closer to dedicated, so it keeps more of its budget.
_UNIFIED_USABLE = 0.65
_DISCRETE_USABLE = 0.90


def usable_budget(hw: Hardware) -> float | None:
    """Memory actually available to a model, in GB.

    Apple reports its unified pool as ``vram_gb`` too, so the vendor has to be
    checked before that figure is treated as a dedicated card's.
    """
    shared = hw.unified or hw.gpu_vendor == "apple"
    pool = hw.vram_gb or hw.ram_gb
    if not pool:
        return None
    return pool * (_UNIFIED_USABLE if shared else _DISCRETE_USABLE)


def fits_hardware(hw: Hardware, model: LocalModelRec) -> bool:
    """True if ``model`` comfortably fits the detected VRAM/RAM budget.

    Two ways to fail, and the old check applied neither strictly enough:
    the machine has to meet the model's stated minimum, *and* — where the
    download size is known — the weights have to fit in what is actually
    usable. ``min_gb <= budget`` alone let an 8 GB entry pass on an 8 GB
    machine with nothing left for the OS.
    """
    budget = hw.vram_gb or hw.ram_gb
    if budget is None or model.min_gb > budget:
        return False
    if model.size_gb:
        usable = usable_budget(hw)
        if usable is not None and model.size_gb > usable:
            return False
    return True


# GPU-addressable memory (GB) -> how big a context window to let a local model
# claim, in tokens. A KV cache for these GQA-era coding models runs roughly
# 100-200 KB per token, so the rungs below hand it about a quarter of the pool
# and leave the rest for weights: 16GB -> 32K costs ~4GB of cache, 48GB -> 128K
# costs ~12GB. Ollama degrades by spilling rather than crashing if a specific
# model is heavier than the estimate, and any explicit `context_windows` entry
# bypasses this entirely.
_CONTEXT_TIERS: tuple[tuple[float, int], ...] = (
    (6, 8_192),
    (10, 16_384),
    (20, 32_768),
    (40, 65_536),
    (80, 131_072),
)
_MAX_CONTEXT = 262_144
# No GPU and no readable RAM figure: assume a modest laptop rather than
# guessing high, since guessing high is what allocates memory that isn't there.
_UNKNOWN_CONTEXT = 16_384


def context_budget(hw: Hardware) -> int:
    """Largest context window this machine should let a local model claim.

    Covers discrete VRAM (NVIDIA, AMD) and unified memory (Apple Silicon,
    NVIDIA Grace/Jetson) identically — :func:`detect_hardware` has already
    normalized both into ``vram_gb``.
    """
    budget = hw.vram_gb or hw.ram_gb
    if budget is None:
        return _UNKNOWN_CONTEXT
    for ceiling, window in _CONTEXT_TIERS:
        if budget < ceiling:
            return window
    return _MAX_CONTEXT


@functools.lru_cache(maxsize=1)
def auto_context_budget() -> int:
    """:func:`context_budget` for the current machine, probed once per process
    (hardware detection shells out to nvidia-smi/rocm-smi)."""
    return context_budget(detect_hardware())


def hardware_summary(hw: Hardware) -> str:
    shared = hw.unified or hw.gpu_vendor == "apple"
    mem = f"{hw.vram_gb:.0f}GB unified memory" if hw.vram_gb and shared else (
        f"{hw.vram_gb:.0f}GB VRAM" if hw.vram_gb and hw.gpu_vendor in ("nvidia", "amd") else
        (f"{hw.ram_gb:.0f}GB RAM" if hw.ram_gb else "unknown memory")
    )
    gpu = {"apple": "Apple Silicon", "nvidia": "NVIDIA GPU", "amd": "AMD GPU"}.get(hw.gpu_vendor or "", "CPU only")
    return f"{hw.os_name} · {gpu} · {mem}"
