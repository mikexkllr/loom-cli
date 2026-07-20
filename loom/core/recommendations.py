"""Hardware-aware local model recommendations, plus a pointer to good cloud
coding models — used by the onboarding wizard and ``/model``.

The local-model table is a best-effort, hand-curated snapshot (see
``_LOCAL_TIERS`` below) rather than a live lookup — there's no stable API for
"which Ollama model is good at coding right now". Update the table as new
models displace old ones; keep entries small enough to be pull-able without a
long wait, and prefer widely-benchmarked coding-tuned models.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class LocalModelRec:
    tag: str  # ollama pull tag, e.g. "qwen2.5-coder:32b"
    min_gb: float  # minimum unified RAM (Apple) or VRAM (NVIDIA) / RAM (CPU) to run comfortably
    blurb: str


# Ordered smallest -> largest. min_gb is the "runs comfortably at 4-bit quant"
# threshold; pick the largest entry whose min_gb fits the detected hardware.
_LOCAL_TIERS: tuple[LocalModelRec, ...] = (
    LocalModelRec("qwen3.5:2b", 4, "tiny — CPU-only laptops, fast but weak"),
    LocalModelRec("qwen3.5:4b", 8, "small — good recon/chat on 8GB machines"),
    LocalModelRec("qwen3.5:9b", 12, "current small-model sweet spot on 12-16GB"),
    LocalModelRec("devstral-small-2:24b", 24, "agent-first Mistral coder — 68% SWE-bench Verified, 384K ctx"),
    LocalModelRec("qwen3-coder:30b-a3b", 24, "MoE — 3B active params, fast agentic coding"),
    LocalModelRec("qwen3.6:27b", 24, "current best dense local coder — 256K context"),
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


def detect_hardware() -> Hardware:
    os_name = platform.system()
    ram_gb = _detect_ram_gb()
    if os_name == "Darwin" and platform.machine() == "arm64":
        # Apple Silicon: unified memory *is* the GPU's memory pool.
        return Hardware(os_name, ram_gb, "apple", ram_gb)
    vram_gb = _detect_nvidia_vram_gb()
    if vram_gb:
        return Hardware(os_name, ram_gb, "nvidia", vram_gb)
    vram_gb = _detect_amd_vram_gb()
    if vram_gb:
        return Hardware(os_name, ram_gb, "amd", vram_gb)
    return Hardware(os_name, ram_gb, None, None)


def recommend_local_models(hw: Hardware, *, top_n: int = 3) -> list[LocalModelRec]:
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


def hardware_summary(hw: Hardware) -> str:
    mem = f"{hw.vram_gb:.0f}GB VRAM" if hw.vram_gb and hw.gpu_vendor in ("nvidia", "amd") else (
        f"{hw.ram_gb:.0f}GB unified memory" if hw.ram_gb and hw.gpu_vendor == "apple" else
        (f"{hw.ram_gb:.0f}GB RAM" if hw.ram_gb else "unknown memory")
    )
    gpu = {"apple": "Apple Silicon", "nvidia": "NVIDIA GPU", "amd": "AMD GPU"}.get(hw.gpu_vendor or "", "CPU only")
    return f"{hw.os_name} · {gpu} · {mem}"
