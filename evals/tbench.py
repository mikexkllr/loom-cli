#!/usr/bin/env python3
"""Run Loom (and reference harnesses) on Terminal-Bench 2.1 via Harbor.

    uv tool install harbor                       # once; needs Docker running
    python3 evals/tbench.py build                # wheel + pinned requirements
    python3 evals/tbench.py run loom-hybrid --tasks pilot -n 2
    python3 evals/tbench.py run terminus-argon --tasks pilot -n 4
    python3 evals/tbench.py report               # → evals/results/REPORT.md

Configurations (see CONFIGS): the same models in different harnesses, so the
harness is the only variable. API keys come from the environment, or from the
``env`` block of ~/.loom/settings.json — they are passed to Harbor's process
environment only and never printed or written to job files.

Harbor job output lands in evals/jobs/ (gitignored, large). ``report``
condenses it into evals/results/, which is committed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
BUILD = EVALS / ".build"
JOBS = EVALS / "jobs"
RESULTS = EVALS / "results"
DATASET = "terminal-bench/terminal-bench-2-1"
UV_VERSION = "0.9.5"

ORCH = "google/gemini-4-argon"
LOCAL = "ollama/qwen3.8:27b-mlx"

# name → (harbor agent, model, extra harbor args, description)
CONFIGS: dict[str, tuple[str, str, list[str], str]] = {
    "loom-hybrid": (
        "evals.loom_agent:LoomAgent", ORCH, ["--ak", f"subagent_model={LOCAL}"],
        "Loom as configured: Gemini 4 Argon orchestrator, Qwen3.8-27B (local, MLX) subagents",
    ),
    "loom-cloud": (
        "evals.loom_agent:LoomAgent", ORCH, ["--ak", "subagent_model=google:gemini-4-argon"],
        "Loom with every role on Gemini 4 Argon (isolates the cost of local routing)",
    ),
    "terminus-argon": (
        "terminus-2", "gemini/gemini-4-argon", [],
        "Terminus-2 (Terminal-Bench reference harness) on Gemini 4 Argon",
    ),
    "terminus-qwen": (
        "terminus-2", "ollama_chat/qwen3.8:27b-mlx",
        ["--ak", "api_base=http://localhost:11434",
         "--ak", 'llm_call_kwargs={"num_ctx": 131072}'],
        "Terminus-2 on the same local Qwen3.8-27B (Qwen's self-reported harness)",
    ),
}

# Tasks whose *grader* cannot run here, for any agent — excluded from every
# config's score and listed in the report.
BROKEN_TASKS = {
    "qemu-alpine-ssh": "tests/test.sh apt-installs curl; the Debian security mirror 404s "
    "those package versions (2026-11-19), so the grader dies before testing anything",
}

# Published numbers for the same models (fetched 2026-11-19; see README.md).
PUBLISHED = [
    ("Vals AI harness", "Gemini 4 Argon", "TB 2.1, 89 tasks", 57.7, "independent"),
    ("Vals AI harness", "Qwen3.8-27B", "TB 2.1, 89 tasks", 58.4, "independent"),
    ("Terminus", "Qwen3.8-27B", "TB 2.1, 89 tasks", 73.0, "self-reported by Qwen"),
    ("Antigravity CLI", "Gemini 4 Argon (high)", "TB 4.1, 66 tasks", 55.5, "official leaderboard"),
]


# --------------------------------------------------------------------- helpers


def _dataset_dir() -> Path:
    """Local copy of the dataset (task.toml metadata is needed for subsets)."""
    cache = EVALS / ".dataset"
    target = cache / DATASET.split("/")[-1]
    if not target.exists():
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run(["harbor", "download", DATASET, "-o", str(cache)], check=True)
    return target


def _tasks_meta() -> dict[str, dict]:
    out = {}
    for f in sorted(_dataset_dir().glob("*/task.toml")):
        d = tomllib.loads(f.read_text())
        out[f.parent.name] = {
            "difficulty": d["metadata"]["difficulty"],
            "category": d["metadata"].get("category", ""),
            "timeout": d["agent"]["timeout_sec"],
        }
    return out


def pilot_subset(n_easy: int = 1, n_medium: int = 7, n_hard: int = 4, seed: int = 2026) -> list[str]:
    """Seeded, difficulty-stratified sample roughly proportional to the full set
    (4 easy / 55 medium / 30 hard). Fixed seed: every config gets the same tasks."""
    by: dict[str, list[str]] = defaultdict(list)
    for name, meta in _tasks_meta().items():
        by[meta["difficulty"]].append(name)
    rng = random.Random(seed)
    return (
        rng.sample(sorted(by["easy"]), n_easy)
        + rng.sample(sorted(by["medium"]), n_medium)
        + rng.sample(sorted(by["hard"]), n_hard)
    )


def _load_keys() -> dict[str, str]:
    env = {}
    try:
        block = json.loads((Path.home() / ".loom" / "settings.json").read_text()).get("env", {})
    except (OSError, ValueError):
        block = {}
    for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(key):
            env[key] = os.environ[key]
        elif block.get(key):
            env[key] = block[key]
    # LiteLLM (Terminus-2) reads GEMINI_API_KEY; Loom reads GOOGLE_API_KEY.
    if "GOOGLE_API_KEY" in env:
        env.setdefault("GEMINI_API_KEY", env["GOOGLE_API_KEY"])
    return env


# -------------------------------------------------------------------- commands


def _fetch_uv(out: Path) -> None:
    """Static uv for each container architecture, uploaded by the adapter so a
    task image needs neither curl nor a working package mirror."""
    for arch in ("x86_64", "aarch64"):
        if (out / f"uv-{arch}").exists():
            continue
        url = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-{arch}-unknown-linux-musl.tar.gz"
        with tempfile.TemporaryDirectory() as tmp:
            tgz = Path(tmp) / "uv.tgz"
            subprocess.run(["curl", "-fsSL", "-o", str(tgz), url], check=True)
            with tarfile.open(tgz) as tf:
                member = next(m for m in tf.getmembers() if m.name.endswith("/uv"))
                (out / f"uv-{arch}").write_bytes(tf.extractfile(member).read())


def cmd_build(args) -> int:
    out = Path(getattr(args, "out", None) or BUILD).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    subprocess.run(["uv", "build", "--wheel", "-o", str(out)], cwd=ROOT, check=True)
    subprocess.run(
        ["uv", "export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project", "-q",
         "-o", str(out / "requirements.txt")],
        cwd=ROOT, check=True,
    )
    _fetch_uv(out)
    print(f"built → {out}")
    return 0


def cmd_run(args) -> int:
    agent, model, extra, _ = CONFIGS[args.config]
    if args.tasks == "pilot":
        tasks = pilot_subset()
    elif args.tasks == "all":
        tasks = []
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    scope = args.tasks if args.tasks in ("pilot", "all") else "custom"
    label = f"{args.config}-{args.tag}" if args.tag else args.config
    job = args.job_name or f"{label}-{scope}"
    cmd = [
        "harbor", "run", "-p", str(_dataset_dir()), "-a", agent, "-m", model, *extra,
        "-o", str(JOBS), "--job-name", job, "-n", str(args.n), "-y", "-q",
    ]
    if agent.startswith("evals."):
        wheel_dir = Path(args.wheel_dir).resolve() if args.wheel_dir else BUILD
        if not (wheel_dir / "requirements.txt").exists():
            raise SystemExit(f"no build in {wheel_dir} — run `tbench.py build --out {wheel_dir}`")
        _fetch_uv(wheel_dir)
        cmd += ["--ak", f"wheel_dir={wheel_dir}"]
    for t in tasks:
        cmd += ["-i", t]
    env = {**os.environ, **_load_keys(), "PYTHONPATH": str(ROOT)}
    print(f"▸ {args.config}: {len(tasks) or 'all'} tasks → {JOBS / job}", flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def _trials(job_dir: Path):
    for trial in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        res = trial / "result.json"
        if not res.exists():
            continue
        yield trial, json.loads(res.read_text())


def _reward(result: dict) -> float | None:
    vr = result.get("verifier_result") or {}
    rewards = vr.get("rewards") or {}
    if "reward" in rewards:
        return float(rewards["reward"])
    return None


def _loom_cost(trial: Path) -> float | None:
    """Cloud USD for one Loom trial from its per-call usage log, priced with
    Loom's own table. Covers trials the timeout killed before Loom printed a
    receipt, and reprices old runs when the table gains a model."""
    log = trial / "agent" / "usage.jsonl"
    if not log.exists():
        return None
    sys.path.insert(0, str(ROOT))
    from loom.core.usage import cost_usd

    total = 0.0
    for line in log.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("event") == "llm" and not rec.get("local"):
            total += cost_usd(rec.get("model") or "", rec.get("input") or 0, rec.get("output") or 0,
                              cache_read_tokens=rec.get("cache_read") or 0,
                              cache_write_tokens=rec.get("cache_write") or 0)
    return round(total, 4)


def collect() -> dict[str, dict[str, dict]]:
    """config → task → row, from every job directory under evals/jobs."""
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    jobs = sorted(JOBS.iterdir()) if JOBS.exists() else []
    # Retries last, so they can replace the rows they were rerun for.
    jobs.sort(key=lambda d: d.name.endswith("-retry"))
    for job_dir in jobs:
        if not job_dir.is_dir() or job_dir.name.startswith("debug"):
            continue
        # Job dirs are "<config>[-<tag>]-<scope>"; the report row is everything
        # before the scope, so tagged runs (e.g. "loom-cloud-fixed") get their own row.
        config = job_dir.name.rsplit("-", 1)[0] if "-" in job_dir.name else job_dir.name
        retry = job_dir.name.endswith("-retry")
        for trial, res in _trials(job_dir):
            task = res.get("task_name", trial.name.split("__")[0]).split("/")[-1]
            # A retry only replaces a trial that never got graded (infra
            # failure before or during setup) — never a real pass or fail.
            if retry and task in rows[config] and rows[config][task]["reward"] is not None:
                continue
            ctx = (res.get("agent_result") or {})
            exc = res.get("exception_info") or {}
            phase = res.get("agent_execution") or {}
            started, finished = phase.get("started_at"), phase.get("finished_at")
            secs = None
            if started and finished:
                from datetime import datetime

                secs = (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
            rows[config][task] = {
                "reward": _reward(res),
                "exception": exc.get("exception_type"),
                "cost": _loom_cost(trial) if (trial / "agent" / "usage.jsonl").exists()
                else ctx.get("cost_usd"),
                "in": ctx.get("n_input_tokens"),
                "out": ctx.get("n_output_tokens"),
                "meta": ctx.get("metadata") or {},
                "seconds": secs,
                "trial": str(trial.relative_to(EVALS)),
            }
    return rows


def cmd_report(_args) -> int:
    rows = collect()
    meta = _tasks_meta()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    lines = [f"# Terminal-Bench 2.1 — Loom vs reference harnesses ({date.today()})", ""]
    lines += ["| Config | Graded | Passed | Pass rate | Hit timeout | Ungraded (infra) | "
              "Cloud cost | Median agent time |",
              "|---|---|---|---|---|---|---|---|"]
    for config, tasks in rows.items():
        tasks = {t: r for t, r in tasks.items() if t not in BROKEN_TASKS}
        graded = [r for r in tasks.values() if r["reward"] is not None]
        n = len(graded)
        passed = sum(1 for r in graded if r["reward"] >= 1)
        timeouts = sum(1 for r in graded if r["exception"] and "Timeout" in r["exception"])
        cost = sum(r["cost"] or 0 for r in tasks.values())
        times = sorted(r["seconds"] for r in graded if r["seconds"])
        med = times[len(times) // 2] / 60 if times else 0
        rate = f"{passed / n:.0%}" if n else "—"
        lines.append(f"| {config} | {n} | {passed} | {rate} | {timeouts} | {len(tasks) - n} | "
                     f"${cost:.2f} | {med:.1f} min |")
    lines += ["", "Loom cost: every cloud call in the trial's usage log, priced with Loom's "
              "table (Gemini 4 Argon at list $4/$20 per MTok, cached input 95% off) — "
              "includes trials the timeout killed. Terminus cost: LiteLLM's own estimate, "
              "which omits Gemini's thinking tokens.", ""]
    lines += [f"- Excluded `{t}`: {why}" for t, why in BROKEN_TASKS.items()]
    lines += ["", "## Per task", ""]
    configs = list(rows)
    lines.append("| Task | Difficulty | " + " | ".join(configs) + " |")
    lines.append("|---|---|" + "---|" * len(configs))
    for task in sorted({t for c in rows.values() for t in c}):
        cells = []
        for c in configs:
            r = rows[c].get(task)
            if r is None:
                cells.append("—")
                continue
            mark = "⚠️" if r["reward"] is None else ("✅" if r["reward"] >= 1 else "❌")
            note = f" {r['exception']}" if r["exception"] else ""
            mins = f" {r['seconds'] / 60:.0f}m" if r["seconds"] else ""
            cells.append(f"{mark}{mins}{note}")
        lines.append(f"| {task} | {meta.get(task, {}).get('difficulty', '')} | " + " | ".join(cells) + " |")
    lines += ["", "## Published numbers, same models", "",
              "| Harness | Model | Benchmark | Score | Source type |", "|---|---|---|---|---|"]
    for h, m, b, s, t in PUBLISHED:
        lines.append(f"| {h} | {m} | {b} | {s}% | {t} |")
    (RESULTS / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", help="build dir (default evals/.build)")
    r = sub.add_parser("run")
    r.add_argument("config", choices=list(CONFIGS))
    r.add_argument("--tasks", default="pilot", help="pilot | all | comma-separated task names")
    r.add_argument("-n", type=int, default=2, help="concurrent trials")
    r.add_argument("--job-name")
    r.add_argument("--tag", help="suffix for the job/report row, e.g. 'fixed'")
    r.add_argument("--wheel-dir", help="Loom build to install (default evals/.build)")
    sub.add_parser("report")
    sub.add_parser("pilot")
    args = p.parse_args()
    if args.cmd == "pilot":
        print("\n".join(pilot_subset()))
        return 0
    return {"build": cmd_build, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
