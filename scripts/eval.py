#!/usr/bin/env python3
"""Loom eval harness — measure task success, cost, and local/cloud routing.

    python scripts/eval.py                 # run all tasks in evals/tasks.yaml
    python scripts/eval.py --only add-function
    python scripts/eval.py --airgap        # run the fleet in airgap mode

For each task: copy its fixture to a temp dir, run the Loom orchestrator on
the prompt (auto-approved, headless), run the task's `check` command, and
record pass/fail, wall time, tokens, cloud cost, and escalation count. A
markdown report lands in evals/report.md — publish the numbers.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVALS = ROOT / "evals"


def run_task(task: dict, *, airgap: bool, timeout: int) -> dict:
    from loom.core import settings as settings_mod
    from loom.core.usage import UsageTracker
    from loom.middleware import policy
    from loom.tools import sandbox

    workdir = Path(tempfile.mkdtemp(prefix=f"loom-eval-{task['name']}-"))
    shutil.copytree(EVALS / task["fixture"], workdir, dirs_exist_ok=True)

    sandbox.set_root(workdir)
    policy.auto_approve.set(True)
    settings = settings_mod.load_settings(workdir)
    tracker = UsageTracker(settings.models)

    result = {"name": task["name"], "passed": False, "error": "", "seconds": 0.0}
    start = time.monotonic()
    try:
        from loom.core.orchestrator import build_orchestrator

        bundle = build_orchestrator(settings, airgap=airgap, cwd=str(workdir))
        bundle.agent.invoke(
            {"messages": [("user", task["prompt"])]},
            config={"callbacks": [tracker], "recursion_limit": 80},
        )
    except Exception as exc:  # noqa: BLE001 — record and move on
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["seconds"] = time.monotonic() - start

    if not result["error"]:
        check = subprocess.run(
            task["check"], shell=True, cwd=workdir, capture_output=True, text=True, timeout=timeout
        )
        result["passed"] = check.returncode == 0
        if not result["passed"]:
            result["error"] = (check.stdout + check.stderr)[-500:]

    u = tracker.session
    ci, co = u.tokens(u.cloud)
    li, lo = u.tokens(u.local)
    result.update(
        cloud_in=ci, cloud_out=co, local_in=li, local_out=lo,
        cloud_cost=round(u.cloud_cost, 4),
        all_cloud_est=round(u.all_cloud_estimate(settings.models.orchestrator), 4),
        workdir=str(workdir),
    )
    return result


def write_report(results: list[dict], path: Path) -> None:
    lines = [
        "# Loom eval report",
        "",
        "| Task | Pass | Time | Cloud tokens | Local tokens | Cloud cost | All-cloud est. |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {'✅' if r['passed'] else '❌'} | {r['seconds']:.0f}s "
            f"| {r['cloud_in'] + r['cloud_out']:,} | {r['local_in'] + r['local_out']:,} "
            f"| ${r['cloud_cost']:.3f} | ${r['all_cloud_est']:.3f} |"
        )
    passed = sum(r["passed"] for r in results)
    total_cost = sum(r["cloud_cost"] for r in results)
    total_est = sum(r["all_cloud_est"] for r in results)
    saving = (1 - total_cost / total_est) * 100 if total_est else 0.0
    lines += [
        "",
        f"**{passed}/{len(results)} passed** · cloud spend ${total_cost:.3f} vs all-cloud est. "
        f"${total_est:.3f} ({saving:.0f}% saved by local routing)",
        "",
    ]
    for r in results:
        if r["error"]:
            lines += [f"### {r['name']} — error/output", "```", r["error"], "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Run a single task by name")
    parser.add_argument("--airgap", action="store_true", help="Run in airgap mode")
    parser.add_argument("--timeout", type=int, default=120, help="Check-command timeout (s)")
    args = parser.parse_args()

    tasks = yaml.safe_load((EVALS / "tasks.yaml").read_text())
    if args.only:
        tasks = [t for t in tasks if t["name"] == args.only]
        if not tasks:
            print(f"no task named {args.only!r}", file=sys.stderr)
            return 2

    results = []
    for task in tasks:
        print(f"▸ {task['name']} …", flush=True)
        r = run_task(task, airgap=args.airgap, timeout=args.timeout)
        status = "PASS" if r["passed"] else f"FAIL ({r['error'][:80]})"
        print(f"  {status} · {r['seconds']:.0f}s · ${r['cloud_cost']:.3f} cloud")
        results.append(r)

    report = EVALS / "report.md"
    write_report(results, report)
    print(f"\nreport → {report}")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
