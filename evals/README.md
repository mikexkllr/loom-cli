# Loom evals — Terminal-Bench 2.1 and SWE-bench Pro

Loom is measured on **[Terminal-Bench 2.1](https://www.tbench.ai/)**: 89 tasks
(4 easy / 55 medium / 30 hard) in Docker containers, graded by hidden tests
that the agent never sees. It is the benchmark the coding-agent leaderboards
use to compare *harnesses* (Claude Code, Codex CLI, Gemini CLI, OpenCode,
Terminus …), not only models, so Loom is scored the same way they are: the CLI is
installed into each task container, handed the task text, and left alone until
it exits or the task's agent timeout fires (official timeouts, no multiplier).

## Files

| File | What |
|---|---|
| `loom_agent.py` | Harbor installed-agent adapter. Builds nothing itself: installs the wheel from `.build/` with versions pinned to `uv.lock`, writes a `LOOM_HOME/settings.json` with the model routing, runs `loom --yolo --root $(pwd) "<task>"`, and parses `LOOM_USAGE_LOG` for per-role tokens and Loom's own cost receipt. |
| `tbench.py` | Runner: `build`, `run <config>`, `report`, `pilot`. |
| `results/<bench>/` | Committed: `REPORT.md` (tables) and `results.json` (per-trial rows), one folder per benchmark (`tb21`, `swepro`). |
| `jobs/`, `.build/`, `.dataset/` | Gitignored Harbor output, wheel, downloaded tasks. |

## Running

```bash
uv tool install harbor          # once; Docker Desktop must be running
python3 evals/tbench.py build   # after any code change
python3 evals/tbench.py run loom-hybrid --tasks pilot -n 2
python3 evals/tbench.py run terminus-argon --tasks pilot -n 2
python3 evals/tbench.py report
```

`--bench swepro` (before the subcommand) switches every command to SWE-bench Pro
(`scale-ai/swe-bench-pro`, 731 tasks, 50-minute limit); its pilot is one task per
repository, same seed. `mini-argon` runs mini-swe-agent, SWE-bench's reference
harness, on the same model:

```bash
python3 evals/tbench.py --bench swepro run loom-cloud --tasks pilot -n 1
python3 evals/tbench.py --bench swepro run mini-argon --tasks pilot -n 1
python3 evals/tbench.py --bench swepro report
```

`--tasks` takes `pilot` (seeded 12-task stratified sample: 1 easy, 7 medium, 4
hard, `seed=2026`), `all`, or a comma-separated list of task names.

Keys: `GOOGLE_API_KEY` (and others) are read from the environment, falling back
to the `env` block of `~/.loom/settings.json`. They are passed to Harbor's
process environment and to each `loom` exec. They are never written into the container.

Local models: Ollama stays on the host. Containers reach it at
`http://host.docker.internal:11434`, which works with Docker Desktop and an Ollama bound to
localhost. `num_ctx` is pinned to 131072. Auto-detection inside a 2 GB
container would otherwise size the KV cache from the container, not the Mac.

## Configurations

| Config | Harness | Orchestrator | Subagents / worker |
|---|---|---|---|
| `loom-hybrid` | Loom | Gemini 4 Argon | Qwen3.8-27B local (MLX, nvfp4) |
| `loom-cloud` | Loom | Gemini 4 Argon | Gemini 4 Argon |
| `terminus-argon` | Terminus-2 | Gemini 4 Argon | — (single agent) |
| `terminus-qwen` | Terminus-2 | Qwen3.8-27B local | — (single agent) |
| `mini-argon` | mini-swe-agent | Gemini 4 Argon | — (single agent) |

Same models, different harness. The harness is the variable being measured.

## Published numbers for the same models

Fetched 2026-11-19.

| Harness | Model | Benchmark | Score | Type |
|---|---|---|---|---|
| Vals AI | Gemini 4 Argon | TB 2.1 (89) | 57.7% | independent |
| Vals AI | Qwen3.8-27B | TB 2.1 (89) | 58.4% | independent |
| Terminus | Qwen3.8-27B | TB 2.1 (89) | 73.0% | self-reported (Qwen model card) |
| Antigravity CLI | Gemini 4 Argon (high) | TB 4.1 (66) | 55.5% ±2.8 | official leaderboard |

Sources: [tbench.ai leaderboard](https://www.tbench.ai/leaderboard),
[BenchLM TB2 mirror (Vals runs)](https://benchlm.ai/benchmarks/terminal-bench-2),
[MorphLLM Qwen3.8 page](https://www.morphllm.com/qwen-3-8).

The official tbench.ai board has no 2.0 or 2.1 rows for either model. 2.1
numbers are Vals runs or vendor reports. On 12 pilot tasks one task is
±8 percentage points and the 95% interval is roughly ±28 pp. Compare pilot configs **with each
other** on identical tasks, and against the published full-set numbers only
for direction.

## Caveats

- Docker Desktop here has 8 GB of RAM, so concurrency stays at 2 per config.
  Contention adds wall time, which matters because tasks are timeout-bound.
- Images are run as-is on Apple Silicon (aarch64 Docker). Leaderboard runs
  are usually x86 cloud VMs.
- The local model is shared by every concurrent Loom trial through one Ollama
  daemon, so local throughput is the hybrid config's bottleneck.
