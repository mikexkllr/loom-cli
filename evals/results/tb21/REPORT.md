# Terminal-Bench 2.1 — Loom vs reference harnesses (2026-11-19)

| Config | Graded | Passed | Pass rate | Hit timeout | Ungraded (infra) | Cloud cost | Median agent time |
|---|---|---|---|---|---|---|---|
| loom-cloud-fixed | 10 | 8 | 80% | 3 | 1 | $10.53 | 15.0 min |
| loom-cloud | 11 | 7 | 64% | 3 | 0 | $8.91 | 8.1 min |
| loom-hybrid | 11 | 3 | 27% | 8 | 0 | $0.83 | 15.0 min |
| terminus-argon | 11 | 3 | 27% | 7 | 0 | $0.55 | 15.0 min |

Loom cost: every cloud call in the trial's usage log, priced with Loom's table (Gemini 4 Argon at list $4/$20 per MTok, cached input 95% off) — includes trials the timeout killed. Terminus cost: LiteLLM's own estimate, which omits Gemini's thinking tokens.

- Excluded `qemu-alpine-ssh`: tests/test.sh apt-installs curl; the Debian security mirror 404s those package versions (2026-11-19), so the grader dies before testing anything

## Per task

| Task | Difficulty | loom-cloud-fixed | loom-cloud | loom-hybrid | terminus-argon |
|---|---|---|---|---|---|
| chess-best-move | medium | ✅ 4m | ✅ 4m | ✅ 6m | ❌ 15m AgentTimeoutError |
| cobol-modernization | easy | ✅ 15m AgentTimeoutError | ✅ 15m AgentTimeoutError | ❌ 15m AgentTimeoutError | ❌ 15m AgentTimeoutError |
| distribution-search | medium | ✅ 4m | ✅ 5m | ✅ 60m AgentTimeoutError | ✅ 14m |
| git-leak-recovery | medium | ✅ 3m | ✅ 8m | ✅ 7m | ✅ 11m |
| make-doom-for-mips | hard | ❌ 15m AgentTimeoutError | ❌ 15m AgentTimeoutError | ❌ 15m AgentTimeoutError | ❌ 15m AgentTimeoutError |
| mteb-retrieve | medium | ✅ 26m | ❌ 9m | ❌ 30m AgentTimeoutError | ❌ 30m AgentTimeoutError |
| password-recovery | hard | ✅ 5m | ✅ 7m | ❌ 15m AgentTimeoutError | ❌ 15m AgentTimeoutError |
| path-tracing-reverse | hard | ✅ 28m | ✅ 11m | ❌ 30m AgentTimeoutError | ❌ 8m |
| pypi-server | medium | ✅ 4m | ✅ 7m | ❌ 15m AgentTimeoutError | ✅ 3m |
| qemu-alpine-ssh | medium | ⚠️ NonZeroAgentExitCodeError | ❌ 2m | ⚠️ NonZeroAgentExitCodeError | ❌ 15m AgentTimeoutError |
| torch-pipeline-parallelism | hard | ❌ 15m AgentTimeoutError | ❌ 15m AgentTimeoutError | ❌ 178m AgentTimeoutError | ❌ 15m AgentTimeoutError |
| vulnerable-secret | medium | ⚠️ NetworkConnectionError | ❌ 0m | ❌ 0m | ❌ 15m AgentTimeoutError |

## Published numbers, same models

| Harness | Model | Benchmark | Score | Source type |
|---|---|---|---|---|
| Vals AI harness | Gemini 4 Argon | TB 2.1, 89 tasks | 57.7% | independent |
| Vals AI harness | Qwen3.8-27B | TB 2.1, 89 tasks | 58.4% | independent |
| Terminus | Qwen3.8-27B | TB 2.1, 89 tasks | 73.0% | self-reported by Qwen |
| Antigravity CLI | Gemini 4 Argon (high) | TB 4.1, 66 tasks | 55.5% | official leaderboard |
