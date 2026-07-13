# Changelog

## 0.2.0 — 2026-07-13

### Added
- **Playwright MCP + tester subagent** — persistent MCP sessions on a
  background event loop; the tester drives a real browser through user
  journeys and the orchestrator must verify frontend-visible changes
  end-to-end before declaring done. Evidence reports land in
  `.loom/verifications/`.
- **Claude Code-style REPL** — `⏺`/`⎿` rendering, token-level streaming,
  `>` prompt, compact banner, and the standard slash-command set
  (`/status /mcp /compact /cost /doctor /init /memory /export /hooks
  /vim /theme /resume /undo /airgap` + `/config` alias).
- **Cost receipts** — per-model token tracking (local vs cloud) with an
  all-cloud comparison after every turn; `/cost` per-model breakdown.
- **Model picker** — `/model` shows every role; `/model <role>` picks
  interactively from installed Ollama models; `/model <role> <model>` sets.
- **Resume** — SQLite-backed session persistence (`.loom/sessions.db`)
  with `/resume`.
- **Git safety** — unified-diff previews on write approvals; per-turn
  snapshots with `/undo`.
- **Context ergonomics** — `@file` mentions, compact repo map, and the
  project memory file (`LOOM.md`/`CLAUDE.md`/`AGENTS.md`) sent on the
  first turn.
- **Airgap mode** (`--airgap` / `/airgap`) — raw code never reaches the
  cloud: local subagents read files, the cloud orchestrator plans from
  summaries, escalation disabled.
- **Cloud fallback** — when Ollama is unavailable, local roles run on
  `cloud_fallback` (default `claude-haiku-4-5`) with a loud warning;
  `--local-only`/`--airgap` fail fast instead.
- **`loom doctor`** and an eval harness (`scripts/eval.py` + `evals/`).

### Fixed
- Root `[PROMPT]` argument no longer swallows subcommands
  (`loom models status` previously ran as a free-form task).
- Bare permission rules are now tool-name globs (`browser_*`).

## 0.1.0

Initial release: hybrid local/cloud orchestrator + six subagents,
layered settings.json, permissions, hooks, interactive REPL.
