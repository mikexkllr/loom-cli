# Loom — Agent Guide

Loom (repo `dual_mind_ai`, GitHub `mikexkllr/loom-cli`) is a hybrid local/cloud multi-agent CLI coding assistant built on deepagents/LangGraph. A strong cloud orchestrator plans and routes; specialized subagents — mostly local Ollama models — handle isolated, bounded subtasks in their own context windows and return only summaries.

## Toolchain

- Tests: `.venv/bin/python -m pytest -q` — bare `python`/`pytest` are NOT on PATH; always use the venv interpreter.
- Lint: `ruff check .` — smoke: `scripts/smoke.py`.
- CI runs `uv sync --locked`, `ruff check .`, `pytest -q`, `scripts/smoke.py` — commit `uv.lock` with any `pyproject.toml` change.
- deepagents is pinned `>=0.7,<0.8`. For API questions beyond what the venv install shows, check upstream source (github.com/langchain-ai/deepagents, `libs/deepagents/deepagents/`), not a local install.

## Working agreements

- Commit and push to `main` directly (established flow).
- The main dev machine cannot run local Ollama models — never assume Ollama is available; cloud-fallback paths must always work without it.
- Loom's own UI must stay Claude Code/opencode-style (⏺/⎿ bullets, `>` prompt, Claude Code slash commands).
- Product strategy: differentiate on cost receipts, airgap privacy, and mandatory browser-verified E2E testing — not feature parity with Claude Code.

## Memories

Durable, hard-won knowledge lives in `memories/` (migrated from Claude Code's memory store on 2026-08-03; gitignored — local dev notes, not committed). Read the relevant file before working in its area:

| File | Read when |
|---|---|
| [memories/loom-project-context.md](memories/loom-project-context.md) | Any work in this repo — device constraints, deepagents 0.7 gotchas |
| [memories/loom-enforce-routing-structurally.md](memories/loom-enforce-routing-structurally.md) | "Make the orchestrator/subagent stop doing X" — tool allowlists, exclusion, read budget |
| [memories/loom-cloud-fallback-needs-anthropic-key.md](memories/loom-cloud-fallback-needs-anthropic-key.md) | Provider/auth failures, OpenCode Zen/Go credentials, free models, Go region gates |
| [memories/loom-test-end-to-end.md](memories/loom-test-end-to-end.md) | Testing Loom — why unit tests miss real bugs, the four test layers incl. pty REPL testing |
| [memories/loom-approval-prompts-race-the-renderer.md](memories/loom-approval-prompts-race-the-renderer.md) | Anything that prints during a turn — tool calls run on LangGraph worker threads, so prompts race the stream |
| [memories/loom-privacy-telemetry.md](memories/loom-privacy-telemetry.md) | Privacy modes, Sentry/Langfuse telemetry — consent gates, fail-closed invariants, scrub rules |

When you learn something durable about this project, update the matching memory file (and add a row here if you create a new one).
