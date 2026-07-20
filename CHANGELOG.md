# Changelog

## Unreleased

### Changed
- **Setup wizard now starts from your existing config.** Re-running
  `/setup` (or `loom setup`) shows the current role → model table and which
  credentials are already on file (masked), and every pre-existing API
  key/endpoint gets an explicit "already set (…abcd) — keep it?" prompt
  instead of being silently reused, so keys can finally be rotated from the
  wizard. Values entered earlier in the same run are still reused without
  re-prompting.
- **Bedrock routing flag renamed `CLAUDE_CODE_USE_BEDROCK` →
  `LOOM_USE_BEDROCK`.** Loom's env block is applied to the process
  environment (and inherited by `execute` subshells), so reusing Claude
  Code's own variable could silently reconfigure a Claude Code running
  inside a Loom session — and vice versa. Legacy settings.json env blocks
  are translated on the fly and the Claude Code name is never exported.
  `ANTHROPIC_BEDROCK_BASE_URL` (an Anthropic SDK variable) still opts in.
- **`task`, `write_todos`, and `consult` can no longer be re-gated by
  accident.** These coordination tools are now always allowed (deny rules
  still win): a user `permissions.allow` list replaces the packaged one
  wholesale, which used to put "approve task?" prompts in front of every
  subagent spawn and todo update.
- **Refreshed default models to the July 2026 lineup**, verified against
  provider docs and the Ollama library:
  - *Anthropic:* orchestrator/escalation `claude-sonnet-4-6` →
    `claude-sonnet-5` (current Sonnet; same $3/$15, intro pricing through
    Aug 2026). Advisor (`claude-opus-4-8`) and cloud fallback
    (`claude-haiku-4-5`) were already current.
  - *OpenAI:* `gpt-5.2` / `gpt-5.2-codex` / `gpt-5-nano` → the GPT-5.6
    family (`gpt-5.6-terra` main, `gpt-5.6-sol` flagship, `gpt-5.6-luna`
    light), with pricing added to the cost receipts.
  - *Google:* `gemini-3-pro` / `gemini-3-flash` → `gemini-3.5-flash`
    (stable agentic mainline) + `gemini-3.1-pro-preview` (flagship).
  - *Local:* editor `deepseek-coder:14b` → `qwen3.6:27b` (best current
    dense local coder, 256K ctx); small roles `qwen3:4b`/`qwen3:14b` →
    `qwen3.5:4b`/`qwen3.5:9b`. The hardware-recommendation table now spans
    `qwen3.5:2b` → `qwen3-coder-next` (80B-A3B) and adds
    `devstral-small-2:24b` (68% SWE-bench Verified); stale `qwen2.5-coder`,
    `devstral:24b`, and `llama3.3:70b` tiers dropped. Non-Qwen alternatives
    `gemma4:e4b`, `gemma4:12b`, and `glm-4.7-flash` (strongest 30B-class
    MoE) are listed alongside the Qwen picks in `/model` and onboarding,
    with the recommendation list widened from 3 to 4 entries.

  Existing `~/.loom/config.yaml` files are untouched — run `/setup`,
  `/model`, or `loom config set` to adopt the new defaults.

### Added
- **Claude Code-style approval selector.** Tool approvals are no longer a
  bare yes/no: pick `1` yes, `2` yes — don't ask again for this tool this
  session, or `3` no — and tell Loom what to do differently. Decline
  feedback is routed back to the model in the blocking tool message
  ("The user says to do this instead: …") so the next attempt adjusts
  course instead of retrying blind.
- **Reasoning/thinking is streamed.** Models that emit reasoning
  (Anthropic thinking blocks, Ollama/OpenAI-compat `reasoning_content`)
  now stream it live in dim `✻ thinking…` blocks, from every model in the
  run — orchestrator, subagents, and advisor alike. `ui.show_thinking` now
  defaults to `true`; set it to `false` to hide reasoning again.
- **Every streamed block is attributed to its model.** Token streams from
  any model other than the orchestrator open with a
  `[role · model (⌂ local / ☁ cloud)]` header, so subagent and advisor
  output — and whether it's billed cloud or free local — is visible while
  it streams, not just after. Nested-graph steps surface via
  `subgraphs=True` streaming where the installed langgraph supports it.
- **Inline diffs for file edits.** `write_file`/`edit_file` tool calls
  render their unified diff under the call line whenever the approval
  prompt isn't about to show the same diff itself (yolo, accept-edits,
  allow-listed, or session-approved tools).
- **Cloud vs local is visible everywhere a model acts.** The banner,
  bottom toolbar, and `/status` badge models with `⌂ local` / `☁ cloud`;
  `task`/`consult` tool-call lines show which model the work is delegated
  to (`task(…) → ollama/qwen3.6:27b (⌂ local)`), and subagent output labels
  carry the same badge. Badges are fallback-aware: when Ollama is down and
  a role runs on the billed cloud fallback, it shows as `☁ cloud`.
- **Claude Code-style plan mode.** Plan mode is now a first-class mode in the
  Shift+Tab cycle (`default → accept-edits → plan → yolo`) and `/mode plan`.
  After a planning turn, Loom presents the plan with an approve-&-execute
  gate ("yes + auto-accept edits" / "yes + manual approval" / "keep
  planning"); approving flips plan mode off, rebuilds the write-capable
  agent, and implements the plan in the same thread. `/plan` also accepts
  explicit `on`/`off`, and entering plan/yolo now clears the other modes
  instead of stacking.

### Security
- **Permissions, hooks, and `/undo` now enforce inside subagents.** deepagents
  builds a fresh middleware stack per subagent, so the orchestrator-level
  policy gate never saw the `write_file`/`edit_file`/`execute` calls that
  actually happen inside editor/bash/general-purpose. Every subagent now
  carries its own `PolicyMiddleware`; delegation (`task`) is allowed by
  default and approval happens at the real write/execute instead.
- **Closed the hidden `general-purpose` subagent hole.** deepagents auto-adds
  an unrestricted general-purpose subagent (orchestrator model, full
  filesystem + shell, no policy middleware) whenever no subagent carries that
  exact name — silently bypassing plan mode's read-only guarantee and
  airgap's "raw code never reaches the cloud" guarantee. Loom's fallback
  subagent now claims the reserved name (`general` → `general-purpose`,
  legacy config keys still work), survives every run mode, and is rebuilt
  read-only in plan mode / pinned local in local-only and airgap.
- **Read-only subagents are enforced, not just prompted.** explorer, searcher,
  and reviewer lose `write_file`/`edit_file`/`delete`/`execute` via
  middleware; the editor and tester lose `execute`.
- **The `delete` filesystem tool is now covered** by the orchestrator's tool
  exclusions, airgap's deny list, the default ask-list, `delete(path/**)`
  permission specifiers, and pre-delete `/undo` snapshots.

### Changed
- `deepagents` is now pinned `>=0.6,<0.7` (the code relies on 0.6-era APIs:
  `deepagents.backends`, per-subagent middleware, the general-purpose
  override).
- **Model pulls go through the daemon's HTTP API** (`POST /api/pull`, streamed
  per-layer progress bars) at the configured `ollama_endpoint` — remote
  daemons now work end-to-end, and the `ollama` CLI binary is no longer
  required for `loom models pull`, the wizard, or `/model`.
- **Daemon reachability, not binary presence, gates the Ollama UX** —
  `loom models status`, `loom doctor`, `/models`, and the wizard treat a
  reachable remote daemon as healthy, and only show install instructions when
  neither a binary nor a daemon exists.

### Fixed
- **`/model <role> <local-model>` offers to pull a missing tag on the spot**
  (and the interactive picker now lists hardware-fitting recommendations
  alongside installed models). Previously a missing tag was saved silently and
  the role quietly ran on the billed cloud fallback.
- The setup wizard checks the pull's result instead of ignoring it, and warns
  that the role runs on the cloud fallback until the pull succeeds.
- `loom models status` and `missing_models` now apply the same `:latest` tag
  normalization the runtime uses — a config `ollama/qwen3` no longer reports
  missing when the daemon serves `qwen3:latest`.

### Added
- **Setup wizard** (`loom setup` / `/setup`) — configure every model role
  (orchestrator/advisor/escalation/subagents) from the UI: pick a provider,
  enter credentials, pick a model, all written straight to `settings.json`
  and reloaded live. Auto-launches on a true first run.
- **Hardware-aware local model recommendations** — detects OS, RAM, and
  Apple Silicon/NVIDIA/AMD GPU + VRAM, and suggests an Ollama coding model
  that actually fits, with an offer to `ollama pull` on the spot
  (`loom/core/recommendations.py`).
- **New providers**: AWS Bedrock (Anthropic), OpenAI-compatible custom
  endpoints, OpenCode Zen, OpenCode Go, and Google Vertex AI — alongside the
  existing Anthropic, OpenAI, and Google AI Studio. Full catalog in
  `loom/core/providers.py`.
- `settings.json` can now carry a top-level `models` key (deep-merged on top
  of `config.yaml`, at either the user or project layer) — what `/setup`
  writes, but also settable by hand.

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
