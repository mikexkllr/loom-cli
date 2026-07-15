# Loom

[![CI](https://github.com/mikexkllr/loom-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/mikexkllr/loom-cli/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue)](#python)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A hybrid local/cloud multi-agent CLI coding assistant. A strong **cloud
orchestrator** plans and routes; specialized **subagents** — mostly local
[Ollama](https://ollama.com) models — handle isolated, bounded subtasks in their
own context windows and return only summaries.

> The thesis (validated by both Anthropic's Claude Code and OpenAI's Codex CLI):
> subagents don't make the model smarter — they **preserve the orchestrator's
> context quality** by quarantining noisy tool output, logs, and file content
> into isolated windows that hand back only a summary.

## Contents

- [Why Ollama for local models](#why-ollama-for-local-models)
- [Install](#install)
- [Setup wizard](#setup-wizard)
- [Usage](#usage)
- [Configuration](#configuration)
- [Settings (`settings.json`)](#settings-settingsjson)
- [The fleet](#the-fleet)
- [How it stays clean](#how-it-stays-clean)
- [Architecture](#architecture)
- [Tests](#tests)
- [deepagents compatibility](#deepagents-compatibility)
- [Python](#python)
- [Notes on model names](#notes-on-model-names)

## Why Ollama for local models

Ollama is the cross-platform backend, so Loom needs **no per-platform model
code**:

| Platform | Acceleration | How |
|---|---|---|
| macOS (Apple Silicon) | Metal / MLX-class | Ollama uses the Metal backend automatically |
| Linux / Windows (NVIDIA) | CUDA | Ollama uses the CUDA backend automatically |
| Linux / Windows (AMD) | ROCm | Ollama uses the ROCm backend automatically on supported GPUs |
| Any | CPU fallback | automatic |

Installing and running a model is one command (`loom models pull`). If you'd
rather run models *natively* through MLX on a Mac, install the optional `mlx`
extra — but Ollama already gives you Metal acceleration out of the box, so this
is rarely needed.

## Install

```bash
pip install -e .            # or: pip install -e ".[dev]"
```

Then run `loom` — on a true first run (no `settings.json` anywhere yet) it
launches the **setup wizard** automatically; run it again any time with
`/setup` (REPL) or `loom setup` (shell). See [Setup wizard](#setup-wizard)
below for what it configures and which providers it supports.

Prefer to configure by hand? Set provider keys for whichever cloud models you
use:

```bash
export ANTHROPIC_API_KEY=...      # claude-* (orchestrator / advisor / reviewer)
export OPENAI_API_KEY=...         # gpt-* / o3 (optional)
export GOOGLE_API_KEY=...         # gemini-* (optional)
export LOOM_SEARCH_API_KEY=...    # optional web_search (Tavily)
```

Install Ollama and pull the local models named in your config:

```bash
loom models status     # check daemon + what's missing
loom models pull       # pull everything missing in one shot
```

## Setup wizard

`loom setup` (or `/setup` in the REPL) walks through every model role —
`orchestrator`, `advisor`, `escalation`, and each subagent — and writes the
result straight to `settings.json`, reloading it live. No manual YAML/JSON
editing required.

- **Quick setup** (default): pick one local Ollama model for the
  local-leaning roles (explorer/editor/bash/searcher/general/tester) and one
  cloud provider for the cloud-leaning roles (orchestrator/advisor/escalation/
  reviewer) — the provider's strongest model goes to `advisor`, its
  lightest/cheapest to `reviewer`.
- **Advanced**: configure every role individually — local or cloud, any
  provider, any model id.
- **Local models**: lists what's already installed via Ollama, plus
  hardware-aware recommendations (detects OS, RAM, and Apple Silicon/NVIDIA/
  AMD GPU + VRAM — see [`loom/core/recommendations.py`](loom/core/recommendations.py))
  with an offer to `ollama pull` on the spot. The recommendation table is a
  hand-curated, best-effort snapshot — there's no live "best coding model"
  API — so treat it as a sane starting point, not gospel.
- **Credentials**: prompted per provider and written to `settings.json`'s
  `env` block (`loom settings show env` to inspect, `/setup` to redo).
- **Scope**: save to your user settings (`~/.loom/settings.json`) or this
  project's (`.loom/settings.json`) — pass `--scope user|project` to `loom
  setup` to skip that prompt.

### Supported providers

| Provider | Notes |
|---|---|
| **Local (Ollama)** | Free, private. Metal (macOS), CUDA (NVIDIA), or ROCm (AMD) — Ollama picks the right backend automatically. |
| **Anthropic** | Direct API — `ANTHROPIC_API_KEY`. |
| **Anthropic via AWS Bedrock** | `AWS_BEARER_TOKEN_BEDROCK` (or real AWS credentials) + optional `ANTHROPIC_BEDROCK_BASE_URL` for a corporate proxy. Needs `pip install -e ".[bedrock]"`. |
| **OpenAI** | `OPENAI_API_KEY`. |
| **OpenAI-compatible (custom endpoint)** | Any server speaking the OpenAI Chat Completions API — vLLM, LM Studio, Together, Groq, etc. `LOOM_CUSTOM_BASE_URL` + `LOOM_CUSTOM_API_KEY`. |
| **OpenCode Zen** | Curated pay-per-use model gateway (several free models). `OPENCODE_ZEN_API_KEY`. Only OpenAI-shaped Zen models are wired up so far. |
| **OpenCode Go** | $5 first month / $10-mo subscription to curated open models (GLM, Kimi, DeepSeek, MiMo). `OPENCODE_GO_API_KEY`. |
| **Google AI Studio** (Gemini API) | Personal API key, no GCP project — `GOOGLE_API_KEY`. |
| **Google Vertex AI** | GCP project + Application Default Credentials (`gcloud auth application-default login`). Needs `pip install -e ".[vertexai]"`. |

Each provider maps to a config model-string prefix — see
[`loom/core/providers.py`](loom/core/providers.py) for the full catalog (also
used to power the wizard). A few examples, settable directly with `/model
<role> <model>` without the wizard:

```
ollama/qwen2.5-coder:32b     # local
anthropic:claude-sonnet-4-6  # Anthropic direct or Bedrock (env-flag controlled)
openai:gpt-5.2               # OpenAI
zen:glm-5.2                  # OpenCode Zen
go:deepseek-v4-flash         # OpenCode Go
vertexai:gemini-3-pro        # Google Vertex AI
custom:my-self-hosted-model  # any OpenAI-compatible endpoint
```

## Usage

### Interactive UI (the REPL)

Run `loom` with no task to drop into the in-terminal chat UI, styled after
Claude Code / opencode: a compact welcome box, a bare `>` prompt, `⏺` bullets
for assistant text and tool calls, `⎿` lines for tool results, and a live
status toolbar:

```
$ loom
╭──────────────────────────────────────────────────╮
│ ✻ Welcome to Loom!  v0.2.0                       │
│                                                  │
│   /help for help, /status for your current setup │
│                                                  │
│   model: claude-sonnet-4-6 · advisor: opus-4-8   │
│   cwd: /path/to/project                          │
╰──────────────────────────────────────────────────╯
> add a health-check endpoint to the API
⏺ I'll add the endpoint and verify it end-to-end.
⏺ task(subagent: editor, add /health route)
  ⎿ added GET /health to src/api/routes.py … +2 lines
⏺ task(subagent: tester, verify http://localhost:3000/health)
  ⎿ PASS — page shows {"status":"ok"}
```

Slash commands (Claude Code-compatible where it makes sense):

| Command | What it does |
|---|---|
| `/help` | list all commands |
| `/status` | version, models, modes, MCP, persistence, session cost |
| `/model` | show every role's model + installed Ollama models |
| `/model <role>` | interactive picker (installed local models + cloud) |
| `/model <role> <model>` | assign any model to any role |
| `/setup [roles…]` | full setup wizard: providers, credentials, hardware-aware local picks |
| `/agents` | subagents and their models |
| `/models` | local Ollama daemon + model status |
| `/mcp` | MCP servers, connection state, tools |
| `/permissions` | active allow/ask/deny rules |
| `/config`, `/settings` | show or set settings (`/settings ui.theme light`) |
| `/hooks` | configured tool hooks |
| `/compact` | summarize the conversation, free up context |
| `/cost` | per-model cost receipt: cloud spend vs free local tokens |
| `/resume` | list past sessions and continue one (SQLite-backed) |
| `/undo` | roll back the last turn's file writes |
| `/airgap` | toggle airgap mode — raw code never reaches the cloud |
| `/clear` | reset the conversation |
| `/init` | analyze the codebase, write a `LOOM.md` memory file |
| `/memory` | show the project memory file |
| `/export [path]` | save the transcript to markdown |
| `/doctor` | health-check your setup (ollama, keys, npx, MCP) |
| `/theme`, `/vim` | UI theme · vim editing mode |
| `/mode [m]` | approval mode: default / accept-edits / yolo (Shift+Tab cycles) |
| `/loop [N] <task>` | iterate until done — optional `--until "pytest -q"` gate |
| `/plan`, `/local`, `/yolo` | toggle plan / local-only / full auto-approve |
| `/cwd`, `/exit` | sandbox root · quit |

The project memory file (`LOOM.md`, or `CLAUDE.md`/`AGENTS.md` if present) and
a compact repo map are sent with your first message each session, and `@path`
mentions in your prompt inline that file's contents. Assistant text streams
token-by-token. When a tool needs approval (per your permission rules), Loom
asks inline with a unified diff preview for file writes — unless `/yolo` is
on — and every write is snapshotted so `/undo` can roll a turn back.

### Approval modes

Like Claude Code, **Shift+Tab** cycles the permission mode (also `/mode`):

| Mode | Behavior |
|---|---|
| `default` | every ask-rule tool prompts you (with a diff preview for writes) |
| `accept-edits` | file writes auto-approve; shell and everything else still ask |
| `yolo` | everything auto-approves (`/yolo`, `--yolo`) |

`deny` rules always win — even in yolo, `execute(sudo *)` stays blocked.

### Loop mode

Iterate on a task autonomously until it's actually done:

```
/loop 8 fix the failing test suite --until "pytest -q"
loom --loop 8 --until "pytest -q" "fix the failing test suite"
```

Each iteration runs a full agent turn. With `--until`, the check command
decides completion and its failure output is fed into the next iteration;
without it, the loop stops when the agent reports the task complete
(`LOOP_COMPLETE`). Ctrl-C stops the loop; `/undo` rolls back the last turn.
Pair with `accept-edits` or `yolo` so approvals don't pause the loop.

### Cost receipts

After every turn Loom prints a receipt — the measurable version of the hybrid
pitch:

```
✻ $0.052 cloud (10.0k in / 1.5k out) + 98.0k local tokens (free) · all-cloud est. $0.443 · session $0.052
```

The `all-cloud est.` prices the free local tokens at the cloud orchestrator's
rates: what this task would have cost on an all-cloud agent. `/cost` breaks
the session down per model.

### Airgap mode (`--airgap` / `/airgap`)

For code that must not leave the machine: local subagents do ALL file
reading, the cloud orchestrator loses its `read_file` tool and plans from
distilled summaries only, cloud-backed subagents are dropped, and the
prompt-size escalation to cloud is disabled. You keep the strong cloud
planner without ever uploading source code. (`--local-only` is the stricter
variant: no cloud calls at all.)

### Evals

`python scripts/eval.py` runs the task suite in [evals/tasks.yaml](evals/tasks.yaml)
against fixture projects — each task runs headless, its `check` command
decides pass/fail, and the report records wall time, local/cloud token split,
actual cloud spend, and the all-cloud estimate. Use it to validate model
routing choices (`/model editor …`) with numbers instead of vibes.

### One-shot

```bash
loom "refactor the auth module to use JWT"            # main entry
loom --plan "add pagination to all list endpoints"    # read-only plan first
loom --local-only "explain this function"             # no cloud calls at all
loom --yolo "run the test suite and fix failures"     # don't ask before tools
loom --advisor-threshold high "redesign the DB schema"# only consult advisor on high risk
loom config set orchestrator gpt-4o                   # reconfigure model routing
loom settings set ui.theme light                      # reconfigure UI/permissions/…
loom agents list                                      # subagents + assigned models
loom doctor                                           # health-check: ollama, keys, npx, MCP
```

**No Ollama? Loom still works.** If the daemon isn't running (or a model
isn't pulled), local roles automatically run on a cheap cloud model
(`cloud_fallback`, default `claude-haiku-4-5`) for the session — Loom tells
you loudly, and the cost receipts show the spend. `--local-only` and
`--airgap` refuse to fall back and fail fast with instructions instead.

## Configuration

Lives at `~/.loom/config.yaml` (seeded from
[`loom/config/default_config.yaml`](loom/config/default_config.yaml) on first
run). Model strings accept both `provider/model` (LiteLLM-style) and
`provider:model` (LangChain-style); the `ollama/` prefix marks a local model.

```yaml
orchestrator: claude-sonnet-4-6
subagents:
  explorer: ollama/qwen3:4b
  editor:   ollama/deepseek-coder:14b
  bash:     ollama/qwen3:14b
  searcher: ollama/qwen3:4b
  reviewer: claude-haiku-4-5
  general:  ollama/qwen3:14b
  tester:   ollama/qwen3:14b
advisor: claude-opus-4-8       # consulted on-demand only
ollama_endpoint: http://localhost:11434
```

## Settings (`settings.json`)

Beyond model routing, Loom is fully configurable through a layered
`settings.json` — the same idea as Claude Code. Layers are deep-merged, later
wins:

| Layer | Path | Commit? |
|---|---|---|
| packaged defaults | `loom/config/default_settings.json` | — |
| legacy models | `~/.loom/config.yaml` | — |
| user | `~/.loom/settings.json` | no |
| project | `.loom/settings.json` | yes |
| local | `.loom/settings.local.json` | no (gitignore) |

`loom settings init` drops a starter file in your project. Sections:

```jsonc
{
  "permissions": {
    "default_mode": "ask",                    // allow | ask | deny
    "allow": ["read_file", "ls", "grep_tool"],
    "ask":   ["write_file", "edit_file", "execute"],
    "deny":  ["execute(rm -rf /*)", "execute(sudo *)"]
  },
  "hooks": {
    "pre_tool_use":  [{ "matcher": "write_file|edit_file", "command": "…" }],
    "post_tool_use": [{ "matcher": "write_file", "command": "black -q ." }]
  },
  "env": { "ANTHROPIC_API_KEY": "…" },
  "ui": { "theme": "dark", "streaming": true, "prompt_symbol": ">" },
  "mcp_servers": {
    "playwright": { "command": "npx", "args": ["@playwright/mcp@latest"] }
  }
}
```

- **Permissions** — rule syntax `tool` or `tool(glob)`; e.g. `execute(git *)`,
  `write_file(src/**)`, `*`. Evaluated **deny > allow > ask > default_mode**.
  An `ask` decision prompts you inline in the REPL (or auto-denies headless,
  unless `--yolo`). Enforced by `PolicyMiddleware` around every tool call.
- **Hooks** — shell commands run around tool events; a `pre_tool_use` hook that
  exits non-zero **blocks** the tool. Commands get the event as JSON on stdin
  plus `LOOM_TOOL_NAME` / `LOOM_TOOL_INPUT` in env. Great for auto-format,
  lint-gates, or audit logging.
- **env** — injected before any model call (`setdefault`, so your real shell
  env still wins).
- **models** — same shape as `config.yaml` (`orchestrator`, `advisor`,
  `escalation_model`, `subagents: {...}`); deep-merges on top of it regardless
  of layer. This is what [`/setup`](#setup-wizard) writes — role assignments
  work identically whether they land in the user or project layer.
- **ui** — theme, streaming, tool-call visibility, prompt symbol, banner.
- **mcp_servers** — MCP servers to connect (stdio subprocess or
  `streamable_http`/`sse` via `url`). Sessions are held open for the whole
  process on a background event loop, so stateful servers — the bundled
  [Playwright MCP](https://github.com/microsoft/playwright-mcp) browser above
  all — keep their state across tool calls. Playwright ships enabled by
  default (needs `npx`); disable with
  `{"mcp_servers": {"playwright": {"enabled": false}}}`. Its `browser_*`
  tools power the `tester` subagent and are allowed by default; other
  servers' tools go to `general` and follow normal permission rules.

Edit from the CLI (`loom settings set permissions.default_mode allow`), from the
REPL (`/settings ui.theme light`), or by hand.

## The fleet

| Agent | Default model | Tools | Mode |
|---|---|---|---|
| `explorer` | local small | `ls`, `read_file`, `glob`, `grep` | read-only |
| `editor`   | local mid   | `read_file`, `write_file`, `edit_file` | write |
| `bash`     | local mid   | `execute` (sandboxed shell), `write_file` | write |
| `searcher` | local small | `grep`, `glob`, `web_search` (optional) | read-only |
| `reviewer` | cloud cheap | `read_file`, `grep` | read-only |
| `general`  | local mid   | all tools | fallback |
| `tester`   | local mid   | `browser_*` (Playwright MCP) | write |

**Advisor** (`consult` tool): the strongest cloud model, called on-demand at
decision gates. It only advises — it never acts. Auto-consultation is gated by
`--advisor-threshold` (low/medium/high).

**Reviewer**: dispatched after significant writes; returns a structured
`ReviewVerdict` (risk low/medium/high, approve/flag, issue list). High risk or no
approval → Loom stops and surfaces it for human sign-off.

**Tester**: whenever a change touches anything a user can see or interact with
from the frontend, the orchestrator is required to verify it end-to-end from
the user's perspective before declaring done — bash starts the dev server,
then the tester drives a real browser through the exact user journey
(navigate, click, type) and judges each step only by what the page visibly
shows, returning PASS/FAIL with evidence. A FAIL means the task is not done.
The tester only appears in the fleet when the Playwright MCP server connects;
without it the orchestrator must say E2E testing was skipped and how to verify
manually.

## How it stays clean

- **Prompt-size guard** — if a local subagent's prompt nears its context window,
  that single call auto-escalates to `escalation_model` (cloud) instead of
  failing. See [`middleware/prompt_size_guard.py`](loom/middleware/prompt_size_guard.py).
- **Artifact store** — tool output over `artifact_offload_tokens` is written to
  `.loom/artifacts/` and replaced in-context with a path reference.
- **Summarization** — the orchestrator auto-compacts at `compaction_threshold`
  (default 70%) of its window.
- **Worktree isolation** — parallel write agents each get their own git worktree
  so edits never collide. Falls back to in-place when not a git repo.

## Architecture

```
loom/
├── core/
│   ├── orchestrator.py     # create_deep_agent() wiring subagents + middleware
│   ├── advisor.py          # consult() tool + ReviewVerdict risk model
│   ├── model_router.py     # provider resolution + escalation logic
│   ├── artifact_store.py   # large-output offload + summarization middleware
│   ├── worktree.py         # git worktree isolation
│   ├── ollama.py           # local model status / pull
│   ├── config.py           # config.yaml (model routing) loader + validation
│   ├── settings.py         # layered settings.json loader
│   ├── permissions.py      # allow/ask/deny rule engine
│   ├── hooks.py            # pre/post tool-use hook runner
│   ├── mcp.py              # persistent MCP sessions (Playwright browser)
│   ├── usage.py            # token tracking + cost receipts
│   ├── sessions.py         # SQLite persistence + /resume index
│   ├── undo.py             # per-turn file snapshots for /undo
│   └── repomap.py          # repo map + @file mention expansion
├── subagents/              # explorer, editor, bash, searcher, reviewer, general, tester
├── middleware/
│   ├── prompt_size_guard.py
│   └── policy.py           # enforce permissions + run hooks per tool call
├── tools/                  # sandboxed fs / shell / search tools
├── ui/
│   ├── repl.py             # interactive terminal chat loop
│   ├── slash.py            # /command registry
│   └── theme.py            # Rich themes from ui settings
├── cli/main.py             # Typer app + Rich streaming + loom doctor
├── config/
│   ├── default_config.yaml   # model routing defaults
│   └── default_settings.json # permissions / hooks / env / ui / mcp defaults
evals/                      # eval tasks + fixtures (scripts/eval.py)
```

Built on [`deepagents`](https://docs.langchain.com/oss/python/deepagents) /
LangChain. The orchestrator gets `write_todos` + `task` (delegation) +
`compact_conversation` from the deepagents middleware stack; Loom adds the
`consult` tool, the prompt-size guard, and tuned summarization.

## Tests

```bash
pytest                    # unit tests
python scripts/smoke.py   # wiring smoke test — no API calls, no Ollama
```

Tests that don't need the heavy LLM stack (config, model routing, worktree) run
standalone; those that exercise tools/subagents `importorskip` the optional deps
so the suite degrades gracefully.

`scripts/smoke.py` exercises the runtime-integration surfaces against a **stub
agent** — permission gating in `PolicyMiddleware.wrap_tool_call`, the
prompt-size escalation decision, hook blocking, and the REPL turn/stream loop —
so you can confirm the deepagents wiring is sound right after
`pip install -e .`, without spending tokens or standing up local models.

## deepagents compatibility

Loom follows current deepagents / LangChain 1.0 best practices:

- **Middleware** subclass `AgentMiddleware` and implement the `wrap_tool_call` /
  `wrap_model_call` hooks (plus their `awrap_*` async variants), returning a
  `ToolMessage` to short-circuit a gated tool. Tool identity is read from
  `request.call` (`{name, args, id}`), matching the real `ToolCallRequest`.
- **Subagents** are dicts with `name` / `description` / `system_prompt` /
  `tools` / `model` / `middleware`; the reviewer uses `response_format` for a
  structured verdict.
- **Persistence** uses a LangGraph checkpointer + `thread_id`: the REPL attaches
  an `InMemorySaver` and sends only the new turn each round (the graph keeps the
  transcript), so conversations survive and can resume. Falls back to resending
  the full transcript if no checkpointer is available.

## Python

Requires **Python ≥ 3.11**; tested against 3.11–3.14. Local models run through
Ollama, so there's no per-platform build step.

## Notes on model names

Model ids in the default config (`claude-sonnet-4-6`, `claude-haiku-4-5`,
`claude-opus-4-8`, etc.) are placeholders you can change at any time with
`loom config set`. Use whatever your provider account has access to.
