# Loom

A hybrid local/cloud multi-agent CLI coding assistant. A strong **cloud
orchestrator** plans and routes; specialized **subagents** — mostly local
[Ollama](https://ollama.com) models — handle isolated, bounded subtasks in their
own context windows and return only summaries.

> The thesis (validated by both Anthropic's Claude Code and OpenAI's Codex CLI):
> subagents don't make the model smarter — they **preserve the orchestrator's
> context quality** by quarantining noisy tool output, logs, and file content
> into isolated windows that hand back only a summary.

## Why Ollama for local models

Ollama is the cross-platform backend, so Loom needs **no per-platform model
code**:

| Platform | Acceleration | How |
|---|---|---|
| macOS (Apple Silicon) | Metal / MLX-class | Ollama uses the Metal backend automatically |
| Linux / Windows | CUDA | Ollama uses the CUDA backend automatically |
| Any | CPU fallback | automatic |

Installing and running a model is one command (`loom models pull`). If you'd
rather run models *natively* through MLX on a Mac, install the optional `mlx`
extra — but Ollama already gives you Metal acceleration out of the box, so this
is rarely needed.

## Install

```bash
pip install -e .            # or: pip install -e ".[dev]"
```

Set provider keys for whichever cloud models you use:

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

## Usage

### Interactive UI (the REPL)

Run `loom` with no task to drop into the in-terminal chat UI, styled after
Claude Code / opencode: a compact welcome box, a bare `>` prompt, `⏺` bullets
for assistant text and tool calls, `⎿` lines for tool results, and a live
status toolbar:

```
$ loom
╭──────────────────────────────────────────────────╮
│ ✻ Welcome to Loom!  v0.1.0                       │
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
| `/status` | version, models, modes, MCP, session usage |
| `/model [name]` | show or switch the orchestrator model |
| `/agents` | subagents and their models |
| `/models` | local Ollama daemon + model status |
| `/mcp` | MCP servers, connection state, tools |
| `/permissions` | active allow/ask/deny rules |
| `/config`, `/settings` | show or set settings (`/settings ui.theme light`) |
| `/hooks` | configured tool hooks |
| `/compact` | summarize the conversation, free up context |
| `/cost` | token usage this session |
| `/clear` | reset the conversation |
| `/init` | analyze the codebase, write a `LOOM.md` memory file |
| `/memory` | show the project memory file |
| `/export [path]` | save the transcript to markdown |
| `/doctor` | health-check your setup (ollama, keys, npx, MCP) |
| `/theme`, `/vim` | UI theme · vim editing mode |
| `/plan`, `/local`, `/yolo` | toggle plan / local-only / auto-approve |
| `/cwd`, `/exit` | sandbox root · quit |

The project memory file (`LOOM.md`, or `CLAUDE.md`/`AGENTS.md` if present) is
sent with your first message each session. When a tool needs approval (per
your permission rules), Loom asks inline — unless `/yolo` is on.

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
```

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
│   └── hooks.py            # pre/post tool-use hook runner
├── subagents/              # explorer, editor, bash, searcher, reviewer, general, tester
├── middleware/
│   ├── prompt_size_guard.py
│   └── policy.py           # enforce permissions + run hooks per tool call
├── tools/                  # sandboxed fs / shell / search tools
├── ui/
│   ├── repl.py             # interactive terminal chat loop
│   ├── slash.py            # /command registry
│   └── theme.py            # Rich themes from ui settings
├── cli/main.py             # Typer app + Rich streaming
└── config/
    ├── default_config.yaml   # model routing defaults
    └── default_settings.json # permissions / hooks / env / ui defaults
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
