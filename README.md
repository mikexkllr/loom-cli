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

```bash
loom "refactor the auth module to use JWT"            # main entry
loom --plan "add pagination to all list endpoints"    # read-only plan first
loom --local-only "explain this function"             # no cloud calls at all
loom --advisor-threshold high "redesign the DB schema"# only consult advisor on high risk
loom config set orchestrator gpt-4o                   # reconfigure on the fly
loom config show
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
advisor: claude-opus-4-8       # consulted on-demand only
ollama_endpoint: http://localhost:11434
```

## The fleet

| Agent | Default model | Tools | Mode |
|---|---|---|---|
| `explorer` | local small | `ls`, `read_file`, `glob`, `grep` | read-only |
| `editor`   | local mid   | `read_file`, `write_file`, `edit_file` | write |
| `bash`     | local mid   | `execute` (sandboxed shell), `write_file` | write |
| `searcher` | local small | `grep`, `glob`, `web_search` (optional) | read-only |
| `reviewer` | cloud cheap | `read_file`, `grep` | read-only |
| `general`  | local mid   | all tools | fallback |

**Advisor** (`consult` tool): the strongest cloud model, called on-demand at
decision gates. It only advises — it never acts. Auto-consultation is gated by
`--advisor-threshold` (low/medium/high).

**Reviewer**: dispatched after significant writes; returns a structured
`ReviewVerdict` (risk low/medium/high, approve/flag, issue list). High risk or no
approval → Loom stops and surfaces it for human sign-off.

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
│   └── config.py           # config.yaml loader + validation
├── subagents/              # explorer, editor, bash, searcher, reviewer, general
├── middleware/
│   └── prompt_size_guard.py
├── tools/                  # sandboxed fs / shell / search tools
├── cli/main.py             # Typer app + Rich streaming
└── config/default_config.yaml
```

Built on [`deepagents`](https://docs.langchain.com/oss/python/deepagents) /
LangChain. The orchestrator gets `write_todos` + `task` (delegation) +
`compact_conversation` from the deepagents middleware stack; Loom adds the
`consult` tool, the prompt-size guard, and tuned summarization.

## Tests

```bash
pytest
```

Tests that don't need the heavy LLM stack (config, model routing, worktree) run
standalone; those that exercise tools/subagents `importorskip` the optional deps
so the suite degrades gracefully.

## Notes on model names

Model ids in the default config (`claude-sonnet-4-6`, `claude-haiku-4-5`,
`claude-opus-4-8`, etc.) are placeholders you can change at any time with
`loom config set`. Use whatever your provider account has access to.
