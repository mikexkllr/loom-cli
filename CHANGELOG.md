# Changelog

## Unreleased

### Fixed

Found by driving the whole app end to end — the real CLI, the real graph, real
tools on a real project, and live runs against a real provider. Every one of these
was invisible to the unit tests that passed over it.

- **The read budget was advice, not enforcement.** Withdrawing `read_file` from
  the model's schema does not stop the call: `ToolNode` still holds every tool the
  agent was built with, so an over-budget read emitted anyway — which is what a
  model does after watching four such calls succeed in its visible history —
  executed and returned the file. The budget is now enforced twice, withdrawn at
  the model call and refused at the tool call, with the refusal naming the
  subagent to delegate to. Counting stops at the call being judged, so a batch of
  parallel reads degrades in order instead of being rejected wholesale.
  `/cost` and `/status` report withheld and refused separately: a refusal means the
  model reached for a tool it could no longer see, which is the signal that
  withdrawing it was never going to be enough.
- **The advisor's tokens were billed to the orchestrator.** Role attribution walks
  the callback run tree, which works only for calls that nest under the `task`
  run. A model invoked by hand from inside a tool does not: the config a tool is
  handed belongs to the tool *node*, so `consult`'s call to the advisor landed
  beside the `consult` run rather than under it and the walk sailed straight past
  the role. Both hand-invoked callers — the advisor and `/compact`'s summariser —
  now state their role outright via run metadata, which is read before the walk
  because an explicit claim beats an inference.
- **`/compact` made the orchestrator look greedy.** Its summarisation is
  housekeeping on the transcript, not work on the task, so it is now a third
  bucket: charged for and shown as its own `/cost` row, but kept out of the
  delegation ratio and the delegated-role count. Compacting a session no longer
  moves the number the receipt leads with.
- **`--local-only` and `--airgap` refused to start without a cloud key.** The
  cloud-backed roles in a hybrid fleet — the shipped config's `reviewer` trails a
  cloud advisor — were filtered out of the fleet *after* their models were built,
  and building a model validates its credentials. So the two modes whose entire
  promise is that nothing leaves the machine died on a missing
  `OPENCODE_ZEN_API_KEY`, on exactly the machine most likely to be asking for
  them. Cloud roles are now skipped before construction.
- **A free model was billed as if it were Sonnet.** Prices were looked up on the
  name the provider reported, and a gateway can answer a free-tier request under
  the *upstream* model's name — OpenCode Zen serves `deepseek-v4-flash-free` and
  reports `deepseek-v4-flash`. Unknown name, so it fell to the Sonnet-tier
  default: a live session on nothing but free models invoiced $0.032. The reported
  name still wins whenever Loom recognises it, because that is what actually ran
  and includes a cloud fallback the config never named; when it doesn't, the
  role's configured model answers instead. A `-free`/`:free` suffix now prices at
  zero, and provider prefixes no longer hide a known price
  (`zen/claude-haiku-4-5` was priced as an unknown model).
- **An estimated price looked exactly like a real one.** An unknown model is still
  charged at the Sonnet-tier default — a receipt with a hole in it is worse — but
  it now reads `~$0.008` with a footnote naming the assumption, instead of
  presenting a fabricated figure with the same confidence as a billed one.
- **`loom "fix the tests" --yolo` failed** with `No such command '--yolo'`. Click
  disables interspersed arguments for groups so a subcommand keeps its own flags;
  in prompt form there is no subcommand to shield, and the most obvious way to
  type the command was the one that didn't work. Flags before the prompt still
  work, and a subcommand's own flags are still left for it.
- **Nothing told the agents where the filesystem root was.** The backend mounts
  the project at `/`, undocumented in any prompt: a live run had the explorer
  waste a call grepping an invented `/home/user`, then close its report by
  "correcting" the user — "the file is at `/src/billing.py`, *not*
  `src/billing.py`". Both spellings are the same file. The subagent preamble, the
  reviewer's prompt, and the orchestrator's reading rule now say so.
- **`/cost` folded role and model names mid-word** in an 80-column terminal
  (`orchestra`/`tor`, `claude-so`/`nnet-5`). Local/cloud moved onto the model as a
  badge, the way every other Loom surface shows it, which frees the column the
  names needed. `/status` gained thousands separators, stopped saying "read
  budget" twice in one line, and now carries the `~` estimate marker too — the
  same figure read as billed there and estimated in the receipt.
- **The OpenCode provider notes were wrong**, in the wizard and the README. Both
  claimed MiniMax/Qwen-style models "are Anthropic-shaped and aren't wired up
  yet"; every one of them answers over the gateways' OpenAI-compatible API, tool
  calls included (checked live on `minimax-m3`, `qwen3.7-plus`, `kimi-k3`,
  `glm-5.2`, `mimo-v2.5`, `hy3`). The notes were steering users away from models
  that work. Also documented: one `OPENCODE_API_KEY` covers Zen and Go together,
  and Go being a flat subscription means its per-token figures are estimates.

### Changed

- **Migrated to deepagents 0.7** (`>=0.7,<0.8`, LangChain `>=1.3.14`,
  `langchain-anthropic >=1.5.3`). 0.7 stops shipping prompts and stops assuming
  a toolset, which moves both to the harness. Four of its breaking changes were
  silent failures for Loom:
  - *`write_todos` disappeared.* `TodoListMiddleware` left the default stack, so
    the orchestrator prompt was instructing the model to call a tool that no
    longer existed. It's added back explicitly, with Loom's own planning prose
    instead of LangChain's ~600 generic words: a todo is written as the work
    *plus the subagent that will do it*, so the plan and the routing are one
    artifact. Subagents deliberately don't get it — their work is already bounded
    by the task they were handed.
  - *The built-in prompts went empty.* `BASE_AGENT_PROMPT` is blank and the prose
    describing the filesystem / `task` / summarization tools is gone. Every
    system prompt in Loom was rewritten to carry its own tool inventory, method,
    and output contract. New `tests/test_prompt_contract.py` binds each prompt to
    its role's real allowlist, since a prompt naming a tool the role can't call
    is a dead end no smoke test catches.
  - *`delete` is now handed out recursively* whenever the backend supports it,
    and classified as an ordinary write — so any rule permitting writes to a
    directory also permitted erasing that subtree. No Loom role gets it (`bash`
    can `rm` through `execute`, which the policy gate prompts for by name).
  - *`write_file` silently replaces an existing file* instead of erroring. Every
    write-capable prompt now says so and steers to `edit_file`.
- **Per-role tool allowlists replace post-hoc filtering.** 0.7 lets a
  caller-supplied middleware instance replace the default with the same `.name`,
  so each agent now gets a `FilesystemMiddleware(tools=[...])` holding exactly
  its role's tools — they are never constructed and their schemas never reach the
  model, rather than being injected and stripped. The orchestrator's holds one
  tool: `read_file`. `editor` lost `execute` and `bash` lost `write_file` /
  `edit_file`; that split is what makes a delegation legible. Specs declare
  `fs_tools` once and both the allowlist and the last-mile
  `ToolExclusionMiddleware` derive from it — the exclusion layer stays because
  the allowlist cannot express "no `read_file`", which is what `--airgap` needs.
- **The orchestrator's own reads are now metered.** Its search and write tools
  were already gone, but a strong cloud model handed `read_file` still swept a
  dozen files "just to be sure" — the exact context pollution subagents exist to
  prevent, at cloud prices. Prompt wording doesn't hold that line, so
  `orchestrator_read_budget` (default 4) direct reads per user turn are enforced
  by *removing the tool* for the rest of the turn
  ([`middleware/delegation_guard.py`](loom/middleware/delegation_guard.py)). The
  system prompt states the budget, so the tool vanishing reads as the rule
  working rather than a broken harness — and because the note lives in the stable
  prompt prefix, it never invalidates the provider's prompt cache. `0` forbids
  direct reads (what `--airgap` does); `-1` removes the cap.
- **Cost receipts were wrong in three ways, all fixed.**
  - *Cached input was billed at the uncached rate.* Providers charge three
    different rates for input (uncached, cache write at a premium, cache read at
    a tenth) and `usage_metadata["input_tokens"]` is the sum of all three. With
    prompt caching on — which deepagents enables for Anthropic by default — the
    receipt overstated a long conversation by most of an order of magnitude.
    Tokens are now split and priced per rate, and the cached share is shown.
  - *There was no way to see whether the orchestrator was doing too much.* Usage
    was keyed by model name, which cannot separate the orchestrator from a
    subagent sharing its model. Attribution now comes from the callback run tree:
    every model call beneath a `task` belongs to that subagent, and `consult`
    calls to the advisor. The receipt reports `orchestrator N% of tokens, M
    delegated roles`, and `/cost` breaks the session down per role — calls,
    cached share, and cost each.
  - *"Saved vs all-cloud" priced local tokens against `config.orchestrator`*,
    even when that was itself a local model, inventing a saving in `--local-only`
    runs where nothing was billed. The baseline is now always a billed model,
    named in the receipt, falling back to a stated default when every configured
    role is local.
  Local/cloud classification also prefers the run's `ls_provider` over the
  model-name heuristic, so an Ollama tag without a colon (`gpt-oss`,
  `llama3.2`) stops being charged as cloud.
- **`compaction_threshold` finally reaches the agents.** The builder in
  `artifact_store.py` was never wired into `create_deep_agent`, so the documented
  knob did nothing and every agent used deepagents' default — which derives its
  trigger from the model's published profile and, for a profile-less ChatOllama,
  falls back to a flat 170K tokens a 32K local model can never reach before
  overflowing. Each agent now compacts at `compaction_threshold` of *its own*
  detected window, and evicted history is offloaded through the shared backend
  into `.loom/sessions/conversation_history/` where it stays re-readable. Local
  roles also get their `grep` match cap and large-result eviction threshold
  scaled to their window.
- **The `consult` tool forwards its run config**, so the advisor's billed cloud
  tokens actually appear in the receipt and are attributed to the advisor rather
  than vanishing.
- **The prompt-size guard now counts tool schemas.** deepagents' filesystem and
  `task` tools run to a few thousand tokens of JSON — a tenth of a small local
  model's window before a single message — so leaving them out under-counted
  exactly the calls most likely to overflow.

- **The reviewer now runs on the Advisor's model.** It was pinned to a cheap
  cloud model (`claude-haiku-4-5`), which made the one gate that decides
  whether a change needs human sign-off the weakest model in the fleet. Roles
  can now *inherit*: `subagents.reviewer` is unset in the packaged config and
  falls through to `advisor`, while every other unassigned role still trails
  `general-purpose`. An explicit `subagents.<role>` entry always wins, so
  pinning the reviewer to something cheaper or local is a one-line override —
  and existing `~/.loom/config.yaml` files, which were seeded with an explicit
  `reviewer:`, keep exactly the model they have until that line is removed.
  New `LoomConfig.model_for()` / `loom.subagents.model_for()` resolve a role to
  its effective model; `agents list`, the local-only/airgap subagent filter,
  and the role planner all use them instead of reading `config.subagents`
  directly (an unassigned role has a real model, it just isn't in the file).
  A local `advisor` is now covered by the local-first role planner too, since
  the reviewer depends on it.
- **`max_local_context` is sized from the machine instead of a fixed 64K.**
  Left unset (the new default) it comes from GPU-addressable memory — 8GB →
  16K, 16GB → 32K, 24GB → 64K, 48GB → 128K, 80GB+ → 256K — so a small laptop
  stops being handed a KV cache it can't allocate and a big box stops being
  capped below what it can carry. `detect_hardware()` now recognizes NVIDIA
  parts with **unified memory** rather than discrete VRAM (Jetson/Tegra —
  Orin, Thor — and the Grace superchips GH200 / GB10 / DGX Spark), which
  `nvidia-smi` either can't see or under-reports; those boxes previously
  looked GPU-less and got the smallest possible budget. Apple Silicon, NVIDIA
  and AMD discrete VRAM, and NVIDIA unified parts now all size identically.
  Setting `max_local_context` explicitly still overrides everything.

- **Cloud is now the last resort, not the first, whenever a local model can do
  the job.** Three paths used to hand work to a billed model on a machine with
  a perfectly healthy Ollama, and all three now stay local:
  - *A role whose model isn't pulled.* `apply_cloud_fallback` sent that role
    straight to `cloud_fallback` — so a user who had pulled the 4B and 9B but
    not the 27B editor model silently paid for every edit while two local
    models sat idle. Roles now fall to the closest **local** stand-in first
    (nearest context window, cheapest-adequate), and reach `cloud_fallback`
    only when the daemon is down or serving nothing usable. `--local-only` and
    `--airgap` get the substitution too, so a missing tag no longer breaks a
    role outright in the modes that can't fall back.
  - *An oversized prompt.* The prompt-size guard escalated straight to
    `escalation_model` (cloud). Overflowing a 4B model's window is a capacity
    problem, not a difficulty one, so it now climbs a two-rung ladder — the
    roomiest served **local** model first, cloud only when none has the
    headroom. `/stats` reports the two counts separately
    (`N local→local (free) · M local→cloud`).
  - *An unknown context window.* Any local model without a `context_windows`
    entry was assumed to hold 32K, which both shrank the `num_ctx` handed to
    Ollama and escalated calls the model could have held. Loom now reads the
    real context length from the daemon's `/api/show` at startup, capped by the
    new `max_local_context` (default 64K) so the KV cache still fits in memory.
    An explicit `context_windows` entry always wins.

  New [`loom/core/local_pool.py`](loom/core/local_pool.py) holds the local-first
  logic behind one daemon probe per session. `OrchestratorBundle` gained
  `substitutions` and `active_config`; the REPL badges, `/stats`, and both
  `doctor` implementations now distinguish "covered locally (free)" from
  "running on the cloud fallback (billed)" instead of reporting every missing
  model as a billed fallback.

### Added
- **Playwright browser setup, one command.** New
  [`loom/core/playwright_setup.py`](loom/core/playwright_setup.py) detects
  whether Playwright's browser binaries are downloaded (the step
  `npx @playwright/mcp` doesn't do for you — without it the `tester` subagent
  connects fine but fails at the first `browser_*` call) and installs them
  with `loom playwright install` / `loom playwright status`, mirroring
  `loom models pull`'s streamed-output pattern. `doctor` (CLI and `/doctor`)
  now reports browser-install state alongside the existing `npx` check, and
  the setup wizard (`loom setup`, `/setup`, and the true-first-run
  auto-launch) offers to install the browser right after model roles are
  assigned. New `/playwright` REPL command (`/playwright install` to fix).
- **Model picker now shows every model it can, live.** New
  [`loom/core/model_catalog.py`](loom/core/model_catalog.py) queries each
  cloud provider's own "list models" endpoint (Anthropic's Models API,
  OpenAI's, Google AI Studio's ListModels, and OpenCode Zen/Go's — the last
  two need no API key, so they're always queried) and feeds the numbered
  picker in `/setup` and `/model <role>` from the live result, falling back
  to the provider's hardcoded example models when there's no listing
  endpoint (Bedrock, Vertex AI — both need SDK-level credential machinery),
  no credentials yet, or the request fails. Zen/Go model families that are
  Anthropic-shaped rather than OpenAI-shaped on those gateways (MiniMax,
  Qwen — see the existing provider notes) are filtered out so the picker
  never offers a model id that would silently break. The local side of the
  picker (`/setup`'s local-model prompt and `/model <role>`) now lists
  Loom's *entire* curated Ollama catalog — not just the top hardware-fit
  picks — each annotated installed / fits-your-hardware / may-not-fit
  (there's still no stable public API for "every model in the Ollama
  library", so this stays a hand-maintained snapshot, not a live query).
  Quick setup can now optionally pick a specific model per tier
  (main/flagship/light) from the same live-or-example catalog instead of
  always taking the provider's built-in defaults. Every pick — quick,
  advanced, or `/model` — still lands in the same layered `settings.json`
  (`~/.loom` or `<project>/.loom`) it already did, so it's remembered across
  restarts and new sessions with no separate storage to add.
- **Standalone binaries + curl installer + self-update.** Every push to
  `main` now freezes a `loom` binary (PyInstaller, see
  [`packaging/loom.spec`](packaging/loom.spec)) for macOS (arm64/x64), Linux
  (x64/arm64), and Windows (x64) via
  [`.github/workflows/release.yml`](.github/workflows/release.yml), and
  publishes them as the repo's latest GitHub release with a
  `checksums.txt`. [`scripts/install.sh`](scripts/install.sh) (curl | sh)
  and [`scripts/install.ps1`](scripts/install.ps1) (irm | iex) detect
  OS/arch, verify the checksum, and install to `~/.local/bin` (or
  `%LOCALAPPDATA%\loom\bin` on Windows). The new `loom update` command
  compares the running binary's checksum against the latest release and
  swaps itself in place if it's stale (Windows swap happens via a detached
  helper since it can't replace its own locked `.exe`); source installs get
  a `git pull && uv sync` hint instead. Binary installs also self-check on
  every REPL/one-shot launch (throttled to ≤once/6h, 3s network timeout,
  cached in `~/.loom/update_check.json`, never blocks startup on failure)
  and, on a real terminal, ask **update now or continue with the current
  version** — accepting resumes the same session on the new build
  (`os.execv` on Unix; a child-process relaunch + deferred swap on
  Windows). Piped/non-interactive stdin only ever prints a notice.

### Changed
- **The project is uv-native now.** `uv.lock` is committed; `uv sync`
  replaces `pip install -e .` (dev tools moved to a `[dependency-groups]`
  dev group installed by default); CI runs `uv sync --locked` + `uv run`;
  all in-app hints (missing extras, wizard notes, error messages) point at
  `uv sync --extra …`. `pip install -e .` still works.
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

### Fixed
- **`/model` changes didn't stick — reverted to the standard model on
  restart.** `set_value("models.*")` wrote model routing to `config.yaml`,
  but `settings.json` deep-merges *over* `config.yaml` (see
  [`load_settings`](loom/core/settings.py)), so once `/setup` had written a
  `models` block there, every later `/model` pick persisted to `config.yaml`
  yet was silently shadowed on load — it never took effect in-session and was
  gone on reopen. `/model` (and any `models.*` set) now writes to the same
  winning `settings.json` layer the setup wizard uses
  ([`_set_user_model_value`](loom/core/settings.py)), validated before write,
  so a pick takes effect immediately and survives restarts — matching what the
  picker already claimed to do. `loom config set <key> <value>` now writes the
  same winning `settings.json` layer (it used to write the shadowed
  `config.yaml`), `loom config show` prints the effective merged routing, and
  `loom config path` names both files (defaults + overrides).
- **`/model` vs `/models` was confusing — the plural did something
  unrelated.** `/model` configures routing while `/models` checked the Ollama
  daemon, so the config command's apparent plural reported daemon health
  instead. The status check is now [`/ollama`](loom/ui/slash.py) (which is what
  it inspects), and `/models` is an alias of `/model`, so the plural just opens
  the picker.
- **Subagent output was attributed to the wrong role.** Streamed subagent
  text and tool calls were labelled by reverse-matching the running model name
  against config, which is ambiguous once two roles share a model — most often
  when a local role falls back to the cloud `cloud_fallback`/`reviewer` model:
  an `explorer` on the fallback would render as `[reviewer]`, and its tool
  calls as a bare `[subagent]`. Attribution now binds the nested delegation
  namespace (`tools:<id>`) to the `subagent_type` from the originating `task`
  call, in delegation order (`_attribute_ns` in
  [`loom/ui/repl.py`](loom/ui/repl.py)), so a subagent is labelled by the role
  it actually is — `[explorer · … ]` — even when its model collides with
  another role's. Model-name matching stays as the fallback.
- **The orchestrator did its own recon instead of delegating.** `ls` is now
  stripped from the orchestrator alongside `glob`/`grep` in
  [`_orchestrator_excluded_tools`](loom/core/orchestrator.py). A strong cloud
  orchestrator (e.g. `gpt-5.5`) would map the tree with `ls` and sweep files
  itself for "what is this codebase about"–style questions, never routing to
  the local `explorer` and defeating the context-quarantine design — the
  "delegate, don't investigate yourself" rule was prompt-only and got ignored.
  Removing the tool makes it structural. `read_file` deliberately stays: the
  orchestrator still needs targeted single-file reads to confirm a path a
  subagent named or review a reported change (browsing/searching need a
  directory walk; a confirmation needs a path).
- **Approval prompts sometimes never appeared — tools were silently
  auto-denied.** The confirm callback (and the /yolo, accept-edits, and
  /undo turn-id state) lived in ``contextvars``, but LangGraph executes
  tool calls in worker threads where a fresh context reverts to the
  default: headless deny. The model saw "User declined" without any prompt
  ever being shown. These are now process-global slots visible from every
  thread, and concurrent prompts are serialized behind a lock so parallel
  tool calls can't interleave on the terminal.

### Added
- **Agent skills (SKILL.md folders) via deepagents (`/skills`).** Three
  layered sources — packaged `loom/skills/`, user `~/.loom/skills/`,
  project `.loom/skills/` (later wins on name collisions) — mount into the
  agent's virtual filesystem and load with progressive disclosure: only
  name + description enter the prompt; the full SKILL.md is read when a
  task matches. Ships a `graphify-graph-rag` skill teaching agents to
  prefer graph queries and how to refresh a stale graph. `/skills` lists
  everything discovered.
- **`/graphify` installs its own tooling.** First run offers to install
  the CLI on the spot (`uv tool install graphifyy`, pipx fallback) and
  build the graph in one flow; the MCP server entry gets pinned to the
  resolved binary path so fresh `~/.local/bin` installs work even when
  that directory isn't on PATH.

- **Graphify knowledge graph / GraphRAG integration (`/graphify`).**
  [Graphify](https://github.com/safishamsi/graphify) (`uv tool install
  graphifyy`) builds a tree-sitter knowledge graph of the repo
  (`graphify-out/graph.json`); Loom mounts it as a stdio MCP server
  (`graphify . --mcp`, packaged entry disabled until a graph exists).
  `/graphify build|update` indexes the repo and auto-enables the server;
  `/graphify` shows cli/graph/server status; `/graphify query|path|explain`
  runs one-off CLI queries. Once connected, the orchestrator and the
  explorer/searcher subagents get the read-only `query_graph` / `get_node` /
  `shortest_path` tools (always-allowed, never prompt) plus a system-prompt
  nudge to answer structure questions from the graph instead of
  glob/grep/read sweeps — a subgraph's worth of tokens with file:line
  citations. In airgap mode the tools stay subagent-only.
- **Every tool call names its caller.** Tool-call lines now always end
  with who issued them — `[orchestrator]` or
  `[editor · qwen3.6:27b (⌂ local)]` (from the message's model metadata,
  falling back to `[subagent]` for nested-graph messages that don't say) —
  so orchestrator and subagent activity are no longer indistinguishable.
- **Explicit end-of-turn marker.** Every turn now closes with a
  `✔ turn complete · <receipt>` (or `⏹ turn interrupted`) line — while
  it's absent, Loom is still streaming, so an intermediate message can't
  be mistaken for the final answer.
- **Receipts show % local and money saved.** The per-turn/session receipt
  now reads e.g. `$0.060 cloud + 89.0k local tokens (free) · 88% local,
  saved ~$0.375 vs all-cloud · session $0.060 (saved ~$0.38)`, and
  `/status` gains a "savings" row with the session's local-token share and
  the dollars avoided versus running everything on the cloud orchestrator.
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
- **Selected local models show in the welcome banner, toolbar, and
  `/status`.** They power the subagent roles, so they were invisible next
  to the cloud orchestrator/advisor: the banner now reads
  `model: ☁ claude-sonnet-5 · local: ⌂ qwen3.6:27b, … · advisor: ☁ …`, the
  bottom toolbar lists the same `⌂` tags, and `/status` gains a
  "local models" row. Fallback-aware: tags whose role is live-fallen-back
  to the cloud drop out instead of claiming to run locally.
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
