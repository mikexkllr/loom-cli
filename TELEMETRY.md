# Telemetry, privacy, and the ingest proxy

Everything about what Loom sends off a user's machine: the consent model, where
credentials live, why traces go through a Cloudflare Worker instead of straight
to Langfuse, and how to operate it.

If you only read one thing, read [Two secret stores](#two-secret-stores) — it is
the part that looks wrong until you know why.

---

## The three modes

Chosen once in setup (quick *and* advanced), changeable any time with
`/privacy` or `loom privacy`. **The default is `none`.**

| Mode | Label | What leaves the machine |
| --- | --- | --- |
| `none` | none | Nothing. No SDK is imported, no socket opened. |
| `errors` | bug reports | Crashes only, via Sentry: exception type, message, stack trace. No prompts, no code, no locals, no argv, no hostname. |
| `full` | full tracing | Everything `errors` sends, plus complete LLM traces via Langfuse: prompts, completions, tool calls, delegations, tokens, timings. |

`full` is the training corpus for a distilled local orchestrator. It
**necessarily contains source code**, so it takes a second explicit
confirmation behind a warning card, defaulting to no.

Defined in `MODES` in `loom/core/telemetry.py`. Each mode declares both what it
sends *and* what it never sends, and the setup screen prints both — a consent
screen that only lists benefits is not consent.

## Two consent gates

Both must pass before a single byte is sent.

1. **Global mode** — chosen once, stored in `$LOOM_HOME/telemetry.json`.
2. **Per-project answer** — asked the first time Loom runs in a given git repo.
   Agreeing to share crash reports from a hobby project is not agreeing to
   share them from work.

Project identity is the **git work-tree root** (`telemetry.project_key`), so
`repo/backend` and `repo/frontend` are one decision, not two. Non-git
directories key on the resolved path.

Everything fails closed:

| Situation | Effective mode |
| --- | --- |
| Never asked | `none` |
| Global `full`, project never asked | `none` |
| Global `full`, project declined | `none` |
| Global `full`, project accepted | `full` |
| Consent file corrupt or unparseable | `none` |
| `LOOM_TELEMETRY=none` set | `none` |
| `LOOM_TELEMETRY=full` set, stored mode `none` | `none` — the env var can only **tighten** |

That last row is deliberate: a stray environment variable must never be able to
switch on tracing nobody consented to.

## Where consent lives

`$LOOM_HOME/telemetry.json` (default `~/.loom/telemetry.json`), mode `0600`.

**Never** in a project's `.loom/settings.json` — that file is meant to be
committed, and one developer's privacy choice is not a team-wide setting.

---

## Two secret stores

The single most confusing thing here. Two stores hold similarly-named secrets
and they are unrelated.

| Store | Key | Read by | Contains |
| --- | --- | --- | --- |
| **Cloudflare Workers** | `LANGFUSE_SECRET_KEY` | the worker, at runtime | the real Langfuse secret |
| **Cloudflare Workers** | `LANGFUSE_PUBLIC_KEY` | the worker, at runtime | the real Langfuse public key |
| **Cloudflare Workers** | `LOOM_CLIENT_TOKEN` | the worker, at runtime | the token it accepts from binaries |
| **GitHub Actions** | `LOOM_CLIENT_TOKEN` | `packaging/loom.spec` at release | the token it bakes into binaries |
| **GitHub Actions** | `CLOUDFLARE_API_TOKEN` | the deploy workflow | deploys the worker |
| **GitHub Actions** | `CLOUDFLARE_ACCOUNT_ID` | the deploy workflow | account to deploy to |

**The Langfuse secret is only ever in Cloudflare.** It is not in GitHub, not in
the repo, and not in any binary.

A GitHub secret named `LOOM_LANGFUSE_SECRET_KEY` existed briefly, for an earlier
design that baked the Langfuse secret into the binary. Nothing reads it now and
it has been deleted. If you see it referenced anywhere, that reference is stale.

The deploy workflow deliberately does **not** manage worker secrets —
`wrangler deploy` leaves existing secrets alone, so they never pass through a CI
log. Set them once by hand.

---

## Why traces go through a proxy

Sentry's DSN is safe to publish by construction: it ingests events and cannot
read them. So Loom ships one, and `errors` mode is zero-setup.

**Langfuse has no equivalent.** Its API has a single auth scheme, and a project
key pair grants reads as well as writes — there is no ingest-only scope
([Public API docs](https://langfuse.com/docs/api-and-data-platform/features/public-api),
[Security FAQ](https://langfuse.com/security/security-faq)). Baking one into a
public binary would let anyone who downloaded Loom read every full-tracing
user's prompts and source code.

So the binary never gets a Langfuse key:

```
loom binary ──POST /api/public/ingestion──▶ telemetry-proxy ──▶ cloud.langfuse.com
              basic auth:                    (Cloudflare Worker)   real pk / sk
              pk-lf-loom-ingest / lct_…       holds real creds      never shipped
              extractable, WRITE-ONLY
```

The `lct_…` client token being extractable is expected and fine. It authorises
writes and nothing else, and rotating the worker's `LOOM_CLIENT_TOKEN` retires
every binary carrying the old one. **A leaked write token means spam; a leaked
Langfuse secret key means someone reads your users' code.**

Worker source: `telemetry-proxy/src/index.js`. It is public and contains no
credentials — it reads them from `env` at runtime.

### What the proxy will and won't serve

| Path | Method | Behaviour |
| --- | --- | --- |
| `/health`, `/api/public/health` | GET | 200, unauthenticated |
| `/api/public/ingestion` | POST | forwarded |
| `/api/public/otel/v1/traces` | POST | forwarded |
| `/api/public/projects` | GET | forwarded — returns id + name only |
| everything else | any | 404 |

`READ_PATHS` is an **allowlist and must stay one**. Forwarding GETs by prefix
would expose `/api/public/traces`, which is the entire thing the worker exists
to prevent. `/api/public/projects` is allowed only because the SDK's
`auth_check()` and Loom's credential verification both need it.

### Credential precedence

`_init_langfuse` and `_init_sentry` resolve in this order, so anyone can point
telemetry at their own project:

1. real environment (`LANGFUSE_*`, `SENTRY_DSN`)
2. the user's consent record (`/privacy setup`)
3. the bundled defaults (proxy host + baked token)

---

## What a crash report actually contains

Verified by capturing a real event through a fake transport, not by reading the
config. Frames carry `abs_path`, `filename`, `function`, `lineno`, `module` —
and nothing else.

Removed at init (`_init_sentry`):

- `include_local_variables=False` — locals would carry file contents
- `include_source_context=False` — **source lines around the crash**
- `send_default_pii=False`, `max_breadcrumbs=0`, `traces_sample_rate=0.0`
- `default_integrations=False`, `auto_enabling_integrations=False`

Removed again in `scrub_event` (belt and braces — a promise this specific
should not rest on one flag being right):

- frame `vars`, `pre_context`, `context_line`, `post_context`
- `server_name` (hostname), `user`, `request`, `modules`
- `extra["sys.argv"]` — for a coding assistant, argv is the user's prompt
- all breadcrumbs
- `$HOME` and the username redacted from every string
- any dict key matching `_SECRET_MARKERS` replaced with `<redacted>`

Only four Sentry integrations are enabled — `excepthook` (with
`always_run=True`, since the REPL looks interactive), `atexit` (with a silent
callback), `dedupe`, and `threading` (LangGraph runs tool calls on worker
threads, where `sys.excepthook` never fires). The defaults would auto-enable
`langchain`, `langgraph`, `mcp` and `httpx`, patching the exact path Loom uses
to call models, for a user who asked for crashes only.

That explicit set is also the PyInstaller-safe one: a bundled integration
submodule that failed to collect raises `ModuleNotFoundError` out of `init()`,
which the surrounding `except` would swallow into "telemetry silently off" in
the shipped binary.

---

## What actually gets reported

Loom swallows almost everything so that one broken thing never ends a session.
The cost is that a caught error is invisible to whoever has to fix it, so every
such site reports on the way past via `telemetry.report(where, exc)`. The
`where` tag is a fixed label chosen in the source, never user content.

| `where` | Site |
| --- | --- |
| `turn`, `/<command>` | REPL turn and slash-command crashes (`_report_crash`) |
| `turn.stream`, `turn.invoke` | streaming failed; the synchronous retry failed too |
| `bundle.build`, `bundle.dependency` | the orchestrator could not be constructed |
| `startup.*`, `task` | the headless `loom "task"` path |
| `tool` | an exception escaped a tool (`PolicyMiddleware.wrap_tool_call`) |
| `tool.result` | a tool returned an error `ToolMessage` instead of raising |
| `mcp.connect` | an MCP server failed to start and was skipped |
| `compact` | `/compact` could not summarise the transcript |
| `privacy.test` | the synthetic event `/privacy test` sends |

`tool.result` carries only the tool name and the leading exception class name —
never the message body, which routinely quotes a path or a line of the user's
file — and reports once per `(tool, error class)` per process. A coding agent
guesses paths and greps for things that aren't there by design; without that
cap the real failures drown.

Two things had to be true before any of this reached Sentry, and neither was:

- **Telemetry has to be activated.** It was REPL-only, so every headless run
  (`loom "task"`, `--loop`, CI, scripts) reported nothing no matter what mode
  the user had chosen. `_run_task` now activates and flushes like the REPL.
- **The user has to have answered.** With no `telemetry.json` the mode is
  `none` and nothing is sent — correctly, but indistinguishably from "reporting
  is on and nothing broke". `/doctor` now separates *never answered* from
  *answered none*, and the startup consent prompt no longer fails silently.

### Verify it end to end

```
loom privacy test
```

Activates, captures a real `RuntimeError`, flushes, and — in `full` mode —
authenticates against the trace endpoint too. It says which link is missing
when it can't: mode `none`, project not opted in, an SDK that never
initialised, or credentials the host refuses. It is the only check that
exercises the whole path — a configuration screen can only report the first
two.

The trace check is not redundant. The Langfuse SDK uploads on a background
thread and logs **nothing** at any level when the endpoint answers 401, so a
rejected trace and a delivered trace are indistinguishable from the terminal.

### Keys and host are one credential

`langfuse_credentials()` resolves the public key, secret key and host together
and never mixes sources, because they are not independent:

| Credential | Valid only at |
| --- | --- |
| bundled `pk-lf-loom-ingest` + `lct_…` | Loom's ingest proxy |
| a user's own Langfuse pair | Langfuse Cloud, or their self-hosted instance |

Resolving them field by field produced the one combination that authenticates
nowhere. `Consent.langfuse_host` defaulted to `https://cloud.langfuse.com` and
was written into every consent record — including those of users who supplied
no keys at all — so it shadowed the proxy host while the keys still fell
through to the bundle. Every full-tracing user was posting the proxy's
write-only token to the public API, which rejected it:

```
$ curl https://cloud.langfuse.com/api/public/projects -H "authorization: Basic <bundled>"
{"message":"Invalid credentials. Confirm that you've configured the correct host."}   # 401

$ curl -X POST https://loom-telemetry.telemetry-proxy.workers.dev/api/public/otel/v1/traces …
200
```

`LANGFUSE_HOST` in the environment still overrides in every branch — that is
how the proxy gets pointed at a local capture server under test. A stored
`cloud.langfuse.com` with no keys beside it is read as unset, so records
written by older builds heal on load rather than needing a reset.

---

## Operations

### Deploy the worker

Automatic on pushes to `main` touching `telemetry-proxy/**`
(`.github/workflows/deploy-telemetry-proxy.yml`), plus `workflow_dispatch`. The
workflow re-checks `/health` afterwards, because a deploy that succeeds while
serving 500s is the failure worth catching.

Manually:

```bash
cd telemetry-proxy && wrangler deploy
```

`wrangler-action` is pinned to wrangler `4.118.0`. It defaults to v3, which
cannot read `wrangler.jsonc` and fails with a misleading `Missing entry-point`
instead of a config-format error.

### Set worker secrets (once)

```bash
cd telemetry-proxy
wrangler secret put LOOM_CLIENT_TOKEN
wrangler secret put LANGFUSE_PUBLIC_KEY
wrangler secret put LANGFUSE_SECRET_KEY
```

### Rotate the client token

Old binaries lose the ability to write, which is the point.

```bash
NEW="lct_$(openssl rand -hex 24)"
printf '%s' "$NEW" | wrangler secret put LOOM_CLIENT_TOKEN   # worker accepts it
printf '%s' "$NEW" | gh secret set LOOM_CLIENT_TOKEN         # next release bakes it
```

### Verify the deployment

```bash
U=https://loom-telemetry.telemetry-proxy.workers.dev
TOKEN=<the client token>

curl -s $U/health
curl -s -o /dev/null -w '%{http_code}\n' -u "pk:$TOKEN" "$U/api/public/traces?limit=1"   # must be 404
curl -s -o /dev/null -w '%{http_code}\n' -X POST -u 'pk:WRONG' $U/api/public/ingestion -d '{}'  # must be 401
curl -s -o /dev/null -w '%{http_code}\n' -X POST -u "pk:$TOKEN" \
  -H 'content-type: application/json' $U/api/public/ingestion -d '{"batch":[]}'          # must be 207
```

A `404` on `/api/public/traces` **with a valid write token** is the security
property. If that ever returns 200, stop and fix it before anything else.

### Local development

`telemetry-proxy/.dev.vars` (gitignored) holds local values:

```bash
cd telemetry-proxy && wrangler dev --port 8799
```

Point the real SDK at it to exercise the whole path:

```bash
LANGFUSE_HOST=http://localhost:8799 \
LANGFUSE_PUBLIC_KEY=pk-lf-loom-ingest \
LANGFUSE_SECRET_KEY=lct_local_dev_token \
  python -c "from langfuse import get_client; print(get_client().auth_check())"
```

`loom/_built.py` (gitignored) simulates the binary's baked token on a dev
machine. Delete it to test the source-install fallback path.

---

## Testing

`tests/conftest.py` has an autouse `_no_real_telemetry` fixture that blanks all
three bundled credentials for **every** test. Two reasons, and both were real
bugs:

- Without it, any test reaching `telemetry.activate()` starts a live Sentry
  client against the production DSN. The suite was opening a session on Loom's
  real Sentry project on every full run.
- The baked token in `loom/_built.py` is gitignored, so it exists only on a
  machine that has built a release. Tests that branched on "is a bundled
  credential available" passed in CI and failed locally.

Tests that are *about* bundled credentials set obvious fakes themselves
(`https://bundled@example.invalid/2`). **Never point a test at a real DSN or
key.**

```bash
uv run pytest -q          # 776 passing
uv run ruff check loom/ tests/
```

---

## Still open

**Trace deletion on request.** `full` mode uploads source code, so a user should
be able to ask for their traces to be removed. That needs an anonymous install
id stamped on each trace from the Loom side before it can be honoured. Nothing
implements this yet.

**Retention.** No retention policy is configured on the Langfuse project. Traces
accumulate indefinitely.

---

## File map

| Path | Role |
| --- | --- |
| `loom/core/telemetry.py` | modes, consent store, gating, Sentry/Langfuse init, `scrub_event` |
| `loom/core/telemetry_setup.py` | `sentry` / `langfuse` CLI bridges used during setup |
| `loom/ui/privacy.py` | the wizard step, per-project gate, `/privacy` rendering |
| `loom/ui/onboarding.py` | calls the privacy step in quick *and* advanced setup |
| `loom/ui/repl.py` | "run setup now?", per-directory ask, activation, callbacks, crash capture, flush |
| `loom/cli/main.py` | the same activation/capture/flush for headless `loom "task"` runs |
| `loom/middleware/policy.py` | reports tool crashes and error results — the only chokepoint every tool call passes |
| `telemetry-proxy/` | the Cloudflare Worker, its config and README |
| `packaging/loom.spec` | bakes `LOOM_CLIENT_TOKEN` into `loom/_built.py` at freeze time |
| `.github/workflows/deploy-telemetry-proxy.yml` | auto-deploy on worker changes |
| `tests/test_telemetry.py`, `tests/test_privacy.py` | consent gating, scrubbing, wizard behaviour |
