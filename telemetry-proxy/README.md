# Loom telemetry ingest proxy

A Cloudflare Worker that sits between Loom binaries and Langfuse.

## Why this exists

Loom's `full` privacy mode uploads LLM traces — prompts, completions, tool
calls, and therefore source code. Sending those directly to Langfuse would mean
baking a Langfuse key pair into a public binary, and **Langfuse has no
ingest-only credential**: its API has a single auth scheme, and a project key
pair grants reads as well as writes. Anyone who downloaded Loom could extract
the key and read every full-tracing user's uploaded code.

So the binary never gets a Langfuse key. It gets a `lct_…` client token and
talks to this worker, which holds the real credentials as Cloudflare secrets.

```
loom binary ──POST /api/public/ingestion──▶ this worker ──▶ cloud.langfuse.com
              basic auth: pk-lf-loom-ingest / lct_…          real pk / sk
              (extractable, write-only)                      (never shipped)
```

The client token being extractable is expected and fine. It authorises writes
and nothing else, and rotating `LOOM_CLIENT_TOKEN` retires every binary that
carries the old one. A leaked write token means spam; a leaked Langfuse secret
key means someone reads your users' code.

## What it will and won't serve

| Path | Method | Behaviour |
| --- | --- | --- |
| `/health`, `/api/public/health` | GET | 200, unauthenticated — Loom's `/doctor` reachability check |
| `/api/public/ingestion` | POST | forwarded to Langfuse |
| `/api/public/otel/v1/traces` | POST | forwarded to Langfuse |
| `/api/public/projects` | GET | forwarded — returns project id + name only; the SDK's `auth_check()` and Loom's credential verification both need it |
| everything else | any | 404 |

`READ_PATHS` in `src/index.js` is an **allowlist and must stay one**. Forwarding
GETs by prefix would expose `/api/public/traces`, which is the entire thing this
worker exists to prevent.

## Deploying

Pushes to `main` that touch `telemetry-proxy/**` deploy automatically via
`.github/workflows/deploy-telemetry-proxy.yml`. Manually:

```bash
cd telemetry-proxy && wrangler deploy
```

Secrets are set once and are **not** managed by the workflow (`wrangler deploy`
leaves existing secrets alone, so they never pass through a CI log):

```bash
wrangler secret put LOOM_CLIENT_TOKEN     # what binaries send; also a GH Actions secret
wrangler secret put LANGFUSE_PUBLIC_KEY   # real Langfuse project key
wrangler secret put LANGFUSE_SECRET_KEY   # real Langfuse project secret
```

## Rotating the client token

Old binaries stop being able to write, which is the point.

```bash
NEW="lct_$(openssl rand -hex 24)"
printf '%s' "$NEW" | wrangler secret put LOOM_CLIENT_TOKEN
printf '%s' "$NEW" | gh secret set LOOM_CLIENT_TOKEN   # baked into the next release
```

## Local development

`.dev.vars` (gitignored) holds local values, then:

```bash
wrangler dev --port 8799
curl -s localhost:8799/health
```

Point the real SDK at it to test the whole path:

```bash
LANGFUSE_HOST=http://localhost:8799 \
LANGFUSE_PUBLIC_KEY=pk-lf-loom-ingest \
LANGFUSE_SECRET_KEY=lct_local_dev_token \
  python -c "from langfuse import get_client; c=get_client(); print(c.auth_check())"
```

## Still open

Trace deletion on request. Mode 3 uploads source code, so a user should be able
to ask for their traces to be removed — which needs an anonymous install id
stamped on each trace from the Loom side before it can be honoured.
