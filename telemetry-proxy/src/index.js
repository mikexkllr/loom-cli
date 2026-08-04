/**
 * Loom telemetry ingest proxy.
 *
 * Why this exists: Langfuse project keys are read+write — there is no
 * ingest-only scope — so a key baked into a public binary lets anyone who
 * downloads Loom read every full-tracing user's uploaded prompts and source.
 * This worker holds the real secret server-side and speaks just enough of the
 * Langfuse ingest API that the Langfuse SDK can point at it unmodified:
 * Loom sets LANGFUSE_HOST to this worker and bakes a write-only client token
 * in place of the secret key.
 *
 * The token is still extractable from the binary. That is fine and expected —
 * it grants writes only, and rotating LOOM_CLIENT_TOKEN invalidates every old
 * binary's ability to write. A leaked write token means spam; a leaked
 * Langfuse secret key means someone reads your users' source code.
 *
 * Secrets (wrangler secret put):
 *   LOOM_CLIENT_TOKEN   what Loom binaries send as the basic-auth password
 *   LANGFUSE_PUBLIC_KEY real Langfuse project public key
 *   LANGFUSE_SECRET_KEY real Langfuse project secret key
 * Vars:
 *   LANGFUSE_HOST       upstream, defaults to https://cloud.langfuse.com
 */

// Paths the Langfuse SDKs post to. The Python SDK v3 uses /api/public/ingestion;
// the OTLP path is accepted too so switching transports needs no redeploy.
const INGEST_PATHS = ["/api/public/ingestion", "/api/public/otel/v1/traces"];

// The ONE readable path, allowlisted by exact match. The SDK's auth_check()
// and Loom's own credential verification both call it, and it returns only the
// project's id and name — no traces, no observations, no prompts.
//
// This is an allowlist and must stay one: forwarding GETs by prefix would hand
// the client token /api/public/traces, which is the entire thing this worker
// exists to prevent.
const READ_PATHS = ["/api/public/projects"];

// Langfuse batches are small; anything this size is not a trace batch.
const MAX_BODY_BYTES = 5 * 1024 * 1024;

const json = (status, body) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });

/** Constant-time string compare, so the token can't be recovered by timing. */
function tokenMatches(given, expected) {
  if (typeof given !== "string" || typeof expected !== "string") return false;
  const a = new TextEncoder().encode(given);
  const b = new TextEncoder().encode(expected);
  if (a.byteLength !== b.byteLength) return false;
  try {
    return crypto.subtle.timingSafeEqual(a, b);
  } catch {
    // Older runtimes: fall back to a manual constant-time compare rather than
    // to `===`, which would leak length-prefix information under load.
    let diff = 0;
    for (let i = 0; i < a.length; i++) diff |= a[i] ^ b[i];
    return diff === 0;
  }
}

/** The password half of an HTTP basic auth header, or "". */
function basicAuthPassword(header) {
  if (!header || !header.startsWith("Basic ")) return "";
  try {
    const decoded = atob(header.slice(6));
    const separator = decoded.indexOf(":");
    return separator === -1 ? "" : decoded.slice(separator + 1);
  } catch {
    return "";
  }
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // Loom's preflight (`/doctor`, `/privacy`) checks reachability before it
    // promises the user that tracing works. Unauthenticated on purpose: it
    // reveals nothing and must answer even for a binary with a stale token.
    if (request.method === "GET" && (url.pathname === "/health" || url.pathname === "/api/public/health")) {
      return json(200, { status: "ok", service: "loom-telemetry-proxy" });
    }

    const isIngest = INGEST_PATHS.some((p) => url.pathname === p || url.pathname.startsWith(p + "/"));
    const isRead = READ_PATHS.includes(url.pathname);
    if (!isIngest && !isRead) {
      return json(404, { message: "not found" });
    }
    if (isIngest && request.method !== "POST") {
      return json(405, { message: "method not allowed" });
    }
    if (isRead && request.method !== "GET") {
      return json(405, { message: "method not allowed" });
    }

    if (!tokenMatches(basicAuthPassword(request.headers.get("authorization")), env.LOOM_CLIENT_TOKEN)) {
      // Same shape Langfuse returns, so the SDK's own error handling applies
      // and Loom's verify path reports "credentials rejected" as it would
      // against the real API.
      return json(401, { message: "Invalid credentials. Confirm that you've configured the correct host." });
    }

    const declared = Number(request.headers.get("content-length") || 0);
    if (declared > MAX_BODY_BYTES) {
      return json(413, { message: "payload too large" });
    }

    // Per-IP throttle. Guarded because the binding is optional — a deploy
    // without it should still work rather than 500 on every request.
    if (env.RATE_LIMITER) {
      const ip = request.headers.get("cf-connecting-ip") || "unknown";
      const { success } = await env.RATE_LIMITER.limit({ key: ip });
      if (!success) return json(429, { message: "rate limited" });
    }

    const upstream = (env.LANGFUSE_HOST || "https://cloud.langfuse.com").replace(/\/+$/, "");
    const credentials = btoa(`${env.LANGFUSE_PUBLIC_KEY}:${env.LANGFUSE_SECRET_KEY}`);

    let response;
    try {
      response = await fetch(upstream + url.pathname + url.search, {
        method: request.method,
        headers: {
          "content-type": request.headers.get("content-type") || "application/json",
          authorization: `Basic ${credentials}`,
          // Lets you tell proxied traffic apart from anything writing directly.
          "x-loom-proxy": "1",
        },
        body: isIngest ? request.body : undefined,
      });
    } catch (err) {
      // Never surface upstream detail to the client: it is the one place the
      // real Langfuse host and key state could leak back out.
      console.error("upstream ingest failed", err && err.message);
      return json(502, { message: "upstream unavailable" });
    }

    // Pass the body straight through — the SDK reads per-event statuses out of
    // a 207 and will retry the ones that failed.
    return new Response(response.body, {
      status: response.status,
      headers: { "content-type": response.headers.get("content-type") || "application/json" },
    });
  },
};
