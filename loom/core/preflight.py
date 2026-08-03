"""Does this model actually answer?

Configuring a model and being *able to call it* are different things, and the
gap between them used to surface as a failed turn hours later. A key can be
valid while the model is off-limits: wrong region, not opted in, not on your
plan, retired, or simply misspelt. Only a real round-trip tells you.

So :func:`check_model` sends one tiny prompt and reports what came back, in
words that say what to do next. The wizard runs it before it saves, and
``loom doctor --probe`` runs it on demand.
"""

from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass

from loom.core import config as cfg
from loom.core import ollama as ollama_mod

# A probe is a real billed call, so keep it to the smallest thing that proves
# the round-trip works. Single digits of tokens.
PROBE = "Reply with the single word: ok"
TIMEOUT_SECONDS = 25.0


@dataclass(frozen=True)
class Check:
    """One model's verdict."""

    model: str
    ok: bool
    state: str  # ok | auth | forbidden | missing | offline | timeout | error | not-pulled
    detail: str  # what happened, in the provider's words where they help
    hint: str = ""  # what to do about it

    @property
    def is_local(self) -> bool:
        return self.state == "not-pulled" or ":" in self.model and "/" in self.model


def check_model(model_string: str, config: cfg.LoomConfig, *, timeout: float = TIMEOUT_SECONDS) -> Check:
    """Verify one configured model end to end.

    Local models are checked against the Ollama daemon (is it up, is the tag
    pulled) rather than by inference — starting a cold local model can take a
    minute and proves nothing the tag list doesn't already tell us. Cloud
    models get a real call, because that is the only thing that distinguishes
    "key accepted" from "key accepted but you may not use this model".
    """
    if config.is_local(model_string):
        return _check_local(model_string, config)
    return _check_cloud(model_string, config, timeout)


def _check_local(model_string: str, config: cfg.LoomConfig) -> Check:
    from loom.core.model_router import resolve

    tag = resolve(model_string).name
    status = ollama_mod.status(config)
    if not status.running:
        return Check(
            model_string,
            False,
            "offline",
            f"Ollama is not reachable at {status.endpoint}",
            ollama_mod.daemon_hint(status.endpoint) if status.installed else ollama_mod.INSTALL_HINT,
        )
    if not ollama_mod.is_served(tag, status.models):
        return Check(
            model_string,
            False,
            "not-pulled",
            f"`{tag}` is not downloaded",
            f"loom models pull {tag}",
        )
    return Check(model_string, True, "ok", "installed and served")


def _check_cloud(model_string: str, config: cfg.LoomConfig, timeout: float) -> Check:
    from loom.core.model_router import build_model

    try:
        model = build_model(model_string, config)
    except RuntimeError as exc:  # the router's own "env var is not set" message
        return Check(model_string, False, "auth", str(exc), "re-run /setup, or export the key")
    except Exception as exc:
        return Check(model_string, False, "error", _brief(exc))

    # A wedged provider must not hang the wizard behind an un-cancellable call.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(model.invoke, PROBE)
        reply = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return Check(
            model_string, False, "timeout", f"no answer within {timeout:.0f}s", "the provider may be degraded"
        )
    except Exception as exc:
        return _classify(model_string, exc)
    finally:
        pool.shutdown(wait=False)

    text = str(getattr(reply, "content", "") or "").strip()
    if not text:
        # It answered, which is what we were testing; an empty body is odd but
        # not a configuration problem.
        return Check(model_string, True, "ok", "answered (empty body)")
    return Check(model_string, True, "ok", f"answered {text[:40]!r}")


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    if code is None:  # langchain sometimes only leaves it in the text
        match = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
        code = int(match.group(1)) if match else None
    return code


def _provider_message(exc: Exception) -> str:
    """The provider's own explanation, which is usually the only part worth
    reading — "only available hosted in China and requires explicit opt in"
    tells you far more than "403 Forbidden"."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
        if body.get("message"):
            return str(body["message"])
    match = re.search(r"'message':\s*'([^']+)'", str(exc)) or re.search(r'"message":\s*"([^"]+)"', str(exc))
    return match.group(1) if match else ""


def _classify(model_string: str, exc: Exception) -> Check:
    code = _status_code(exc)
    message = _provider_message(exc) or _brief(exc)
    name = type(exc).__name__

    if code in (401, 403) or "authentication" in name.lower() or "permissiondenied" in name.lower():
        # 401 is nearly always the key. 403 nearly always is not — the key was
        # accepted and then this particular model was refused, which is a very
        # different thing to tell someone.
        if code == 401 or "authentication" in name.lower():
            return Check(model_string, False, "auth", message, "the key was rejected — re-run /setup")
        return Check(
            model_string,
            False,
            "forbidden",
            message,
            "your key works, but not for this model — pick another",
        )
    if code == 404 or "notfound" in name.lower():
        return Check(
            model_string, False, "missing", message or "no such model", "check the model id, or pick another"
        )
    if code == 429:
        # Rate-limited means it answered the auth question affirmatively.
        return Check(model_string, True, "ok", "rate-limited, but the key and model are valid")
    if code and code >= 500:
        return Check(model_string, False, "error", message, "the provider is having trouble; try again")
    if "connect" in name.lower() or "timeout" in name.lower():
        return Check(model_string, False, "offline", message, "check your network or the provider's base URL")
    return Check(model_string, False, "error", message)


def _brief(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    return text[:200] or type(exc).__name__


def check_plan(models: dict[str, str], config: cfg.LoomConfig, *, timeout: float = TIMEOUT_SECONDS):
    """Verify every distinct model in a role → model-string plan.

    Yields ``(roles, Check)`` with the roles sharing that model collapsed, so a
    fleet of seven subagents on one tag is probed (and billed) once.
    """
    by_model: dict[str, list[str]] = {}
    for role, model_string in models.items():
        by_model.setdefault(model_string, []).append(role)
    for model_string, roles in by_model.items():
        yield roles, check_model(model_string, config, timeout=timeout)
