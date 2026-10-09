"""Usage tracking + cost receipts — the measurable half of the hybrid pitch.

A LangChain callback handler records every model call (orchestrator and
subagents alike — callbacks propagate into nested runs), attributes it to the
role that made it, classifies it local vs cloud, and prices the cloud tokens.
After each turn the REPL prints a receipt: what this task cost, how much of it
the orchestrator spent on itself, what the free local tokens would have cost on
the cloud, and the session running total.

Three things this module has to get right, each of which it previously got
wrong:

* **Cached input is not full-price input.** Providers bill a cache read at a
  fraction of the base input rate and a cache write at a premium, and
  ``usage_metadata["input_tokens"]`` is the sum of *all* input token types. On a
  long conversation with prompt caching on — which deepagents enables for
  Anthropic by default — pricing every input token at the uncached rate
  overstates the bill by most of an order of magnitude.
* **Who spent it.** Attribution by model name cannot separate the orchestrator
  from a subagent that happens to share its model, which is exactly the question
  worth asking of a delegating architecture. Roles come from the callback run
  tree instead: a ``task`` tool call owns every model call beneath it.
* **What "saved" is measured against.** The counterfactual has to be a cloud
  model. Pricing local tokens against a local orchestrator produced an invented
  saving in local-only mode, where nothing was billed at all.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from loom.core.config import LoomConfig
from loom.core.model_router import resolve

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - allows import without langchain
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass


class Price(NamedTuple):
    """Cloud pricing for one model family.

    ``inp``/``out`` are USD per million tokens. ``cache_read`` and
    ``cache_write`` are multipliers on ``inp``: providers charge a discount to
    replay a cached prefix and a premium to write one. ``estimated`` marks a
    price Loom guessed rather than knows, so the UI can say so instead of
    presenting a fabricated figure with the same confidence as a real one.
    """

    inp: float
    out: float
    cache_read: float = 0.1
    cache_write: float = 1.25
    estimated: bool = False


# USD per million tokens. Cloud models only — local is free.
# Prices per provider pricing pages (2026-07); unknown cloud models fall back to
# Sonnet-tier so receipts stay conservative rather than absent.
CLOUD_PRICES: dict[str, Price] = {
    # Anthropic: cache read 0.1x, 5-minute cache write 1.25x.
    "claude-fable-5": Price(10.0, 50.0),
    "claude-opus-4-8": Price(5.0, 25.0),
    "claude-opus-4-7": Price(5.0, 25.0),
    "claude-opus-4-6": Price(5.0, 25.0),
    "claude-opus-4-5": Price(5.0, 25.0),
    "claude-opus": Price(5.0, 25.0),
    "claude-sonnet-5": Price(3.0, 15.0),
    "claude-sonnet-4-6": Price(3.0, 15.0),
    "claude-sonnet-4-5": Price(3.0, 15.0),
    "claude-sonnet": Price(3.0, 15.0),
    "claude-haiku-4-5": Price(1.0, 5.0),
    "claude-haiku": Price(1.0, 5.0),
    # OpenAI: cached input discounted, no separate write charge.
    "gpt-5.6-sol": Price(5.0, 30.0, cache_write=1.0),
    "gpt-5.6-terra": Price(2.5, 15.0, cache_write=1.0),
    "gpt-5.6-luna": Price(1.0, 6.0, cache_write=1.0),
    "gpt-5.6": Price(5.0, 30.0, cache_write=1.0),  # bare alias routes to Sol
    "gpt-4o": Price(2.5, 10.0, cache_write=1.0),
    "gpt-4.1": Price(2.0, 8.0, cache_write=1.0),
}
_DEFAULT_CLOUD_PRICE = Price(3.0, 15.0, estimated=True)
FREE_PRICE = Price(0.0, 0.0)

# Gateways that publish a no-charge tier mark it in the model id. OpenCode Zen
# uses a "-free" suffix (`deepseek-v4-flash-free`) — and answers such a request
# reporting the *upstream* model's name with the suffix stripped, so this can only
# be decided from the model string the user configured, never from the response.
_FREE_SUFFIXES = ("-free", ":free")


def is_free_model(model_string: str) -> bool:
    """Whether a configured model string names a provider's free tier."""
    return model_string.strip().lower().endswith(_FREE_SUFFIXES)

# Used when nothing in the config offers a cloud model to price the
# "what if this had all run in the cloud" counterfactual against.
DEFAULT_CLOUD_REFERENCE = "claude-sonnet-5"

# Run-metadata key a caller uses to name the role it is spending on behalf of.
ROLE_KEY = "loom_role"

# Opt-in raw log. Set to a file path and every model call, tool call and
# subagent/advisor result is appended to it as one JSON line, so a receipt can
# be checked against the per-call numbers it was summed from.
USAGE_LOG_ENV = "LOOM_USAGE_LOG"


def _result_text(output: Any) -> str:
    """The text a tool handed back, whether it returned a plain value, a message,
    or a LangGraph ``Command`` carrying one (deepagents' ``task`` does)."""
    update = getattr(output, "update", None)
    if isinstance(update, dict):
        messages = update.get("messages") or []
        if messages:
            output = messages[-1]
    content = getattr(output, "content", output)
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
    return "" if content is None else str(content)


def role_metadata(config: Any, role: str) -> dict[str, Any]:
    """``config`` with ``role`` stamped on its run metadata.

    For a model invoked directly rather than through the agent graph — the
    advisor's ``consult``, ``/compact``'s summarizer — where the run tree cannot
    say who spent the tokens. The rest of the config (callbacks, thread, tags) is
    carried through untouched, so the tracker still sees the call at all.
    """
    merged = dict(config or {})
    merged["metadata"] = {**(merged.get("metadata") or {}), ROLE_KEY: role}
    return merged


def price_entry(model_name: str) -> Price:
    """Full pricing for a cloud model, longest-prefix match.

    Accepts a bare model name or a full ``provider/model`` string; the provider
    prefix is dropped before matching, so a free-tier suffix is still seen.
    """
    name = model_name.strip().lower()
    if is_free_model(name):
        return FREE_PRICE
    try:
        name = resolve(name).name
    except Exception:
        pass
    best: Price | None = None
    best_len = -1
    for prefix, price in CLOUD_PRICES.items():
        if name.startswith(prefix) and len(prefix) > best_len:
            best, best_len = price, len(prefix)
    return best or _DEFAULT_CLOUD_PRICE


def price_for(model_name: str) -> tuple[float, float]:
    """(input, output) USD per MTok for a cloud model, longest-prefix match."""
    p = price_entry(model_name)
    return (p.inp, p.out)


def cost_usd(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """USD for one model's tokens, discounting cached input.

    ``input_tokens`` is the provider's total — cached tokens included, matching
    ``usage_metadata`` — so the cached counts are subtracted out and repriced
    rather than added on top.
    """
    p = price_entry(model_name)
    cached = max(0, cache_read_tokens) + max(0, cache_write_tokens)
    uncached = max(0, input_tokens - cached)
    return (
        uncached * p.inp
        + max(0, cache_read_tokens) * p.inp * p.cache_read
        + max(0, cache_write_tokens) * p.inp * p.cache_write
        + output_tokens * p.out
    ) / 1_000_000


@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    # Subsets of input_tokens, not additions to it.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def merge(self, other: "ModelUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.calls += other.calls


class Actor(NamedTuple):
    """Who made a set of model calls.

    ``model`` is what the provider reported serving the call. ``billed_as`` is
    the string it is priced against, which is usually the same — it differs when
    a gateway reports a name Loom has no price for and the role's configured
    model is the better answer (see ``UsageTracker._billing_model``).
    """

    role: str  # "orchestrator", a subagent name, "advisor", or "?"
    model: str
    is_local: bool
    billed_as: str = ""

    @property
    def price_model(self) -> str:
        return self.billed_as or self.model


# The orchestrator's own spending, for the delegation ratio. "?" is folded in
# with it: an unattributed call was made outside any `task`, which in practice
# means the main graph.
_MAIN_ROLES = ("orchestrator", "?")

# Housekeeping on the session itself, not work on the user's task: neither the
# orchestrator reasoning nor a subagent it delegated to. Counted in the cost —
# compacting a long session is not free — but kept out of the delegation ratio,
# which would otherwise blame the orchestrator for every /compact.
_OVERHEAD_ROLES = ("compaction",)


@dataclass
class TurnUsage:
    """Token accounting for one turn (or a whole session), by actor."""

    actors: dict[Actor, ModelUsage] = field(default_factory=dict)

    def add(
        self,
        model: str,
        is_local: bool,
        inp: int,
        out: int,
        *,
        role: str = "?",
        cache_read: int = 0,
        cache_write: int = 0,
        billed_as: str = "",
    ) -> None:
        mu = self.actors.setdefault(Actor(role, model, is_local, billed_as), ModelUsage())
        mu.input_tokens += inp
        mu.output_tokens += out
        mu.cache_read_tokens += cache_read
        mu.cache_write_tokens += cache_write
        mu.calls += 1

    # ----- views -----

    def _by_model(self, *, local: bool) -> dict[str, ModelUsage]:
        out: dict[str, ModelUsage] = {}
        for actor, mu in self.actors.items():
            if actor.is_local is not local:
                continue
            out.setdefault(actor.model, ModelUsage()).merge(mu)
        return out

    @property
    def cloud(self) -> dict[str, ModelUsage]:
        """model -> usage, for every billed call."""
        return self._by_model(local=False)

    @property
    def local(self) -> dict[str, ModelUsage]:
        """model -> usage, for every free call."""
        return self._by_model(local=True)

    def rows(self) -> list[tuple[Actor, ModelUsage]]:
        """Per-actor usage, most expensive first, then most tokens."""
        return sorted(
            self.actors.items(),
            key=lambda kv: (-self.cost_of(kv[0], kv[1]), -kv[1].total_tokens),
        )

    @staticmethod
    def cost_of(actor: Actor, usage: ModelUsage) -> float:
        if actor.is_local:
            return 0.0
        return cost_usd(
            actor.price_model,
            usage.input_tokens,
            usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )

    @staticmethod
    def is_estimated(actor: Actor) -> bool:
        """Whether this row's price is a guess. Callers should say so rather than
        print a fabricated number as if it were billed."""
        return not actor.is_local and price_entry(actor.price_model).estimated

    def has_estimates(self) -> bool:
        return any(self.is_estimated(a) for a in self.actors)

    @property
    def cloud_cost(self) -> float:
        return sum(self.cost_of(a, u) for a, u in self.actors.items())

    @property
    def cache_read_tokens(self) -> int:
        return sum(u.cache_read_tokens for a, u in self.actors.items() if not a.is_local)

    def tokens(self, bucket: dict[str, ModelUsage]) -> tuple[int, int]:
        return (
            sum(u.input_tokens for u in bucket.values()),
            sum(u.output_tokens for u in bucket.values()),
        )

    # ----- delegation ratio -----

    @staticmethod
    def _is_delegated(role: str) -> bool:
        return role not in _MAIN_ROLES and role not in _OVERHEAD_ROLES

    def orchestrator_tokens(self) -> int:
        return sum(u.total_tokens for a, u in self.actors.items() if a.role in _MAIN_ROLES)

    def delegated_tokens(self) -> int:
        return sum(u.total_tokens for a, u in self.actors.items() if self._is_delegated(a.role))

    def overhead_tokens(self) -> int:
        """Session housekeeping — compaction summaries. Real money, but not work
        on the task, so it sits outside the delegation ratio."""
        return sum(u.total_tokens for a, u in self.actors.items() if a.role in _OVERHEAD_ROLES)

    def orchestrator_share(self) -> float:
        """Fraction (0..1) of this turn's task tokens the orchestrator spent on
        itself rather than delegating. High means it is doing the work instead
        of routing it."""
        total = self.orchestrator_tokens() + self.delegated_tokens()
        return self.orchestrator_tokens() / total if total else 0.0

    def orchestrator_cost(self) -> float:
        return sum(
            self.cost_of(a, u) for a, u in self.actors.items() if a.role in _MAIN_ROLES
        )

    def delegations(self) -> int:
        """Distinct roles that ran in their own context this turn — every
        subagent, plus the advisor when it was consulted. The orchestrator caused
        those calls but never held their output, which is the distinction the
        share above is measuring."""
        return len({a.role for a in self.actors if self._is_delegated(a.role)})

    # ----- counterfactual -----

    def all_cloud_estimate(self, reference_model: str) -> float:
        """What this turn would cost if the local tokens ran on the cloud
        reference model instead. Local tokens are priced uncached: a local model
        replays its whole prompt every call, with no provider cache to claim."""
        li, lo = self.tokens(self.local)
        return self.cloud_cost + cost_usd(reference_model, li, lo)

    def local_share(self) -> float:
        """Fraction (0..1) of this bucket's tokens that ran locally for free."""
        ci, co = self.tokens(self.cloud)
        li, lo = self.tokens(self.local)
        total = ci + co + li + lo
        return (li + lo) / total if total else 0.0

    def savings(self, reference_model: str) -> float:
        """USD avoided by running the local tokens locally instead of on the
        cloud reference model."""
        return self.all_cloud_estimate(reference_model) - self.cloud_cost


class UsageTracker(BaseCallbackHandler):
    """Callback handler accumulating token usage per role and model, per turn.

    Attach via ``config={"callbacks": [tracker]}`` on ``agent.stream`` /
    ``invoke`` — LangGraph propagates callbacks into subagent runs, so local
    subagent tokens are counted too.

    Role attribution walks the callback run tree. Every ``task`` tool call is
    recorded against its ``subagent_type``, and any model call whose parent chain
    passes through that tool belongs to that subagent. This is authoritative
    where model names are not: two roles can share a model (a local role fallen
    back to the same cloud model as the reviewer), and the orchestrator's own
    calls are indistinguishable from a subagent's by name alone.

    The tree only answers for calls that actually nest under the tool run, which
    a model invoked *by hand* from inside a tool does not: the config such a tool
    is handed belongs to the tool node, so its model call lands as a sibling of
    the tool run rather than a child, and the walk sails past the role. Callers in
    that position say who they are outright, via ``role_metadata`` — read here
    before the walk, since an explicit claim beats an inferred one.
    """

    # Don't let a telemetry bug kill the agent run.
    raise_error = False

    # Runs tracked per turn before the parent map stops growing. A ceiling, not a
    # budget: the map is cleared every turn and only holds UUID pairs.
    _MAX_TRACKED_RUNS = 20_000

    def __init__(self, config: LoomConfig) -> None:
        super().__init__()
        self.config = config
        self._local_names = self._local_model_names(config)
        self.turns = 0
        self.turn = TurnUsage()
        self.session = TurnUsage()
        # run_id -> parent_run_id, and run_id -> role for the runs that name one.
        self._parent: dict[Any, Any] = {}
        self._role: dict[Any, str] = {}
        self._provider: dict[Any, str] = {}
        self._log_path = os.environ.get(USAGE_LOG_ENV) or None
        self._log_lock = threading.Lock()

    def _billing_model(self, role: str, reported: str) -> str:
        """The model string to price this call against.

        The reported name wins whenever Loom recognises it: it is what actually
        ran, including a cloud fallback the config never named, and pricing that
        is the whole point.

        When Loom has no price for the reported name, the role's configured model
        is the better answer. A gateway can answer a free-tier request under the
        *upstream* model's name — OpenCode Zen serves ``deepseek-v4-flash-free``
        and reports ``deepseek-v4-flash`` — and pricing that upstream name at the
        unknown-model default invented a bill for a session that was free.
        """
        if not price_entry(reported).estimated:
            return reported
        configured = self._role_model(role)
        return configured or reported

    def _role_model(self, role: str) -> str:
        """What the config assigns this role, or "" if it names no model for it.

        Read off :class:`LoomConfig` directly rather than through the subagent
        registry: importing it here would close a cycle (the reviewer spec pulls
        in the advisor, which reports usage through this module).
        """
        config = self.config
        if role in ("orchestrator", "?", "compaction"):
            return config.orchestrator or ""
        if role == "advisor":
            return config.advisor or ""
        try:
            return config.model_for(role, "general-purpose") or ""
        except Exception:
            return config.subagents.get(role, "") or ""

    @staticmethod
    def _local_model_names(config: LoomConfig) -> set[str]:
        names: set[str] = set()
        for model in config.all_models().values():
            try:
                rm = resolve(model)
            except Exception:
                continue
            if rm.is_local:
                names.add(rm.name)
        return names

    # ----- turn lifecycle -----
    def start_turn(self) -> None:
        self.turns += 1
        self.turn = TurnUsage()
        self._parent.clear()
        self._role.clear()
        self._provider.clear()
        self._log({"event": "turn"})

    def _log(self, record: dict[str, Any]) -> None:
        """Append one record to the ``LOOM_USAGE_LOG`` file, if one is set."""
        if not self._log_path:
            return
        try:
            line = json.dumps({"t": round(time.time(), 3), "turn": self.turns, **record})
            with self._log_lock, open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass  # a log that can't be written must not cost the user the turn

    def cloud_reference(self) -> str:
        """A cloud model to price the all-cloud counterfactual against.

        The orchestrator when it is a cloud model; otherwise whichever configured
        role is billed. In local-only mode every role is local, so the comparison
        falls back to a named default — the alternative was pricing local tokens
        against a local model and reporting a saving that never existed.
        """
        for model in (
            self.config.orchestrator,
            self.config.advisor,
            self.config.cloud_fallback,
            self.config.escalation_model,
        ):
            if model and not self.config.is_local(model):
                try:
                    return resolve(model).name
                except Exception:
                    return model
        return DEFAULT_CLOUD_REFERENCE

    # ----- run-tree bookkeeping -----

    def _remember(self, run_id: Any, parent_run_id: Any) -> None:
        if run_id is None or len(self._parent) >= self._MAX_TRACKED_RUNS:
            return
        self._parent[run_id] = parent_run_id

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        **kwargs: Any,
    ) -> None:
        try:
            run_id = kwargs.get("run_id")
            self._remember(run_id, kwargs.get("parent_run_id"))
            name = (serialized or {}).get("name") or kwargs.get("name") or ""
            if self._log_path:
                caller, thread = self._owner(kwargs.get("parent_run_id"))
                self._log({"event": "tool", "role": caller, "thread": thread, "name": name})
            if name == "task":
                inputs = kwargs.get("inputs") or {}
                sub = inputs.get("subagent_type") if isinstance(inputs, dict) else None
                self._role[run_id] = str(sub or "general-purpose")
            elif name == "consult":
                self._role[run_id] = "advisor"
        except Exception:
            pass

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Log how much a subagent or the advisor handed back — the summary the
        orchestrator keeps, set against the tokens spent producing it."""
        try:
            run_id = kwargs.get("run_id")
            role = self._role.get(run_id)
            if role is None or not self._log_path:
                return
            self._log(
                {"event": "result", "role": role, "thread": str(run_id),
                 "chars": len(_result_text(output))}
            )
        except Exception:
            pass

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        **kwargs: Any,
    ) -> None:
        try:
            self._remember(kwargs.get("run_id"), kwargs.get("parent_run_id"))
        except Exception:
            pass

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: Any,
        **kwargs: Any,
    ) -> None:
        try:
            run_id = kwargs.get("run_id")
            self._remember(run_id, kwargs.get("parent_run_id"))
            meta = kwargs.get("metadata") or {}
            provider = str(meta.get("ls_provider") or "").lower()
            if provider and run_id is not None:
                self._provider[run_id] = provider
            # An explicit claim from the caller, for calls the run tree can't place.
            role = meta.get(ROLE_KEY)
            if role and run_id is not None:
                self._role[run_id] = str(role)
        except Exception:
            pass

    # `on_llm_start` fires for non-chat models; keep the tree complete.
    def on_llm_start(self, serialized: dict[str, Any] | None, prompts: Any, **kwargs: Any) -> None:
        self.on_chat_model_start(serialized, prompts, **kwargs)

    def role_for(self, run_id: Any) -> str:
        """The role that owns ``run_id``, by walking up the run tree to the
        nearest ``task`` (or ``consult``) call. Defaults to the orchestrator: a
        model call under no delegation is the main graph's own."""
        return self._owner(run_id)[0]

    def _owner(self, run_id: Any) -> tuple[str, str]:
        """``(role, thread)`` for ``run_id``. The thread is the run that claimed
        the role — one ``task`` call is one subagent context window — or
        ``"main"`` for the orchestrator's own."""
        seen: set[Any] = set()
        cur = run_id
        while cur is not None and cur not in seen:
            seen.add(cur)
            role = self._role.get(cur)
            if role is not None:
                return role, str(cur)
            cur = self._parent.get(cur)
        return "orchestrator", "main"

    # ----- LangChain callback hook -----
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            self._record(response, kwargs.get("run_id"))
        except Exception:
            pass  # never break the run over accounting

    def _record(self, response: Any, run_id: Any = None) -> None:
        role, thread = self._owner(run_id)
        provider = self._provider.get(run_id, "")
        for generations in getattr(response, "generations", []) or []:
            for gen in generations:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                meta = getattr(msg, "usage_metadata", None) or {}
                inp = int(meta.get("input_tokens", 0) or 0)
                out = int(meta.get("output_tokens", 0) or 0)
                if not inp and not out:
                    continue
                details = meta.get("input_token_details") or {}
                cache_read = int(details.get("cache_read", 0) or 0)
                cache_write = int(details.get("cache_creation", 0) or 0)
                rmeta = getattr(msg, "response_metadata", None) or {}
                model = str(rmeta.get("model_name") or rmeta.get("model") or "unknown")
                is_local = self._is_local(model, provider)
                billed_as = "" if is_local else self._billing_model(role, model)
                for bucket in (self.turn, self.session):
                    bucket.add(
                        model,
                        is_local,
                        inp,
                        out,
                        role=role,
                        cache_read=cache_read,
                        cache_write=cache_write,
                        billed_as=billed_as,
                    )
                self._log(
                    {"event": "llm", "role": role, "thread": thread, "model": model,
                     "local": is_local, "input": inp, "cache_read": cache_read,
                     "cache_write": cache_write, "output": out}
                )

    def _is_local(self, model_name: str, provider: str = "") -> bool:
        """Whether a call was free.

        Precedence matters, and each step exists for a case the next one gets
        wrong:

        1. The config is authoritative. If the user assigned this model to a role
           with an ``ollama/`` prefix, it is local — whatever any provider tag
           says. Trusting the tag over the config bills free tokens as cloud the
           moment a local model is served by anything that does not identify
           itself as "ollama".
        2. Otherwise an ``ollama`` provider tag settles it. This covers the model
           the local-first planner substituted in, which the config never named.
        3. Otherwise fall back to the shape of the name. An Ollama tag without a
           colon (``gpt-oss``) is indistinguishable from a cloud name here, which
           is exactly why the config comes first.
        """
        if model_name in self._local_names:
            return True
        provider = (provider or "").lower()
        if provider in ("ollama", "chatollama"):
            return True
        if provider:
            return False  # a named non-Ollama provider is a network provider
        # Ollama tags look like "qwen3:14b"; cloud names never carry a colon.
        return ":" in model_name and not model_name.startswith("claude")

    # ----- rendering -----
    @staticmethod
    def _fmt_tokens(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    @staticmethod
    def _fmt_usd(v: float) -> str:
        return f"${v:.3f}" if v < 1 else f"${v:.2f}"

    def receipt(self, turn: bool = True) -> str:
        """One-line receipt, e.g.
        ``$0.031 cloud (10.2k in · 8.9k cached / 1.4k out) + 84.0k local (free) ·
        orchestrator 18% of tokens, 4 subagents · 89% local, saved ~$0.26 vs
        all-cloud on claude-sonnet-5 · session $0.14 (saved ~$1.02)``
        """
        u = self.turn if turn else self.session
        ci, co = u.tokens(u.cloud)
        li, lo = u.tokens(u.local)
        parts: list[str] = []
        if ci or co:
            cached = u.cache_read_tokens
            cached_note = f" · {self._fmt_tokens(cached)} cached" if cached else ""
            # "~" when any of it was priced against an unpublished model.
            approx = "~" if u.has_estimates() else ""
            parts.append(
                f"{approx}{self._fmt_usd(u.cloud_cost)} cloud "
                f"({self._fmt_tokens(ci)} in{cached_note} / {self._fmt_tokens(co)} out)"
            )
        if li or lo:
            parts.append(f"{self._fmt_tokens(li + lo)} local tokens (free)")
        if not parts:
            return ""
        line = " + ".join(parts)
        delegations = u.delegations()
        if delegations:
            line += (
                f" · orchestrator {u.orchestrator_share():.0%} of tokens, "
                f"{delegations} delegated role{'s' if delegations != 1 else ''}"
            )
        if li or lo:
            reference = self.cloud_reference()
            saved = u.savings(reference)
            line += (
                f" · {u.local_share():.0%} local, saved ~{self._fmt_usd(saved)} "
                f"vs all-cloud on {reference}"
            )
        if turn:
            # Carries its own marker: the session total can be an estimate even
            # when this turn's is not, and vice versa.
            session_approx = "~" if self.session.has_estimates() else ""
            line += f" · session {session_approx}{self._fmt_usd(self.session.cloud_cost)}"
            session_saved = self.session.savings(self.cloud_reference())
            if session_saved > 0:
                line += f" (saved ~{self._fmt_usd(session_saved)})"
        return line
