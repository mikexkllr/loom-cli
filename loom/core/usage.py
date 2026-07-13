"""Usage tracking + cost receipts — the measurable half of the hybrid pitch.

A LangChain callback handler records every model call (orchestrator and
subagents alike — callbacks propagate into nested runs), classifies it local
vs cloud, and prices the cloud tokens. After each turn the REPL prints a
receipt: what this task cost, what the free local tokens would have cost on
the cloud orchestrator model, and the session running total.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loom.core.config import LoomConfig
from loom.core.model_router import resolve

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - allows import without langchain
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass

# USD per million tokens (input, output). Cloud models only — local is free.
# Prices per Anthropic pricing (2026-06); unknown cloud models fall back to
# Sonnet-tier so receipts stay conservative rather than absent.
CLOUD_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-opus": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4.1": (2.0, 8.0),
}
_DEFAULT_CLOUD_PRICE = (3.0, 15.0)


def price_for(model_name: str) -> tuple[float, float]:
    """(input, output) USD per MTok for a cloud model, longest-prefix match."""
    name = model_name.lower()
    best: tuple[float, float] | None = None
    best_len = -1
    for prefix, price in CLOUD_PRICES.items():
        if name.startswith(prefix) and len(prefix) > best_len:
            best, best_len = price, len(prefix)
    return best or _DEFAULT_CLOUD_PRICE


def cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = price_for(model_name)
    return (input_tokens * inp + output_tokens * out) / 1_000_000


@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


@dataclass
class TurnUsage:
    cloud: dict[str, ModelUsage] = field(default_factory=dict)
    local: dict[str, ModelUsage] = field(default_factory=dict)

    def add(self, model: str, is_local: bool, inp: int, out: int) -> None:
        bucket = self.local if is_local else self.cloud
        mu = bucket.setdefault(model, ModelUsage())
        mu.input_tokens += inp
        mu.output_tokens += out
        mu.calls += 1

    @property
    def cloud_cost(self) -> float:
        return sum(cost_usd(m, u.input_tokens, u.output_tokens) for m, u in self.cloud.items())

    def tokens(self, bucket: dict[str, ModelUsage]) -> tuple[int, int]:
        return (
            sum(u.input_tokens for u in bucket.values()),
            sum(u.output_tokens for u in bucket.values()),
        )

    def all_cloud_estimate(self, reference_model: str) -> float:
        """What this turn would cost if the local tokens ran on the cloud
        reference model (the orchestrator) instead."""
        li, lo = self.tokens(self.local)
        return self.cloud_cost + cost_usd(reference_model, li, lo)


class UsageTracker(BaseCallbackHandler):
    """Callback handler accumulating token usage per model, per turn.

    Attach via ``config={"callbacks": [tracker]}`` on ``agent.stream`` /
    ``invoke`` — LangGraph propagates callbacks into subagent runs, so local
    subagent tokens are counted too.
    """

    # Don't let a telemetry bug kill the agent run.
    raise_error = False

    def __init__(self, config: LoomConfig) -> None:
        super().__init__()
        self.config = config
        self._local_names = self._local_model_names(config)
        self.turns = 0
        self.turn = TurnUsage()
        self.session = TurnUsage()

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

    # ----- LangChain callback hook -----
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            self._record(response)
        except Exception:
            pass  # never break the run over accounting

    def _record(self, response: Any) -> None:
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
                rmeta = getattr(msg, "response_metadata", None) or {}
                model = str(
                    rmeta.get("model_name") or rmeta.get("model") or "unknown"
                )
                self.turn.add(model, self._is_local(model), inp, out)
                self.session.add(model, self._is_local(model), inp, out)

    def _is_local(self, model_name: str) -> bool:
        if model_name in self._local_names:
            return True
        # Ollama tags look like "qwen3:14b"; cloud names never carry a colon.
        return ":" in model_name and not model_name.startswith("claude")

    # ----- rendering -----
    @staticmethod
    def _fmt_tokens(n: int) -> str:
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    def receipt(self, turn: bool = True) -> str:
        """One-line receipt, e.g.
        ``$0.031 cloud (10.2k in / 1.4k out) + 84.0k local (free) · all-cloud est. $0.29 · session $0.14``
        """
        u = self.turn if turn else self.session
        ci, co = u.tokens(u.cloud)
        li, lo = u.tokens(u.local)
        parts: list[str] = []
        if ci or co:
            parts.append(f"${u.cloud_cost:.3f} cloud ({self._fmt_tokens(ci)} in / {self._fmt_tokens(co)} out)")
        if li or lo:
            parts.append(f"{self._fmt_tokens(li + lo)} local tokens (free)")
        if not parts:
            return ""
        line = " + ".join(parts)
        if li or lo:
            est = u.all_cloud_estimate(self.config.orchestrator)
            line += f" · all-cloud est. ${est:.3f}"
        if turn:
            line += f" · session ${self.session.cloud_cost:.3f}"
        return line
