"""Role attribution against REAL LangChain callbacks and a real `task` call.

`test_usage_attribution.py` drives UsageTracker's hooks by hand, which proves the
bookkeeping but not the plumbing: whether `on_tool_start` actually receives the
`task` call's `inputs`, and whether `parent_run_id` really chains from the
subagent's model call back up through the subgraph to that tool run. Both are
LangChain's decisions, not Loom's, so they are pinned against a compiled graph.

This is also where the local/cloud precedence bug showed up: with `ls_provider`
outranking the config, a subagent's free local tokens were billed as cloud the
moment its model was served by anything not literally tagged "ollama".
"""

import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from loom.core import ollama
from loom.core.config import LoomConfig
from loom.core.ollama import OllamaStatus
from loom.core.settings import Permissions, Settings
from loom.core.usage import UsageTracker

ROLES = ("explorer", "editor", "bash", "searcher", "reviewer", "general-purpose", "tester")


class _Scripted(BaseChatModel):
    """Delegates once, then answers. Reports usage so there is something to
    attribute, including the cache split a real Anthropic response carries."""

    tag: str = "orch"
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(self.tag)
        nth = sum(1 for t in self.seen if t == self.tag)
        if self.tag == "sub":
            msg = AIMessage(content="loom/core/config.py:172 load_config()")
            usage = {"input_tokens": 40_000, "output_tokens": 400, "input_token_details": {}}
            model = "qwen3:4b"
        elif nth == 1:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"description": "find the config loader", "subagent_type": "explorer"},
                        "id": "t1",
                    }
                ],
            )
            usage = {
                "input_tokens": 5_000,
                "output_tokens": 300,
                "input_token_details": {"cache_read": 4_000, "cache_creation": 500},
            }
            model = "claude-sonnet-5"
        else:
            msg = AIMessage(content="it lives in loom/core/config.py")
            usage = {
                "input_tokens": 6_000,
                "output_tokens": 200,
                "input_token_details": {"cache_read": 5_500, "cache_creation": 0},
            }
            model = "claude-sonnet-5"
        msg.usage_metadata = usage
        msg.response_metadata = {"model_name": model}
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def turn(monkeypatch):
    """Run one delegating turn through a real compiled agent; yield the tracker."""
    monkeypatch.setattr(
        ollama, "status", lambda cfg: OllamaStatus(True, True, ["qwen3:4b"], "http://x")
    )
    seen: list = []
    orchestrator_model = _Scripted(tag="orch", seen=seen)
    subagent_model = _Scripted(tag="sub", seen=seen)
    monkeypatch.setattr("loom.core.orchestrator.build_model", lambda *a, **k: orchestrator_model)
    monkeypatch.setattr("loom.subagents.base.build_model", lambda *a, **k: subagent_model)

    from loom.core.orchestrator import build_orchestrator

    settings = Settings(
        models=LoomConfig(
            orchestrator="claude-sonnet-5",
            subagents={r: "ollama/qwen3:4b" for r in ROLES},
        ),
        permissions=Permissions(default_mode="allow"),
    )
    bundle = build_orchestrator(settings, cwd=".")
    tracker = UsageTracker(settings.models)
    tracker.start_turn()
    bundle.agent.invoke(
        {"messages": [("user", "where is the config loader?")]},
        config={"callbacks": [tracker]},
    )
    return tracker


def test_the_run_tree_separates_orchestrator_from_subagent(turn):
    roles = {a.role for a in turn.turn.actors}
    assert roles == {"orchestrator", "explorer"}


def test_subagent_tokens_are_not_billed(turn):
    """The config assigned explorer an `ollama/` model, so its tokens are free —
    regardless of what provider tag the run happens to carry."""
    explorer = next(a for a in turn.turn.actors if a.role == "explorer")
    assert explorer.is_local is True
    assert turn.turn.cost_of(explorer, turn.turn.actors[explorer]) == 0.0


def test_the_orchestrators_cached_tokens_survive_the_round_trip(turn):
    assert turn.turn.cache_read_tokens == 9_500
    orch = next(a for a in turn.turn.actors if a.role == "orchestrator")
    usage = turn.turn.actors[orch]
    assert usage.cache_read_tokens == 9_500
    assert usage.cache_write_tokens == 500


def test_the_delegation_ratio_reflects_the_actual_split(turn):
    # 11.5k orchestrator tokens against 40.4k delegated.
    assert turn.turn.delegations() == 1
    assert turn.turn.orchestrator_share() == pytest.approx(11_500 / 51_900, abs=0.01)


def test_the_receipt_reads_as_a_hybrid_turn(turn):
    receipt = turn.receipt()
    assert "local tokens (free)" in receipt
    assert "1 delegated role" in receipt
    assert "cached" in receipt
