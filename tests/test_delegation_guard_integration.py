"""The read budget end-to-end, against a real compiled deepagents agent.

The unit tests exercise DelegationGuard in isolation. This one drives a fake
model through a real ``create_deep_agent`` graph so the middleware ordering
actually matters: the guard has to see the same `read_file` the FilesystemMiddleware
injected, and it has to still be in the stack after deepagents merges Loom's
middleware into its own.
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

ROLES = ("explorer", "editor", "bash", "searcher", "reviewer", "general-purpose", "tester")


def _tool_name(tool) -> str | None:
    if isinstance(tool, dict):
        return tool.get("name") or (tool.get("function") or {}).get("name")
    return getattr(tool, "name", None)


class _ReadingModel(BaseChatModel):
    """Asks to read a file on every turn, and records the tools it was offered.

    ``bind_tools`` is where the agent hands over the tool set, so recording there
    is what lets the test see the guard revoke one mid-run.
    """

    # Shared across the bound copies langchain may make of this model.
    offered: list = []
    max_reads: int = 6

    @property
    def _llm_type(self) -> str:
        return "reading-fake"

    def bind_tools(self, tools, **kwargs):
        self.offered.append({n for n in (_tool_name(t) for t in tools) if n})
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        available = self.offered[-1] if self.offered else set()
        n = len(self.offered)
        if n <= self.max_reads and "read_file" in available:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"file_path": "/pyproject.toml"}, "id": f"r{n}"}
                ],
            )
        else:
            msg = AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _settings(budget: int) -> Settings:
    return Settings(
        models=LoomConfig(
            orchestrator="ollama/qwen3:14b",
            subagents={r: "ollama/qwen3:4b" for r in ROLES},
            orchestrator_read_budget=budget,
        ),
        # The guard, not the approval prompt, is what this test is about.
        permissions=Permissions(default_mode="allow"),
    )


@pytest.fixture(autouse=True)
def _stub_ollama(monkeypatch):
    monkeypatch.setattr(
        ollama, "status", lambda cfg: OllamaStatus(True, True, ["qwen3:4b", "qwen3:14b"], "http://x")
    )


def _run(monkeypatch, budget: int, cwd: str):
    model = _ReadingModel(offered=[])
    monkeypatch.setattr("loom.core.orchestrator.build_model", lambda *a, **k: model)

    from loom.core.orchestrator import build_orchestrator

    bundle = build_orchestrator(_settings(budget), cwd=cwd)
    bundle.agent.invoke({"messages": [("user", "look at the project")]})
    return bundle, model


def test_read_file_disappears_once_the_budget_is_spent(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    bundle, model = _run(monkeypatch, 2, str(tmp_path))

    offered = model.offered
    assert "read_file" in offered[0], "the orchestrator should start with a read budget"
    # Two reads allowed, then the tool is gone for the rest of the turn.
    assert "read_file" in offered[1]
    assert all("read_file" not in o for o in offered[2:]), offered
    # Delegation is never revoked — forcing it is the point.
    assert all("task" in o for o in offered)
    assert bundle.delegation_guard.blocked_count >= 1


def test_an_uncapped_budget_leaves_the_tool_alone(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    bundle, model = _run(monkeypatch, -1, str(tmp_path))
    assert all("read_file" in o for o in model.offered)
    assert bundle.delegation_guard.blocked_count == 0


def test_a_zero_budget_never_offers_read_file(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    _bundle, model = _run(monkeypatch, 0, str(tmp_path))
    assert all("read_file" not in o for o in model.offered), model.offered
