"""The reminder the orchestrator gets back with every `read_file` result.

`ls`/`glob`/`grep` are structurally gone from the orchestrator; `read_file`
stays for targeted confirmation and is never withdrawn. Instead each result ends
with a one-line nudge to hand anything larger to `explorer`. These tests pin that
the nudge lands on read_file results only, in both content shapes LangChain
uses, without ever blocking the read — and, against a real compiled deepagents
graph, that the tool stays offered and the reminder reaches the model.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("langchain")

from langchain_core.messages import ToolMessage

from loom.middleware.delegation_reminder import REMINDER, DelegationReminder


def _request(name="read_file", call_id="c1"):
    return SimpleNamespace(tool_call={"name": name, "args": {"file_path": "/a.py"}, "id": call_id})


def _run(reminder, request, result):
    ran = []

    def handler(r):
        ran.append(r)
        return result

    out = reminder.wrap_tool_call(request, handler)
    assert ran, "the reminder must never stop a read from running"
    return out


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


def test_read_file_result_ends_with_the_reminder():
    reminder = DelegationReminder()
    out = _run(reminder, _request(), ToolMessage(content="1\tprint('hi')", tool_call_id="c1"))
    assert out.content.startswith("1\tprint('hi')")
    assert out.content.endswith(REMINDER)
    assert out.tool_call_id == "c1"
    assert reminder.reminded_count == 1


def test_every_read_is_reminded_there_is_no_budget():
    reminder = DelegationReminder()
    for i in range(10):
        out = _run(reminder, _request(call_id=str(i)), ToolMessage(content="x", tool_call_id=str(i)))
        assert REMINDER in out.content
    assert reminder.reminded_count == 10


def test_list_content_gets_a_text_block():
    reminder = DelegationReminder()
    msg = ToolMessage(content=[{"type": "text", "text": "file body"}], tool_call_id="c1")
    out = _run(reminder, _request(), msg)
    assert out.content[0] == {"type": "text", "text": "file body"}
    assert out.content[-1] == {"type": "text", "text": REMINDER}


def test_other_tools_are_left_alone():
    reminder = DelegationReminder()
    msg = ToolMessage(content="report", tool_call_id="c1")
    out = _run(reminder, _request(name="task"), msg)
    assert out is msg
    assert reminder.reminded_count == 0


def test_non_message_results_pass_through():
    """A Command or a bare string has no content the model reads as a tool
    result; rewriting it would be guesswork."""
    reminder = DelegationReminder()
    assert _run(reminder, _request(), "raw") == "raw"
    assert reminder.reminded_count == 0


def test_the_original_message_is_not_mutated():
    reminder = DelegationReminder()
    msg = ToolMessage(content="body", tool_call_id="c1")
    _run(reminder, _request(), msg)
    assert msg.content == "body"


@pytest.mark.asyncio
async def test_async_path_reminds_too():
    reminder = DelegationReminder()

    async def handler(_r):
        return ToolMessage(content="body", tool_call_id="c1")

    out = await reminder.awrap_tool_call(_request(), handler)
    assert out.content.endswith(REMINDER)


# ---------------------------------------------------------------------------
# End-to-end against a real compiled deepagents agent
#
# Drives a fake model through a real ``create_deep_agent`` graph, so the
# middleware has to sit where the FilesystemMiddleware's read_file actually runs
# and survive deepagents merging Loom's stack into its own.
# ---------------------------------------------------------------------------

pytest.importorskip("deepagents")

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from loom.core import ollama  # noqa: E402
from loom.core.config import LoomConfig  # noqa: E402
from loom.core.ollama import OllamaStatus  # noqa: E402
from loom.core.settings import Permissions, Settings  # noqa: E402

ROLES = ("explorer", "editor", "bash", "searcher", "reviewer", "general-purpose", "tester")


def _tool_name(tool) -> str | None:
    if isinstance(tool, dict):
        return tool.get("name") or (tool.get("function") or {}).get("name")
    return getattr(tool, "name", None)


class _ReadingModel(BaseChatModel):
    """Reads a file on each of its first ``max_reads`` calls, recording the
    tools it was offered and every tool result it was shown."""

    offered: list = []
    seen_results: list = []
    max_reads: int = 6

    @property
    def _llm_type(self) -> str:
        return "reading-fake"

    def bind_tools(self, tools, **kwargs):
        self.offered.append({n for n in (_tool_name(t) for t in tools) if n})
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_results[:] = [m for m in messages if getattr(m, "type", None) == "tool"]
        n = len(self.offered)
        if n <= self.max_reads and "read_file" in (self.offered[-1] if self.offered else set()):
            msg = AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"file_path": "/pyproject.toml"}, "id": f"r{n}"}],
            )
        else:
            msg = AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def _stub_ollama(monkeypatch):
    monkeypatch.setattr(
        ollama, "status", lambda cfg: OllamaStatus(True, True, ["qwen3:4b", "qwen3:14b"], "http://x")
    )


@pytest.fixture
def _rooted(tmp_path):
    """The filesystem backend is rooted at the sandbox root, not at `cwd`; point
    it at tmp_path and put it back so no later test inherits it."""
    from pathlib import Path

    from loom.tools import sandbox

    token = sandbox._ROOT.set(Path(tmp_path).resolve())
    yield tmp_path
    sandbox._ROOT.reset(token)


def _run_agent(monkeypatch, cwd: str, *, airgap: bool = False):
    model = _ReadingModel(offered=[], seen_results=[])
    monkeypatch.setattr("loom.core.orchestrator.build_model", lambda *a, **k: model)

    from loom.core.orchestrator import build_orchestrator

    settings = Settings(
        models=LoomConfig(orchestrator="ollama/qwen3:14b", subagents={r: "ollama/qwen3:4b" for r in ROLES}),
        permissions=Permissions(default_mode="allow"),
    )
    bundle = build_orchestrator(settings, cwd=cwd, airgap=airgap)
    bundle.agent.invoke({"messages": [("user", "look at the project")]})
    return bundle, model


def test_read_file_is_never_withdrawn_and_every_result_is_reminded(monkeypatch, _rooted, _stub_ollama):
    tmp_path = _rooted
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    bundle, model = _run_agent(monkeypatch, str(tmp_path))

    assert all("read_file" in o for o in model.offered), model.offered
    assert all("task" in o for o in model.offered)
    reads = [m for m in model.seen_results if m.name == "read_file"]
    assert len(reads) == model.max_reads
    assert all(REMINDER in str(m.content) for m in reads)
    assert "name='x'" in str(reads[0].content), "the file itself must still come back"
    assert bundle.delegation_reminder.reminded_count == model.max_reads


def test_airgap_has_no_read_file_and_no_reminder(monkeypatch, _rooted, _stub_ollama):
    tmp_path = _rooted
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    bundle, model = _run_agent(monkeypatch, str(tmp_path), airgap=True)
    assert all("read_file" not in o for o in model.offered), model.offered
    assert bundle.delegation_reminder is None
