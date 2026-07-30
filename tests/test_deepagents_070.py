"""Regressions for the deepagents 0.7 migration.

0.7 changed three things Loom silently depended on:

* ``TodoListMiddleware`` left the defaults, deleting ``write_todos`` — a tool the
  orchestrator prompt tells the model to call.
* The built-in prompts went empty, so all tool-usage prose is Loom's to supply.
* A caller-supplied middleware instance now *replaces* the default with the same
  ``.name``, which is how Loom restricts the filesystem toolset structurally
  instead of filtering after injection.

Each of those is invisible to a smoke test and would degrade quietly, so they
are pinned here against the real library.
"""

import pytest

pytest.importorskip("deepagents")

from loom.core import ollama
from loom.core.config import LoomConfig
from loom.core.ollama import OllamaStatus
from loom.core.settings import Settings

ROLES = ("explorer", "editor", "bash", "searcher", "reviewer", "general-purpose", "tester")


def _settings(**kw):
    defaults = dict(
        orchestrator="ollama/qwen3:14b",
        subagents={n: "ollama/qwen3:4b" for n in ROLES},
    )
    defaults.update(kw)
    return Settings(models=LoomConfig(**defaults))


@pytest.fixture(autouse=True)
def _stub_ollama(monkeypatch):
    monkeypatch.setattr(
        ollama, "status", lambda cfg: OllamaStatus(True, True, ["qwen3:4b", "qwen3:14b"], "http://x")
    )


def _capture(monkeypatch):
    """Build an orchestrator and return (main create_agent kwargs, subagent specs)."""
    import deepagents.graph as dg
    import deepagents.middleware.subagents as sam
    import langchain.agents as la

    calls: list[dict] = []
    specs: list[dict] = []
    orig_create = la.create_agent
    orig_sub_init = sam.SubAgentMiddleware.__init__

    def spy_create(*args, **kwargs):
        calls.append(kwargs)
        return orig_create(*args, **kwargs)

    def spy_sub(self, *args, **kwargs):
        for s in kwargs.get("subagents") or (args[1] if len(args) > 1 else []):
            if isinstance(s, dict):
                specs.append(s)
        return orig_sub_init(self, *args, **kwargs)

    monkeypatch.setattr(la, "create_agent", spy_create)
    monkeypatch.setattr(dg, "create_agent", spy_create)
    monkeypatch.setattr(sam, "create_agent", spy_create)
    monkeypatch.setattr(sam.SubAgentMiddleware, "__init__", spy_sub)

    def build(**mode):
        calls.clear()
        specs.clear()
        from loom.core.orchestrator import build_orchestrator

        bundle = build_orchestrator(_settings(), cwd=".", **mode)
        return bundle, calls[-1], {s["name"]: s for s in specs}

    return build


def _mw_names(spec_or_kwargs) -> list[str]:
    return [m.name for m in (spec_or_kwargs.get("middleware") or [])]


def _fs_tools(spec_or_kwargs) -> set[str]:
    for m in spec_or_kwargs.get("middleware") or []:
        if m.name == "FilesystemMiddleware":
            return {t.name for t in m.tools}
    return set()


# ---------------------------------------------------------------------------
# write_todos survived the default-middleware removal
# ---------------------------------------------------------------------------


def test_orchestrator_still_has_write_todos(monkeypatch):
    """0.7 dropped TodoListMiddleware from the defaults. The orchestrator prompt
    instructs the model to call `write_todos`, so Loom must add it back."""
    build = _capture(monkeypatch)
    _bundle, main, _subs = build()
    assert "TodoListMiddleware" in _mw_names(main)


def test_todo_prompt_is_looms_not_langchains(monkeypatch):
    """The built-in prose is generic task tracking; Loom's ties todos to the
    fleet, which is the only reason the orchestrator keeps a list."""
    from loom.core.orchestrator import TODO_SYSTEM_PROMPT

    build = _capture(monkeypatch)
    _bundle, main, _subs = build()
    todo = next(m for m in main["middleware"] if m.name == "TodoListMiddleware")
    assert todo.system_prompt == TODO_SYSTEM_PROMPT
    assert "explorer" in TODO_SYSTEM_PROMPT  # names a subagent, i.e. is about routing


def test_prompt_names_only_tools_that_exist(monkeypatch):
    """The prompt must not advertise a tool 0.7 removed, or one Loom strips."""
    build = _capture(monkeypatch)
    _bundle, main, _subs = build()
    prompt = main["system_prompt"]
    assert "`write_todos`" in prompt
    assert "`task(" in prompt
    for gone in ("`ls`, `glob`", "`write_file`"):
        # They may only appear in the sentence that says they are absent.
        assert gone in prompt
    assert "You have no `ls`" in prompt


# ---------------------------------------------------------------------------
# Structural tool allowlists (the FilesystemMiddleware replacement)
# ---------------------------------------------------------------------------


def test_orchestrator_filesystem_middleware_holds_only_read_file(monkeypatch):
    build = _capture(monkeypatch)
    _bundle, main, _subs = build()
    assert _fs_tools(main) == {"read_file"}


def test_subagent_allowlists_match_their_roles(monkeypatch):
    build = _capture(monkeypatch)
    _bundle, _main, subs = build()
    assert _fs_tools(subs["explorer"]) == {"ls", "read_file", "glob", "grep"}
    assert _fs_tools(subs["searcher"]) == {"ls", "read_file", "glob", "grep"}
    assert _fs_tools(subs["reviewer"]) == {"ls", "read_file", "glob", "grep"}
    assert _fs_tools(subs["editor"]) == {"ls", "read_file", "glob", "grep", "write_file", "edit_file"}
    assert _fs_tools(subs["bash"]) == {"ls", "read_file", "glob", "grep", "execute"}


def test_editor_cannot_run_and_bash_cannot_edit(monkeypatch):
    """The split is the point: one role changes code, another runs it."""
    build = _capture(monkeypatch)
    _bundle, _main, subs = build()
    assert "execute" not in _fs_tools(subs["editor"])
    assert "write_file" not in _fs_tools(subs["bash"])
    assert "edit_file" not in _fs_tools(subs["bash"])


def test_nobody_gets_the_recursive_delete_tool(monkeypatch):
    """0.7 hands the model a recursive `delete` whenever the backend supports
    one, and treats it as an ordinary write. No Loom role needs it."""
    build = _capture(monkeypatch)
    _bundle, main, subs = build()
    assert "delete" not in _fs_tools(main)
    for name, spec in subs.items():
        assert "delete" not in _fs_tools(spec), name


def test_plan_mode_allowlists_are_read_only(monkeypatch):
    build = _capture(monkeypatch)
    _bundle, _main, subs = build(plan=True)
    for name, spec in subs.items():
        assert _fs_tools(spec) <= {"ls", "read_file", "glob", "grep"}, name


def test_hardened_general_purpose_has_no_filesystem_at_all(monkeypatch):
    """With no local model available, `general-purpose` must still claim the
    reserved name (or deepagents resurrects its own unrestricted default) while
    being unable to touch code. FilesystemMiddleware rejects an allowlist without
    `read_file`, so the exclusion layer is what makes this expressible."""
    import deepagents.middleware.subagents as sam

    from loom.middleware.tool_exclusion import ToolExclusionMiddleware
    from loom.subagents.base import ALL_FS_TOOLS

    specs: list[dict] = []
    orig_sub_init = sam.SubAgentMiddleware.__init__

    def spy_sub(self, *args, **kwargs):
        for s in kwargs.get("subagents") or (args[1] if len(args) > 1 else []):
            if isinstance(s, dict):
                specs.append(s)
        return orig_sub_init(self, *args, **kwargs)

    monkeypatch.setattr(sam.SubAgentMiddleware, "__init__", spy_sub)

    from loom.core.orchestrator import build_orchestrator

    # Every role on the cloud: airgap drops them all, so the fallback rebuild has
    # no local model to pin to.
    settings = Settings(
        models=LoomConfig(
            orchestrator="claude-sonnet-5",
            advisor="claude-opus-4-8",
            subagents={r: "claude-haiku-4-5" for r in ROLES},
        )
    )
    bundle = build_orchestrator(settings, cwd=".", airgap=True)
    assert "general-purpose" in bundle.subagent_names

    gp = next(s for s in specs if s["name"] == "general-purpose")
    excluded = {
        e for m in gp["middleware"] if isinstance(m, ToolExclusionMiddleware) for e in m._excluded
    }
    assert ALL_FS_TOOLS <= excluded


def test_airgap_orchestrator_still_loses_read_file(monkeypatch):
    """FilesystemMiddleware refuses an allowlist without read_file, so airgap
    has to strip it at the last mile — that layer must still be present."""
    from loom.middleware.tool_exclusion import ToolExclusionMiddleware

    build = _capture(monkeypatch)
    _bundle, main, _subs = build(airgap=True)
    excluded = {
        e
        for m in main["middleware"]
        if isinstance(m, ToolExclusionMiddleware)
        for e in m._excluded
    }
    assert "read_file" in excluded


# ---------------------------------------------------------------------------
# Everything shares one real backend
# ---------------------------------------------------------------------------


def test_every_agent_shares_the_orchestrator_backend(monkeypatch):
    """A subagent that built its own FilesystemMiddleware without the shared
    backend would silently fall back to an in-memory StateBackend and lose the
    real filesystem — a failure that looks like an empty repository."""
    from deepagents.backends import CompositeBackend

    build = _capture(monkeypatch)
    _bundle, main, subs = build()
    backends = set()
    for spec in [main, *subs.values()]:
        for m in spec.get("middleware") or []:
            if m.name == "FilesystemMiddleware":
                assert isinstance(m.backend, CompositeBackend), spec["name"]
                backends.add(id(m.backend))
    assert len(backends) == 1, "filesystem middleware instances disagree on the backend"


# ---------------------------------------------------------------------------
# Summarization: the config knob now actually reaches the agent
# ---------------------------------------------------------------------------


def _trigger_of(middleware) -> tuple | None:
    """The configured summarization trigger, however deepagents stores it."""
    for holder in (middleware, getattr(middleware, "_lc_helper", None)):
        trigger = getattr(holder, "trigger", None)
        if trigger is not None:
            return trigger
    return None


def test_compaction_threshold_reaches_the_summarizer(monkeypatch):
    """`compaction_threshold` was a documented knob that nothing read, and
    deepagents' own default trigger for a profile-less model is a flat 170K
    tokens — unreachable for a 32K local model, which would overflow instead."""
    build = _capture(monkeypatch)
    _bundle, main, subs = build()
    for spec in [main, subs["explorer"]]:
        summarizer = next(m for m in spec["middleware"] if m.name == "SummarizationMiddleware")
        trigger = _trigger_of(summarizer)
        assert trigger is not None
        assert trigger != ("tokens", 170000), "left at deepagents' profile-less default"
        assert trigger[0] == "tokens" and trigger[1] < 170000


def test_summarization_trigger_scales_with_the_window():
    from loom.core.artifact_store import compaction_trigger

    config = LoomConfig(context_windows={"ollama/qwen3:4b": 32768})
    assert compaction_trigger(config, "ollama/qwen3:4b") == int(32768 * 0.70)
    # Never so small that the first message trips it.
    assert compaction_trigger(LoomConfig(context_windows={"x": 2048}), "x") == 4_000


def test_summarization_middleware_builds_for_a_loom_model_string():
    """deepagents resolves a bare string through init_chat_model, which rejects
    Loom's `ollama/tag` form — the builder has to use Loom's own router."""
    from deepagents.backends import StateBackend

    from loom.core.artifact_store import summarization_middleware

    mw = summarization_middleware(
        LoomConfig(context_windows={"ollama/qwen3:4b": 32768}),
        StateBackend(),
        model_string="ollama/qwen3:4b",
    )
    assert mw is not None
    assert _trigger_of(mw) == ("tokens", int(32768 * 0.70))
