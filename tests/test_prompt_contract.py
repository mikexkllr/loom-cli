"""The prompts are Loom's own now — keep them consistent with the tool sets.

deepagents 0.7 ships an empty base prompt and injects no tool-usage prose, so a
prompt that names a tool the role does not have is a silent dead end: the model
plans around a tool that will never appear. These tests bind each prompt to its
spec's actual allowlist.
"""

import re

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("langchain_core")

from loom.subagents import SPECS
from loom.subagents.base import ALL_FS_TOOLS, ISOLATION_PREAMBLE, WRITE_TOOLS

# Backticked tool mentions, e.g. `edit_file` or `browser_*`.
_MENTION = re.compile(r"`([a-z_]+)`")


def _mentioned_fs_tools(prompt: str) -> set[str]:
    return {m for m in _MENTION.findall(prompt) if m in ALL_FS_TOOLS}


@pytest.mark.parametrize("name", sorted(SPECS))
def test_prompt_only_names_tools_the_role_has(name):
    spec = SPECS[name]
    if name == "reviewer":
        pytest.skip("reviewer prompt is shared with the advisor (REVIEW_SYSTEM)")
    for tool in _mentioned_fs_tools(spec.system_prompt):
        assert tool in spec.fs_tools, f"{name} prompt names `{tool}`, which it cannot call"


@pytest.mark.parametrize("name", sorted(SPECS))
def test_write_capable_prompts_explain_write_file_overwrites(name):
    """0.7 made `write_file` replace an existing file silently instead of
    erroring, so any role holding it has to be told."""
    spec = SPECS[name]
    if "write_file" not in spec.fs_tools or name == "tester":
        pytest.skip("role does not author code files")
    prompt = spec.system_prompt.lower()
    assert "replace" in prompt and "write_file" in prompt


@pytest.mark.parametrize("name", sorted(SPECS))
def test_prompts_carry_the_isolation_contract(name):
    spec = SPECS[name]
    if name == "reviewer":
        pytest.skip("reviewer states its own contract and returns structured output")
    assert spec.system_prompt.startswith(ISOLATION_PREAMBLE)


@pytest.mark.parametrize("name", sorted(SPECS))
def test_prompts_state_a_report_format(name):
    """A subagent's final message is the only thing that survives it."""
    prompt = SPECS[name].system_prompt.lower()
    assert "report" in prompt or "return" in prompt


@pytest.mark.parametrize("name", sorted(SPECS))
def test_descriptions_tell_the_orchestrator_what_comes_back(name):
    """The `task` tool advertises exactly these strings, and they are all the
    orchestrator has to route on. A description that omits what the subagent
    returns is the most common cause of a mis-route."""
    description = SPECS[name].description
    assert len(description) > 80, "too thin to route on"
    assert "eturn" in description, "does not say what it hands back"


def test_no_role_can_delete():
    for name, spec in SPECS.items():
        assert "delete" not in spec.fs_tools, name


def test_read_only_roles_hold_no_write_tools():
    for name in ("explorer", "searcher", "reviewer"):
        assert not (SPECS[name].fs_tools & WRITE_TOOLS), name
        assert SPECS[name].mode == "read-only"


def test_orchestrator_prompt_matches_the_registered_fleet():
    """Every subagent must appear in the routing table, and the table must not
    advertise one that does not exist."""
    from loom.core.orchestrator import orchestrator_system_prompt

    prompt = orchestrator_system_prompt()
    for name in SPECS:
        assert f"`{name}`" in prompt, f"{name} is missing from the fleet listing"
    for named in _MENTION.findall(prompt):
        if named in ("task", "write_todos", "consult", "read_file"):
            continue
        if named in ALL_FS_TOOLS:
            continue  # the sentence that says they are absent
        assert named in SPECS, f"prompt advertises unknown subagent `{named}`"


def test_orchestrator_prompt_does_not_claim_search_tools():
    from loom.core.orchestrator import _orchestrator_excluded_tools, orchestrator_system_prompt

    prompt = orchestrator_system_prompt()
    excluded = _orchestrator_excluded_tools(airgap=False)
    # They are named exactly once, in the sentence declaring they are quarantined.
    assert "You have no" in prompt
    for tool in ("ls", "glob", "grep", "write_file", "edit_file"):
        assert tool in excluded


# ---------------------------------------------------------------------------
# The virtual filesystem root
#
# The backend mounts the project at `/` (virtual_mode=True), and nothing said so.
# A live run had the explorer grep an invented `/home/user`, then close its report
# by "correcting" the user: "the file is at /src/billing.py, not src/billing.py".
# Both spellings are the same file; the prompt now says which.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SPECS))
def test_every_subagent_is_told_where_the_root_is(name):
    prompt = SPECS[name].system_prompt
    assert "project root is `/`" in prompt


def _flat(text: str) -> str:
    """Prompts are hand-wrapped, so match on content rather than line breaks."""
    return " ".join(text.split())


def test_the_root_convention_says_both_spellings_are_one_file():
    preamble = _flat(ISOLATION_PREAMBLE)
    assert "name the same file" in preamble
    assert "correction" in preamble, (
        "the failure was not confusion but a confident wrong correction in the report"
    )


def test_the_root_convention_rules_out_the_wider_filesystem():
    """The invented `/home/user` grep cost a tool call and returned nothing."""
    preamble = _flat(ISOLATION_PREAMBLE)
    assert "Nothing exists outside the root" in preamble
    assert "no home directory" in preamble


def test_the_orchestrator_is_told_the_root_whenever_it_can_read():
    from loom.core.orchestrator import orchestrator_system_prompt

    prompt = orchestrator_system_prompt(can_read=True)
    assert "read_file" in prompt
    assert "project root is `/`" in prompt


def test_the_orchestrator_is_not_told_about_paths_when_it_cannot_read():
    """Airgap mode: no filesystem tools, so path conventions are noise."""
    from loom.core.orchestrator import orchestrator_system_prompt

    assert "project root is `/`" not in orchestrator_system_prompt(can_read=False)


def test_the_prompt_announces_the_reminder_on_read_results():
    """DelegationReminder appends a line to every read_file result, so the prompt
    has to predict it or the model treats it as news to react to."""
    from loom.core.orchestrator import orchestrator_system_prompt

    prompt = _flat(orchestrator_system_prompt())
    assert "Every `read_file` result ends with a one-line reminder" in prompt
    assert "`explorer`" in prompt


# ---------------------------------------------------------------------------
# The virtual root stops at the shell
#
# A live run on 2026-10-09: the orchestrator briefed bash with "`cd /` is the
# project root ... run `bash ./ci.sh` from `/`", and bash ran exactly that. For
# the file tools `/` is the project; in a real shell it is the machine's root, so
# the command left the project and missed the allow rule for ./ci.sh.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["bash", "general-purpose"])
def test_shell_roles_are_told_not_to_cd_to_the_virtual_root(name):
    prompt = _flat(SPECS[name].system_prompt)
    assert "never `cd /`" in prompt
    assert "already starts in the project root" in prompt


@pytest.mark.parametrize("name", ["explorer", "editor", "searcher", "reviewer", "tester"])
def test_roles_without_a_shell_hear_nothing_about_it(name):
    assert "cd /" not in SPECS[name].system_prompt


def test_the_orchestrator_briefs_commands_relative_to_the_root():
    from loom.core.orchestrator import orchestrator_system_prompt

    for can_read in (True, False):
        prompt = _flat(orchestrator_system_prompt(can_read=can_read))
        assert "relative to the project root" in prompt
        assert "machine's real root" in prompt
