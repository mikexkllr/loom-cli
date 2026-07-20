"""Agent skills: source layering, SKILL.md discovery, and the packaged skills."""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")
pytest.importorskip("rich")

from loom.core import skills as skills_mod


def test_builtin_skills_ship_with_loom():
    assert skills_mod.BUILTIN_DIR.is_dir()
    sources = skills_mod.skill_sources(".")
    assert sources and sources[0][0] == "/skills/built_in_skills/"


def test_packaged_graphify_skill_is_discovered(tmp_path):
    found = {s["name"]: s for s in skills_mod.list_skills(tmp_path)}
    skill = found["graphify-graph-rag"]
    assert skill["source"] == "built-in"
    assert "knowledge graph" in skill["description"]


def test_project_skills_layer_and_shadow(tmp_path):
    proj = tmp_path / ".loom" / "skills"
    (proj / "release-checklist").mkdir(parents=True)
    (proj / "release-checklist" / "SKILL.md").write_text(
        "---\nname: release-checklist\ndescription: Steps before tagging a release\n---\n# Release\n"
    )
    # Same name as the packaged skill: project layer wins (last source).
    (proj / "graphify-graph-rag").mkdir()
    (proj / "graphify-graph-rag" / "SKILL.md").write_text(
        "---\nname: graphify-graph-rag\ndescription: project override\n---\n# Override\n"
    )
    sources = skills_mod.skill_sources(tmp_path)
    assert [r for r, _ in sources] == ["/skills/built_in_skills/", "/skills/project/"]
    found = {s["name"]: s for s in skills_mod.list_skills(tmp_path)}
    assert found["release-checklist"]["source"] == "project"
    assert found["graphify-graph-rag"]["description"] == "project override"
    assert found["graphify-graph-rag"]["source"] == "project"


def test_frontmatterless_skill_uses_directory_name(tmp_path):
    proj = tmp_path / ".loom" / "skills" / "bare"
    proj.mkdir(parents=True)
    (proj / "SKILL.md").write_text("# Just markdown, no frontmatter\n")
    found = {s["name"] for s in skills_mod.list_skills(tmp_path)}
    assert "bare" in found


def test_skills_slash_command_lists(tmp_path, capsys):
    from loom.core import settings as st
    from loom.ui import slash
    from loom.ui.repl import Session

    s = Session(st.load_settings(tmp_path), cwd=str(tmp_path))
    assert slash.dispatch(s, "/skills") is True
    out = capsys.readouterr().out
    assert "graphify-graph-rag" in out
