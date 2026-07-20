"""Agent skills — Anthropic-style SKILL.md folders, via deepagents.

deepagents' ``SkillsMiddleware`` implements progressive disclosure: only each
skill's name + description enter the system prompt; the agent reads the full
SKILL.md on demand when a task matches. Loom layers three sources (later
overrides earlier on name collisions):

    packaged   loom/skills/               (ships with Loom, e.g. graphify-graph-rag)
    user       ~/.loom/skills/            (yours, every project)
    project    <project>/.loom/skills/    (commit these for your team)

A skill is a directory holding a ``SKILL.md`` with YAML frontmatter
(``name``, ``description``) followed by markdown instructions. Sources are
mounted into the agent's virtual filesystem under ``/skills/...`` (see
``build_orchestrator``), so skills work the same across backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loom.core.config import PACKAGE_ROOT, USER_CONFIG_DIR

BUILTIN_DIR = PACKAGE_ROOT / "skills"


def skill_sources(cwd: str | Path = ".") -> list[tuple[str, Path]]:
    """(virtual route, real directory) pairs for every existing skills layer,
    in override order (packaged -> user -> project; last one wins)."""
    ordered = [
        # `built_in_skills` is a magic leaf deepagents renders as "Built-in".
        ("/skills/built_in_skills/", BUILTIN_DIR),
        ("/skills/user/", USER_CONFIG_DIR / "skills"),
        ("/skills/project/", Path(cwd).resolve() / ".loom" / "skills"),
    ]
    return [(route, path) for route, path in ordered if path.is_dir()]


def _frontmatter(skill_md: Path) -> dict[str, Any]:
    import yaml

    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    try:
        data = yaml.safe_load(text[3:end])
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def list_skills(cwd: str | Path = ".") -> list[dict[str, str]]:
    """All discoverable skills: {name, description, source, path}. Later
    sources shadow earlier ones with the same name, mirroring load order."""
    by_name: dict[str, dict[str, str]] = {}
    labels = {"/skills/built_in_skills/": "built-in", "/skills/user/": "user", "/skills/project/": "project"}
    for route, root in skill_sources(cwd):
        for skill_md in sorted(root.glob("*/SKILL.md")):
            meta = _frontmatter(skill_md)
            name = str(meta.get("name") or skill_md.parent.name)
            by_name[name] = {
                "name": name,
                "description": str(meta.get("description") or ""),
                "source": labels.get(route, route),
                "path": str(skill_md),
            }
    return sorted(by_name.values(), key=lambda s: s["name"])
