"""Per-turn file snapshots powering /undo.

Before the policy middleware lets a write_file/edit_file through, it snapshots
the target file into ``.loom/undo/<turn_id>/``. ``/undo`` restores the most
recent turn's snapshots — overwritten files get their old bytes back, files
that didn't exist before the turn are deleted.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from loom.core.slot import Slot

# Set by the REPL at the start of each turn; empty = snapshots disabled.
# A process-global Slot, not a contextvar — snapshots must fire from
# LangGraph's tool worker threads too (see loom.core.slot).
current_turn_id: Slot[str] = Slot("")

_INDEX = "index.json"


def _undo_root(cwd: str | Path) -> Path:
    return Path(cwd).resolve() / ".loom" / "undo"


def _turn_dir(cwd: str | Path, turn_id: str) -> Path:
    return _undo_root(cwd) / turn_id


def snapshot(cwd: str | Path, target: str | Path) -> None:
    """Record the pre-write state of ``target`` for the current turn.

    First write per (turn, file) wins — later writes in the same turn keep the
    original pre-turn content so /undo rolls the whole turn back.
    """
    turn_id = current_turn_id.get()
    if not turn_id:
        return
    cwd = Path(cwd).resolve()
    target = Path(target)
    if not target.is_absolute():
        target = cwd / target
    try:
        rel = str(target.resolve().relative_to(cwd))
    except ValueError:
        return  # outside the sandbox — nothing we manage

    tdir = _turn_dir(cwd, turn_id)
    tdir.mkdir(parents=True, exist_ok=True)
    index_path = tdir / _INDEX
    index: dict = json.loads(index_path.read_text()) if index_path.exists() else {"files": {}}
    if rel in index["files"]:
        return  # keep the original pre-turn snapshot

    key = f"f{len(index['files'])}"
    if target.exists():
        shutil.copy2(target, tdir / key)
        index["files"][rel] = {"key": key, "existed": True}
    else:
        index["files"][rel] = {"key": key, "existed": False}
    index_path.write_text(json.dumps(index, indent=2))


def turns(cwd: str | Path) -> list[str]:
    """Snapshot turn ids, oldest first."""
    root = _undo_root(cwd)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / _INDEX).exists())


def undo_last(cwd: str | Path) -> list[str]:
    """Restore the most recent snapshotted turn. Returns the restored paths."""
    cwd = Path(cwd).resolve()
    all_turns = turns(cwd)
    if not all_turns:
        return []
    tdir = _turn_dir(cwd, all_turns[-1])
    index = json.loads((tdir / _INDEX).read_text())
    restored: list[str] = []
    for rel, entry in index["files"].items():
        target = cwd / rel
        if entry["existed"]:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tdir / entry["key"], target)
        elif target.exists():
            target.unlink()  # file was created this turn — undo removes it
        restored.append(rel)
    shutil.rmtree(tdir)
    return restored
