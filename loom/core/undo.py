"""Per-turn file snapshots powering /undo.

Before the policy middleware lets a write_file/edit_file through, it snapshots
the target file into ``.loom/undo/<turn_id>/``. ``/undo`` restores the most
recent turn's snapshots — overwritten files get their old bytes back, files
that didn't exist before the turn are deleted.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

from loom.core.slot import Slot

# Set by the REPL at the start of each turn; empty = snapshots disabled.
# A process-global Slot, not a contextvar — snapshots must fire from
# LangGraph's tool worker threads too (see loom.core.slot).
current_turn_id: Slot[str] = Slot("")

_INDEX = "index.json"

# Parallel subagents write files from separate worker threads, and every write
# snapshots first. The index is read, extended and rewritten, and the backup's
# file name is derived from its length — so two unguarded snapshots could hand
# two files the same backup name and have /undo restore one file's bytes into
# the other.
_LOCK = threading.Lock()

# Turns kept on disk. /undo walks back one turn at a time, so this is the depth
# of the undo history; without a bound `.loom/undo` grew by one directory of
# file copies per turn, forever.
KEEP_TURNS = 50


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

    with _LOCK:
        tdir = _turn_dir(cwd, turn_id)
        tdir.mkdir(parents=True, exist_ok=True)
        index_path = tdir / _INDEX
        if index_path.exists():
            index: dict = json.loads(index_path.read_text())
        else:
            index = {"created": time.time(), "files": {}}
            _prune(cwd, keep=turn_id)
        if rel in index["files"]:
            return  # keep the original pre-turn snapshot

        key = f"f{len(index['files'])}"
        if target.exists():
            shutil.copy2(target, tdir / key)
            index["files"][rel] = {"key": key, "existed": True}
        else:
            index["files"][rel] = {"key": key, "existed": False}
        index_path.write_text(json.dumps(index, indent=2))


def _created(index_path: Path) -> float:
    """When a turn's first snapshot was taken.

    Recorded in the index; an index written before that field existed falls
    back to the file's mtime, which is the time of its last snapshot.
    """
    try:
        created = json.loads(index_path.read_text()).get("created")
        if isinstance(created, (int, float)):
            return float(created)
    except (OSError, ValueError, AttributeError):
        pass
    try:
        return index_path.stat().st_mtime
    except OSError:
        return 0.0


def turns(cwd: str | Path) -> list[str]:
    """Snapshot turn ids, oldest first.

    Ordered by when each turn started snapshotting, not by name. Names end in
    a turn counter, and sorted as text ``…-t10`` comes before ``…-t2`` — so
    from the tenth turn of a session on, /undo rolled back turn nine instead of
    the turn that had just run.
    """
    root = _undo_root(cwd)
    if not root.exists():
        return []
    found = [
        (_created(p / _INDEX), p.name)
        for p in root.iterdir()
        if p.is_dir() and (p / _INDEX).exists()
    ]
    return [name for _, name in sorted(found)]


def _prune(cwd: Path, *, keep: str) -> None:
    """Drop the oldest turns beyond :data:`KEEP_TURNS`. Best-effort."""
    stale = [name for name in turns(cwd) if name != keep]
    for name in stale[: max(0, len(stale) - (KEEP_TURNS - 1))]:
        shutil.rmtree(_turn_dir(cwd, name), ignore_errors=True)


def undo_last(cwd: str | Path) -> list[str]:
    """Restore the most recent snapshotted turn. Returns the restored paths."""
    cwd = Path(cwd).resolve()
    with _LOCK:
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
