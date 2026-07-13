"""Session persistence for /resume.

The LangGraph SQLite checkpointer at ``.loom/sessions.db`` holds the actual
thread state; this module keeps a small human-readable index
(``.loom/sessions.json``) so /resume can list past sessions by title and date.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _loom_dir(cwd: str | Path) -> Path:
    d = Path(cwd).resolve() / ".loom"
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path(cwd: str | Path) -> Path:
    return _loom_dir(cwd) / "sessions.db"


def _index_path(cwd: str | Path) -> Path:
    return _loom_dir(cwd) / "sessions.json"


def make_checkpointer(cwd: str | Path) -> tuple[Any, bool]:
    """(checkpointer, durable). SQLite when available — survives restarts —
    else the in-memory saver (thread state lives only for this process)."""
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(str(db_path(cwd)), check_same_thread=False)
        return SqliteSaver(conn), True
    except Exception:
        pass
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver(), False
    except Exception:
        try:
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver(), False
        except Exception:
            return None, False


def new_thread_id() -> str:
    return f"loom-{datetime.now():%Y%m%d-%H%M%S}"


def load_index(cwd: str | Path) -> list[dict]:
    path = _index_path(cwd)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def record(cwd: str | Path, thread_id: str, first_message: str) -> None:
    """Upsert a session row; ``first_message`` titles new sessions."""
    sessions = load_index(cwd)
    now = datetime.now().isoformat(timespec="seconds")
    for row in sessions:
        if row["thread_id"] == thread_id:
            row["updated"] = now
            row["turns"] = row.get("turns", 0) + 1
            break
    else:
        title = " ".join(first_message.split())[:80]
        sessions.append(
            {"thread_id": thread_id, "title": title, "created": now, "updated": now, "turns": 1}
        )
    sessions = sessions[-50:]  # keep the index bounded
    _index_path(cwd).write_text(json.dumps(sessions, indent=2), encoding="utf-8")
