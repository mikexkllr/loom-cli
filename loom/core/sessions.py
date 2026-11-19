"""Session persistence for /resume.

The LangGraph SQLite checkpointer at ``.loom/sessions.db`` holds the actual
thread state; this module keeps a small human-readable index
(``.loom/sessions.json``) so /resume can list past sessions by title and date.
"""

from __future__ import annotations

import json
import uuid
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
    except Exception as exc:
        # The fallback still gives a working session, but it stops surviving
        # restarts — `/resume` will find nothing. That is a promise quietly
        # withdrawn, so it gets reported even though nothing breaks now.
        from loom.core import telemetry

        telemetry.report("sessions.checkpointer", exc)
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
    """A thread id that sorts by time and is never issued twice.

    The timestamp alone has one-second resolution, and the id is the
    checkpointer's key. `/clear` or `/compact` inside the second a session
    started got the *same* id back, so the "new" thread was the old one with
    its history intact; two `loom "task"` runs started together in one project
    wrote into a single shared conversation.
    """
    return f"loom-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def load_index(cwd: str | Path) -> list[dict]:
    path = _index_path(cwd)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"sessions index is a {type(data).__name__}, not a list")
        return data
    except Exception as exc:
        # An unreadable index reads as "no sessions", which is indistinguishable
        # from a fresh project — and hides that the user's session history is
        # still on disk but no longer listable.
        from loom.core import telemetry

        telemetry.report("sessions.index", exc)
        return []


def record(cwd: str | Path, thread_id: str, first_message: str) -> None:
    """Upsert a session row; ``first_message`` titles new sessions."""
    sessions = [row for row in load_index(cwd) if isinstance(row, dict) and row.get("thread_id")]
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
