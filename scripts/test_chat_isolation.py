"""
Local test for per-browser chat isolation.

Simulates two independent browsers (two different user_ids) against a
throwaway SQLite file and asserts:

  A. Browser A cannot see Browser B's conversations (list scope).
  B. Browser B cannot see Browser A's conversations (list scope).
  C. Refresh preserves each browser's own conversations (list survives
     re-opening the DB file — the SQLite equivalent of a page reload).
  D. New Chat works independently for each browser.
  E. Rename/delete only affect the current browser's conversations
     — including the "attacker knows the integer id" case: A tries
     to rename/delete/read/write B's conversation by id, all fail.
  F. Pre-existing rows with user_id = NULL (legacy) are invisible to
     any real browser.

Run:
    python -m scripts.test_chat_isolation
"""

from __future__ import annotations

import sqlite3
import tempfile
import uuid
from pathlib import Path

from src.chat.store import ChatStore


def _fresh_db_path() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="chat_isolation_"))
    return tmp / "test.db"


def _seed_legacy_row(db_path: Path) -> int:
    """Simulate a row written before the ownership column existed."""
    # Create the table without user_id, then let ChatStore.__init__
    # run the migration path. This mirrors the real upgrade scenario.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE conversations ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " title TEXT NOT NULL,"
        " created_at TEXT NOT NULL,"
        " updated_at TEXT NOT NULL)"
    )
    cur = conn.execute(
        "INSERT INTO conversations (title, created_at, updated_at)"
        " VALUES ('LEGACY - should be invisible', "
        "'2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00')"
    )
    legacy_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return legacy_id


def main() -> None:
    db_path = _fresh_db_path()
    print(f"[setup] temp DB at {db_path}")

    # Seed a pre-ownership row, then instantiate ChatStore (which runs
    # the ALTER TABLE migration).
    legacy_id = _seed_legacy_row(db_path)
    store = ChatStore(db_path)

    user_a = uuid.uuid4().hex
    user_b = uuid.uuid4().hex
    print(f"[setup] user_a = {user_a[:8]}…  user_b = {user_b[:8]}…")

    # --- D: independent New Chat ---------------------------------------
    a1 = store.create_conversation(user_a, "A: my private notes")
    a2 = store.create_conversation(user_a, "A: another chat")
    b1 = store.create_conversation(user_b, "B: my private notes")

    store.add_message(user_a, a1, "user", "A's first question")
    store.add_message(user_b, b1, "user", "B's first question")
    print(f"[D] created A convs {a1},{a2} and B conv {b1}")

    # --- A: list scope -------------------------------------------------
    a_titles = {c.title for c in store.list_conversations(user_a)}
    b_titles = {c.title for c in store.list_conversations(user_b)}
    assert a_titles == {"A: my private notes", "A: another chat"}, (
        f"A leaked: {a_titles}"
    )
    print(f"[A] user_a sees only their own: {a_titles}")

    # --- B: list scope -------------------------------------------------
    assert b_titles == {"B: my private notes"}, f"B leaked: {b_titles}"
    print(f"[B] user_b sees only their own: {b_titles}")

    # --- F: legacy row invisible to both -------------------------------
    all_visible_ids = {c.id for c in store.list_conversations(user_a)} \
        | {c.id for c in store.list_conversations(user_b)}
    assert legacy_id not in all_visible_ids, (
        f"legacy row {legacy_id} leaked into a real user's list"
    )
    print(f"[F] legacy row id={legacy_id} invisible to both users")

    # --- E: A cannot access B's conversation by id ---------------------
    # Read
    stolen_msgs = store.get_messages(user_a, b1)
    assert stolen_msgs == [], (
        f"A stole B's messages by id: {stolen_msgs}"
    )
    stolen_conv = store.get_conversation(user_a, b1)
    assert stolen_conv is None, f"A stole B's conv by id: {stolen_conv}"

    # Rename (should be a no-op returning False; B's title unchanged)
    renamed = store.rename_conversation(user_a, b1, "HACKED BY A")
    assert renamed is False, "rename crossed ownership boundary"
    still_b = store.get_conversation(user_b, b1)
    assert still_b is not None and still_b.title == "B: my private notes", (
        f"B's title changed under attack: {still_b}"
    )

    # Write
    try:
        store.add_message(user_a, b1, "user", "HACKED MESSAGE")
    except PermissionError:
        pass
    else:
        raise AssertionError("add_message crossed ownership boundary")

    # Delete (should be a no-op returning False; B's conv still present)
    deleted = store.delete_conversation(user_a, b1)
    assert deleted is False, "delete crossed ownership boundary"
    still_b = store.get_conversation(user_b, b1)
    assert still_b is not None, "B's conv was deleted by A"
    print("[E] A blocked from read/rename/write/delete on B's conv by id")

    # --- E': B can rename/delete their own ------------------------------
    assert store.rename_conversation(user_b, b1, "B: renamed") is True
    assert store.get_conversation(user_b, b1).title == "B: renamed"
    assert store.delete_conversation(user_b, b1) is True
    assert store.get_conversation(user_b, b1) is None
    print("[E'] B's own rename/delete still work")

    # --- C: "refresh" = re-open the DB file ---------------------------
    del store
    store2 = ChatStore(db_path)
    a_after = {c.title for c in store2.list_conversations(user_a)}
    b_after = {c.title for c in store2.list_conversations(user_b)}
    assert a_after == {"A: my private notes", "A: another chat"}, (
        f"A lost history on refresh: {a_after}"
    )
    assert b_after == set(), f"B saw ghost rows on refresh: {b_after}"
    print(f"[C] refresh preserved A={a_after}, B={b_after}")

    print("\nALL ISOLATION ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
