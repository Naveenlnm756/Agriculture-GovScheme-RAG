"""
SQLite-backed chat persistence for the Streamlit frontend.

Design decisions worth naming so downstream edits don't unpick them:

  1. **Deliberately separate from Chroma.** Chroma holds corpus knowledge
     (embeddings + chunks + metadata). This DB holds user-authored
     conversations. Mixing them would (a) tie chat retention to vector-
     store lifecycle — a corpus rebuild would wipe chat history — and
     (b) let user text creep into retrieval results. Keeping them in
     separate files at separate paths makes both properties impossible
     to violate by accident.

  2. **One SQLite file, WAL mode.** WAL gives concurrent reader safety
     while a single writer commits (Streamlit reruns fire read-side
     queries very often). One file keeps deployment trivial — put the
     file on the Fly.io volume and it survives restarts.

  3. **Assistant payload stored as JSON.** The full `/query` response
     (citations, scheme discovery, latency, retrieved chunks) is
     serialised into a `payload_json` column on the assistant message.
     This lets a reloaded conversation re-render EXACTLY what the user
     saw originally — including scheme cards and source citations —
     without re-hitting the API. A "content" column alone would lose
     that context.

  4. **No context carry-over into the RAG pipeline.** By design (owner
     decision D1, 2026-09-20): the RAG pipeline is stateless. Each
     query gets fresh retrieval. This module stores past turns for
     RENDERING; nothing in this module ever feeds prior turns back
     into the retriever. Loading an old conversation shows the past
     answers verbatim; asking a new question in that conversation
     starts a fresh retrieval pass.

  5. **Per-browser ownership (added 2026-09-20).** Each conversation
     row carries a ``user_id`` — the anonymous 32-hex browser id
     minted by ``src.chat.user_id``. Every read and every mutating
     call requires the caller's ``user_id`` and scopes the SQL with
     ``WHERE user_id = ?``. This is the ONLY isolation mechanism
     between browsers; it is not a shortcut that can be skipped for
     convenience. A user cannot access another user's conversation
     by guessing its integer id — the id AND the owner must match.

     Legacy rows written before this change have ``user_id = NULL``
     and are therefore invisible to every real browser (no real
     UUID ever equals NULL). We keep them on disk rather than
     deleting them so no data is lost.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterator


# --- Value objects ----------------------------------------------------------

@dataclass(frozen=True)
class Conversation:
    id: int
    title: str
    created_at: str   # ISO-8601 UTC
    updated_at: str

    @property
    def created_at_dt(self) -> datetime:
        return datetime.fromisoformat(self.created_at)

    @property
    def updated_at_dt(self) -> datetime:
        return datetime.fromisoformat(self.updated_at)


@dataclass(frozen=True)
class Message:
    id: int
    conversation_id: int
    role: str              # 'user' or 'assistant'
    content: str           # the readable text (query or answer)
    payload_json: str | None
    created_at: str

    @property
    def payload(self) -> dict[str, Any] | None:
        """Decoded assistant payload (or None for user messages)."""
        if not self.payload_json:
            return None
        try:
            return json.loads(self.payload_json)
        except json.JSONDecodeError:
            return None


# --- Schema -----------------------------------------------------------------
#
# `user_id` is TEXT and nullable so pre-existing rows (written before we
# added ownership) can coexist. New inserts always populate it. Every
# read filters on `user_id = ?`, so a NULL row never matches a real
# 32-hex UUID and legacy conversations are invisible to every browser.

# Split into "base" (safe on fresh AND on legacy pre-ownership DBs)
# and "post-migration" (references user_id, so must run only AFTER
# the ALTER TABLE adds the column on old DBs).

_SCHEMA_BASE = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    payload_json    TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conv
    ON messages(conversation_id, created_at);
"""

_SCHEMA_POST_MIGRATION = """
CREATE INDEX IF NOT EXISTS idx_conv_user_updated
    ON conversations(user_id, updated_at DESC);
"""


# --- The store --------------------------------------------------------------

class ChatStore:
    """
    Thin wrapper around a single SQLite file. Instantiate once per
    process; safe to share across Streamlit reruns via
    `st.cache_resource`.

    All conversation-facing methods take a ``user_id`` (the caller's
    anonymous browser id). This is not optional: it is the only thing
    keeping browsers isolated from each other. Do not add convenience
    overloads that drop it.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        # Ensure parent dir exists. On Fly.io the volume mount at /data
        # is guaranteed present; locally the user may not have created
        # data/ yet.
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # A lock to serialise writes from within one process — SQLite
        # accepts one writer at a time and Streamlit reruns can race.
        self._write_lock = Lock()
        self._init_schema()

    # --- Internals -----------------------------------------------------

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """One short-lived connection per operation. `check_same_thread=False`
        because Streamlit callbacks and page renders may run on different
        threads within the same script run. Foreign-key enforcement must
        be re-enabled per-connection (SQLite defaults to OFF)."""
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=5.0,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            # 1. Create tables (safe on fresh DBs and on legacy ones —
            #    CREATE TABLE IF NOT EXISTS is a no-op on the pre-
            #    ownership schema).
            c.executescript(_SCHEMA_BASE)
            # 2. Migrate: `CREATE TABLE IF NOT EXISTS` never adds new
            #    columns to an existing table. If this DB was created
            #    before we introduced ownership, `user_id` will be
            #    missing — add it explicitly, leaving existing rows at
            #    NULL so they become invisible to every real browser id.
            existing_cols = {
                row["name"]
                for row in c.execute("PRAGMA table_info(conversations)")
            }
            if "user_id" not in existing_cols:
                c.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
            # 3. Create the ownership index (must run after the column
            #    is guaranteed to exist).
            c.executescript(_SCHEMA_POST_MIGRATION)
            c.commit()

    @staticmethod
    def _now() -> str:
        # UTC ISO-8601. Timestamps live in one timezone so ordering is
        # trivial and there's no local-time / DST footgun.
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # --- Conversations -------------------------------------------------

    def create_conversation(self, user_id: str, title: str = "New chat") -> int:
        now = self._now()
        with self._write_lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO conversations (user_id, title, created_at, updated_at)"
                " VALUES (?, ?, ?, ?)",
                (user_id, title.strip() or "New chat", now, now),
            )
            c.commit()
            return int(cur.lastrowid)

    def list_conversations(self, user_id: str) -> list[Conversation]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, title, created_at, updated_at"
                " FROM conversations WHERE user_id = ?"
                " ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [
            Conversation(
                id=r["id"], title=r["title"],
                created_at=r["created_at"], updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def get_conversation(
        self, user_id: str, conv_id: int
    ) -> Conversation | None:
        with self._conn() as c:
            r = c.execute(
                "SELECT id, title, created_at, updated_at"
                " FROM conversations WHERE id = ? AND user_id = ?",
                (conv_id, user_id),
            ).fetchone()
        if r is None:
            return None
        return Conversation(
            id=r["id"], title=r["title"],
            created_at=r["created_at"], updated_at=r["updated_at"],
        )

    def rename_conversation(
        self, user_id: str, conv_id: int, new_title: str
    ) -> bool:
        """Rename ``conv_id`` iff it belongs to ``user_id``.

        Returns True if a row was updated. A False return means the
        conversation either does not exist or is owned by someone else;
        the caller should treat those cases identically to avoid
        leaking existence information across users.
        """
        with self._write_lock, self._conn() as c:
            cur = c.execute(
                "UPDATE conversations SET title = ?, updated_at = ?"
                " WHERE id = ? AND user_id = ?",
                (new_title.strip() or "New chat", self._now(),
                 conv_id, user_id),
            )
            c.commit()
            return cur.rowcount > 0

    def delete_conversation(self, user_id: str, conv_id: int) -> bool:
        """Delete ``conv_id`` iff it belongs to ``user_id``.

        Returns True if a row was deleted (see ``rename_conversation``
        for the intent). ON DELETE CASCADE removes the child messages.
        """
        with self._write_lock, self._conn() as c:
            cur = c.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conv_id, user_id),
            )
            c.commit()
            return cur.rowcount > 0

    def touch(self, user_id: str, conv_id: int) -> None:
        """Bump updated_at (used after adding a message so the sidebar
        list re-sorts with the freshly-active conversation on top)."""
        with self._write_lock, self._conn() as c:
            c.execute(
                "UPDATE conversations SET updated_at = ?"
                " WHERE id = ? AND user_id = ?",
                (self._now(), conv_id, user_id),
            )
            c.commit()

    # --- Messages ------------------------------------------------------

    def add_message(
        self,
        user_id: str,
        conversation_id: int,
        role: str,
        content: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Append a message to ``conversation_id`` iff owned by ``user_id``.

        Raises ``PermissionError`` if the conversation is not owned by
        this user. We do NOT silently drop the write — a caller that
        thinks it wrote and got nothing back would be a hard-to-debug
        UI bug. The ownership check is a single indexed lookup so the
        extra round-trip is negligible.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        now = self._now()
        payload_json = json.dumps(payload) if payload is not None else None
        with self._write_lock, self._conn() as c:
            owner_row = c.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if owner_row is None:
                raise PermissionError(
                    f"conversation {conversation_id} is not owned by this user"
                )
            cur = c.execute(
                "INSERT INTO messages"
                " (conversation_id, role, content, payload_json, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, payload_json, now),
            )
            c.execute(
                "UPDATE conversations SET updated_at = ?"
                " WHERE id = ? AND user_id = ?",
                (now, conversation_id, user_id),
            )
            c.commit()
            return int(cur.lastrowid)

    def get_messages(
        self, user_id: str, conversation_id: int
    ) -> list[Message]:
        """Return messages for ``conversation_id`` iff owned by ``user_id``.

        Non-owners get an empty list (indistinguishable from an empty
        conversation) so we do not leak "this conversation exists but
        belongs to someone else".
        """
        with self._conn() as c:
            owner_row = c.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if owner_row is None:
                return []
            rows = c.execute(
                "SELECT id, conversation_id, role, content, payload_json,"
                " created_at FROM messages WHERE conversation_id = ?"
                " ORDER BY created_at ASC, id ASC",
                (conversation_id,),
            ).fetchall()
        return [
            Message(
                id=r["id"], conversation_id=r["conversation_id"],
                role=r["role"], content=r["content"],
                payload_json=r["payload_json"], created_at=r["created_at"],
            )
            for r in rows
        ]

    def message_count(self, user_id: str, conversation_id: int) -> int:
        with self._conn() as c:
            owner_row = c.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if owner_row is None:
                return 0
            r = c.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return int(r["n"] or 0)


# --- Small helper the UI reaches for ----------------------------------------

def title_from_query(query: str, max_len: int = 48) -> str:
    """First user message becomes the conversation title. Trim to the
    first sentence-y unit so the sidebar stays readable."""
    q = (query or "").strip().replace("\n", " ")
    if not q:
        return "New chat"
    if len(q) <= max_len:
        return q
    cut = q[:max_len].rsplit(" ", 1)[0]
    return (cut or q[:max_len]) + "…"
