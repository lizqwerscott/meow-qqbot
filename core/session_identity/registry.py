"""Persistent mapping between canonical identities and legacy storage facts."""

from __future__ import annotations

import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable

from .identity import (
    ConversationRef,
    DeliveryTarget,
    SessionKind,
    build_chat_session_key,
    build_workspace_slug,
)


class SessionIdentityRegistry:
    """Own the durable identity mapping used by all storage adapters."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS session_identities (
                    session_key TEXT PRIMARY KEY,
                    session_kind TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    channel TEXT,
                    account_id TEXT,
                    chat_type TEXT,
                    target_id TEXT,
                    legacy_key TEXT,
                    workspace_slug TEXT,
                    hindsight_document_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    migration_run_id TEXT,
                    created_at REAL NOT NULL,
                    migrated_at REAL,
                    UNIQUE(agent_id, channel, account_id, chat_type, target_id)
                );
                CREATE INDEX IF NOT EXISTS idx_session_identity_legacy
                    ON session_identities(legacy_key);
                CREATE INDEX IF NOT EXISTS idx_session_identity_state
                    ON session_identities(state);
                """)

    @staticmethod
    def _document_id() -> str:
        return f"hdoc_{secrets.token_urlsafe(9).rstrip('=')}"

    def register_chat(
        self,
        target: DeliveryTarget,
        *,
        agent_id: str = "main",
        legacy_key: str | None = None,
        workspace_slug: str | None = None,
        hindsight_document_id: str | None = None,
        state: str = "planned",
        migration_run_id: str | None = None,
    ) -> ConversationRef:
        session_key = build_chat_session_key(target, agent_id=agent_id)
        now = time.time()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM session_identities WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            if row is not None:
                existing_target = DeliveryTarget(
                    row["channel"],
                    row["account_id"],
                    row["chat_type"],
                    row["target_id"],
                )
                if existing_target != target:
                    raise ValueError("session identity target mismatch")
                if any(
                    value is not None
                    for value in (
                        legacy_key,
                        workspace_slug,
                        hindsight_document_id,
                        migration_run_id,
                    )
                ):
                    self._conn.execute(
                        """
                        UPDATE session_identities
                        SET legacy_key = COALESCE(?, legacy_key),
                            workspace_slug = COALESCE(?, workspace_slug),
                            hindsight_document_id = COALESCE(?, hindsight_document_id),
                            migration_run_id = COALESCE(?, migration_run_id)
                        WHERE session_key = ?
                        """,
                        (
                            legacy_key,
                            workspace_slug,
                            hindsight_document_id,
                            migration_run_id,
                            session_key,
                        ),
                    )
                    row = self._conn.execute(
                        "SELECT * FROM session_identities WHERE session_key = ?",
                        (session_key,),
                    ).fetchone()
                return self._row_to_ref(row)
            document_id = hindsight_document_id or self._document_id()
            self._conn.execute(
                """
                INSERT INTO session_identities (
                    session_key, session_kind, agent_id, channel, account_id,
                    chat_type, target_id, legacy_key, workspace_slug,
                    hindsight_document_id, state, migration_run_id, created_at,
                    migrated_at
                ) VALUES (?, 'chat', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_key,
                    agent_id,
                    target.channel,
                    target.account_id,
                    target.chat_type,
                    target.target_id,
                    legacy_key,
                    workspace_slug or build_workspace_slug(target),
                    document_id,
                    state,
                    migration_run_id,
                    now,
                    now if state in {"verified", "cutover"} else None,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM session_identities WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        return self._row_to_ref(row)

    def register_internal(
        self,
        session_key: str,
        *,
        session_kind: SessionKind,
        agent_id: str = "main",
        legacy_key: str | None = None,
        hindsight_document_id: str | None = None,
        state: str = "planned",
        migration_run_id: str | None = None,
    ) -> ConversationRef:
        if session_kind == "chat":
            raise ValueError("use register_chat for chat sessions")
        now = time.time()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM session_identities WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO session_identities (
                        session_key, session_kind, agent_id, legacy_key,
                        hindsight_document_id, state, migration_run_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_key,
                        session_kind,
                        agent_id,
                        legacy_key,
                        hindsight_document_id or self._document_id(),
                        state,
                        migration_run_id,
                        now,
                    ),
                )
                row = self._conn.execute(
                    "SELECT * FROM session_identities WHERE session_key = ?",
                    (session_key,),
                ).fetchone()
        return self._row_to_ref(row)

    def set_state(self, session_key: str, state: str) -> None:
        with self._lock, self._conn:
            updated = self._conn.execute(
                "UPDATE session_identities SET state = ?, migrated_at = ? WHERE session_key = ?",
                (
                    state,
                    time.time() if state in {"verified", "cutover"} else None,
                    session_key,
                ),
            ).rowcount
        if not updated:
            raise KeyError(session_key)

    def get(self, session_key: str) -> ConversationRef | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM session_identities WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        return self._row_to_ref(row) if row is not None else None

    def find_by_target(
        self, target: DeliveryTarget, *, agent_id: str = "main"
    ) -> ConversationRef | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM session_identities
                WHERE agent_id = ? AND channel = ? AND account_id = ?
                  AND chat_type = ? AND target_id = ?
                """,
                (agent_id, *target.catalog_key),
            ).fetchone()
        return self._row_to_ref(row) if row is not None else None

    def find_by_legacy(self, legacy_key: str) -> list[ConversationRef]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM session_identities WHERE legacy_key = ? ORDER BY session_key",
                (legacy_key,),
            ).fetchall()
        return [self._row_to_ref(row) for row in rows]

    def list_all(self, *, states: Iterable[str] | None = None) -> list[ConversationRef]:
        with self._lock:
            if states:
                values = tuple(states)
                placeholders = ",".join("?" for _ in values)
                rows = self._conn.execute(
                    f"SELECT * FROM session_identities WHERE state IN ({placeholders}) ORDER BY session_key",
                    values,
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM session_identities ORDER BY session_key"
                ).fetchall()
        return [self._row_to_ref(row) for row in rows]

    def missing_migration_keys(
        self,
        expected_keys: Iterable[str],
        *,
        allowed_states: Iterable[str] = ("verified", "cutover"),
    ) -> tuple[str, ...]:
        """Return expected migrated keys absent from an allowed registry state."""
        expected = {str(key) for key in expected_keys if key}
        if not expected:
            return ()
        states = tuple(str(state) for state in allowed_states)
        placeholders = ",".join("?" for _ in states)
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_key FROM session_identities "
                f"WHERE state IN ({placeholders}) AND session_key IN "
                f"({','.join('?' for _ in expected)})",
                (*states, *sorted(expected)),
            ).fetchall()
        present = {str(row["session_key"]) for row in rows}
        return tuple(sorted(expected - present))

    @staticmethod
    def _row_to_ref(row: sqlite3.Row) -> ConversationRef:
        target = None
        if row["session_kind"] == "chat":
            target = DeliveryTarget(
                row["channel"], row["account_id"], row["chat_type"], row["target_id"]
            )
        legacy = (row["legacy_key"],) if row["legacy_key"] else ()
        return ConversationRef(
            session_key=row["session_key"],
            target=target,
            session_kind=row["session_kind"],
            legacy_session_keys=legacy,
            hindsight_document_id=row["hindsight_document_id"],
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
