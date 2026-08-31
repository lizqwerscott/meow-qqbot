"""Durable archive membership for timeline event identities."""

import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable


class ArchiveLedger:
    """Persist the fact that a source identity entered a committed archive.

    The ledger deliberately stores membership separately from archive payloads.
    Archive files may be duplicated or retained independently, while a source
    event can belong to at most one committed batch per chat.
    """

    def __init__(self, db_path: str):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS archive_batches (
                batch_id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at REAL NOT NULL,
                committed_at REAL,
                records_hash TEXT
            )
            """)
        self._ensure_batch_column("records_hash", "TEXT")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS archive_membership (
                chat_id TEXT NOT NULL,
                identity TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                archived_at REAL NOT NULL,
                PRIMARY KEY (chat_id, identity),
                FOREIGN KEY (batch_id) REFERENCES archive_batches(batch_id)
            )
            """)
        self._conn.commit()

    def _ensure_batch_column(self, name: str, definition: str) -> None:
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(archive_batches)")
        }
        if name not in columns:
            self._conn.execute(
                f"ALTER TABLE archive_batches ADD COLUMN {name} {definition}"
            )

    def prepare_batch(
        self, batch_id: str, chat_id: str, records_hash: str | None = None
    ) -> None:
        if not batch_id or not chat_id:
            raise ValueError("batch_id and chat_id are required")
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT chat_id, records_hash FROM archive_batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if existing is not None and existing[0] != chat_id:
                raise RuntimeError(f"archive batch belongs to another chat: {batch_id}")
            if (
                existing is not None
                and records_hash is not None
                and existing[1] is not None
                and existing[1] != records_hash
            ):
                raise RuntimeError(f"archive batch hash mismatch: {batch_id}")
            self._conn.execute(
                """
                INSERT INTO archive_batches(
                    batch_id, chat_id, state, created_at, records_hash
                )
                VALUES (?, ?, 'prepared', ?, ?)
                ON CONFLICT(batch_id) DO NOTHING
                """,
                (batch_id, chat_id, time.time(), records_hash),
            )
            if existing is not None and existing[1] is None and records_hash:
                self._conn.execute(
                    "UPDATE archive_batches SET records_hash = ? WHERE batch_id = ?",
                    (records_hash, batch_id),
                )

    def recover_batch(
        self, batch_id: str, chat_id: str, records_hash: str | None = None
    ) -> str:
        """Validate or create a prepared batch while recovering an operation."""
        self.prepare_batch(batch_id, chat_id, records_hash)
        with self._lock:
            row = self._conn.execute(
                "SELECT state FROM archive_batches WHERE batch_id = ? AND chat_id = ?",
                (batch_id, chat_id),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"archive batch was not persisted: {batch_id}")
        return str(row[0])

    def commit_membership(
        self,
        batch_id: str,
        chat_id: str,
        identities: Iterable[str],
        records_hash: str | None = None,
    ) -> None:
        identities = tuple(
            dict.fromkeys(identity for identity in identities if identity)
        )
        if not batch_id or not chat_id:
            raise ValueError("batch_id and chat_id are required")
        self.prepare_batch(batch_id, chat_id, records_hash)
        with self._lock, self._conn:
            existing_batch = self._conn.execute(
                "SELECT chat_id FROM archive_batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if existing_batch is not None and existing_batch[0] != chat_id:
                raise RuntimeError(f"archive batch belongs to another chat: {batch_id}")
            conflicts = self._conn.execute(
                """
                SELECT identity, batch_id FROM archive_membership
                 WHERE chat_id = ? AND identity IN ({})
                   AND batch_id != ?
                """.format(",".join("?" for _ in identities) or "NULL"),
                (chat_id, *identities, batch_id),
            ).fetchall()
            if conflicts:
                identity, owner_batch_id = conflicts[0]
                raise RuntimeError(
                    f"archive identity already committed: {identity} ({owner_batch_id})"
                )
            self._conn.execute(
                """
                INSERT INTO archive_batches(batch_id, chat_id, state, created_at)
                VALUES (?, ?, 'prepared', ?)
                ON CONFLICT(batch_id) DO NOTHING
                """,
                (batch_id, chat_id, time.time()),
            )
            archived_at = time.time()
            self._conn.executemany(
                """
                INSERT OR IGNORE INTO archive_membership
                    (chat_id, identity, batch_id, archived_at)
                VALUES (?, ?, ?, ?)
                """,
                [(chat_id, identity, batch_id, archived_at) for identity in identities],
            )
            self._conn.execute(
                """
                UPDATE archive_batches
                   SET state = 'committed', committed_at = ?
                 WHERE batch_id = ? AND chat_id = ?
                """,
                (archived_at, batch_id, chat_id),
            )

    def is_archived(self, chat_id: str, identity: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT 1 FROM archive_membership
                 WHERE chat_id = ? AND identity = ?
                """,
                (chat_id, identity),
            ).fetchone()
        return row is not None

    def archived_identities(self, chat_id: str, identities: Iterable[str]) -> set[str]:
        identities = tuple(
            dict.fromkeys(identity for identity in identities if identity)
        )
        if not identities:
            return set()
        placeholders = ",".join("?" for _ in identities)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT identity FROM archive_membership
                 WHERE chat_id = ? AND identity IN ({placeholders})
                """,
                (chat_id, *identities),
            ).fetchall()
        return {str(row[0]) for row in rows}

    def batch_ids_for_identities(
        self, chat_id: str, identities: Iterable[str]
    ) -> dict[str, str]:
        identities = tuple(
            dict.fromkeys(identity for identity in identities if identity)
        )
        if not identities:
            return {}
        placeholders = ",".join("?" for _ in identities)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT identity, batch_id FROM archive_membership
                 WHERE chat_id = ? AND identity IN ({placeholders})
                """,
                (chat_id, *identities),
            ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    def committed_batch_count(self, chat_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) FROM archive_batches
                 WHERE chat_id = ? AND state = 'committed'
                """,
                (chat_id,),
            ).fetchone()
        return int(row[0])

    def latest_committed_batch(self, chat_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT batch_id FROM archive_batches
                 WHERE chat_id = ? AND state = 'committed'
                 ORDER BY committed_at DESC LIMIT 1
                """,
                (chat_id,),
            ).fetchone()
        return str(row[0]) if row is not None else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
