"""Durable archive membership and batch metadata derived from the event ledger."""

import asyncio
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ArchiveTurnRecord:
    turn_id: str
    turn_sequence: int
    source_date: str
    event_count: int
    estimated_tokens: int


@dataclass(frozen=True)
class ArchiveBatch:
    batch_id: str
    operation_id: str
    chat_id: str
    state: str
    captured_cutoff_seq: int
    source_hash: str
    turn_count: int
    event_count: int
    source_dates: tuple[str, ...]
    created_at: float
    committed_at: float = 0.0


class ArchiveIndex:
    """Small metadata projection for logical cold partitions."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str = "data/archive_index.sqlite3") -> None:
        self._path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def _ensure_open(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        async with self._lock:
            if self._conn is not None:
                return self._conn
            if self._path != ":memory:":
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS archive_index_schema (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS archive_batches (
                    batch_id TEXT PRIMARY KEY,
                    operation_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    captured_cutoff_seq INTEGER NOT NULL,
                    source_hash TEXT NOT NULL,
                    turn_count INTEGER NOT NULL,
                    event_count INTEGER NOT NULL,
                    source_dates TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    committed_at REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS archive_batch_turns (
                    batch_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    turn_sequence INTEGER NOT NULL,
                    source_date TEXT NOT NULL,
                    event_count INTEGER NOT NULL,
                    estimated_tokens INTEGER NOT NULL,
                    PRIMARY KEY (batch_id, turn_id),
                    FOREIGN KEY (batch_id) REFERENCES archive_batches(batch_id)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS archive_batch_events (
                    batch_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    PRIMARY KEY (batch_id, event_id),
                    UNIQUE (chat_id, event_id),
                    FOREIGN KEY (batch_id) REFERENCES archive_batches(batch_id)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS archive_exports (
                    batch_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    path TEXT NOT NULL DEFAULT '',
                    content_hash TEXT NOT NULL DEFAULT '',
                    manifest_hash TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (batch_id) REFERENCES archive_batches(batch_id)
                        ON DELETE CASCADE
                );
                """)
            row = self._conn.execute(
                "SELECT version FROM archive_index_schema LIMIT 1"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO archive_index_schema(version) VALUES (?)",
                    (self.SCHEMA_VERSION,),
                )
            elif int(row["version"]) != self.SCHEMA_VERSION:
                raise RuntimeError("unsupported archive index schema version")
            self._conn.commit()
        return self._conn

    @staticmethod
    def membership_hash(event_ids: Iterable[str]) -> str:
        return hashlib.sha256(
            json.dumps(sorted(set(event_ids)), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ArchiveBatch:
        return ArchiveBatch(
            batch_id=row["batch_id"],
            operation_id=row["operation_id"],
            chat_id=row["chat_id"],
            state=row["state"],
            captured_cutoff_seq=int(row["captured_cutoff_seq"]),
            source_hash=row["source_hash"],
            turn_count=int(row["turn_count"]),
            event_count=int(row["event_count"]),
            source_dates=tuple(json.loads(row["source_dates"] or "[]")),
            created_at=float(row["created_at"]),
            committed_at=float(row["committed_at"]),
        )

    async def prepare_batch(
        self,
        *,
        batch_id: str,
        operation_id: str,
        chat_id: str,
        captured_cutoff_seq: int,
        turn_records: Sequence[ArchiveTurnRecord],
        event_ids: Sequence[tuple[str, str]],
    ) -> ArchiveBatch:
        if not batch_id or not operation_id or not chat_id:
            raise ValueError("batch identity is required")
        normalized_events = tuple(dict.fromkeys(event_ids))
        source_hash = self.membership_hash(
            event_id for event_id, _ in normalized_events
        )
        dates = tuple(sorted({record.source_date for record in turn_records}))
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT * FROM archive_batches WHERE batch_id = ?", (batch_id,)
                ).fetchone()
                if existing is not None:
                    batch = self._from_row(existing)
                    if (
                        batch.chat_id != chat_id
                        or batch.source_hash != source_hash
                        or batch.operation_id != operation_id
                    ):
                        raise RuntimeError("archive batch identity collision")
                    conn.commit()
                    return batch
                now = time.time()
                conn.execute(
                    """
                    INSERT INTO archive_batches
                        (batch_id, operation_id, chat_id, state,
                         captured_cutoff_seq, source_hash, turn_count, event_count,
                         source_dates, created_at)
                    VALUES (?, ?, ?, 'prepared', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        operation_id,
                        chat_id,
                        int(captured_cutoff_seq),
                        source_hash,
                        len(turn_records),
                        len(normalized_events),
                        json.dumps(dates, ensure_ascii=False),
                        now,
                    ),
                )
                conn.executemany(
                    """
                    INSERT INTO archive_batch_turns
                        (batch_id, turn_id, turn_sequence, source_date,
                         event_count, estimated_tokens)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            batch_id,
                            record.turn_id,
                            record.turn_sequence,
                            record.source_date,
                            record.event_count,
                            record.estimated_tokens,
                        )
                        for record in turn_records
                    ],
                )
                conn.executemany(
                    "INSERT INTO archive_batch_events(batch_id, chat_id, event_id, turn_id) VALUES (?, ?, ?, ?)",
                    [
                        (batch_id, chat_id, event_id, turn_id)
                        for event_id, turn_id in normalized_events
                    ],
                )
                conn.commit()
                return ArchiveBatch(
                    batch_id=batch_id,
                    operation_id=operation_id,
                    chat_id=chat_id,
                    state="prepared",
                    captured_cutoff_seq=int(captured_cutoff_seq),
                    source_hash=source_hash,
                    turn_count=len(turn_records),
                    event_count=len(normalized_events),
                    source_dates=dates,
                    created_at=now,
                )
            except BaseException:
                conn.rollback()
                raise

    async def mark_state(self, batch_id: str, state: str) -> ArchiveBatch:
        if state not in {"prepared", "committed", "export_degraded", "soft_deleted"}:
            raise ValueError("invalid archive batch state")
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM archive_batches WHERE batch_id = ?", (batch_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(batch_id)
                current = self._from_row(row)
                if current.state == "soft_deleted" and state != "soft_deleted":
                    conn.commit()
                    return current
                if current.state == "export_degraded" and state == "prepared":
                    conn.commit()
                    return current
                if current.state == "committed" and state == "prepared":
                    conn.commit()
                    return current
                committed_at = current.committed_at
                if state == "committed" and not committed_at:
                    committed_at = time.time()
                conn.execute(
                    "UPDATE archive_batches SET state = ?, committed_at = ? WHERE batch_id = ?",
                    (state, committed_at, batch_id),
                )
                conn.commit()
                return ArchiveBatch(
                    **{
                        **current.__dict__,
                        "state": state,
                        "committed_at": committed_at,
                    }
                )
            except BaseException:
                conn.rollback()
                raise

    async def get(self, batch_id: str) -> ArchiveBatch | None:
        conn = await self._ensure_open()
        async with self._lock:
            row = conn.execute(
                "SELECT * FROM archive_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    async def list_for_webui(self, chat_id: str) -> list[dict[str, Any]]:
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                "SELECT * FROM archive_batches WHERE chat_id = ? ORDER BY created_at DESC",
                (chat_id,),
            ).fetchall()
            export_rows = {
                str(row["batch_id"]): dict(row)
                for row in conn.execute(
                    "SELECT * FROM archive_exports WHERE batch_id IN "
                    "(SELECT batch_id FROM archive_batches WHERE chat_id = ?)",
                    (chat_id,),
                ).fetchall()
            }
        result = []
        for row in rows:
            item = dict(self._from_row(row).__dict__)
            export = export_rows.get(str(row["batch_id"]))
            item.update(
                {
                    "export_status": export["status"] if export else "disabled",
                    "export_path": export["path"] if export else "",
                    "export_content_hash": export["content_hash"] if export else "",
                    "export_manifest_hash": export["manifest_hash"] if export else "",
                    "export_error": export["error"] if export else "",
                }
            )
            result.append(item)
        return result

    async def record_export(
        self,
        batch_id: str,
        *,
        status: str,
        path: str = "",
        content_hash: str = "",
        manifest_hash: str = "",
        error: str = "",
    ) -> None:
        if status not in {"disabled", "exported", "failed", "deferred"}:
            raise ValueError("invalid archive export status")
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                """
                INSERT INTO archive_exports
                    (batch_id, status, path, content_hash, manifest_hash, error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    status = excluded.status,
                    path = excluded.path,
                    content_hash = excluded.content_hash,
                    manifest_hash = excluded.manifest_hash,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    batch_id,
                    status,
                    path,
                    content_hash,
                    manifest_hash,
                    error[:1000],
                    time.time(),
                ),
            )
            conn.commit()

    async def turns_for_batch(self, batch_id: str) -> list[ArchiveTurnRecord]:
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                """
                SELECT turn_id, turn_sequence, source_date, event_count,
                       estimated_tokens
                  FROM archive_batch_turns
                 WHERE batch_id = ?
                 ORDER BY turn_sequence
                """,
                (batch_id,),
            ).fetchall()
        return [
            ArchiveTurnRecord(
                turn_id=str(row["turn_id"]),
                turn_sequence=int(row["turn_sequence"]),
                source_date=str(row["source_date"]),
                event_count=int(row["event_count"]),
                estimated_tokens=int(row["estimated_tokens"]),
            )
            for row in rows
        ]

    async def list_pending(self, chat_id: str | None = None) -> list[ArchiveBatch]:
        conn = await self._ensure_open()
        async with self._lock:
            if chat_id is None:
                rows = conn.execute(
                    "SELECT * FROM archive_batches WHERE state = 'prepared' ORDER BY created_at"
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM archive_batches
                     WHERE state = 'prepared' AND chat_id = ?
                     ORDER BY created_at
                    """,
                    (chat_id,),
                ).fetchall()
        return [self._from_row(row) for row in rows]

    async def chat_ids(self) -> list[str]:
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                "SELECT DISTINCT chat_id FROM archive_batches ORDER BY chat_id"
            ).fetchall()
        return [str(row["chat_id"]) for row in rows]

    async def event_ids(self, batch_id: str) -> frozenset[str]:
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                "SELECT event_id FROM archive_batch_events WHERE batch_id = ?",
                (batch_id,),
            ).fetchall()
        return frozenset(str(row["event_id"]) for row in rows)

    async def committed_event_ids(self, chat_id: str) -> frozenset[str]:
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                """
                SELECT event_id FROM archive_batch_events e
                JOIN archive_batches b ON b.batch_id = e.batch_id
                WHERE b.chat_id = ? AND b.committed_at > 0
                """,
                (chat_id,),
            ).fetchall()
        return frozenset(str(row["event_id"]) for row in rows)

    async def clear_chat(self, chat_id: str) -> None:
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                """
                DELETE FROM archive_batch_events
                 WHERE chat_id = ?
                    OR batch_id IN (
                        SELECT batch_id FROM archive_batches WHERE chat_id = ?
                    )
                """,
                (chat_id, chat_id),
            )
            conn.execute(
                """
                DELETE FROM archive_batch_turns
                 WHERE batch_id IN (
                    SELECT batch_id FROM archive_batches WHERE chat_id = ?
                 )
                """,
                (chat_id,),
            )
            conn.execute("DELETE FROM archive_batches WHERE chat_id = ?", (chat_id,))
            conn.commit()

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
