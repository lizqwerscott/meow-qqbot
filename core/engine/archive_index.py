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
    turn_kind: str = "unknown"


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

    SCHEMA_VERSION = 2

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
                    turn_kind TEXT NOT NULL DEFAULT 'unknown',
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
            columns = {
                str(row["name"])
                for row in self._conn.execute(
                    "PRAGMA table_info(archive_batch_turns)"
                ).fetchall()
            }
            if "turn_kind" not in columns:
                self._conn.execute(
                    "ALTER TABLE archive_batch_turns ADD COLUMN turn_kind "
                    "TEXT NOT NULL DEFAULT 'unknown'"
                )
            row = self._conn.execute(
                "SELECT version FROM archive_index_schema LIMIT 1"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO archive_index_schema(version) VALUES (?)",
                    (self.SCHEMA_VERSION,),
                )
            elif int(row["version"]) == 1:
                self._conn.execute(
                    "UPDATE archive_index_schema SET version = ?",
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
                         event_count, estimated_tokens, turn_kind)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            batch_id,
                            record.turn_id,
                            record.turn_sequence,
                            record.source_date,
                            record.event_count,
                            record.estimated_tokens,
                            record.turn_kind,
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

    async def list_for_webui(
        self,
        chat_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        conn = await self._ensure_open()
        async with self._lock:
            params: tuple[Any, ...] = (chat_id,)
            state_clause = ""
            if state is not None:
                state_clause = " AND state = ?"
                params += (state,)
            limit_clause = ""
            if limit is not None:
                limit_clause = " LIMIT ? OFFSET ?"
                params += (max(1, int(limit)), max(0, int(offset)))
            rows = conn.execute(
                "SELECT * FROM archive_batches WHERE chat_id = ?"
                + state_clause
                + " ORDER BY created_at DESC"
                + limit_clause,
                params,
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

    async def count_for_webui(self, chat_id: str, *, state: str | None = None) -> int:
        conn = await self._ensure_open()
        async with self._lock:
            params: tuple[Any, ...] = (chat_id,)
            state_clause = ""
            if state is not None:
                state_clause = " AND state = ?"
                params += (state,)
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM archive_batches WHERE chat_id = ?"
                    + state_clause,
                    params,
                ).fetchone()[0]
            )

    async def chat_summaries_for_webui(
        self,
        *,
        query: str = "",
        limit: int | None = None,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return committed archive session summaries without loading batches."""
        conn = await self._ensure_open()
        async with self._lock:
            params: tuple[Any, ...] = ()
            where = " WHERE state = 'committed'"
            if query:
                where += " AND chat_id LIKE ?"
                params += (f"%{query}%",)
            total = int(
                conn.execute(
                    "SELECT COUNT(DISTINCT chat_id) FROM archive_batches" + where,
                    params,
                ).fetchone()[0]
            )
            paging = ""
            if limit is not None:
                paging = " LIMIT ? OFFSET ?"
                params += (max(1, int(limit)), max(0, int(offset)))
            rows = conn.execute(
                """
                SELECT chat_id, COUNT(*) AS archive_count,
                       MAX(committed_at) AS latest_archive,
                       COALESCE(SUM(event_count), 0) AS total_size
                  FROM archive_batches
                """
                + where
                + " GROUP BY chat_id ORDER BY latest_archive DESC, chat_id"
                + paging,
                params,
            ).fetchall()
        return (
            [
                {
                    "chat_id": str(row["chat_id"]),
                    "archive_count": int(row["archive_count"]),
                    "latest_archive": (
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(row["latest_archive"])
                        )
                        if row["latest_archive"]
                        else "-"
                    ),
                    "total_size": int(row["total_size"]),
                    "ledger_archive": True,
                }
                for row in rows
            ],
            total,
        )

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
        records, _ = await self.turns_for_batch_page(batch_id)
        return records

    async def count_turns_for_batch(self, batch_id: str) -> int:
        conn = await self._ensure_open()
        async with self._lock:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM archive_batch_turns WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()[0]
            )

    async def turns_for_batch_page(
        self, batch_id: str, *, limit: int | None = None, offset: int = 0
    ) -> tuple[list[ArchiveTurnRecord], int]:
        conn = await self._ensure_open()
        async with self._lock:
            params: tuple[Any, ...] = (batch_id,)
            paging = ""
            if limit is not None:
                paging = " LIMIT ? OFFSET ?"
                params += (max(1, int(limit)), max(0, int(offset)))
            total = int(
                conn.execute(
                    "SELECT COUNT(*) FROM archive_batch_turns WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()[0]
            )
            rows = conn.execute(
                """
                SELECT turn_id, turn_sequence, source_date, event_count,
                       estimated_tokens, turn_kind
                  FROM archive_batch_turns
                 WHERE batch_id = ?
                 ORDER BY turn_sequence DESC
                """ + paging,
                params,
            ).fetchall()
        return (
            [
                ArchiveTurnRecord(
                    turn_id=str(row["turn_id"]),
                    turn_sequence=int(row["turn_sequence"]),
                    source_date=str(row["source_date"]),
                    event_count=int(row["event_count"]),
                    estimated_tokens=int(row["estimated_tokens"]),
                    turn_kind=str(row["turn_kind"] or "unknown"),
                )
                for row in rows
            ],
            total,
        )

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

    async def status(self) -> dict[str, int]:
        """Return archive and export counters for rollout monitoring."""
        conn = await self._ensure_open()
        async with self._lock:
            row = conn.execute(
                """
                SELECT COUNT(*) AS batch_count,
                       COUNT(DISTINCT chat_id) AS chat_count,
                       COALESCE(SUM(state = 'prepared'), 0) AS pending_count,
                       COALESCE(SUM(state IN ('committed', 'export_degraded', 'soft_deleted')), 0) AS committed_count,
                       COALESCE(SUM(event_count), 0) AS event_count
                  FROM archive_batches
                """
            ).fetchone()
            export_row = conn.execute(
                """
                SELECT COALESCE(SUM(status = 'exported'), 0) AS exported_count,
                       COALESCE(SUM(status = 'failed'), 0) AS failed_count
                  FROM archive_exports
                """
            ).fetchone()
        return {
            "chat_count": int(row["chat_count"]),
            "batch_count": int(row["batch_count"]),
            "pending_count": int(row["pending_count"]),
            "committed_count": int(row["committed_count"]),
            "event_count": int(row["event_count"]),
            "exported_count": int(export_row["exported_count"]),
            "export_failed_count": int(export_row["failed_count"]),
        }

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
