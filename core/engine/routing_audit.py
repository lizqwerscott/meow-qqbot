"""Persistent, content-free audit records for deterministic mode routing."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RoutingAuditRecord:
    id: str
    created_at: float
    chat_id: str
    message_id: str
    source: str
    intent: str
    mode: str
    reason_code: str
    reason: str
    capability_profile: str
    policy_version: str
    scheduler_revision: int
    work_plan_hint: str | None
    trace: tuple[str, ...]


class RoutingAuditStore:
    """SQLite audit trail with bounded retention and no conversation content."""

    def __init__(
        self,
        path: str = "data/orchestration.sqlite",
        *,
        retention_seconds: float = 30 * 24 * 60 * 60,
        max_rows: int = 50_000,
    ) -> None:
        self.path = str(path)
        self.retention_seconds = max(0.0, retention_seconds)
        self.max_rows = max(1, max_rows)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def _open(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        async with self._lock:
            if self._conn is not None:
                return self._conn
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS routing_audit (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    chat_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    capability_profile TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    scheduler_revision INTEGER NOT NULL,
                    work_plan_hint TEXT,
                    trace_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_routing_audit_created
                    ON routing_audit(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_routing_audit_filters
                    ON routing_audit(mode, reason_code, source, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_routing_audit_chat
                    ON routing_audit(chat_id, created_at DESC);
                """)
            self._conn.commit()
        return self._conn

    @staticmethod
    def _record(row: sqlite3.Row) -> RoutingAuditRecord:
        try:
            trace = tuple(str(item) for item in json.loads(row["trace_json"]))
        except (TypeError, ValueError):
            trace = ()
        return RoutingAuditRecord(
            id=str(row["id"]),
            created_at=float(row["created_at"]),
            chat_id=str(row["chat_id"]),
            message_id=str(row["message_id"]),
            source=str(row["source"]),
            intent=str(row["intent"]),
            mode=str(row["mode"]),
            reason_code=str(row["reason_code"]),
            reason=str(row["reason"]),
            capability_profile=str(row["capability_profile"]),
            policy_version=str(row["policy_version"]),
            scheduler_revision=int(row["scheduler_revision"]),
            work_plan_hint=(
                str(row["work_plan_hint"]) if row["work_plan_hint"] else None
            ),
            trace=trace,
        )

    async def append(
        self,
        *,
        chat_id: str,
        message_id: str,
        source: str,
        intent: str,
        mode: str,
        reason_code: str,
        reason: str,
        capability_profile: str,
        policy_version: str,
        scheduler_revision: int,
        work_plan_hint: str | None,
        trace: tuple[str, ...],
    ) -> RoutingAuditRecord:
        if not chat_id or not message_id:
            raise ValueError("chat_id and message_id are required")
        record = RoutingAuditRecord(
            id=uuid.uuid4().hex,
            created_at=time.time(),
            chat_id=chat_id,
            message_id=message_id,
            source=source,
            intent=intent,
            mode=mode,
            reason_code=reason_code,
            reason=reason,
            capability_profile=capability_profile,
            policy_version=policy_version,
            scheduler_revision=max(0, int(scheduler_revision)),
            work_plan_hint=work_plan_hint or None,
            trace=tuple(trace),
        )
        conn = await self._open()
        async with self._lock:
            conn.execute(
                """
                INSERT INTO routing_audit (
                    id, created_at, chat_id, message_id, source, intent, mode,
                    reason_code, reason, capability_profile, policy_version,
                    scheduler_revision, work_plan_hint, trace_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.created_at,
                    record.chat_id,
                    record.message_id,
                    record.source,
                    record.intent,
                    record.mode,
                    record.reason_code,
                    record.reason,
                    record.capability_profile,
                    record.policy_version,
                    record.scheduler_revision,
                    record.work_plan_hint,
                    json.dumps(record.trace, ensure_ascii=True),
                ),
            )
            self._prune(conn, record.created_at)
            conn.commit()
        return record

    def _prune(self, conn: sqlite3.Connection, now: float) -> None:
        if self.retention_seconds:
            conn.execute(
                "DELETE FROM routing_audit WHERE created_at < ?",
                (now - self.retention_seconds,),
            )
        conn.execute(
            """
            DELETE FROM routing_audit
            WHERE id IN (
                SELECT id FROM routing_audit
                ORDER BY created_at DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (self.max_rows,),
        )

    async def list_records(
        self,
        *,
        mode: str | None = None,
        reason_code: str | None = None,
        source: str | None = None,
        chat_prefix: str | None = None,
        limit: int = 100,
    ) -> list[RoutingAuditRecord]:
        conn = await self._open()
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("mode", mode),
            ("reason_code", reason_code),
            ("source", source),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if chat_prefix:
            clauses.append("chat_id LIKE ?")
            params.append(f"{chat_prefix}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 500)))
        async with self._lock:
            rows = conn.execute(
                f"SELECT * FROM routing_audit {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._record(row) for row in rows]

    async def get(self, record_id: str) -> RoutingAuditRecord | None:
        conn = await self._open()
        async with self._lock:
            row = conn.execute(
                "SELECT * FROM routing_audit WHERE id = ?", (record_id,)
            ).fetchone()
        return self._record(row) if row is not None else None

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
