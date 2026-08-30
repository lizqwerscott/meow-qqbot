"""Durable storage for runtime setting revisions and target metadata."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
DEFAULT_AUDIT_RETENTION = 2000


def _load_object(raw: str, label: str) -> dict[str, Any]:
    def collect(items: list[tuple[str, Any]]):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeSettingsError(f"duplicate key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=collect)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeSettingsError(f"{label} is invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeSettingsError(f"{label} must be an object")
    return value


class RuntimeSettingsError(RuntimeError):
    """Raised when the runtime settings database cannot be used safely."""


class RuntimeSettingsConflict(RuntimeError):
    """Raised when an update was based on a stale revision."""

    def __init__(self, domain: str, expected: int, actual: int):
        super().__init__(
            f"{domain} revision conflict: expected {expected}, got {actual}"
        )
        self.domain = domain
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class RuntimeSettingsRecord:
    domain: str
    revision: int
    overrides: dict[str, Any]
    updated_at: float
    source: str
    schema_version: int = SCHEMA_VERSION
    status: str = "applied"


@dataclass(frozen=True)
class AuditRecord:
    id: int
    domain: str
    previous_revision: int
    new_revision: int
    change: dict[str, Any]
    source: str
    remote_ip: str | None
    created_at: float
    outcome: str = "success"
    failure_class: str | None = None


@dataclass(frozen=True)
class EngagementTarget:
    chat_id: str
    verification_status: str
    first_observed_at: float | None
    last_observed_at: float | None
    verified_at: float | None
    created_at: float
    updated_at: float


class RuntimeSettingsStore:
    """Own SQLite transactions for runtime settings and target metadata."""

    def __init__(
        self,
        path: str = "data/runtime_settings.sqlite3",
        *,
        audit_retention: int = DEFAULT_AUDIT_RETENTION,
    ):
        if audit_retention < 1:
            raise ValueError("audit_retention must be positive")
        self._path = path
        self._audit_retention = int(audit_retention)
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
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS runtime_settings_schema "
                    "(version INTEGER NOT NULL)"
                )
                row = conn.execute(
                    "SELECT version FROM runtime_settings_schema LIMIT 1"
                ).fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO runtime_settings_schema(version) VALUES (?)",
                        (SCHEMA_VERSION,),
                    )
                elif int(row["version"]) > SCHEMA_VERSION:
                    raise RuntimeSettingsError(
                        f"unsupported runtime settings schema: {row['version']}"
                    )
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS runtime_settings (
                        domain TEXT PRIMARY KEY,
                        revision INTEGER NOT NULL,
                        override_json TEXT NOT NULL,
                        updated_at REAL NOT NULL,
                        source TEXT NOT NULL,
                        schema_version INTEGER NOT NULL,
                        status TEXT NOT NULL DEFAULT 'applied'
                    )
                    """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS runtime_settings_audit (
                        id INTEGER PRIMARY KEY,
                        domain TEXT NOT NULL,
                        previous_revision INTEGER NOT NULL,
                        new_revision INTEGER NOT NULL,
                        change_json TEXT NOT NULL,
                        source TEXT NOT NULL,
                        remote_ip TEXT,
                        created_at REAL NOT NULL,
                        outcome TEXT NOT NULL DEFAULT 'success',
                        failure_class TEXT
                    )
                    """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS engagement_targets (
                        chat_id TEXT PRIMARY KEY,
                        verification_status TEXT NOT NULL,
                        first_observed_at REAL,
                        last_observed_at REAL,
                        verified_at REAL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """)
                audit_columns = {
                    row["name"]
                    for row in conn.execute(
                        "PRAGMA table_info(runtime_settings_audit)"
                    ).fetchall()
                }
                if "outcome" not in audit_columns:
                    conn.execute(
                        "ALTER TABLE runtime_settings_audit ADD COLUMN "
                        "outcome TEXT NOT NULL DEFAULT 'success'"
                    )
                if "failure_class" not in audit_columns:
                    conn.execute(
                        "ALTER TABLE runtime_settings_audit ADD COLUMN "
                        "failure_class TEXT"
                    )
                if row is not None and int(row["version"]) < SCHEMA_VERSION:
                    conn.execute(
                        "UPDATE runtime_settings_schema SET version = ?",
                        (SCHEMA_VERSION,),
                    )
                    conn.execute(
                        "UPDATE runtime_settings SET schema_version = ?",
                        (SCHEMA_VERSION,),
                    )
                conn.commit()
            except Exception:
                conn.close()
                raise
            self._conn = conn
        return self._conn

    @staticmethod
    def _record(row: sqlite3.Row | None, domain: str) -> RuntimeSettingsRecord:
        if row is None:
            return RuntimeSettingsRecord(domain, 0, {}, 0.0, "bootstrap")
        value = _load_object(row["override_json"], "runtime settings override")
        return RuntimeSettingsRecord(
            domain=domain,
            revision=int(row["revision"]),
            overrides=value,
            updated_at=float(row["updated_at"]),
            source=str(row["source"]),
            schema_version=int(row["schema_version"]),
            status=str(row["status"]),
        )

    async def get(self, domain: str) -> RuntimeSettingsRecord:
        if not domain:
            raise ValueError("domain is required")
        conn = await self._ensure_open()
        async with self._lock:
            row = conn.execute(
                "SELECT * FROM runtime_settings WHERE domain = ?", (domain,)
            ).fetchone()
        return self._record(row, domain)

    async def commit(
        self,
        domain: str,
        *,
        expected_revision: int,
        overrides: dict[str, Any],
        source: str,
        change: dict[str, Any],
        remote_ip: str | None = None,
        now: float | None = None,
    ) -> RuntimeSettingsRecord:
        conn = await self._ensure_open()
        timestamp = time.time() if now is None else float(now)
        override_json = json.dumps(
            overrides,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        change_json = json.dumps(
            change,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        async with self._lock:
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT revision FROM runtime_settings WHERE domain = ?",
                    (domain,),
                ).fetchone()
                current_revision = int(row["revision"]) if row else 0
                if current_revision != expected_revision:
                    conn.rollback()
                    raise RuntimeSettingsConflict(
                        domain, expected_revision, current_revision
                    )
                new_revision = expected_revision + 1
                conn.execute(
                    """
                    INSERT INTO runtime_settings
                        (domain, revision, override_json, updated_at, source,
                         schema_version, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'applied')
                    ON CONFLICT(domain) DO UPDATE SET
                        revision = excluded.revision,
                        override_json = excluded.override_json,
                        updated_at = excluded.updated_at,
                        source = excluded.source,
                        schema_version = excluded.schema_version,
                        status = excluded.status
                    """,
                    (
                        domain,
                        new_revision,
                        override_json,
                        timestamp,
                        source,
                        SCHEMA_VERSION,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO runtime_settings_audit
                        (domain, previous_revision, new_revision, change_json,
                         source, remote_ip, created_at, outcome, failure_class)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'success', NULL)
                    """,
                    (
                        domain,
                        expected_revision,
                        new_revision,
                        change_json,
                        source,
                        remote_ip,
                        timestamp,
                    ),
                )
                self._prune_audit(conn, domain)
                conn.commit()
            except RuntimeSettingsConflict:
                raise
            except Exception:
                conn.rollback()
                raise
            row = conn.execute(
                "SELECT * FROM runtime_settings WHERE domain = ?", (domain,)
            ).fetchone()
        return self._record(row, domain)

    def _prune_audit(self, conn: sqlite3.Connection, domain: str) -> None:
        conn.execute(
            """
            DELETE FROM runtime_settings_audit
            WHERE domain = ? AND id NOT IN (
                SELECT id FROM runtime_settings_audit
                WHERE domain = ? ORDER BY id DESC LIMIT ?
            )
            """,
            (domain, domain, self._audit_retention),
        )

    async def append_audit(
        self,
        domain: str,
        *,
        previous_revision: int,
        new_revision: int,
        change: dict[str, Any],
        source: str,
        remote_ip: str | None = None,
        outcome: str = "success",
        failure_class: str | None = None,
        now: float | None = None,
    ) -> AuditRecord:
        if outcome not in {"success", "failure"}:
            raise ValueError("invalid audit outcome")
        if outcome == "success" and failure_class is not None:
            raise ValueError("success audit cannot have a failure class")
        if outcome == "failure" and not failure_class:
            raise ValueError("failure audit requires a failure class")
        timestamp = time.time() if now is None else float(now)
        change_json = json.dumps(
            change,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        conn = await self._ensure_open()
        async with self._lock:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO runtime_settings_audit
                        (domain, previous_revision, new_revision, change_json,
                         source, remote_ip, created_at, outcome, failure_class)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        domain,
                        int(previous_revision),
                        int(new_revision),
                        change_json,
                        source,
                        remote_ip,
                        timestamp,
                        outcome,
                        failure_class,
                    ),
                )
                self._prune_audit(conn, domain)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            row = conn.execute(
                "SELECT * FROM runtime_settings_audit WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return self._audit_record(row)

    @staticmethod
    def _audit_record(row: sqlite3.Row) -> AuditRecord:
        change = _load_object(row["change_json"], "runtime settings audit")
        return AuditRecord(
            id=int(row["id"]),
            domain=str(row["domain"]),
            previous_revision=int(row["previous_revision"]),
            new_revision=int(row["new_revision"]),
            change=change,
            source=str(row["source"]),
            remote_ip=row["remote_ip"],
            created_at=float(row["created_at"]),
            outcome=str(row["outcome"]),
            failure_class=row["failure_class"],
        )

    async def list_audit(
        self,
        domain: str,
        *,
        limit: int = 50,
        before_id: int | None = None,
    ) -> list[AuditRecord]:
        conn = await self._ensure_open()
        bounded_limit = max(1, min(int(limit), 200))
        async with self._lock:
            if before_id is None:
                rows = conn.execute(
                    """
                    SELECT * FROM runtime_settings_audit
                    WHERE domain = ? ORDER BY id DESC LIMIT ?
                    """,
                    (domain, bounded_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM runtime_settings_audit
                    WHERE domain = ? AND id < ? ORDER BY id DESC LIMIT ?
                    """,
                    (domain, int(before_id), bounded_limit),
                ).fetchall()
        return [self._audit_record(row) for row in rows]

    async def ensure_target(
        self, chat_id: str, *, now: float | None = None
    ) -> EngagementTarget:
        timestamp = time.time() if now is None else float(now)
        if not chat_id:
            raise ValueError("chat_id is required")
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                """
                INSERT INTO engagement_targets
                    (chat_id, verification_status, created_at, updated_at)
                VALUES (?, 'unverified', ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (chat_id, timestamp, timestamp),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM engagement_targets WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        return self._target(row)

    @staticmethod
    def _target(row: sqlite3.Row) -> EngagementTarget:
        return EngagementTarget(
            chat_id=str(row["chat_id"]),
            verification_status=str(row["verification_status"]),
            first_observed_at=(
                float(row["first_observed_at"])
                if row["first_observed_at"] is not None
                else None
            ),
            last_observed_at=(
                float(row["last_observed_at"])
                if row["last_observed_at"] is not None
                else None
            ),
            verified_at=(
                float(row["verified_at"]) if row["verified_at"] is not None else None
            ),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    async def get_target(self, chat_id: str) -> EngagementTarget | None:
        conn = await self._ensure_open()
        async with self._lock:
            row = conn.execute(
                "SELECT * FROM engagement_targets WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        return self._target(row) if row else None

    async def set_target_status(
        self, chat_id: str, status: str, *, now: float | None = None
    ) -> EngagementTarget:
        if status not in {"unverified", "verified", "removed"}:
            raise ValueError("invalid target status")
        target = await self.ensure_target(chat_id, now=now)
        timestamp = time.time() if now is None else float(now)
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                """
                UPDATE engagement_targets
                SET verification_status = ?, verified_at = ?, updated_at = ?
                WHERE chat_id = ?
                """,
                (
                    status,
                    timestamp if status == "verified" else None,
                    timestamp,
                    chat_id,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM engagement_targets WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        return self._target(row)

    async def mark_observed(
        self, chat_id: str, *, at: float | None = None
    ) -> EngagementTarget:
        timestamp = time.time() if at is None else float(at)
        target = await self.ensure_target(chat_id, now=timestamp)
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                """
                UPDATE engagement_targets
                SET first_observed_at = COALESCE(first_observed_at, ?),
                    last_observed_at = ?, updated_at = ?
                WHERE chat_id = ?
                """,
                (timestamp, timestamp, timestamp, chat_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM engagement_targets WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        return self._target(row)

    async def list_targets(self, *, limit: int = 100) -> list[EngagementTarget]:
        conn = await self._ensure_open()
        bounded_limit = max(1, min(int(limit), 200))
        async with self._lock:
            rows = conn.execute(
                "SELECT * FROM engagement_targets ORDER BY chat_id LIMIT ?",
                (bounded_limit,),
            ).fetchall()
        return [self._target(row) for row in rows]

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
