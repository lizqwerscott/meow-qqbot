"""Durable admission side-effect outbox.

The local conversation history is the admission commit point. This outbox
bridges that commit to best-effort external observers such as Hindsight and
learners, so a process restart can resume effects without duplicating them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

_log = logging.getLogger(__name__)


class AdmissionOutbox:
    """Small SQLite-backed queue keyed by message and side-effect kind."""

    def __init__(
        self,
        path: str = "data/admission_outbox.sqlite3",
        *,
        succeeded_retention_seconds: float = 7 * 24 * 3600,
        max_succeeded_rows: int = 10_000,
    ) -> None:
        self._path = Path(path)
        self._succeeded_retention_seconds = max(0.0, succeeded_retention_seconds)
        self._max_succeeded_rows = max(0, max_succeeded_rows)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._process_lock = asyncio.Lock()

    async def _ensure_open(self) -> sqlite3.Connection:
        async with self._lock:
            if self._conn is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(self._path, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS admission_outbox (
                        chat_id TEXT NOT NULL,
                        message_id TEXT NOT NULL,
                        effect_type TEXT NOT NULL,
                        admission_order INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        lease_id TEXT NOT NULL DEFAULT '',
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (chat_id, message_id, effect_type)
                    )
                    """)
                columns = {
                    row[1]
                    for row in self._conn.execute(
                        "PRAGMA table_info(admission_outbox)"
                    ).fetchall()
                }
                if "admission_order" not in columns:
                    self._conn.execute(
                        "ALTER TABLE admission_outbox ADD COLUMN "
                        "admission_order INTEGER NOT NULL DEFAULT 0"
                    )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_admission_outbox_ready "
                    "ON admission_outbox(status, updated_at)"
                )
                self._conn.commit()
            return self._conn

    async def prepare(
        self, chat_id: str, message_id: str, payload: dict[str, Any]
    ) -> bool:
        conn = await self._ensure_open()
        encoded = json.dumps(payload, ensure_ascii=False)
        now = time.time()
        created = False
        async with self._lock:
            row = conn.execute(
                "SELECT COALESCE(MAX(admission_order), 0) + 1 AS next_order "
                "FROM admission_outbox WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            admission_order = int(row["next_order"])
            for effect_type in ("hindsight", "learner"):
                cursor = conn.execute(
                    """
                    INSERT INTO admission_outbox
                        (chat_id, message_id, effect_type, admission_order,
                         status, payload, updated_at)
                    VALUES (?, ?, ?, ?, 'prepared', ?, ?)
                    ON CONFLICT(chat_id, message_id, effect_type) DO NOTHING
                    """,
                    (chat_id, message_id, effect_type, admission_order, encoded, now),
                )
                created = created or cursor.rowcount > 0
            conn.commit()
        return created

    async def mark_ready(self, chat_id: str, message_id: str) -> None:
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                """
                UPDATE admission_outbox
                   SET status = 'ready', lease_id = '', updated_at = ?
                 WHERE chat_id = ? AND message_id = ? AND status = 'prepared'
                """,
                (time.time(), chat_id, message_id),
            )
            conn.commit()

    async def cancel(self, chat_id: str, message_id: str) -> None:
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                "DELETE FROM admission_outbox WHERE chat_id = ? AND message_id = ?",
                (chat_id, message_id),
            )
            conn.commit()

    async def claim_ready(
        self, limit: int = 50, lease_seconds: int = 300
    ) -> list[dict[str, Any]]:
        conn = await self._ensure_open()
        now = time.time()
        lease_id = uuid.uuid4().hex
        async with self._lock:
            rows = conn.execute(
                """
                SELECT chat_id, message_id, effect_type, payload, attempts
                  FROM admission_outbox
                 WHERE status = 'ready'
                    OR (status = 'processing' AND updated_at < ?)
                 ORDER BY chat_id,
                          CASE WHEN admission_order = 0 THEN 0 ELSE 1 END,
                          CASE WHEN admission_order = 0 THEN updated_at END,
                          admission_order,
                          message_id,
                          effect_type
                 LIMIT ?
                """,
                (now, limit),
            ).fetchall()
            claimed: list[dict[str, Any]] = []
            for row in rows:
                lease_expires_at = now + lease_seconds if lease_seconds else now
                conn.execute(
                    """
                    UPDATE admission_outbox
                       SET status = 'processing', lease_id = ?,
                           attempts = attempts + 1, updated_at = ?
                     WHERE chat_id = ? AND message_id = ? AND effect_type = ?
                    """,
                    (
                        lease_id,
                        lease_expires_at,
                        row["chat_id"],
                        row["message_id"],
                        row["effect_type"],
                    ),
                )
                claimed.append(
                    {
                        "chat_id": row["chat_id"],
                        "message_id": row["message_id"],
                        "effect_type": row["effect_type"],
                        "payload": json.loads(row["payload"]),
                        "attempts": row["attempts"] + 1,
                        "lease_id": lease_id,
                    }
                )
            conn.commit()
            return claimed

    async def complete(
        self,
        item: dict[str, Any],
        *,
        succeeded: bool,
        permanent: bool = False,
    ) -> None:
        conn = await self._ensure_open()
        status = "succeeded" if succeeded or permanent else "ready"
        async with self._lock:
            if status == "succeeded":
                conn.execute(
                    """
                    UPDATE admission_outbox
                       SET status = 'succeeded', lease_id = '', updated_at = ?
                     WHERE chat_id = ? AND message_id = ? AND effect_type = ?
                       AND lease_id = ?
                    """,
                    (
                        time.time(),
                        item["chat_id"],
                        item["message_id"],
                        item["effect_type"],
                        item["lease_id"],
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE admission_outbox
                       SET status = 'ready', lease_id = '', updated_at = ?
                     WHERE chat_id = ? AND message_id = ? AND effect_type = ?
                       AND lease_id = ?
                    """,
                    (
                        time.time(),
                        item["chat_id"],
                        item["message_id"],
                        item["effect_type"],
                        item["lease_id"],
                    ),
                )
            conn.commit()

    async def recover_prepared(
        self, is_admitted: Callable[[str, str], Awaitable[bool | None]]
    ) -> int:
        """Promote crash-window rows whose local history commit succeeded.

        ``None`` means admission is currently in progress in this process;
        leave the prepared rows untouched until the admission finishes.
        """
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute("""
                SELECT DISTINCT chat_id, message_id
                  FROM admission_outbox
                 WHERE status = 'prepared'
                """).fetchall()
        recovered = 0
        for row in rows:
            admitted = await is_admitted(row["chat_id"], row["message_id"])
            if admitted is None:
                continue
            if admitted:
                await self.mark_ready(row["chat_id"], row["message_id"])
                recovered += 1
            else:
                await self.cancel(row["chat_id"], row["message_id"])
        return recovered

    async def pending_count(self) -> int:
        conn = await self._ensure_open()
        async with self._lock:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM admission_outbox "
                "WHERE status IN ('prepared', 'ready', 'processing')"
            ).fetchone()
            return int(row["count"])

    async def prune_succeeded(self) -> int:
        """Bound completed rows while retaining a recent duplicate guard."""
        conn = await self._ensure_open()
        cutoff = time.time() - self._succeeded_retention_seconds
        async with self._lock:
            removed = conn.execute(
                """
                DELETE FROM admission_outbox
                 WHERE status = 'succeeded' AND updated_at < ?
                """,
                (cutoff,),
            ).rowcount
            removed += conn.execute(
                """
                DELETE FROM admission_outbox
                 WHERE status = 'succeeded'
                   AND rowid NOT IN (
                       SELECT rowid
                         FROM admission_outbox
                        WHERE status = 'succeeded'
                        ORDER BY updated_at DESC, rowid DESC
                        LIMIT ?
                   )
                """,
                (self._max_succeeded_rows,),
            ).rowcount
            conn.commit()
            return removed

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    async def process(
        self,
        handlers: dict[str, Callable[[dict[str, Any]], Awaitable[bool]]],
        *,
        limit: int = 50,
    ) -> int:
        async with self._process_lock:
            processed = 0
            claimed = await self.claim_ready(limit=limit)
            try:
                for item in claimed:
                    handler = handlers.get(item["effect_type"])
                    if handler is None:
                        await self.complete(item, succeeded=True, permanent=True)
                        continue
                    try:
                        payload = dict(item["payload"])
                        payload["idempotency_key"] = ":".join(
                            (
                                "admission",
                                item["chat_id"],
                                item["message_id"],
                                item["effect_type"],
                            )
                        )
                        succeeded = bool(await handler(payload))
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        _log.exception(
                            "admission side effect failed: %s/%s/%s",
                            item["chat_id"],
                            item["message_id"],
                            item["effect_type"],
                        )
                        succeeded = False
                    await self.complete(item, succeeded=succeeded)
                    processed += 1
                await self.prune_succeeded()
                return processed
            except asyncio.CancelledError:
                for item in claimed:
                    try:
                        await self.complete(item, succeeded=False)
                    except Exception:
                        _log.exception(
                            "释放取消的 admission lease 失败: %s/%s/%s",
                            item["chat_id"],
                            item["message_id"],
                            item["effect_type"],
                        )
                raise
