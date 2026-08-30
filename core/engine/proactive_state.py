"""Durable state for opt-in proactive group scheduling.

Only proactive scheduling policy state and aggregate hourly metrics are stored here. Conversation contents, model protocol, and delivery records remain owned by their existing stores.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProactiveState:
    chat_id: str
    proactive_window_started_at: float = 0.0
    proactive_turns_in_window: int = 0
    proactive_cooldown_until: float = 0.0
    next_due_at: float = 0.0


class ProactiveStateStore:
    """Small SQLite store for restart-safe proactive scheduling state."""

    def __init__(self, path: str = "data/proactive_state.sqlite3"):
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
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS proactive_state (
                    chat_id TEXT PRIMARY KEY,
                    proactive_window_started_at REAL NOT NULL DEFAULT 0,
                    proactive_turns_in_window INTEGER NOT NULL DEFAULT 0,
                    proactive_cooldown_until REAL NOT NULL DEFAULT 0,
                    next_due_at REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS proactive_metrics (
                    namespace TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    bucket_start INTEGER NOT NULL,
                    count INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (namespace, metric, bucket_start)
                )
                """
            )
            self._conn.commit()
        return self._conn

    @staticmethod
    def _state(row: sqlite3.Row | None, chat_id: str) -> ProactiveState:
        if row is None:
            return ProactiveState(chat_id=chat_id)
        return ProactiveState(
            chat_id=chat_id,
            proactive_window_started_at=float(row["proactive_window_started_at"]),
            proactive_turns_in_window=int(row["proactive_turns_in_window"]),
            proactive_cooldown_until=float(row["proactive_cooldown_until"]),
            next_due_at=float(row["next_due_at"]),
        )

    async def get(self, chat_id: str) -> ProactiveState:
        if not chat_id:
            raise ValueError("chat_id is required")
        conn = await self._ensure_open()
        async with self._lock:
            row = conn.execute(
                "SELECT * FROM proactive_state WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        return self._state(row, chat_id)

    async def save(
        self,
        chat_id: str,
        *,
        proactive_window_started_at: float,
        proactive_turns_in_window: int,
        proactive_cooldown_until: float,
        next_due_at: float | None = None,
    ) -> ProactiveState:
        if not chat_id:
            raise ValueError("chat_id is required")
        conn = await self._ensure_open()
        now = time.time()
        async with self._lock:
            previous = conn.execute(
                "SELECT next_due_at FROM proactive_state WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            due = (
                float(previous["next_due_at"])
                if next_due_at is None and previous is not None
                else float(next_due_at or 0.0)
            )
            conn.execute(
                """
                INSERT INTO proactive_state
                    (chat_id, proactive_window_started_at,
                     proactive_turns_in_window, proactive_cooldown_until,
                     next_due_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    proactive_window_started_at = excluded.proactive_window_started_at,
                    proactive_turns_in_window = excluded.proactive_turns_in_window,
                    proactive_cooldown_until = excluded.proactive_cooldown_until,
                    next_due_at = excluded.next_due_at,
                    updated_at = excluded.updated_at
                """,
                (
                    chat_id,
                    float(proactive_window_started_at),
                    max(0, int(proactive_turns_in_window)),
                    float(proactive_cooldown_until),
                    due,
                    now,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM proactive_state WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        return self._state(row, chat_id)

    async def increment_metric(
        self,
        namespace: str,
        metric: str,
        amount: int = 1,
        *,
        at: float | None = None,
    ) -> None:
        """Add to an hourly UTC bucket for durable operational reporting."""
        if not namespace or not metric:
            raise ValueError("namespace and metric are required")
        if amount == 0:
            return
        conn = await self._ensure_open()
        timestamp = time.time() if at is None else float(at)
        bucket = int(timestamp // 3600) * 3600
        async with self._lock:
            conn.execute(
                """
                INSERT INTO proactive_metrics
                    (namespace, metric, bucket_start, count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, metric, bucket_start) DO UPDATE SET
                    count = proactive_metrics.count + excluded.count,
                    updated_at = excluded.updated_at
                """,
                (namespace, metric, bucket, int(amount), time.time()),
            )
            conn.commit()

    async def metric_totals(
        self, namespace: str, *, since: float | None = None
    ) -> dict[str, int]:
        """Return durable totals, optionally limited to a wall-clock window."""
        conn = await self._ensure_open()
        async with self._lock:
            if since is None:
                rows = conn.execute(
                    """
                    SELECT metric, SUM(count) AS total
                    FROM proactive_metrics WHERE namespace = ?
                    GROUP BY metric
                    """,
                    (namespace,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT metric, SUM(count) AS total
                    FROM proactive_metrics
                    WHERE namespace = ? AND bucket_start >= ?
                    GROUP BY metric
                    """,
                    (namespace, int(float(since) // 3600) * 3600),
                ).fetchall()
        return {str(row["metric"]): int(row["total"]) for row in rows}

    async def metric_history(
        self, namespace: str, *, since: float, until: float | None = None
    ) -> list[dict[str, int]]:
        """Return hourly metric buckets for an operational reporting surface."""
        conn = await self._ensure_open()
        start = int(float(since) // 3600) * 3600
        async with self._lock:
            if until is None:
                rows = conn.execute(
                    """
                    SELECT metric, bucket_start, count
                    FROM proactive_metrics
                    WHERE namespace = ? AND bucket_start >= ?
                    ORDER BY bucket_start, metric
                    """,
                    (namespace, start),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT metric, bucket_start, count
                    FROM proactive_metrics
                    WHERE namespace = ? AND bucket_start >= ? AND bucket_start < ?
                    ORDER BY bucket_start, metric
                    """,
                    (namespace, start, float(until)),
                ).fetchall()
        return [
            {
                "metric": str(row["metric"]),
                "bucket_start": int(row["bucket_start"]),
                "count": int(row["count"]),
            }
            for row in rows
        ]

    async def set_next_due(self, chat_id: str, next_due_at: float) -> ProactiveState:
        """Persist only scheduler timing while retaining budget state."""
        state = await self.get(chat_id)
        return await self.save(
            chat_id,
            proactive_window_started_at=state.proactive_window_started_at,
            proactive_turns_in_window=state.proactive_turns_in_window,
            proactive_cooldown_until=state.proactive_cooldown_until,
            next_due_at=next_due_at,
        )

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
