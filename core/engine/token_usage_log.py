"""Durable token/cost usage ledger.

``CostTracker`` keeps per-process in-memory aggregates; this append-only SQLite
store keeps one row per LLM call so consumption can be analysed by source
(model / provider / chat / day / subsystem) even after restarts.

Deliberately synchronous: it is written from ``CostTracker.record_turn`` on the
request hot path, and one small ``INSERT`` under a ``threading.Lock`` is cheaper
and safer than scheduling an asyncio task that could be dropped on shutdown.
Reads are read-only; records are retained forever (``retention_days = 0``).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)

# Source vocabulary. Only the first two are produced today; the rest are
# reserved so auxiliary call sites can adopt the ledger without a migration.
SOURCE_TURN_REPLY = "turn:reply"
SOURCE_TURN_AMBIENT = "turn:ambient"
SOURCE_TURN_STEER = "turn:steer"
SOURCE_TURN_PASSIVE = "turn:passive"
SOURCE_AUX_COMPACTION = "aux:compaction_summary"
# reserved: aux:semantic_summary, aux:vision, aux:media_summary, aux:learner,
#           aux:auto_review
SOURCE_UNKNOWN = "unknown"

_BREAKDOWN_COLUMNS = {
    "model": "model",
    "source": "source",
    "provider": "provider",
    "chat": "chat_id",
}


@dataclass(frozen=True)
class TokenUsageRecord:
    record_id: str
    recorded_at: float
    chat_id: str
    turn_id: str
    model: str
    provider: str
    source: str
    turn_kind: str
    intent: str
    scope_generation: int
    prompt_tokens: int
    completion_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    usage_present: bool
    cache_usage_present: bool
    estimated_prompt_tokens: int
    cost: float
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "recorded_at": self.recorded_at,
            "chat_id": self.chat_id,
            "turn_id": self.turn_id,
            "model": self.model,
            "provider": self.provider,
            "source": self.source,
            "turn_kind": self.turn_kind,
            "intent": self.intent,
            "scope_generation": self.scope_generation,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "usage_present": self.usage_present,
            "cache_usage_present": self.cache_usage_present,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "cost": self.cost,
            "elapsed_ms": self.elapsed_ms,
        }


class TokenUsageLog:
    """Append-only SQLite ledger of per-call token usage and cost."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str = "data/token_usage.sqlite3"):
        self._path = path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()

    # ── 连接与 schema ──

    def _ensure_open(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        with self._lock:
            if self._conn is not None:
                return self._conn
            if self._path != ":memory:":
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS token_usage_log_schema (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS token_usage_records (
                    record_id TEXT PRIMARY KEY,
                    recorded_at REAL NOT NULL,
                    chat_id TEXT NOT NULL DEFAULT '',
                    turn_id TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    provider TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'unknown',
                    turn_kind TEXT NOT NULL DEFAULT '',
                    intent TEXT NOT NULL DEFAULT '',
                    scope_generation INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_miss_tokens INTEGER NOT NULL DEFAULT 0,
                    usage_present INTEGER NOT NULL DEFAULT 0,
                    cache_usage_present INTEGER NOT NULL DEFAULT 0,
                    estimated_prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    cost REAL NOT NULL DEFAULT 0,
                    elapsed_ms REAL NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_token_usage_recorded_at
                    ON token_usage_records(recorded_at);
                CREATE INDEX IF NOT EXISTS idx_token_usage_model
                    ON token_usage_records(model, recorded_at);
                CREATE INDEX IF NOT EXISTS idx_token_usage_source
                    ON token_usage_records(source, recorded_at);
                CREATE INDEX IF NOT EXISTS idx_token_usage_chat
                    ON token_usage_records(chat_id, recorded_at);
                """)
            row = conn.execute("SELECT version FROM token_usage_log_schema").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO token_usage_log_schema (version) VALUES (?)",
                    (self.SCHEMA_VERSION,),
                )
            elif int(row["version"]) != self.SCHEMA_VERSION:
                conn.close()
                raise RuntimeError(
                    f"unsupported token usage log schema version: {row['version']}"
                )
            conn.commit()
            self._conn = conn
        return self._conn

    # ── 写入 ──

    def record(
        self,
        *,
        chat_id: str = "",
        turn_id: str = "",
        model: str = "",
        provider: str = "",
        source: str = SOURCE_UNKNOWN,
        turn_kind: str = "",
        intent: str = "",
        scope_generation: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_hit_tokens: int = 0,
        cache_miss_tokens: int = 0,
        usage_present: bool = False,
        cache_usage_present: bool = False,
        estimated_prompt_tokens: int = 0,
        cost: float = 0.0,
        elapsed_ms: float = 0.0,
        recorded_at: Optional[float] = None,
    ) -> TokenUsageRecord:
        conn = self._ensure_open()
        recorded_at = time.time() if recorded_at is None else float(recorded_at)
        record_id = f"usage:{time.time_ns()}"
        values = (
            record_id,
            recorded_at,
            str(chat_id or ""),
            str(turn_id or ""),
            str(model or ""),
            str(provider or ""),
            str(source or SOURCE_UNKNOWN),
            str(turn_kind or ""),
            str(intent or ""),
            int(scope_generation or 0),
            int(prompt_tokens or 0),
            int(completion_tokens or 0),
            int(cache_hit_tokens or 0),
            int(cache_miss_tokens or 0),
            int(bool(usage_present)),
            int(bool(cache_usage_present)),
            int(estimated_prompt_tokens or 0),
            float(cost or 0.0),
            float(elapsed_ms or 0.0),
        )
        with self._lock:
            conn.execute(
                """
                INSERT INTO token_usage_records (
                    record_id, recorded_at, chat_id, turn_id, model, provider,
                    source, turn_kind, intent, scope_generation, prompt_tokens,
                    completion_tokens, cache_hit_tokens, cache_miss_tokens,
                    usage_present, cache_usage_present, estimated_prompt_tokens,
                    cost, elapsed_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            conn.commit()
        return TokenUsageRecord(*values)  # type: ignore[arg-type]

    # ── 查询（只读） ──

    def _time_clause(
        self, since: Optional[float], until: Optional[float]
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("recorded_at >= ?")
            params.append(float(since))
        if until is not None:
            clauses.append("recorded_at <= ?")
            params.append(float(until))
        return (" AND ".join(clauses), params)

    def summary(
        self, *, since: Optional[float] = None, until: Optional[float] = None
    ) -> dict[str, Any]:
        conn = self._ensure_open()
        where, params = self._time_clause(since, until)
        clause = f" WHERE {where}" if where else ""
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS record_count,
                   COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                   COALESCE(SUM(cache_hit_tokens), 0) AS cache_hit_tokens,
                   COALESCE(SUM(cache_miss_tokens), 0) AS cache_miss_tokens,
                   COALESCE(SUM(cost), 0) AS cost,
                   COALESCE(SUM(usage_present), 0) AS usage_present_count,
                   COALESCE(SUM(cache_usage_present), 0) AS cache_usage_present_count,
                   MIN(recorded_at) AS earliest,
                   MAX(recorded_at) AS latest
              FROM token_usage_records{clause}
            """,
            params,
        ).fetchone()
        return {
            "record_count": int(row["record_count"]),
            "prompt_tokens": int(row["prompt_tokens"]),
            "completion_tokens": int(row["completion_tokens"]),
            "cache_hit_tokens": int(row["cache_hit_tokens"]),
            "cache_miss_tokens": int(row["cache_miss_tokens"]),
            "cost": float(row["cost"]),
            "usage_present_count": int(row["usage_present_count"]),
            "cache_usage_present_count": int(row["cache_usage_present_count"]),
            "earliest": row["earliest"],
            "latest": row["latest"],
        }

    def breakdown(
        self,
        dimension: str,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if dimension == "day":
            expression = "strftime('%Y-%m-%d', recorded_at, 'unixepoch', 'localtime')"
        else:
            expression = _BREAKDOWN_COLUMNS.get(dimension, "")
            if not expression:
                raise ValueError(f"unsupported breakdown dimension: {dimension}")
        conn = self._ensure_open()
        where, params = self._time_clause(since, until)
        clause = f" WHERE {where}" if where else ""
        rows = conn.execute(
            f"""
            SELECT {expression} AS key,
                   COUNT(*) AS record_count,
                   COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                   COALESCE(SUM(cost), 0) AS cost
              FROM token_usage_records{clause}
             GROUP BY key
             ORDER BY cost DESC, record_count DESC
             LIMIT ?
            """,
            [*params, max(1, int(limit))],
        ).fetchall()
        return [
            {
                "key": str(row["key"]),
                "record_count": int(row["record_count"]),
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "cost": float(row["cost"]),
            }
            for row in rows
        ]

    def count(
        self,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
        source: Optional[str] = None,
        model: Optional[str] = None,
    ) -> int:
        conn = self._ensure_open()
        where, params = self._time_clause(since, until)
        clauses = [where] if where else []
        if source:
            clauses.append("source = ?")
            params.append(str(source))
        if model:
            clauses.append("model = ?")
            params.append(str(model))
        clause = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM token_usage_records{clause}", params
        ).fetchone()
        return int(row["n"])

    def recent(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        since: Optional[float] = None,
        until: Optional[float] = None,
        source: Optional[str] = None,
        model: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        conn = self._ensure_open()
        where, params = self._time_clause(since, until)
        clauses = [where] if where else []
        if source:
            clauses.append("source = ?")
            params.append(str(source))
        if model:
            clauses.append("model = ?")
            params.append(str(model))
        clause = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"""
            SELECT * FROM token_usage_records{clause}
             ORDER BY recorded_at DESC
             LIMIT ? OFFSET ?
            """,
            [*params, max(1, int(limit)), max(0, int(offset))],
        ).fetchall()
        return [dict(row) for row in rows]

    def status(self) -> dict[str, int | float]:
        summary = self.summary()
        return {
            "record_count": summary["record_count"],
            "usage_present_count": summary["usage_present_count"],
            "cache_usage_present_count": summary["cache_usage_present_count"],
            "total_cost": summary["cost"],
            "earliest": summary["earliest"] or 0.0,
            "latest": summary["latest"] or 0.0,
        }

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
