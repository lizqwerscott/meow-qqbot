"""Non-sensitive audit records for the prompt actually assembled."""

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PromptContextReport:
    report_id: str
    chat_id: str
    turn_id: str
    source: str
    generation: int
    history_count: int
    protocol_count: int
    total_message_count: int
    estimated_tokens: int
    fallback_reason: str
    recorded_at: float
    attempt_id: str = ""
    projection_version: int = 0
    prompt_hash: str = ""
    summary_dates: tuple[str, ...] = ()
    summary_count: int = 0
    degraded_reason: str = ""
    scope_key: str = ""
    truncated_event_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "chat_id": self.chat_id,
            "turn_id": self.turn_id,
            "source": self.source,
            "generation": self.generation,
            "history_count": self.history_count,
            "protocol_count": self.protocol_count,
            "total_message_count": self.total_message_count,
            "estimated_tokens": self.estimated_tokens,
            "fallback_reason": self.fallback_reason,
            "recorded_at": self.recorded_at,
            "attempt_id": self.attempt_id,
            "projection_version": self.projection_version,
            "prompt_hash": self.prompt_hash,
            "summary_dates": list(self.summary_dates),
            "summary_count": self.summary_count,
            "degraded_reason": self.degraded_reason,
            "scope_key": self.scope_key,
            "truncated_event_ids": list(self.truncated_event_ids),
        }


class PromptContextReportStore:
    """Small append-only store; never persists prompt bodies."""

    def __init__(self, path: str = "data/prompt_context_reports.sqlite3") -> None:
        self._path = path
        self._conn: Optional[sqlite3.Connection] = None
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
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_context_reports (
                    report_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    history_count INTEGER NOT NULL,
                    protocol_count INTEGER NOT NULL,
                    total_message_count INTEGER NOT NULL,
                    estimated_tokens INTEGER NOT NULL,
                    fallback_reason TEXT NOT NULL DEFAULT '',
                    recorded_at REAL NOT NULL,
                    attempt_id TEXT NOT NULL DEFAULT '',
                    projection_version INTEGER NOT NULL DEFAULT 0,
                    prompt_hash TEXT NOT NULL DEFAULT '',
                    summary_dates TEXT NOT NULL DEFAULT '[]',
                    summary_count INTEGER NOT NULL DEFAULT 0,
                    degraded_reason TEXT NOT NULL DEFAULT '',
                    scope_key TEXT NOT NULL DEFAULT '',
                    truncated_event_ids TEXT NOT NULL DEFAULT '[]'
                )
                """)
            columns = {
                row["name"]
                for row in self._conn.execute(
                    "PRAGMA table_info(prompt_context_reports)"
                ).fetchall()
            }
            additions = {
                "attempt_id": "TEXT NOT NULL DEFAULT ''",
                "projection_version": "INTEGER NOT NULL DEFAULT 0",
                "prompt_hash": "TEXT NOT NULL DEFAULT ''",
                "summary_dates": "TEXT NOT NULL DEFAULT '[]'",
                "summary_count": "INTEGER NOT NULL DEFAULT 0",
                "degraded_reason": "TEXT NOT NULL DEFAULT ''",
                "scope_key": "TEXT NOT NULL DEFAULT ''",
                "truncated_event_ids": "TEXT NOT NULL DEFAULT '[]'",
            }
            for name, definition in additions.items():
                if name not in columns:
                    self._conn.execute(
                        f"ALTER TABLE prompt_context_reports ADD COLUMN {name} {definition}"
                    )
            self._conn.commit()
        return self._conn

    async def record(
        self,
        *,
        chat_id: str,
        turn_id: str,
        source: str,
        generation: int = 0,
        history_count: int = 0,
        protocol_count: int = 0,
        total_message_count: int = 0,
        estimated_tokens: int = 0,
        fallback_reason: str = "",
        attempt_id: str = "",
        projection_version: int = 0,
        prompt_hash: str = "",
        summary_dates: tuple[str, ...] = (),
        summary_count: int = 0,
        degraded_reason: str = "",
        scope_key: str = "",
        truncated_event_ids: tuple[str, ...] = (),
    ) -> PromptContextReport:
        report = PromptContextReport(
            report_id=f"prompt:{chat_id}:{turn_id}:{time.time_ns()}",
            chat_id=chat_id,
            turn_id=turn_id,
            source=source,
            generation=max(0, int(generation)),
            history_count=max(0, int(history_count)),
            protocol_count=max(0, int(protocol_count)),
            total_message_count=max(0, int(total_message_count)),
            estimated_tokens=max(0, int(estimated_tokens)),
            fallback_reason=str(fallback_reason or "")[:500],
            recorded_at=time.time(),
            attempt_id=str(attempt_id or "")[:200],
            projection_version=max(0, int(projection_version)),
            prompt_hash=str(prompt_hash or "")[:128],
            summary_dates=tuple(
                str(date)[:32] for date in summary_dates if str(date).strip()
            )[:32],
            summary_count=max(0, int(summary_count)),
            degraded_reason=str(degraded_reason or "")[:500],
            scope_key=str(scope_key or "")[:300],
            truncated_event_ids=tuple(
                str(event_id)[:200]
                for event_id in truncated_event_ids
                if str(event_id).strip()
            )[:200],
        )
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                """
                INSERT INTO prompt_context_reports
                    (report_id, chat_id, turn_id, source, generation,
                     history_count, protocol_count, total_message_count,
                    estimated_tokens, fallback_reason, recorded_at, attempt_id,
                    projection_version, prompt_hash, summary_dates, summary_count,
                    degraded_reason, scope_key, truncated_event_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    report.chat_id,
                    report.turn_id,
                    report.source,
                    report.generation,
                    report.history_count,
                    report.protocol_count,
                    report.total_message_count,
                    report.estimated_tokens,
                    report.fallback_reason,
                    report.recorded_at,
                    report.attempt_id,
                    report.projection_version,
                    report.prompt_hash,
                    json.dumps(report.summary_dates, ensure_ascii=False),
                    report.summary_count,
                    report.degraded_reason,
                    report.scope_key,
                    json.dumps(report.truncated_event_ids, ensure_ascii=False),
                ),
            )
            conn.commit()
        return report

    async def list_for_webui(
        self, chat_id: str, *, limit: int = 100
    ) -> list[PromptContextReport]:
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                """
                SELECT * FROM prompt_context_reports
                 WHERE chat_id = ? ORDER BY recorded_at DESC LIMIT ?
                """,
                (chat_id, max(1, int(limit))),
            ).fetchall()
        return [
            PromptContextReport(
                report_id=row["report_id"],
                chat_id=row["chat_id"],
                turn_id=row["turn_id"],
                source=row["source"],
                generation=int(row["generation"]),
                history_count=int(row["history_count"]),
                protocol_count=int(row["protocol_count"]),
                total_message_count=int(row["total_message_count"]),
                estimated_tokens=int(row["estimated_tokens"]),
                fallback_reason=row["fallback_reason"],
                recorded_at=float(row["recorded_at"]),
                attempt_id=row["attempt_id"],
                projection_version=int(row["projection_version"]),
                prompt_hash=row["prompt_hash"],
                summary_dates=tuple(json.loads(row["summary_dates"] or "[]")),
                summary_count=int(row["summary_count"]),
                degraded_reason=row["degraded_reason"],
                scope_key=row["scope_key"],
                truncated_event_ids=tuple(
                    json.loads(row["truncated_event_ids"] or "[]")
                ),
            )
            for row in rows
        ]

    async def clear_chat(self, chat_id: str) -> None:
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                "DELETE FROM prompt_context_reports WHERE chat_id = ?", (chat_id,)
            )
            conn.commit()

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
