"""Durable conversation timeline projection.

The timeline contains only admitted user input and transport-accepted visible
output. Assistant/tool protocol messages and failed or unknown deliveries stay
outside this projection.
"""

import asyncio
import hashlib
import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from core.managers.chat_message import strip_content_prefix


@dataclass(frozen=True)
class TimelineEvent:
    chat_id: str
    seq: int
    event_id: str
    role: str
    content: str
    message_id: str = ""
    sender_id: str = ""
    event_kind: str = "user_message"
    delivery_kind: str = ""
    timestamp: float = 0.0
    session_kind: str = "chat"
    token_count: int = 0

    def to_history_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "delivery_kind": self.delivery_kind,
            "session_kind": self.session_kind,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class TimelineMigrationReport:
    """Non-content readiness result for retiring legacy history reads."""

    chat_id: str
    timeline_visible_count: int
    legacy_visible_count: int
    missing_legacy_visible_count: int
    extra_timeline_visible_count: int
    legacy_protocol_count: int

    @property
    def ready_for_legacy_read_removal(self) -> bool:
        return (
            self.missing_legacy_visible_count == 0 and self.legacy_protocol_count == 0
        )

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "chat_id": self.chat_id,
            "timeline_visible_count": self.timeline_visible_count,
            "legacy_visible_count": self.legacy_visible_count,
            "missing_legacy_visible_count": self.missing_legacy_visible_count,
            "extra_timeline_visible_count": self.extra_timeline_visible_count,
            "legacy_protocol_count": self.legacy_protocol_count,
            "ready_for_legacy_read_removal": self.ready_for_legacy_read_removal,
        }


@dataclass(frozen=True)
class TimelineMigrationSummary:
    """Content-free migration readiness across all discovered sessions."""

    session_count: int
    sessions_with_missing_legacy_visible: int
    sessions_with_legacy_protocol: int
    sessions_ready_for_legacy_read_removal: int
    sessions_with_scan_errors: int = 0

    @property
    def ready_for_legacy_read_removal(self) -> bool:
        return (
            self.sessions_with_scan_errors == 0
            and self.sessions_ready_for_legacy_read_removal == self.session_count
        )

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "session_count": self.session_count,
            "sessions_with_missing_legacy_visible": self.sessions_with_missing_legacy_visible,
            "sessions_with_legacy_protocol": self.sessions_with_legacy_protocol,
            "sessions_ready_for_legacy_read_removal": self.sessions_ready_for_legacy_read_removal,
            "sessions_with_scan_errors": self.sessions_with_scan_errors,
            "ready_for_legacy_read_removal": self.ready_for_legacy_read_removal,
        }


class ConversationTimeline:
    """Append-only per-chat timeline with idempotent event keys."""

    def __init__(self, path: str = "data/conversation_timeline.sqlite3"):
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
                CREATE TABLE IF NOT EXISTS conversation_timeline (
                    chat_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    message_id TEXT NOT NULL DEFAULT '',
                    sender_id TEXT NOT NULL DEFAULT '',
                    event_kind TEXT NOT NULL,
                    delivery_kind TEXT NOT NULL DEFAULT '',
                    timestamp REAL NOT NULL,
                    session_kind TEXT NOT NULL DEFAULT 'chat',
                    token_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (chat_id, seq),
                    UNIQUE (chat_id, event_id)
                )
                """)
            columns = {
                row["name"]
                for row in self._conn.execute(
                    "PRAGMA table_info(conversation_timeline)"
                ).fetchall()
            }
            if "session_kind" not in columns:
                self._conn.execute(
                    "ALTER TABLE conversation_timeline ADD COLUMN session_kind TEXT NOT NULL DEFAULT 'chat'"
                )
            if "token_count" not in columns:
                self._conn.execute(
                    "ALTER TABLE conversation_timeline ADD COLUMN token_count INTEGER NOT NULL DEFAULT 0"
                )
                self._conn.execute(
                    "UPDATE conversation_timeline SET token_count = LENGTH(content) / 4"
                )
            self._conn.commit()
        return self._conn

    @staticmethod
    def _event(row: sqlite3.Row) -> TimelineEvent:
        return TimelineEvent(
            chat_id=row["chat_id"],
            seq=int(row["seq"]),
            event_id=row["event_id"],
            role=row["role"],
            content=row["content"],
            message_id=row["message_id"],
            sender_id=row["sender_id"],
            event_kind=row["event_kind"],
            delivery_kind=row["delivery_kind"],
            timestamp=float(row["timestamp"]),
            session_kind=(
                row["session_kind"] if "session_kind" in row.keys() else "chat"
            ),
            token_count=(
                int(row["token_count"])
                if "token_count" in row.keys()
                else max(0, len(row["content"]) // 4)
            ),
        )

    async def append(
        self,
        *,
        chat_id: str,
        event_id: str,
        role: str,
        content: str,
        event_kind: str,
        message_id: str = "",
        sender_id: str = "",
        delivery_kind: str = "",
        timestamp: float | None = None,
        session_kind: str = "chat",
        token_count: int | None = None,
    ) -> TimelineEvent:
        """Append one event or return the existing event for the same event key."""
        if not chat_id or not event_id:
            raise ValueError("chat_id and event_id are required")
        if role not in {"user", "assistant"}:
            raise ValueError(f"invalid timeline role: {role}")
        if event_kind not in {"user_message", "delivery"}:
            raise ValueError(f"invalid timeline event kind: {event_kind}")
        if not session_kind:
            session_kind = "chat"
        conn = await self._ensure_open()
        now = time.time() if timestamp is None else timestamp
        tokens = (
            max(0, len(content) // 4) if token_count is None else max(0, token_count)
        )
        async with self._lock:
            existing = conn.execute(
                "SELECT * FROM conversation_timeline WHERE chat_id = ? AND event_id = ?",
                (chat_id, event_id),
            ).fetchone()
            if existing is not None:
                return self._event(existing)
            if session_kind == "chat":
                previous_kind = conn.execute(
                    """
                    SELECT session_kind FROM conversation_timeline
                     WHERE chat_id = ? AND role = 'user'
                     ORDER BY seq DESC LIMIT 1
                    """,
                    (chat_id,),
                ).fetchone()
                if previous_kind is not None:
                    session_kind = previous_kind["session_kind"]
            next_seq = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM conversation_timeline WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()[0]
            conn.execute(
                """
                INSERT INTO conversation_timeline
                    (chat_id, seq, event_id, role, content, message_id,
                     sender_id, event_kind, delivery_kind, timestamp,
                     session_kind, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    next_seq,
                    event_id,
                    role,
                    content,
                    message_id,
                    sender_id,
                    event_kind,
                    delivery_kind,
                    now,
                    session_kind,
                    tokens,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM conversation_timeline WHERE chat_id = ? AND event_id = ?",
                (chat_id, event_id),
            ).fetchone()
        return self._event(row)

    async def append_user_message(
        self,
        *,
        chat_id: str,
        message_id: str,
        content: str,
        sender_id: str,
        timestamp: float,
        session_kind: str = "chat",
    ) -> TimelineEvent:
        return await self.append(
            chat_id=chat_id,
            event_id=f"user:{message_id}",
            role="user",
            content=content,
            event_kind="user_message",
            message_id=message_id,
            sender_id=sender_id,
            timestamp=timestamp,
            session_kind=session_kind,
        )

    async def append_accepted_delivery(
        self,
        *,
        chat_id: str,
        delivery_id: str,
        content: str,
        delivery_kind: str,
        message_id: str = "",
        timestamp: float | None = None,
        session_kind: str = "chat",
    ) -> TimelineEvent:
        return await self.append(
            chat_id=chat_id,
            event_id=f"delivery:{delivery_id}",
            role="assistant",
            content=content,
            event_kind="delivery",
            delivery_kind=delivery_kind,
            message_id=message_id,
            timestamp=timestamp,
            session_kind=session_kind,
        )

    async def repair_from_legacy_history(
        self, chat_id: str, messages: Sequence[dict[str, Any]]
    ) -> int:
        """Backfill missing visible events from legacy chat history.

        Only user messages and assistant messages without tool calls are
        migrated. Repeated calls are idempotent by event ID; assistant content
        already represented by an accepted delivery is not duplicated. Tool
        protocol entries remain outside the visible timeline.
        """
        if not chat_id or not messages:
            return 0
        existing_events = await self.snapshot(chat_id)
        existing_event_ids = {event.event_id for event in existing_events}
        accepted_assistant_content = {
            self._content_fingerprint(event.role, event.content)
            for event in existing_events
            if (
                event.role == "assistant"
                and event.content
                and event.event_id.startswith("delivery:")
            )
        }
        migrated = 0
        for index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("raw_content", message.get("content", ""))
            if role == "user" and "raw_content" not in message:
                content = strip_content_prefix(str(content or ""))
            if role == "user":
                event_id = message.get("message_id")
                if event_id:
                    event_id = f"user:{event_id}"
                else:
                    event_id = self._legacy_event_id("user", index, message)
                if event_id in existing_event_ids:
                    continue
                event = await self.append(
                    chat_id=chat_id,
                    event_id=event_id,
                    role="user",
                    content=str(content or ""),
                    event_kind="user_message",
                    message_id=str(message.get("message_id") or ""),
                    sender_id=str(message.get("sender_id") or ""),
                    timestamp=self._legacy_timestamp(message),
                )
                existing_event_ids.add(event.event_id)
                migrated += 1
            elif role == "assistant" and content and not message.get("tool_calls"):
                event_id = self._legacy_event_id("assistant", index, message)
                fingerprint = self._content_fingerprint("assistant", str(content))
                if (
                    event_id in existing_event_ids
                    or fingerprint in accepted_assistant_content
                ):
                    continue
                event = await self.append(
                    chat_id=chat_id,
                    event_id=event_id,
                    role="assistant",
                    content=str(content),
                    event_kind="delivery",
                    delivery_kind="response",
                    message_id=str(message.get("message_id") or ""),
                    timestamp=self._legacy_timestamp(message),
                )
                existing_event_ids.add(event.event_id)
                migrated += 1
        return migrated

    async def migration_report(
        self, chat_id: str, legacy_messages: Sequence[dict[str, Any]]
    ) -> TimelineMigrationReport:
        """Compare visible legacy history with the timeline without exposing content."""
        events = await self.snapshot(chat_id)
        timeline_visible = Counter(
            self._content_fingerprint(event.role, event.content)
            for event in events
            if event.role in {"user", "assistant"} and event.content
        )
        legacy_visible = Counter()
        legacy_protocol_count = 0
        for message in legacy_messages:
            role = message.get("role")
            if role == "tool" or (role == "assistant" and message.get("tool_calls")):
                legacy_protocol_count += 1
                continue
            if role in {"user", "assistant"}:
                content = message.get("raw_content", message.get("content", ""))
                if content:
                    legacy_visible[self._content_fingerprint(role, str(content))] += 1
        missing = legacy_visible - timeline_visible
        extra = timeline_visible - legacy_visible
        return TimelineMigrationReport(
            chat_id=chat_id,
            timeline_visible_count=sum(timeline_visible.values()),
            legacy_visible_count=sum(legacy_visible.values()),
            missing_legacy_visible_count=sum(missing.values()),
            extra_timeline_visible_count=sum(extra.values()),
            legacy_protocol_count=legacy_protocol_count,
        )

    @staticmethod
    def migration_summary(
        reports: Sequence[TimelineMigrationReport], *, scan_errors: int = 0
    ) -> TimelineMigrationSummary:
        """Aggregate reports without retaining session IDs or message content."""
        return TimelineMigrationSummary(
            session_count=len(reports) + scan_errors,
            sessions_with_missing_legacy_visible=sum(
                report.missing_legacy_visible_count > 0 for report in reports
            ),
            sessions_with_legacy_protocol=sum(
                report.legacy_protocol_count > 0 for report in reports
            ),
            sessions_ready_for_legacy_read_removal=sum(
                report.ready_for_legacy_read_removal for report in reports
            ),
            sessions_with_scan_errors=scan_errors,
        )

    @staticmethod
    def _content_fingerprint(role: str, content: str) -> tuple[str, str]:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return role, digest

    @staticmethod
    def _legacy_event_id(role: str, index: int, message: dict[str, Any]) -> str:
        digest = hashlib.sha256(
            json.dumps(message, ensure_ascii=False, sort_keys=True, default=str).encode(
                "utf-8"
            )
        ).hexdigest()[:24]
        return f"legacy:{role}:{index}:{digest}"

    @staticmethod
    def _legacy_timestamp(message: dict[str, Any]) -> float:
        try:
            return float(message.get("timestamp") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    async def snapshot(
        self, chat_id: str, *, upto_seq: int | None = None
    ) -> tuple[TimelineEvent, ...]:
        """Read a stable sequence snapshot for a new turn."""
        conn = await self._ensure_open()
        async with self._lock:
            if upto_seq is None:
                rows = conn.execute(
                    "SELECT * FROM conversation_timeline WHERE chat_id = ? ORDER BY seq",
                    (chat_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM conversation_timeline
                     WHERE chat_id = ? AND seq <= ? ORDER BY seq
                    """,
                    (chat_id, upto_seq),
                ).fetchall()
        return tuple(self._event(row) for row in rows)

    async def latest_seq(self, chat_id: str) -> int:
        conn = await self._ensure_open()
        async with self._lock:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM conversation_timeline WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        return int(row[0])

    async def chat_ids(self) -> list[str]:
        """Return chats represented by the visible timeline projection."""
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                "SELECT DISTINCT chat_id FROM conversation_timeline ORDER BY chat_id"
            ).fetchall()
        return [str(row[0]) for row in rows]

    async def history(self, chat_id: str, max_events: int | None = None) -> list[dict]:
        events = await self.snapshot(chat_id)
        if max_events is not None:
            events = events[-max_events:] if max_events > 0 else ()
        return [event.to_history_dict() for event in events]

    async def session_summary(self, chat_id: str) -> dict:
        """Return aggregate session metadata without reading event content."""
        conn = await self._ensure_open()
        async with self._lock:
            row = conn.execute(
                """
                SELECT COUNT(*) AS message_count,
                       COALESCE(MAX(timestamp), 0) AS last_activity,
                       COALESCE(SUM(LENGTH(content)), 0) AS content_chars,
                       COALESCE(SUM(token_count), 0) AS estimated_tokens
                  FROM conversation_timeline
                 WHERE chat_id = ?
                """,
                (chat_id,),
            ).fetchone()
        return {
            "message_count": int(row["message_count"]),
            "last_activity": float(row["last_activity"]),
            "estimated_tokens": int(row["estimated_tokens"]),
        }

    async def clear_chat(self, chat_id: str) -> None:
        if not chat_id:
            raise ValueError("chat_id is required")
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                "DELETE FROM conversation_timeline WHERE chat_id = ?", (chat_id,)
            )
            conn.commit()

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
