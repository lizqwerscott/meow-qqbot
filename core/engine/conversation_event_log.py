"""Append-only conversation facts and turn integrity.

This module is the first persistence seam for the unified conversation
ledger. It records immutable events and derives turn state from those events;
archive and prompt projections are deliberately outside this module.
"""

import asyncio
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo


class EventKind(StrEnum):
    USER_MESSAGE = "user_message"
    ACCEPTED_DELIVERY = "accepted_delivery"
    ASSISTANT_TOOL_CALL = "assistant_tool_call"
    TOOL_RESULT = "tool_result"
    DELIVERY_RECEIPT = "delivery_receipt"
    TURN_TERMINAL = "turn_terminal"
    SYSTEM_EVENT = "system_event"


class TurnStatus(StrEnum):
    OPEN = "open"
    WAITING_TOOL = "waiting_tool"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    BLOCKED = "blocked"
    INCOMPLETE = "incomplete"


_TERMINAL_STATUSES = frozenset(
    {
        TurnStatus.COMPLETED,
        TurnStatus.FAILED,
        TurnStatus.ABORTED,
        TurnStatus.BLOCKED,
        TurnStatus.INCOMPLETE,
    }
)
_VISIBLE_KINDS = frozenset({EventKind.USER_MESSAGE, EventKind.ACCEPTED_DELIVERY})


class EventLogInvariantError(RuntimeError):
    """Raised when an event would violate ledger or turn invariants."""


@dataclass(frozen=True)
class ConversationEvent:
    chat_id: str
    turn_id: str
    event_id: str
    role: str
    kind: EventKind | str
    content: str = ""
    turn_sequence: int = 0
    event_seq: int = 0
    timestamp: float = 0.0
    source_date: str = ""
    message_id: str = ""
    sender_id: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    reasoning_content: str = ""
    terminal_status: str = ""
    session_kind: str = "chat"
    token_count: int = 0

    def __post_init__(self) -> None:
        if not self.chat_id or not self.turn_id or not self.event_id:
            raise ValueError("chat_id, turn_id and event_id are required")
        if self.role not in {"user", "assistant", "tool", "system"}:
            raise ValueError(f"invalid event role: {self.role}")
        if not isinstance(self.kind, EventKind):
            object.__setattr__(self, "kind", EventKind(self.kind))
        if self.turn_sequence < 0 or self.event_seq < 0:
            raise ValueError("event sequence values cannot be negative")
        if self.token_count < 0:
            raise ValueError("token_count cannot be negative")
        if not self.session_kind:
            object.__setattr__(self, "session_kind", "chat")
        if self.kind is EventKind.USER_MESSAGE and self.role != "user":
            raise ValueError("user_message must have user role")
        if self.kind is EventKind.ACCEPTED_DELIVERY and self.role != "assistant":
            raise ValueError("accepted_delivery must have assistant role")
        if self.kind is EventKind.ASSISTANT_TOOL_CALL and self.role != "assistant":
            raise ValueError("assistant_tool_call must have assistant role")
        if self.kind is EventKind.TOOL_RESULT:
            if self.role != "tool" or not self.tool_call_id:
                raise ValueError("tool_result requires tool role and tool_call_id")
        if self.kind is EventKind.TURN_TERMINAL:
            if self.role != "system":
                raise ValueError("turn_terminal must have system role")
            if self.terminal_status not in _TERMINAL_STATUSES:
                raise ValueError("turn_terminal requires a terminal status")

    @property
    def is_internal(self) -> bool:
        return self.kind not in _VISIBLE_KINDS

    def to_history_dict(self) -> dict[str, Any]:
        result = {
            "chat_id": self.chat_id,
            "turn_id": self.turn_id,
            "turn_sequence": self.turn_sequence,
            "event_seq": self.event_seq,
            "event_id": self.event_id,
            "role": self.role,
            "kind": str(self.kind),
            "content": self.content,
            "timestamp": self.timestamp,
            "source_date": self.source_date,
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "session_kind": self.session_kind,
            "token_count": self.token_count,
        }
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_name:
            result["tool_name"] = self.tool_name
        if self.tool_calls:
            result["tool_calls"] = [dict(call) for call in self.tool_calls]
        if self.reasoning_content:
            result["reasoning_content"] = self.reasoning_content
        if self.terminal_status:
            result["terminal_status"] = self.terminal_status
        return result

    def to_wire(self) -> dict[str, Any]:
        if self.role == "assistant":
            message: dict[str, Any] = {
                "role": "assistant",
                "content": self.content or None,
            }
            if self.reasoning_content:
                message["reasoning_content"] = self.reasoning_content
            if self.tool_calls:
                message["tool_calls"] = [dict(call) for call in self.tool_calls]
            return message
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content,
            }
        if self.role == "user":
            return {"role": "user", "content": self.content}
        raise EventLogInvariantError(f"event cannot be compiled to wire: {self.role}")


@dataclass(frozen=True)
class ConversationTurn:
    chat_id: str
    turn_id: str
    turn_sequence: int
    status: TurnStatus
    started_seq: int
    ended_seq: int = 0
    terminal_event_id: str = ""
    source_date: str = ""
    event_count: int = 0
    updated_at: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


@dataclass(frozen=True)
class ConversationEventSnapshot:
    chat_id: str
    events: tuple[ConversationEvent, ...]
    cutoff_seq: int


@dataclass(frozen=True)
class ConversationTurnSnapshot:
    chat_id: str
    turns: tuple[ConversationTurn, ...]
    cutoff_seq: int


@dataclass(frozen=True)
class TurnIntegrityReport:
    turn_id: str
    valid: bool
    status: str
    event_count: int
    tool_call_ids: tuple[str, ...] = ()
    tool_result_ids: tuple[str, ...] = ()
    missing_tool_result_ids: tuple[str, ...] = ()
    duplicate_tool_result_ids: tuple[str, ...] = ()
    reason: str = ""


class ConversationEventLog:
    """Deep module for immutable conversation events and turn validation."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str = "data/conversation_event_log.sqlite3",
        *,
        timezone_name: str = "Asia/Shanghai",
    ):
        self._path = path
        self._timezone = ZoneInfo(timezone_name)
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
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversation_events (
                    chat_id TEXT NOT NULL,
                    event_seq INTEGER NOT NULL,
                    event_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    turn_sequence INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    timestamp REAL NOT NULL,
                    source_date TEXT NOT NULL,
                    message_id TEXT NOT NULL DEFAULT '',
                    sender_id TEXT NOT NULL DEFAULT '',
                    tool_call_id TEXT NOT NULL DEFAULT '',
                    tool_name TEXT NOT NULL DEFAULT '',
                    tool_calls TEXT NOT NULL DEFAULT '[]',
                    reasoning_content TEXT NOT NULL DEFAULT '',
                    terminal_status TEXT NOT NULL DEFAULT '',
                    session_kind TEXT NOT NULL DEFAULT 'chat',
                    token_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (chat_id, event_seq),
                    UNIQUE (chat_id, event_id)
                );
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    chat_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    turn_sequence INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    started_seq INTEGER NOT NULL,
                    ended_seq INTEGER NOT NULL DEFAULT 0,
                    terminal_event_id TEXT NOT NULL DEFAULT '',
                    source_date TEXT NOT NULL,
                    event_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (chat_id, turn_id),
                    UNIQUE (chat_id, turn_sequence),
                    UNIQUE (turn_id)
                );
                CREATE TABLE IF NOT EXISTS conversation_event_log_schema (
                    version INTEGER NOT NULL
                );
                """)
            row = self._conn.execute(
                "SELECT version FROM conversation_event_log_schema LIMIT 1"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO conversation_event_log_schema(version) VALUES (?)",
                    (self.SCHEMA_VERSION,),
                )
            elif int(row["version"]) != self.SCHEMA_VERSION:
                raise EventLogInvariantError(
                    f"unsupported event log schema version: {row['version']}"
                )
            self._conn.commit()
        return self._conn

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> ConversationEvent:
        try:
            tool_calls = json.loads(row["tool_calls"] or "[]")
        except json.JSONDecodeError as exc:
            raise EventLogInvariantError("invalid persisted tool_calls JSON") from exc
        if not isinstance(tool_calls, list) or not all(
            isinstance(call, dict) for call in tool_calls
        ):
            raise EventLogInvariantError("persisted tool_calls must be objects")
        return ConversationEvent(
            chat_id=row["chat_id"],
            turn_id=row["turn_id"],
            event_id=row["event_id"],
            role=row["role"],
            kind=row["kind"],
            content=row["content"],
            turn_sequence=int(row["turn_sequence"]),
            event_seq=int(row["event_seq"]),
            timestamp=float(row["timestamp"]),
            source_date=row["source_date"],
            message_id=row["message_id"],
            sender_id=row["sender_id"],
            tool_call_id=row["tool_call_id"],
            tool_name=row["tool_name"],
            tool_calls=tuple(tool_calls),
            reasoning_content=row["reasoning_content"],
            terminal_status=row["terminal_status"],
            session_kind=row["session_kind"],
            token_count=int(row["token_count"]),
        )

    @staticmethod
    def _turn_from_row(row: sqlite3.Row) -> ConversationTurn:
        return ConversationTurn(
            chat_id=row["chat_id"],
            turn_id=row["turn_id"],
            turn_sequence=int(row["turn_sequence"]),
            status=TurnStatus(row["status"]),
            started_seq=int(row["started_seq"]),
            ended_seq=int(row["ended_seq"]),
            terminal_event_id=row["terminal_event_id"],
            source_date=row["source_date"],
            event_count=int(row["event_count"]),
            updated_at=float(row["updated_at"]),
        )

    def _source_date(self, event: ConversationEvent) -> str:
        if event.source_date:
            return event.source_date
        timestamp = event.timestamp or time.time()
        return datetime.fromtimestamp(timestamp, self._timezone).date().isoformat()

    @staticmethod
    def _fingerprint(event: ConversationEvent) -> str:
        payload = {
            "chat_id": event.chat_id,
            "turn_id": event.turn_id,
            "event_id": event.event_id,
            "role": event.role,
            "kind": str(event.kind),
            "content": event.content,
            "timestamp": event.timestamp,
            "source_date": event.source_date,
            "message_id": event.message_id,
            "sender_id": event.sender_id,
            "tool_call_id": event.tool_call_id,
            "tool_name": event.tool_name,
            "tool_calls": event.tool_calls,
            "reasoning_content": event.reasoning_content,
            "terminal_status": event.terminal_status,
            "session_kind": event.session_kind,
            "token_count": event.token_count,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _event_status(event: ConversationEvent, current: TurnStatus) -> TurnStatus:
        if event.kind is EventKind.TURN_TERMINAL:
            return TurnStatus(event.terminal_status)
        if event.kind is EventKind.ASSISTANT_TOOL_CALL:
            return TurnStatus.WAITING_TOOL
        if event.kind is EventKind.TOOL_RESULT:
            return TurnStatus.OPEN
        return current

    async def append_event(self, event: ConversationEvent) -> ConversationEvent:
        """Append an immutable event, returning the existing event idempotently."""
        conn = await self._ensure_open()
        source_date = self._source_date(event)
        normalized = replace(
            event,
            timestamp=float(event.timestamp or time.time()),
            source_date=source_date,
            token_count=(event.token_count or max(0, len(event.content) // 4)),
        )
        async with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing_row = conn.execute(
                    "SELECT * FROM conversation_events WHERE chat_id = ? AND event_id = ?",
                    (normalized.chat_id, normalized.event_id),
                ).fetchone()
                if existing_row is not None:
                    existing = self._event_from_row(existing_row)
                    comparison = normalized
                    if not event.timestamp and not event.source_date:
                        comparison = replace(
                            normalized,
                            timestamp=existing.timestamp,
                            source_date=existing.source_date,
                            token_count=existing.token_count,
                        )
                    if self._fingerprint(existing) != self._fingerprint(comparison):
                        raise EventLogInvariantError(
                            f"event identity collision: {normalized.event_id}"
                        )
                    conn.commit()
                    return existing

                turn_row = conn.execute(
                    "SELECT * FROM conversation_turns WHERE turn_id = ?",
                    (normalized.turn_id,),
                ).fetchone()
                if turn_row is None:
                    turn_sequence = normalized.turn_sequence or int(
                        conn.execute(
                            "SELECT COALESCE(MAX(turn_sequence), 0) + 1 "
                            "FROM conversation_turns WHERE chat_id = ?",
                            (normalized.chat_id,),
                        ).fetchone()[0]
                    )
                    current_status = TurnStatus.OPEN
                    if normalized.kind is EventKind.ASSISTANT_TOOL_CALL:
                        current_status = TurnStatus.WAITING_TOOL
                    conn.execute(
                        """
                        INSERT INTO conversation_turns
                            (chat_id, turn_id, turn_sequence, status, started_seq,
                             source_date, event_count, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                        """,
                        (
                            normalized.chat_id,
                            normalized.turn_id,
                            turn_sequence,
                            current_status,
                            0,
                            source_date,
                            normalized.timestamp,
                            normalized.timestamp,
                        ),
                    )
                    turn_row = conn.execute(
                        "SELECT * FROM conversation_turns WHERE turn_id = ?",
                        (normalized.turn_id,),
                    ).fetchone()
                else:
                    normalized = replace(
                        normalized, source_date=str(turn_row["source_date"])
                    )
                turn = self._turn_from_row(turn_row)
                if (
                    normalized.turn_sequence
                    and normalized.turn_sequence != turn.turn_sequence
                ):
                    raise EventLogInvariantError(
                        "turn sequence does not match turn index"
                    )
                if turn.is_terminal:
                    raise EventLogInvariantError(
                        f"cannot append to terminal turn: {normalized.turn_id}"
                    )
                event_seq = int(
                    conn.execute(
                        "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM conversation_events "
                        "WHERE chat_id = ?",
                        (normalized.chat_id,),
                    ).fetchone()[0]
                )
                turn_event_seq = int(
                    conn.execute(
                        "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM conversation_events "
                        "WHERE turn_id = ?",
                        (normalized.turn_id,),
                    ).fetchone()[0]
                )
                committed = replace(
                    normalized,
                    turn_sequence=turn.turn_sequence,
                    event_seq=event_seq,
                    timestamp=normalized.timestamp or time.time(),
                )
                new_status = self._event_status(committed, turn.status)
                if committed.kind is EventKind.TURN_TERMINAL:
                    terminal_event_id = committed.event_id
                    ended_seq = event_seq
                else:
                    terminal_event_id = turn.terminal_event_id
                    ended_seq = turn.ended_seq
                conn.execute(
                    """
                    INSERT INTO conversation_events
                        (chat_id, event_seq, event_id, turn_id, turn_sequence, role,
                         kind, content, timestamp, source_date, message_id, sender_id,
                         tool_call_id, tool_name, tool_calls, reasoning_content,
                         terminal_status,
                         session_kind, token_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        committed.chat_id,
                        event_seq,
                        committed.event_id,
                        committed.turn_id,
                        committed.turn_sequence,
                        committed.role,
                        committed.kind,
                        committed.content,
                        committed.timestamp,
                        committed.source_date,
                        committed.message_id,
                        committed.sender_id,
                        committed.tool_call_id,
                        committed.tool_name,
                        json.dumps(
                            committed.tool_calls, ensure_ascii=False, sort_keys=True
                        ),
                        committed.reasoning_content,
                        committed.terminal_status,
                        committed.session_kind,
                        committed.token_count,
                    ),
                )
                conn.execute(
                    """
                    UPDATE conversation_turns
                       SET status = ?, started_seq = CASE WHEN started_seq = 0 THEN ? ELSE started_seq END,
                           ended_seq = ?, terminal_event_id = ?, event_count = event_count + 1,
                           updated_at = ?
                     WHERE turn_id = ?
                    """,
                    (
                        new_status,
                        event_seq,
                        ended_seq,
                        terminal_event_id,
                        committed.timestamp,
                        committed.turn_id,
                    ),
                )
                conn.commit()
                return committed
            except BaseException:
                conn.rollback()
                raise

    async def append_user_message(
        self,
        *,
        chat_id: str,
        turn_id: str,
        message_id: str,
        content: str,
        sender_id: str = "",
        timestamp: float = 0.0,
        session_kind: str = "chat",
    ) -> ConversationEvent:
        return await self.append_event(
            ConversationEvent(
                chat_id=chat_id,
                turn_id=turn_id,
                event_id=f"user:{message_id}",
                role="user",
                kind=EventKind.USER_MESSAGE,
                content=content,
                message_id=message_id,
                sender_id=sender_id,
                timestamp=timestamp,
                session_kind=session_kind,
            )
        )

    async def append_accepted_delivery(
        self,
        *,
        chat_id: str,
        turn_id: str,
        delivery_id: str,
        content: str,
        message_id: str = "",
        timestamp: float = 0.0,
        session_kind: str = "chat",
    ) -> ConversationEvent:
        return await self.append_event(
            ConversationEvent(
                chat_id=chat_id,
                turn_id=turn_id,
                event_id=f"delivery:{delivery_id}",
                role="assistant",
                kind=EventKind.ACCEPTED_DELIVERY,
                content=content,
                message_id=message_id,
                timestamp=timestamp,
                session_kind=session_kind,
            )
        )

    async def append_late_delivery_event(
        self,
        *,
        chat_id: str,
        original_turn_id: str,
        delivery_id: str,
        content: str,
        message_id: str = "",
        delivery_kind: str = "response",
        timestamp: float = 0.0,
        session_kind: str = "late_orphan",
    ) -> ConversationEvent:
        """Record an accepted delivery that missed its terminal turn boundary."""
        if not original_turn_id or not delivery_id:
            raise ValueError("late delivery lineage is required")
        orphan_turn_id = f"orphan-delivery:{original_turn_id}:{delivery_id}"
        payload = json.dumps(
            {
                "lineage": {
                    "original_turn_id": original_turn_id,
                    "delivery_id": delivery_id,
                },
                "content": content,
                "delivery_kind": delivery_kind,
                "message_id": message_id,
                "reason": "accepted_after_terminal",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        event = await self.append_event(
            ConversationEvent(
                chat_id=chat_id,
                turn_id=orphan_turn_id,
                event_id=f"late-delivery:{delivery_id}",
                role="system",
                kind=EventKind.SYSTEM_EVENT,
                content=payload,
                timestamp=timestamp,
                session_kind=session_kind,
            )
        )
        await self.append_turn_terminal(
            chat_id=chat_id,
            turn_id=orphan_turn_id,
            status=TurnStatus.COMPLETED,
            timestamp=timestamp,
            event_id=f"late-delivery-terminal:{delivery_id}",
        )
        return event

    async def append_turn_terminal(
        self,
        *,
        chat_id: str,
        turn_id: str,
        status: TurnStatus | str = TurnStatus.COMPLETED,
        timestamp: float = 0.0,
        event_id: str = "",
    ) -> ConversationEvent:
        terminal_status = TurnStatus(status)
        return await self.append_event(
            ConversationEvent(
                chat_id=chat_id,
                turn_id=turn_id,
                event_id=event_id or f"terminal:{turn_id}",
                role="system",
                kind=EventKind.TURN_TERMINAL,
                terminal_status=terminal_status,
                timestamp=timestamp,
            )
        )

    async def repair_from_legacy_history(
        self,
        chat_id: str,
        messages: Sequence[dict[str, Any]],
        *,
        session_kind: str = "chat",
        source_id: str = "",
    ) -> int:
        """Import legacy messages while preserving recognizable turn order."""
        if not chat_id or not messages:
            return 0
        existing = await self.snapshot_events(chat_id, include_internal=True)
        existing_ids = {event.event_id for event in existing.events}
        current_turn_id = ""
        current_terminal = False
        imported = 0

        async def close_previous_turn(timestamp: float) -> None:
            nonlocal current_terminal
            if not current_turn_id or current_terminal:
                return
            report = await self.validate_turn(current_turn_id)
            status = TurnStatus.COMPLETED if report.valid else TurnStatus.INCOMPLETE
            await self.append_turn_terminal(
                chat_id=chat_id,
                turn_id=current_turn_id,
                status=status,
                timestamp=timestamp,
            )
            current_terminal = True

        for index, message in enumerate(messages):
            role = str(message.get("role") or "")
            content = str(message.get("raw_content", message.get("content", "")) or "")
            timestamp = self._legacy_timestamp(message)
            if role == "user":
                await close_previous_turn(timestamp)
                message_id = str(message.get("message_id") or "")
                current_turn_id = message_id or f"legacy-turn:{chat_id}:{index}"
                current_terminal = False
                event_id = (
                    f"user:{message_id}"
                    if message_id
                    else (
                        f"legacy:{source_id}:user:{index}"
                        if source_id
                        else f"legacy:user:{index}"
                    )
                )
                if event_id not in existing_ids:
                    await self.append_event(
                        ConversationEvent(
                            chat_id=chat_id,
                            turn_id=current_turn_id,
                            event_id=event_id,
                            role="user",
                            kind=EventKind.USER_MESSAGE,
                            content=content,
                            message_id=message_id,
                            sender_id=str(message.get("sender_id") or ""),
                            timestamp=timestamp,
                            session_kind=session_kind,
                        )
                    )
                    existing_ids.add(event_id)
                    imported += 1
                continue

            if role == "assistant":
                if not current_turn_id:
                    current_turn_id = f"legacy-turn:{chat_id}:{index}"
                tool_calls = message.get("tool_calls") or ()
                if isinstance(tool_calls, list):
                    tool_calls = tuple(
                        call for call in tool_calls if isinstance(call, dict)
                    )
                else:
                    tool_calls = ()
                if tool_calls:
                    event_id = str(message.get("event_id") or "") or (
                        f"legacy:{source_id}:assistant:{index}"
                        if source_id
                        else f"legacy:assistant:{index}"
                    )
                    if event_id not in existing_ids:
                        await self.append_event(
                            ConversationEvent(
                                chat_id=chat_id,
                                turn_id=current_turn_id,
                                event_id=event_id,
                                role="assistant",
                                kind=EventKind.ASSISTANT_TOOL_CALL,
                                content=content,
                                tool_calls=tool_calls,
                                reasoning_content=str(
                                    message.get("reasoning_content") or ""
                                ),
                                timestamp=timestamp,
                                session_kind=session_kind,
                            )
                        )
                        existing_ids.add(event_id)
                        imported += 1
                elif content:
                    event_id = str(message.get("event_id") or "") or (
                        f"legacy:{source_id}:delivery:{index}"
                        if source_id
                        else f"legacy:delivery:{index}"
                    )
                    if event_id not in existing_ids:
                        await self.append_event(
                            ConversationEvent(
                                chat_id=chat_id,
                                turn_id=current_turn_id,
                                event_id=event_id,
                                role="assistant",
                                kind=EventKind.ACCEPTED_DELIVERY,
                                content=content,
                                timestamp=timestamp,
                                session_kind=session_kind,
                            )
                        )
                        existing_ids.add(event_id)
                        imported += 1
                    await close_previous_turn(timestamp)
                continue

            if role == "tool":
                if not current_turn_id:
                    current_turn_id = f"legacy-turn:{chat_id}:{index}"
                tool_call_id = str(message.get("tool_call_id") or "")
                if not tool_call_id:
                    continue
                event_id = str(message.get("event_id") or "") or (
                    f"legacy:{source_id}:tool:{index}:{tool_call_id}"
                    if source_id
                    else f"legacy:tool:{index}:{tool_call_id}"
                )
                if event_id not in existing_ids:
                    await self.append_event(
                        ConversationEvent(
                            chat_id=chat_id,
                            turn_id=current_turn_id,
                            event_id=event_id,
                            role="tool",
                            kind=EventKind.TOOL_RESULT,
                            content=content,
                            tool_call_id=tool_call_id,
                            tool_name=str(message.get("tool_name") or ""),
                            timestamp=timestamp,
                            session_kind=session_kind,
                        )
                    )
                    existing_ids.add(event_id)
                    imported += 1
        return imported

    @staticmethod
    def _legacy_timestamp(message: dict[str, Any]) -> float:
        try:
            return float(message.get("timestamp") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    async def snapshot_events(
        self,
        chat_id: str,
        *,
        upto_seq: int | None = None,
        include_internal: bool = False,
        event_ids: Sequence[str] | None = None,
        turn_ids: Sequence[str] | None = None,
    ) -> ConversationEventSnapshot:
        conn = await self._ensure_open()
        async with self._lock:
            cutoff = int(
                upto_seq
                if upto_seq is not None
                else conn.execute(
                    "SELECT COALESCE(MAX(event_seq), 0) FROM conversation_events "
                    "WHERE chat_id = ?",
                    (chat_id,),
                ).fetchone()[0]
            )
            kind_clause = ""
            kind_params: tuple[str, ...] = ()
            if not include_internal:
                kind_clause = " AND kind IN (?, ?)"
                kind_params = (
                    EventKind.USER_MESSAGE,
                    EventKind.ACCEPTED_DELIVERY,
                )
            turn_clause = ""
            turn_params: tuple[str, ...] = ()
            if turn_ids is not None:
                turn_params = tuple(
                    dict.fromkeys(str(item) for item in turn_ids if item)
                )
                if not turn_params:
                    return ConversationEventSnapshot(chat_id, (), cutoff)
                turn_clause = (
                    " AND turn_id IN (" + ", ".join("?" for _ in turn_params) + ")"
                )
            if event_ids is None:
                rows = conn.execute(
                    "SELECT * FROM conversation_events WHERE chat_id = ? "
                    "AND event_seq <= ?"
                    + kind_clause
                    + turn_clause
                    + " ORDER BY event_seq",
                    (chat_id, cutoff, *kind_params, *turn_params),
                ).fetchall()
            else:
                selected_ids = tuple(
                    dict.fromkeys(str(item) for item in event_ids if item)
                )
                rows = []
                for offset in range(0, len(selected_ids), 500):
                    chunk = selected_ids[offset : offset + 500]
                    placeholders = ", ".join("?" for _ in chunk)
                    rows.extend(
                        conn.execute(
                            "SELECT * FROM conversation_events WHERE chat_id = ? "
                            "AND event_seq <= ? AND event_id IN ("
                            + placeholders
                            + ")"
                            + kind_clause
                            + turn_clause,
                            (chat_id, cutoff, *chunk, *kind_params, *turn_params),
                        ).fetchall()
                    )
                rows.sort(key=lambda row: int(row["event_seq"]))
        events = tuple(self._event_from_row(row) for row in rows)
        if not include_internal:
            events = tuple(event for event in events if event.kind in _VISIBLE_KINDS)
        return ConversationEventSnapshot(chat_id, events, cutoff)

    async def event_ids(
        self,
        chat_id: str,
        *,
        upto_seq: int | None = None,
        include_internal: bool = False,
    ) -> tuple[str, ...]:
        """Read only event identities for projection planning, never event bodies."""
        conn = await self._ensure_open()
        async with self._lock:
            cutoff = int(
                upto_seq
                if upto_seq is not None
                else conn.execute(
                    "SELECT COALESCE(MAX(event_seq), 0) FROM conversation_events "
                    "WHERE chat_id = ?",
                    (chat_id,),
                ).fetchone()[0]
            )
            params: tuple[Any, ...] = (chat_id, cutoff)
            kind_clause = ""
            if not include_internal:
                kind_clause = " AND kind IN (?, ?)"
                params += (EventKind.USER_MESSAGE, EventKind.ACCEPTED_DELIVERY)
            rows = conn.execute(
                "SELECT event_id FROM conversation_events WHERE chat_id = ? "
                "AND event_seq <= ?" + kind_clause + " ORDER BY event_seq",
                params,
            ).fetchall()
        return tuple(str(row["event_id"]) for row in rows)

    async def latest_event_seq(self, chat_id: str) -> int:
        """Return the current watermark without materializing event contents."""
        conn = await self._ensure_open()
        async with self._lock:
            return int(
                conn.execute(
                    "SELECT COALESCE(MAX(event_seq), 0) FROM conversation_events "
                    "WHERE chat_id = ?",
                    (chat_id,),
                ).fetchone()[0]
            )

    async def snapshot_turns(
        self,
        chat_id: str,
        *,
        upto_turn_sequence: int | None = None,
        include_internal: bool = False,
    ) -> ConversationTurnSnapshot:
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                "SELECT * FROM conversation_turns WHERE chat_id = ? "
                "AND (? IS NULL OR turn_sequence <= ?) ORDER BY turn_sequence",
                (chat_id, upto_turn_sequence, upto_turn_sequence),
            ).fetchall()
            cutoff = int(
                conn.execute(
                    "SELECT COALESCE(MAX(event_seq), 0) FROM conversation_events "
                    "WHERE chat_id = ?",
                    (chat_id,),
                ).fetchone()[0]
            )
        turns = tuple(self._turn_from_row(row) for row in rows)
        if not include_internal:
            async with self._lock:
                visible_rows = conn.execute(
                    "SELECT DISTINCT turn_id FROM conversation_events "
                    "WHERE chat_id = ? AND kind IN (?, ?)",
                    (chat_id, EventKind.USER_MESSAGE, EventKind.ACCEPTED_DELIVERY),
                ).fetchall()
            visible_turn_ids = {str(row["turn_id"]) for row in visible_rows}
            turns = tuple(turn for turn in turns if turn.turn_id in visible_turn_ids)
        return ConversationTurnSnapshot(chat_id, turns, cutoff)

    async def validate_turn(self, turn_id: str) -> TurnIntegrityReport:
        conn = await self._ensure_open()
        async with self._lock:
            turn_row = conn.execute(
                "SELECT * FROM conversation_turns WHERE turn_id = ?", (turn_id,)
            ).fetchone()
            rows = conn.execute(
                "SELECT * FROM conversation_events WHERE turn_id = ? ORDER BY event_seq",
                (turn_id,),
            ).fetchall()
        if turn_row is None:
            return TurnIntegrityReport(turn_id, False, "", 0, reason="turn_not_found")
        events = tuple(self._event_from_row(row) for row in rows)
        turn_status = TurnStatus(turn_row["status"])
        terminal_events = tuple(
            event for event in events if event.kind is EventKind.TURN_TERMINAL
        )
        call_ids: list[str] = []
        result_ids: list[str] = []
        duplicate_results: list[str] = []
        seen_calls: set[str] = set()
        seen_results: set[str] = set()
        call_positions: dict[str, int] = {}
        reason = ""
        if turn_status in _TERMINAL_STATUSES:
            if len(terminal_events) != 1:
                reason = (
                    "terminal_event_missing"
                    if not terminal_events
                    else "duplicate_terminal_event"
                )
            elif terminal_events[0].terminal_status != turn_status.value:
                reason = "terminal_status_mismatch"
        elif terminal_events:
            reason = "terminal_event_on_open_turn"
        for position, event in enumerate(events):
            if event.kind is EventKind.ASSISTANT_TOOL_CALL:
                try:
                    event.to_wire()
                except (AttributeError, TypeError, ValueError, EventLogInvariantError):
                    reason = reason or "assistant_wire_uncompilable"
                    continue
                calls = event.tool_calls or (
                    ({"id": event.tool_call_id, "function": {"name": event.tool_name}},)
                    if event.tool_call_id
                    else ()
                )
                for call in calls:
                    call_id = str(call.get("id") or "")
                    if not call_id:
                        reason = "tool_call_missing_id"
                    elif call_id in seen_calls:
                        reason = "duplicate_tool_call_id"
                    else:
                        seen_calls.add(call_id)
                        call_ids.append(call_id)
                        call_positions[call_id] = position
            elif event.kind is EventKind.TOOL_RESULT:
                try:
                    event.to_wire()
                except (AttributeError, TypeError, ValueError, EventLogInvariantError):
                    reason = reason or "tool_wire_uncompilable"
                call_id = event.tool_call_id
                result_ids.append(call_id)
                if call_id in seen_results:
                    duplicate_results.append(call_id)
                    reason = reason or "duplicate_tool_result"
                seen_results.add(call_id)
                if call_id not in seen_calls:
                    reason = reason or "tool_result_without_call"
                elif position <= call_positions[call_id]:
                    reason = reason or "tool_result_before_call"
        missing = tuple(call_id for call_id in call_ids if call_id not in seen_results)
        if missing and TurnStatus(turn_row["status"]) in _TERMINAL_STATUSES:
            reason = reason or "missing_tool_result"
        valid = not reason and not duplicate_results
        if valid and events:
            expected_turn_sequence = events[0].turn_sequence
            if any(event.turn_sequence != expected_turn_sequence for event in events):
                valid = False
                reason = "turn_sequence_mismatch"
        return TurnIntegrityReport(
            turn_id=turn_id,
            valid=valid,
            status=turn_row["status"],
            event_count=len(events),
            tool_call_ids=tuple(call_ids),
            tool_result_ids=tuple(result_ids),
            missing_tool_result_ids=missing,
            duplicate_tool_result_ids=tuple(duplicate_results),
            reason=reason,
        )

    async def chat_ids(self) -> list[str]:
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                "SELECT DISTINCT chat_id FROM conversation_events ORDER BY chat_id"
            ).fetchall()
        return [str(row[0]) for row in rows]

    async def session_summary(self, chat_id: str) -> dict[str, Any]:
        """Return non-content counters for lists and diagnostics."""
        snapshot = await self.snapshot_events(chat_id, include_internal=True)
        visible = [event for event in snapshot.events if not event.is_internal]
        return {
            "message_count": len(visible),
            "event_count": len(snapshot.events),
            "protocol_count": sum(1 for event in snapshot.events if event.is_internal),
            "wire_count": sum(
                1
                for event in snapshot.events
                if event.kind in {EventKind.ASSISTANT_TOOL_CALL, EventKind.TOOL_RESULT}
            ),
            "estimated_tokens": sum(event.token_count for event in snapshot.events),
            "last_activity": max(
                (event.timestamp for event in snapshot.events), default=0.0
            ),
        }

    async def history(
        self,
        chat_id: str,
        *,
        include_internal: bool = False,
        turn_ids: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return a WebUI-safe serialized ledger view."""
        snapshot = await self.snapshot_events(
            chat_id, include_internal=include_internal, turn_ids=turn_ids
        )
        return [event.to_history_dict() for event in snapshot.events]

    async def protocol_turn_index(self, chat_id: str) -> list[dict[str, Any]]:
        """Return protocol Turn metadata without materializing event bodies."""
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                """
                SELECT turn_id, MAX(turn_sequence) AS turn_sequence,
                       MIN(source_date) AS source_date, COUNT(*) AS event_count
                  FROM conversation_events
                 WHERE chat_id = ? AND kind IN (?, ?)
                 GROUP BY turn_id
                 ORDER BY turn_sequence DESC, turn_id DESC
                """,
                (chat_id, EventKind.ASSISTANT_TOOL_CALL, EventKind.TOOL_RESULT),
            ).fetchall()
        return [
            {
                "turn_id": str(row["turn_id"]),
                "turn_sequence": int(row["turn_sequence"]),
                "source_date": str(row["source_date"]),
                "event_count": int(row["event_count"]),
            }
            for row in rows
        ]

    async def protocol_snapshot(self, turn_id: str) -> tuple[ConversationEvent, ...]:
        """Return one turn's assistant/tool wire events from the core ledger."""
        if not turn_id:
            return ()
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                "SELECT * FROM conversation_events WHERE turn_id = ? "
                "AND kind IN (?, ?) ORDER BY event_seq",
                (
                    turn_id,
                    EventKind.ASSISTANT_TOOL_CALL,
                    EventKind.TOOL_RESULT,
                ),
            ).fetchall()
        events = tuple(self._event_from_row(row) for row in rows)
        if events:
            report = await self.validate_turn(turn_id)
            if not report.valid and report.status in _TERMINAL_STATUSES:
                raise EventLogInvariantError(
                    f"cannot compile invalid terminal turn protocol: {turn_id}"
                )
        return self._canonical_protocol_events(events)

    @staticmethod
    def _canonical_protocol_events(
        events: Sequence[ConversationEvent],
    ) -> tuple[ConversationEvent, ...]:
        """Keep assistant call groups intact while ordering results by call order."""
        canonical: list[ConversationEvent] = []
        index = 0
        while index < len(events):
            assistant = events[index]
            if assistant.kind is not EventKind.ASSISTANT_TOOL_CALL:
                index += 1
                continue
            canonical.append(assistant)
            call_ids = tuple(
                str(call.get("id") or "")
                for call in assistant.tool_calls
                if call.get("id")
            ) or ((assistant.tool_call_id,) if assistant.tool_call_id else ())
            index += 1
            group_results: dict[str, ConversationEvent] = {}
            while index < len(events) and events[index].kind is EventKind.TOOL_RESULT:
                result = events[index]
                group_results[result.tool_call_id] = result
                index += 1
            canonical.extend(
                group_results[call_id]
                for call_id in call_ids
                if call_id in group_results
            )
        return tuple(canonical)

    async def protocol_wire(self, turn_id: str) -> list[dict[str, Any]]:
        """Compile one ledger turn into provider assistant/tool messages."""
        return [event.to_wire() for event in await self.protocol_snapshot(turn_id)]

    async def clear_chat(self, chat_id: str) -> None:
        """Clear ledger facts for an explicit session reset."""
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "DELETE FROM conversation_events WHERE chat_id = ?", (chat_id,)
                )
                conn.execute(
                    "DELETE FROM conversation_turns WHERE chat_id = ?", (chat_id,)
                )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
