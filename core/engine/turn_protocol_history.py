"""Turn-scoped assistant/tool protocol history.

This projection is intentionally separate from the shared conversation
 timeline. It stores only protocol data needed to continue one turn and never
 participates in ordinary conversation snapshots.
"""

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


class ProtocolInvariantError(RuntimeError):
    """Raised when a tool result cannot be paired with a known tool call."""


@dataclass(frozen=True)
class ProtocolEvent:
    turn_id: str
    seq: int
    event_id: str
    role: str
    content: str
    tool_call_id: str = ""
    tool_name: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    reasoning_content: str = ""
    timestamp: float = 0.0

    def to_wire(self) -> dict[str, Any]:
        """Project this protocol event into the provider message shape."""
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
        raise ProtocolInvariantError(f"invalid protocol role: {self.role}")


class TurnProtocolHistory:
    """Durable, turn-isolated protocol event store."""

    def __init__(self, path: str = "data/turn_protocol_history.sqlite3"):
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
                CREATE TABLE IF NOT EXISTS turn_protocol_history (
                    turn_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL DEFAULT '',
                    tool_name TEXT NOT NULL DEFAULT '',
                    tool_calls TEXT NOT NULL DEFAULT '[]',
                    reasoning_content TEXT NOT NULL DEFAULT '',
                    timestamp REAL NOT NULL,
                    PRIMARY KEY (turn_id, seq),
                    UNIQUE (turn_id, event_id)
                )
                """)
            self._conn.commit()
        return self._conn

    @staticmethod
    def _event(row: sqlite3.Row) -> ProtocolEvent:
        try:
            calls = json.loads(row["tool_calls"] or "[]")
        except json.JSONDecodeError as exc:
            raise ProtocolInvariantError("invalid persisted tool_calls JSON") from exc
        if not isinstance(calls, list) or not all(
            isinstance(call, dict) for call in calls
        ):
            raise ProtocolInvariantError(
                "persisted tool_calls must be a list of objects"
            )
        return ProtocolEvent(
            turn_id=row["turn_id"],
            seq=int(row["seq"]),
            event_id=row["event_id"],
            role=row["role"],
            content=row["content"],
            tool_call_id=row["tool_call_id"],
            tool_name=row["tool_name"],
            tool_calls=tuple(calls),
            reasoning_content=row["reasoning_content"],
            timestamp=float(row["timestamp"]),
        )

    async def append_assistant(
        self,
        *,
        turn_id: str,
        event_id: str,
        content: str,
        tool_calls: Sequence[dict[str, Any]] = (),
        reasoning_content: str = "",
        timestamp: float | None = None,
    ) -> ProtocolEvent:
        calls = tuple(dict(call) for call in tool_calls)
        return await self._append(
            turn_id=turn_id,
            event_id=event_id,
            role="assistant",
            content=content,
            tool_calls=calls,
            reasoning_content=reasoning_content,
            timestamp=timestamp,
        )

    async def append_tool_result(
        self,
        *,
        turn_id: str,
        event_id: str,
        tool_call_id: str,
        tool_name: str,
        content: str,
        timestamp: float | None = None,
    ) -> ProtocolEvent:
        if not tool_call_id:
            raise ProtocolInvariantError("tool result requires tool_call_id")
        conn = await self._ensure_open()
        async with self._lock:
            assistant_calls = conn.execute(
                """
                SELECT tool_calls FROM turn_protocol_history
                 WHERE turn_id = ? AND role = 'assistant'
                """,
                (turn_id,),
            ).fetchall()
            known_ids = {
                call.get("id")
                for row in assistant_calls
                for call in json.loads(row["tool_calls"] or "[]")
                if isinstance(call, dict)
            }
            if tool_call_id not in known_ids:
                raise ProtocolInvariantError(
                    f"tool result has no assistant call: {tool_call_id}"
                )
            duplicate = conn.execute(
                "SELECT * FROM turn_protocol_history WHERE turn_id = ? AND event_id = ?",
                (turn_id, event_id),
            ).fetchone()
            if duplicate is not None:
                return self._event(duplicate)
            existing_result = conn.execute(
                """
                SELECT 1 FROM turn_protocol_history
                 WHERE turn_id = ? AND role = 'tool' AND tool_call_id = ?
                """,
                (turn_id, tool_call_id),
            ).fetchone()
            if existing_result is not None:
                raise ProtocolInvariantError(f"duplicate tool result: {tool_call_id}")
            return await self._append_locked(
                conn,
                turn_id=turn_id,
                event_id=event_id,
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                timestamp=timestamp,
            )

    async def _append(
        self,
        *,
        turn_id: str,
        event_id: str,
        role: str,
        content: str,
        tool_calls: Sequence[dict[str, Any]] = (),
        reasoning_content: str = "",
        tool_call_id: str = "",
        tool_name: str = "",
        timestamp: float | None = None,
    ) -> ProtocolEvent:
        conn = await self._ensure_open()
        async with self._lock:
            return await self._append_locked(
                conn,
                turn_id=turn_id,
                event_id=event_id,
                role=role,
                content=content,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                timestamp=timestamp,
            )

    async def _append_locked(
        self,
        conn: sqlite3.Connection,
        *,
        turn_id: str,
        event_id: str,
        role: str,
        content: str,
        tool_calls: Sequence[dict[str, Any]] = (),
        reasoning_content: str = "",
        tool_call_id: str = "",
        tool_name: str = "",
        timestamp: float | None = None,
    ) -> ProtocolEvent:
        if not turn_id or not event_id:
            raise ValueError("turn_id and event_id are required")
        if role not in {"assistant", "tool"}:
            raise ValueError(f"invalid protocol role: {role}")
        existing = conn.execute(
            "SELECT * FROM turn_protocol_history WHERE turn_id = ? AND event_id = ?",
            (turn_id, event_id),
        ).fetchone()
        if existing is not None:
            return self._event(existing)
        seq = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM turn_protocol_history WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()[0]
        now = time.time() if timestamp is None else timestamp
        conn.execute(
            """
            INSERT INTO turn_protocol_history
                (turn_id, seq, event_id, role, content, tool_call_id,
                 tool_name, tool_calls, reasoning_content, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn_id,
                seq,
                event_id,
                role,
                content,
                tool_call_id,
                tool_name,
                json.dumps(list(tool_calls), ensure_ascii=False, sort_keys=True),
                reasoning_content,
                now,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM turn_protocol_history WHERE turn_id = ? AND event_id = ?",
            (turn_id, event_id),
        ).fetchone()
        return self._event(row)

    async def snapshot(self, turn_id: str) -> tuple[ProtocolEvent, ...]:
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                "SELECT * FROM turn_protocol_history WHERE turn_id = ? ORDER BY seq",
                (turn_id,),
            ).fetchall()
        return tuple(self._event(row) for row in rows)

    @staticmethod
    def to_wire_messages(events: Sequence[ProtocolEvent]) -> list[dict[str, Any]]:
        """Project a frozen turn snapshot without exposing storage metadata."""
        return [event.to_wire() for event in events]

    async def snapshot_wire(self, turn_id: str) -> list[dict[str, Any]]:
        """Read one turn's protocol snapshot in provider wire form."""
        return self.to_wire_messages(await self.snapshot(turn_id))

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
