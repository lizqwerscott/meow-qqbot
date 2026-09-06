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
    chat_id: str = ""
    token_count: int = 0

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

    def to_history_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "turn_id": self.turn_id,
        }
        if self.tool_calls:
            result["tool_calls"] = [dict(call) for call in self.tool_calls]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_name:
            result["tool_name"] = self.tool_name
        if self.reasoning_content:
            result["reasoning_content"] = self.reasoning_content
        return result


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
                    chat_id TEXT NOT NULL DEFAULT '',
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
                    token_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (chat_id, turn_id, seq),
                    UNIQUE (chat_id, turn_id, event_id)
                )
                """)
            columns = {
                row["name"]
                for row in self._conn.execute(
                    "PRAGMA table_info(turn_protocol_history)"
                ).fetchall()
            }
            if "chat_id" not in columns:
                self._conn.execute(
                    "ALTER TABLE turn_protocol_history ADD COLUMN chat_id TEXT NOT NULL DEFAULT ''"
                )
            if "token_count" not in columns:
                self._conn.execute(
                    "ALTER TABLE turn_protocol_history ADD COLUMN token_count INTEGER NOT NULL DEFAULT 0"
                )
                self._conn.execute("""
                    UPDATE turn_protocol_history
                       SET token_count = (
                           LENGTH(content)
                           + LENGTH(tool_calls)
                           + LENGTH(reasoning_content)
                       ) / 4
                    """)
            primary_key = {
                row["name"]: int(row["pk"])
                for row in self._conn.execute(
                    "PRAGMA table_info(turn_protocol_history)"
                ).fetchall()
                if int(row["pk"])
            }
            if primary_key != {"chat_id": 1, "turn_id": 2, "seq": 3}:
                self._conn.execute(
                    "ALTER TABLE turn_protocol_history "
                    "RENAME TO turn_protocol_history_legacy"
                )
                self._conn.execute("""
                    CREATE TABLE turn_protocol_history (
                        chat_id TEXT NOT NULL DEFAULT '',
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
                        token_count INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (chat_id, turn_id, seq),
                        UNIQUE (chat_id, turn_id, event_id)
                    )
                    """)
                self._conn.execute("""
                    INSERT INTO turn_protocol_history
                        (chat_id, turn_id, seq, event_id, role, content,
                         tool_call_id, tool_name, tool_calls, reasoning_content,
                         timestamp, token_count)
                    SELECT chat_id, turn_id, seq, event_id, role, content,
                           tool_call_id, tool_name, tool_calls,
                           reasoning_content, timestamp, token_count
                      FROM turn_protocol_history_legacy
                    """)
                self._conn.execute("DROP TABLE turn_protocol_history_legacy")
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
            chat_id=row["chat_id"] if "chat_id" in row.keys() else "",
            token_count=(
                int(row["token_count"])
                if "token_count" in row.keys()
                else TurnProtocolHistory._estimate_tokens(
                    row["content"], row["tool_calls"], row["reasoning_content"]
                )
            ),
        )

    @staticmethod
    def _estimate_tokens(
        content: str, tool_calls: str = "", reasoning_content: str = ""
    ) -> int:
        return max(
            0,
            (len(content or "") + len(tool_calls or "") + len(reasoning_content or ""))
            // 4,
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
        chat_id: str = "",
    ) -> ProtocolEvent:
        calls = tuple(dict(call) for call in tool_calls)
        return await self._append(
            turn_id=turn_id,
            chat_id=chat_id,
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
        chat_id: str = "",
    ) -> ProtocolEvent:
        if not tool_call_id:
            raise ProtocolInvariantError("tool result requires tool_call_id")
        conn = await self._ensure_open()
        async with self._lock:
            assistant_calls = conn.execute(
                """
                SELECT tool_calls FROM turn_protocol_history
                 WHERE chat_id = ? AND turn_id = ? AND role = 'assistant'
                """,
                (chat_id, turn_id),
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
                "SELECT * FROM turn_protocol_history "
                "WHERE chat_id = ? AND turn_id = ? AND event_id = ?",
                (chat_id, turn_id, event_id),
            ).fetchone()
            if duplicate is not None:
                return self._event(duplicate)
            existing_result = conn.execute(
                """
                SELECT 1 FROM turn_protocol_history
                 WHERE chat_id = ? AND turn_id = ?
                   AND role = 'tool' AND tool_call_id = ?
                """,
                (chat_id, turn_id, tool_call_id),
            ).fetchone()
            if existing_result is not None:
                raise ProtocolInvariantError(f"duplicate tool result: {tool_call_id}")
            return await self._append_locked(
                conn,
                turn_id=turn_id,
                chat_id=chat_id,
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
        chat_id: str = "",
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
                chat_id=chat_id,
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
        chat_id: str = "",
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
            "SELECT * FROM turn_protocol_history "
            "WHERE chat_id = ? AND turn_id = ? AND event_id = ?",
            (chat_id, turn_id, event_id),
        ).fetchone()
        if existing is not None:
            return self._event(existing)
        seq = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM turn_protocol_history "
            "WHERE chat_id = ? AND turn_id = ?",
            (chat_id, turn_id),
        ).fetchone()[0]
        now = time.time() if timestamp is None else timestamp
        token_count = self._estimate_tokens(
            content,
            json.dumps(list(tool_calls), ensure_ascii=False, sort_keys=True),
            reasoning_content,
        )
        conn.execute(
            """
            INSERT INTO turn_protocol_history
                (turn_id, chat_id, seq, event_id, role, content, tool_call_id,
                 tool_name, tool_calls, reasoning_content, timestamp, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn_id,
                chat_id,
                seq,
                event_id,
                role,
                content,
                tool_call_id,
                tool_name,
                json.dumps(list(tool_calls), ensure_ascii=False, sort_keys=True),
                reasoning_content,
                now,
                token_count,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM turn_protocol_history "
            "WHERE chat_id = ? AND turn_id = ? AND event_id = ?",
            (chat_id, turn_id, event_id),
        ).fetchone()
        return self._event(row)

    async def snapshot(
        self, turn_id: str, *, chat_id: str | None = None
    ) -> tuple[ProtocolEvent, ...]:
        conn = await self._ensure_open()
        async with self._lock:
            params: tuple[Any, ...] = (turn_id,)
            scope = ""
            if chat_id is not None:
                scope = " AND chat_id = ?"
                params += (chat_id,)
            else:
                chat_rows = conn.execute(
                    "SELECT DISTINCT chat_id FROM turn_protocol_history "
                    "WHERE turn_id = ?",
                    (turn_id,),
                ).fetchall()
                if len(chat_rows) > 1:
                    raise ProtocolInvariantError(
                        "chat_id is required for an ambiguous turn"
                    )
            rows = conn.execute(
                "SELECT * FROM turn_protocol_history WHERE turn_id = ?"
                + scope
                + " ORDER BY seq",
                params,
            ).fetchall()
        return tuple(self._event(row) for row in rows)

    @staticmethod
    def to_wire_messages(events: Sequence[ProtocolEvent]) -> list[dict[str, Any]]:
        """Project a frozen turn snapshot without exposing storage metadata."""
        return [event.to_wire() for event in events]

    async def snapshot_wire(
        self, turn_id: str, *, chat_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Read one turn's protocol snapshot in provider wire form."""
        return self.to_wire_messages(await self.snapshot(turn_id, chat_id=chat_id))

    async def history(
        self, chat_id: str, max_events: int | None = None
    ) -> list[dict[str, Any]]:
        """Return protocol events for one chat in chronological order."""
        conn = await self._ensure_open()
        if max_events is not None:
            max_events = int(max_events)
            if max_events <= 0:
                return []
        async with self._lock:
            if max_events is None:
                rows = conn.execute(
                    """
                    SELECT * FROM turn_protocol_history
                     WHERE chat_id = ? ORDER BY timestamp, turn_id, seq
                    """,
                    (chat_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM turn_protocol_history
                     WHERE chat_id = ? ORDER BY timestamp DESC, turn_id DESC, seq DESC
                     LIMIT ?
                    """,
                    (chat_id, max_events),
                ).fetchall()
                rows.reverse()
        events = [self._event(row) for row in rows]
        return [event.to_history_dict() for event in events]

    async def chat_ids(self) -> list[str]:
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute("""
                SELECT DISTINCT chat_id FROM turn_protocol_history
                 WHERE chat_id != '' ORDER BY chat_id
                """).fetchall()
        return [str(row["chat_id"]) for row in rows]

    async def session_summary(self, chat_id: str) -> dict[str, int | float]:
        conn = await self._ensure_open()
        async with self._lock:
            row = conn.execute(
                """
                SELECT COUNT(*) AS message_count,
                       COALESCE(MAX(timestamp), 0) AS last_activity,
                       COALESCE(SUM(token_count), 0) AS estimated_tokens
                  FROM turn_protocol_history
                 WHERE chat_id = ?
                """,
                (chat_id,),
            ).fetchone()
        return {
            "message_count": int(row["message_count"]),
            "last_activity": float(row["last_activity"]),
            "estimated_tokens": int(row["estimated_tokens"]),
        }

    async def delete_chat(self, chat_id: str) -> None:
        """Delete all protocol events belonging to one chat."""
        if not chat_id:
            raise ValueError("chat_id is required")
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                "DELETE FROM turn_protocol_history WHERE chat_id = ?", (chat_id,)
            )
            conn.commit()

    async def claim_orphan_turns(self, chat_id: str, turn_ids: Sequence[str]) -> int:
        """Attach legacy rows whose turn ID identifies a message in this chat."""
        if not chat_id or not turn_ids:
            return 0
        normalized_ids = tuple(
            dict.fromkeys(str(turn_id) for turn_id in turn_ids if turn_id)
        )
        if not normalized_ids:
            return 0
        conn = await self._ensure_open()
        placeholders = ", ".join("?" for _ in normalized_ids)
        async with self._lock:
            cursor = conn.execute(
                f"""
                UPDATE turn_protocol_history
                   SET chat_id = ?
                 WHERE chat_id = '' AND turn_id IN ({placeholders})
                """,
                (chat_id, *normalized_ids),
            )
            conn.commit()
            return cursor.rowcount

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
