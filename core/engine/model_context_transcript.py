"""Durable, scope-aware provider context projection."""

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional, Sequence

from core.managers.session_manager import InboundIntent


class ModelContextInvariantError(RuntimeError):
    """Raised when a provider context projection would be invalid."""


class ModelContextLimitError(ModelContextInvariantError):
    """Raised when a projection cannot be read within its configured bound."""


class ModelContextScopeKind(StrEnum):
    PRIVATE_CONVERSATION = InboundIntent.PRIVATE_CONVERSATION
    DIRECT_TASK = InboundIntent.DIRECT_TASK


@dataclass(frozen=True)
class ModelContextScope:
    chat_id: str
    principal_id: str
    task_correlation_id: str = ""
    generation: int = 1
    kind: ModelContextScopeKind = ModelContextScopeKind.PRIVATE_CONVERSATION

    def __post_init__(self) -> None:
        if not self.chat_id or not self.principal_id:
            raise ValueError("model context scope requires chat_id and principal_id")
        if self.generation < 1:
            raise ValueError("model context generation must be positive")
        if (
            self.kind is ModelContextScopeKind.DIRECT_TASK
            and not self.task_correlation_id
        ):
            raise ValueError("direct task scope requires task_correlation_id")
        if (
            self.kind is ModelContextScopeKind.PRIVATE_CONVERSATION
            and self.task_correlation_id
        ):
            raise ValueError("private conversation scope cannot have task correlation")

    @classmethod
    def for_intent(
        cls,
        *,
        chat_id: str,
        principal_id: str,
        intent: InboundIntent,
        task_correlation_id: str = "",
    ) -> Optional["ModelContextScope"]:
        if intent is InboundIntent.PRIVATE_CONVERSATION:
            return cls(chat_id=chat_id, principal_id=principal_id)
        if intent is InboundIntent.DIRECT_TASK and task_correlation_id:
            return cls(
                chat_id=chat_id,
                principal_id=principal_id,
                task_correlation_id=task_correlation_id,
                kind=ModelContextScopeKind.DIRECT_TASK,
            )
        return None

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.chat_id,
            self.principal_id,
            self.task_correlation_id,
            str(self.kind),
        )


@dataclass(frozen=True)
class ModelContextEvent:
    scope: ModelContextScope
    seq: int
    event_id: str
    role: str
    content: str
    source_turn_id: str = ""
    source_event_ids: tuple[str, ...] = ()
    tool_call_id: str = ""
    tool_name: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    reasoning_content: str = ""
    compacted: bool = False
    sender_id: str = ""
    timestamp: float = 0.0

    def to_wire(self) -> dict[str, Any]:
        if self.role == "user":
            timestamp = datetime.fromtimestamp(self.timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            sender = self.sender_id or "未知"
            return {
                "role": "user",
                "content": f"[{sender} 在 {timestamp}]: {self.content}",
            }
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
        raise ModelContextInvariantError(f"invalid model context role: {self.role}")


@dataclass(frozen=True)
class ModelContextSnapshot:
    scope: ModelContextScope
    events: tuple[ModelContextEvent, ...] = ()

    @property
    def source_event_ids(self) -> frozenset[str]:
        return frozenset(
            event_id for event in self.events for event_id in event.source_event_ids
        )

    def to_wire(self) -> list[dict[str, Any]]:
        return [event.to_wire() for event in self.events]


class ModelContextTranscript:
    """SQLite-backed model context with explicit scope and generation."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str = "data/model_context_transcript.sqlite3",
        *,
        max_events: int = 512,
        max_tokens: int = 24000,
    ):
        self._path = path
        self._max_events = max(1, int(max_events))
        self._max_tokens = max(1, int(max_tokens))
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._schema_status = "uninitialized"

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
                CREATE TABLE IF NOT EXISTS model_context_schema (
                    version INTEGER NOT NULL
                );
                INSERT INTO model_context_schema(version)
                    SELECT {schema_version} WHERE NOT EXISTS
                    (SELECT 1 FROM model_context_schema);
                CREATE TABLE IF NOT EXISTS model_context_scopes (
                    chat_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    task_correlation_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (chat_id, principal_id, task_correlation_id, kind)
                );
                CREATE TABLE IF NOT EXISTS model_context_events (
                    chat_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    task_correlation_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    seq INTEGER NOT NULL,
                    event_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_turn_id TEXT NOT NULL DEFAULT '',
                    source_event_ids TEXT NOT NULL DEFAULT '[]',
                    tool_call_id TEXT NOT NULL DEFAULT '',
                    tool_name TEXT NOT NULL DEFAULT '',
                    tool_calls TEXT NOT NULL DEFAULT '[]',
                    reasoning_content TEXT NOT NULL DEFAULT '',
                    compacted INTEGER NOT NULL DEFAULT 0,
                    sender_id TEXT NOT NULL DEFAULT '',
                    timestamp REAL NOT NULL,
                    PRIMARY KEY (
                        chat_id, principal_id, task_correlation_id, kind,
                        generation, seq
                    ),
                    UNIQUE (
                        chat_id, principal_id, task_correlation_id, kind,
                        generation, event_id
                    )
                );
                """.format(schema_version=self.SCHEMA_VERSION))
            try:
                schema_row = self._conn.execute(
                    "SELECT version FROM model_context_schema ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
                version = int(schema_row["version"]) if schema_row else 0
                if version != self.SCHEMA_VERSION:
                    raise ModelContextInvariantError(
                        f"unsupported model context schema version: {version}"
                    )
                required_columns = {
                    "model_context_scopes": {
                        "chat_id",
                        "principal_id",
                        "task_correlation_id",
                        "kind",
                        "generation",
                        "fingerprint",
                    },
                    "model_context_events": {
                        "chat_id",
                        "principal_id",
                        "task_correlation_id",
                        "kind",
                        "generation",
                        "seq",
                        "event_id",
                        "role",
                        "content",
                        "source_turn_id",
                        "source_event_ids",
                        "tool_call_id",
                        "tool_name",
                        "tool_calls",
                        "reasoning_content",
                        "compacted",
                        "sender_id",
                        "timestamp",
                    },
                }
                for table, columns in required_columns.items():
                    actual = {
                        row["name"]
                        for row in self._conn.execute(f"PRAGMA table_info({table})")
                    }
                    if not columns.issubset(actual):
                        raise ModelContextInvariantError(
                            f"incomplete model context schema: {table}"
                        )
                self._conn.commit()
                self._schema_status = "ready"
            except Exception:
                self._conn.close()
                self._conn = None
                self._schema_status = "error"
                raise
        return self._conn

    @staticmethod
    def _scope_values(scope: ModelContextScope) -> tuple[str, str, str, str]:
        return (
            scope.chat_id,
            scope.principal_id,
            scope.task_correlation_id,
            str(scope.kind),
        )

    @classmethod
    def _row_event(cls, row: sqlite3.Row) -> ModelContextEvent:
        try:
            source_event_ids = tuple(json.loads(row["source_event_ids"] or "[]"))
            tool_calls = tuple(json.loads(row["tool_calls"] or "[]"))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ModelContextInvariantError(
                "invalid persisted model context JSON"
            ) from exc
        if not all(isinstance(value, str) for value in source_event_ids):
            raise ModelContextInvariantError("source_event_ids must contain strings")
        if not all(isinstance(value, dict) for value in tool_calls):
            raise ModelContextInvariantError("tool_calls must contain objects")
        scope = ModelContextScope(
            chat_id=row["chat_id"],
            principal_id=row["principal_id"],
            task_correlation_id=row["task_correlation_id"],
            generation=int(row["generation"]),
            kind=ModelContextScopeKind(row["kind"]),
        )
        return ModelContextEvent(
            scope=scope,
            seq=int(row["seq"]),
            event_id=row["event_id"],
            role=row["role"],
            content=row["content"],
            source_turn_id=row["source_turn_id"],
            source_event_ids=source_event_ids,
            tool_call_id=row["tool_call_id"],
            tool_name=row["tool_name"],
            tool_calls=tool_calls,
            reasoning_content=row["reasoning_content"],
            compacted=bool(row["compacted"]),
            sender_id=row["sender_id"],
            timestamp=float(row["timestamp"]),
        )

    async def current_scope(self, scope: ModelContextScope) -> ModelContextScope:
        conn = await self._ensure_open()
        async with self._lock:
            values = self._scope_values(scope)
            row = conn.execute(
                """
                SELECT generation FROM model_context_scopes
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ?
                """,
                values,
            ).fetchone()
            if row is None:
                now = time.time()
                conn.execute(
                    """
                    INSERT INTO model_context_scopes
                        (chat_id, principal_id, task_correlation_id, kind,
                         generation, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (*values, scope.generation, now, now),
                )
                conn.commit()
                return scope
            return replace(scope, generation=int(row["generation"]))

    @staticmethod
    def _fingerprint_parts(fingerprint: str) -> tuple[str, str]:
        base, separator, provider = fingerprint.partition("\nprovider=")
        return base, provider if separator else ""

    async def ensure_generation(
        self,
        scope: ModelContextScope,
        fingerprint: str,
        *,
        provider_identity: Optional[str] = None,
    ) -> ModelContextScope:
        """Keep a stable prompt identity in one generation."""
        if not fingerprint and provider_identity is None:
            return await self.current_scope(scope)
        conn = await self._ensure_open()
        async with self._lock:
            values = self._scope_values(scope)
            row = conn.execute(
                """
                SELECT generation, fingerprint FROM model_context_scopes
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ?
                """,
                values,
            ).fetchone()
            if row is None:
                now = time.time()
                stored_fingerprint = (
                    f"{fingerprint}\nprovider={provider_identity}"
                    if provider_identity is not None
                    else fingerprint
                )
                conn.execute(
                    """
                    INSERT INTO model_context_scopes
                        (chat_id, principal_id, task_correlation_id, kind,
                         generation, fingerprint, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (*values, scope.generation, stored_fingerprint, now, now),
                )
                conn.commit()
                return scope
            generation = int(row["generation"])
            existing_base, existing_provider = self._fingerprint_parts(
                row["fingerprint"]
            )
            generation_changed = bool(
                fingerprint and existing_base and existing_base != fingerprint
            )
            if provider_identity is not None and existing_provider != provider_identity:
                generation_changed = True
            if generation_changed:
                conn.execute(
                    """
                    DELETE FROM model_context_events
                     WHERE chat_id = ? AND principal_id = ?
                       AND task_correlation_id = ? AND kind = ? AND generation = ?
                    """,
                    (*values, generation),
                )
                generation += 1
            if provider_identity is not None:
                stored_fingerprint = f"{fingerprint}\nprovider={provider_identity}"
            elif generation_changed or not row["fingerprint"]:
                stored_fingerprint = fingerprint
            else:
                stored_fingerprint = row["fingerprint"]
            conn.execute(
                """
                UPDATE model_context_scopes
                   SET generation = ?, fingerprint = ?, updated_at = ?
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ?
                """,
                (generation, stored_fingerprint, time.time(), *values),
            )
            conn.commit()
            return replace(scope, generation=generation)

    async def snapshot(
        self,
        scope: ModelContextScope,
        *,
        max_events: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelContextSnapshot:
        current = await self.current_scope(scope)
        conn = await self._ensure_open()
        event_limit = max_events or self._max_events
        token_limit = max_tokens or self._max_tokens
        async with self._lock:
            values = (*self._scope_values(current), current.generation)
            rows = conn.execute(
                """
                SELECT * FROM model_context_events
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ? AND generation = ?
                 ORDER BY seq LIMIT ?
                """,
                (*values, event_limit + 1),
            ).fetchall()
        if len(rows) > event_limit:
            raise ModelContextLimitError("model context event limit exceeded")
        events = tuple(self._row_event(row) for row in rows)
        estimated_tokens = sum(self._estimate_wire_tokens(event) for event in events)
        if estimated_tokens > token_limit:
            raise ModelContextLimitError("model context token limit exceeded")
        return ModelContextSnapshot(current, events)

    @staticmethod
    def _estimate_wire_tokens(event: ModelContextEvent) -> int:
        wire = json.dumps(
            event.to_wire(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return max(1, len(wire) // 4)

    def _prune_locked(
        self,
        conn: sqlite3.Connection,
        scope: ModelContextScope,
    ) -> None:
        values = (*self._scope_values(scope), scope.generation)
        while True:
            rows = conn.execute(
                """
                SELECT *
                  FROM model_context_events
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ? AND generation = ?
                 ORDER BY seq
                """,
                values,
            ).fetchall()
            estimated_tokens = sum(
                self._estimate_wire_tokens(self._row_event(row)) for row in rows
            )
            if len(rows) <= self._max_events and estimated_tokens <= self._max_tokens:
                return
            oldest_turn_id = next(
                (row["source_turn_id"] for row in rows if row["source_turn_id"]),
                None,
            )
            if oldest_turn_id is None:
                return
            conn.execute(
                """
                DELETE FROM model_context_events
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ? AND generation = ?
                   AND source_turn_id = ?
                """,
                (*values, oldest_turn_id),
            )

    @staticmethod
    def _validate_history(events: Sequence[ModelContextEvent]) -> None:
        by_turn: dict[str, list[ModelContextEvent]] = {}
        for event in events:
            by_turn.setdefault(event.source_turn_id, []).append(event)
        for turn_events in by_turn.values():
            protocol_events = [
                event for event in turn_events if event.role in {"assistant", "tool"}
            ]
            if protocol_events:
                ModelContextTranscript._validate_protocol(protocol_events)

    async def compact(
        self,
        scope: ModelContextScope,
        *,
        summary: str,
        source_turn_ids: Sequence[str],
        source_event_ids: Sequence[str] = (),
        replacement_event_id: str = "",
    ) -> ModelContextSnapshot:
        """Replace settled turns with one provider-visible summary event."""
        if not summary.strip():
            raise ValueError("compaction summary is required")
        if not source_turn_ids and not source_event_ids:
            raise ValueError("compaction sources are required")
        replacement_event_id = replacement_event_id or f"summary:{time.time_ns()}"
        current = await self.current_scope(scope)
        conn = await self._ensure_open()
        async with self._lock:
            values = (*self._scope_values(current), current.generation)
            rows = conn.execute(
                """
                SELECT * FROM model_context_events
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ? AND generation = ?
                 ORDER BY seq
                """,
                values,
            ).fetchall()
            events = [self._row_event(row) for row in rows]
            selected_turn_ids = set(source_turn_ids)
            selected_event_ids = set(source_event_ids)
            selected = [
                event
                for event in events
                if event.source_turn_id in selected_turn_ids
                or event.event_id in selected_event_ids
                or selected_event_ids.intersection(event.source_event_ids)
            ]
            if not selected:
                raise ModelContextInvariantError("compaction sources not found")
            selected_turn_ids.update(
                event.source_turn_id for event in selected if event.source_turn_id
            )
            remaining = [
                event
                for event in events
                if event.source_turn_id not in selected_turn_ids
            ]
            summary_event = ModelContextEvent(
                scope=current,
                seq=max((event.seq for event in remaining), default=0) + 1,
                event_id=f"compaction:{current.generation}:{replacement_event_id}",
                role="assistant",
                content=summary.strip(),
                source_turn_id=f"compaction:{replacement_event_id}",
                source_event_ids=tuple(
                    sorted(
                        {
                            source_id
                            for event in selected
                            for source_id in (
                                event.source_event_ids or (event.event_id,)
                            )
                        }
                    )
                ),
                compacted=True,
                timestamp=time.time(),
            )
            remaining.append(summary_event)
            remaining.sort(key=lambda event: event.seq)
            self._validate_history(remaining)
            try:
                conn.execute(
                    """
                    DELETE FROM model_context_events
                     WHERE chat_id = ? AND principal_id = ?
                       AND task_correlation_id = ? AND kind = ? AND generation = ?
                    """,
                    values,
                )
                for seq, event in enumerate(remaining, start=1):
                    conn.execute(
                        """
                        INSERT INTO model_context_events
                            (chat_id, principal_id, task_correlation_id, kind,
                             generation, seq, event_id, role, content, source_turn_id,
                             source_event_ids, tool_call_id, tool_name, tool_calls,
                             reasoning_content, compacted, sender_id, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            *self._scope_values(current),
                            current.generation,
                            seq,
                            event.event_id,
                            event.role,
                            event.content,
                            event.source_turn_id,
                            json.dumps(
                                list(event.source_event_ids), ensure_ascii=False
                            ),
                            event.tool_call_id,
                            event.tool_name,
                            json.dumps(list(event.tool_calls), ensure_ascii=False),
                            event.reasoning_content,
                            int(event.compacted),
                            event.sender_id,
                            event.timestamp or time.time(),
                        ),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return await self.snapshot(current)

    @staticmethod
    def _validate_protocol(protocol_events: Sequence[Any]) -> None:
        known_calls: set[str] = set()
        result_ids: set[str] = set()
        for event in protocol_events:
            if event.role == "assistant":
                for call in event.tool_calls:
                    call_id = call.get("id")
                    if not call_id or call_id in known_calls:
                        raise ModelContextInvariantError(
                            "invalid or duplicate assistant tool call"
                        )
                    known_calls.add(call_id)
            elif event.role == "tool":
                if not event.tool_call_id or event.tool_call_id not in known_calls:
                    raise ModelContextInvariantError(
                        "orphan tool result in model context"
                    )
                if event.tool_call_id in result_ids:
                    raise ModelContextInvariantError(
                        "duplicate tool result in model context"
                    )
                result_ids.add(event.tool_call_id)
            else:
                raise ModelContextInvariantError(f"invalid protocol role: {event.role}")
        if known_calls != result_ids:
            raise ModelContextInvariantError("incomplete assistant/tool protocol pair")

    async def append_turn(
        self,
        scope: ModelContextScope,
        *,
        turn_id: str,
        user_events: Sequence[Any],
        protocol_events: Sequence[Any],
        compacted: bool = False,
    ) -> ModelContextSnapshot:
        if not turn_id:
            raise ValueError("turn_id is required")
        if not protocol_events:
            raise ModelContextInvariantError("completed turn has no protocol events")
        self._validate_protocol(protocol_events)
        current = await self.current_scope(scope)
        if current.generation != scope.generation:
            raise ModelContextInvariantError("stale model context generation")
        conn = await self._ensure_open()
        async with self._lock:
            try:
                values = self._scope_values(current)
                next_seq = conn.execute(
                    """
                    SELECT COALESCE(MAX(seq), 0) + 1 FROM model_context_events
                     WHERE chat_id = ? AND principal_id = ?
                       AND task_correlation_id = ? AND kind = ? AND generation = ?
                    """,
                    (*values, current.generation),
                ).fetchone()[0]
                pending = []
                for event in user_events:
                    if event.role != "user":
                        raise ModelContextInvariantError(
                            "model context user event must have user role"
                        )
                    pending.append(
                        ModelContextEvent(
                            scope=current,
                            seq=next_seq,
                            event_id=f"timeline:{event.event_id}",
                            role="user",
                            content=event.content,
                            source_turn_id=turn_id,
                            source_event_ids=(event.event_id,),
                            sender_id=event.sender_id,
                            timestamp=event.timestamp,
                        )
                    )
                    next_seq += 1
                for event in protocol_events:
                    pending.append(
                        ModelContextEvent(
                            scope=current,
                            seq=next_seq,
                            event_id=f"protocol:{turn_id}:{event.event_id}",
                            role=event.role,
                            content=event.content,
                            source_turn_id=turn_id,
                            source_event_ids=(event.event_id,),
                            tool_call_id=event.tool_call_id,
                            tool_name=event.tool_name,
                            tool_calls=tuple(dict(call) for call in event.tool_calls),
                            reasoning_content=event.reasoning_content,
                            compacted=compacted,
                            timestamp=event.timestamp,
                        )
                    )
                    next_seq += 1
                for event in pending:
                    existing = conn.execute(
                        """
                        SELECT * FROM model_context_events
                         WHERE chat_id = ? AND principal_id = ?
                           AND task_correlation_id = ? AND kind = ?
                           AND generation = ? AND event_id = ?
                        """,
                        (*values, current.generation, event.event_id),
                    ).fetchone()
                    if existing is not None:
                        persisted = self._row_event(existing)
                        if persisted.to_wire() != event.to_wire():
                            raise ModelContextInvariantError(
                                f"idempotency key collision: {event.event_id}"
                            )
                        continue
                    conn.execute(
                        """
                        INSERT INTO model_context_events
                            (chat_id, principal_id, task_correlation_id, kind,
                             generation, seq, event_id, role, content, source_turn_id,
                             source_event_ids, tool_call_id, tool_name, tool_calls,
                             reasoning_content, compacted, sender_id, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            *values,
                            current.generation,
                            event.seq,
                            event.event_id,
                            event.role,
                            event.content,
                            event.source_turn_id,
                            json.dumps(
                                list(event.source_event_ids), ensure_ascii=False
                            ),
                            event.tool_call_id,
                            event.tool_name,
                            json.dumps(
                                list(event.tool_calls),
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            event.reasoning_content,
                            int(event.compacted),
                            event.sender_id,
                            event.timestamp or time.time(),
                        ),
                    )
                self._prune_locked(conn, current)
                conn.execute(
                    """
                    UPDATE model_context_scopes SET updated_at = ?
                     WHERE chat_id = ? AND principal_id = ?
                       AND task_correlation_id = ? AND kind = ?
                    """,
                    (time.time(), *values),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return await self.snapshot(current)

    async def bump_generation(
        self, scope: ModelContextScope, *, fingerprint: str = ""
    ) -> ModelContextScope:
        current = await self.current_scope(scope)
        conn = await self._ensure_open()
        async with self._lock:
            values = self._scope_values(current)
            conn.execute(
                """
                DELETE FROM model_context_events
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ? AND generation = ?
                """,
                (*values, current.generation),
            )
            conn.execute(
                """
                UPDATE model_context_scopes
                   SET generation = ?, fingerprint = ?, updated_at = ?
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ?
                """,
                (current.generation + 1, fingerprint, time.time(), *values),
            )
            conn.commit()
        return replace(current, generation=current.generation + 1)

    async def clear(self, scope: ModelContextScope) -> None:
        current = await self.current_scope(scope)
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                """
                DELETE FROM model_context_events
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ? AND generation = ?
                """,
                (*self._scope_values(current), current.generation),
            )
            conn.commit()

    async def close_scope(self, scope: ModelContextScope) -> None:
        """Delete a scope and all generations so it cannot be reused."""
        conn = await self._ensure_open()
        async with self._lock:
            values = self._scope_values(scope)
            conn.execute(
                """
                DELETE FROM model_context_events
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ?
                """,
                values,
            )
            conn.execute(
                """
                DELETE FROM model_context_scopes
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ?
                """,
                values,
            )
            conn.commit()

    async def status(self) -> dict[str, int | str]:
        conn = await self._ensure_open()
        async with self._lock:
            scopes = conn.execute(
                "SELECT COUNT(*) FROM model_context_scopes"
            ).fetchone()[0]
            events = conn.execute(
                "SELECT COUNT(*) FROM model_context_events"
            ).fetchone()[0]
        return {
            "schema_version": self.SCHEMA_VERSION,
            "migration_status": self._schema_status,
            "scope_count": int(scopes),
            "event_count": int(events),
        }

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
