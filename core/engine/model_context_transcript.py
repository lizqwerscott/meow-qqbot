"""Durable, scope-aware provider context projection."""

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Sequence

from core.managers.session_manager import InboundIntent


class ModelContextInvariantError(RuntimeError):
    """Raised when a provider context projection would be invalid."""


class ModelContextLimitError(ModelContextInvariantError):
    """Raised when a projection cannot be read within its configured bound."""


class ModelContextConcurrentMutationError(ModelContextInvariantError):
    """Raised when compaction races with a newer transcript append."""


class ModelContextCompactionInProgressError(ModelContextInvariantError):
    """Raised when a scope already has an active compaction operation."""


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
    operation: str = "append"

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


@dataclass(frozen=True)
class ModelContextCompressionResult:
    scope: ModelContextScope
    snapshot: ModelContextSnapshot
    tier: int
    changed: bool
    operation: str
    before_tokens: int
    after_tokens: int
    saved_tokens: int
    reason: str = ""
    usage: Optional[dict[str, Any]] = None
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class ModelContextUsage:
    """Provider usage attached to one projection generation."""

    scope: ModelContextScope
    provider: str
    model: str
    prompt_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    usage_present: bool
    source: str = "provider"
    turn_id: str = ""
    elapsed_ms: float = 0.0
    recorded_at: float = 0.0


@dataclass(frozen=True)
class ModelContextIncident:
    scope: ModelContextScope
    kind: str
    provider: str = ""
    model: str = ""
    recovered: bool = False
    detail: str = ""
    elapsed_ms: float = 0.0
    recorded_at: float = 0.0


class ModelContextTranscript:
    """SQLite-backed model context with explicit scope and generation."""

    SCHEMA_VERSION = 7

    def __init__(
        self,
        path: str = "data/model_context_transcript.sqlite3",
        *,
        max_events: int = 512,
        max_tokens: int = 24000,
        compaction_enabled: bool = False,
        compaction_tier1_ratio: float = 0.60,
        compaction_tier2_ratio: float = 0.80,
        compaction_tier3_ratio: float = 0.95,
        compaction_keep_recent_tokens: int = 4096,
        compaction_snip_max_chars: int = 1200,
        compaction_max_summary_tokens: int = 500,
    ):
        self._path = path
        self._max_events = max(1, int(max_events))
        self._max_tokens = max(1, int(max_tokens))
        self._compaction_enabled = bool(compaction_enabled)
        self._tier1_ratio = float(compaction_tier1_ratio)
        self._tier2_ratio = float(compaction_tier2_ratio)
        self._tier3_ratio = float(compaction_tier3_ratio)
        if not 0 < self._tier1_ratio < self._tier2_ratio < self._tier3_ratio <= 1:
            raise ValueError("invalid model context compaction tier ratios")
        self._compaction_keep_recent_tokens = max(1, int(compaction_keep_recent_tokens))
        self._compaction_snip_max_chars = max(64, int(compaction_snip_max_chars))
        self._compaction_max_summary_tokens = max(1, int(compaction_max_summary_tokens))
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._schema_status = "uninitialized"
        self._repair_report: dict[str, int | str] = {}

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
                    operation TEXT NOT NULL DEFAULT 'append',
                    PRIMARY KEY (
                        chat_id, principal_id, task_correlation_id, kind,
                        generation, seq
                    ),
                    UNIQUE (
                        chat_id, principal_id, task_correlation_id, kind,
                        generation, event_id
                    )
                );
                CREATE TABLE IF NOT EXISTS model_context_compactions (
                    operation_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    task_correlation_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_event_ids TEXT NOT NULL DEFAULT '[]',
                    replacement_event_id TEXT NOT NULL DEFAULT '',
                    before_tokens INTEGER NOT NULL DEFAULT 0,
                    after_tokens INTEGER NOT NULL DEFAULT 0,
                    saved_tokens INTEGER NOT NULL DEFAULT 0,
                    reason TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    summary_prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    summary_completion_tokens INTEGER NOT NULL DEFAULT 0,
                    elapsed_ms REAL NOT NULL DEFAULT 0,
                    usage_json TEXT NOT NULL DEFAULT '{{}}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_context_usage (
                    observation_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    task_correlation_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    provider TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_miss_tokens INTEGER NOT NULL DEFAULT 0,
                    usage_present INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'provider',
                    turn_id TEXT NOT NULL DEFAULT '',
                    elapsed_ms REAL NOT NULL DEFAULT 0,
                    recorded_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_context_incidents (
                    incident_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    task_correlation_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    incident_kind TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    recovered INTEGER NOT NULL DEFAULT 0,
                    detail TEXT NOT NULL DEFAULT '',
                    elapsed_ms REAL NOT NULL DEFAULT 0,
                    recorded_at REAL NOT NULL
                );
                """.format(schema_version=self.SCHEMA_VERSION))
            try:
                schema_row = self._conn.execute(
                    "SELECT version FROM model_context_schema ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
                version = int(schema_row["version"]) if schema_row else 0
                if version > self.SCHEMA_VERSION or version < 1:
                    raise ModelContextInvariantError(
                        f"unsupported model context schema version: {version}"
                    )
                if version == 1:
                    actual_events = {
                        row["name"]
                        for row in self._conn.execute(
                            "PRAGMA table_info(model_context_events)"
                        )
                    }
                    if "operation" not in actual_events:
                        self._conn.execute(
                            "ALTER TABLE model_context_events "
                            "ADD COLUMN operation TEXT NOT NULL DEFAULT 'append'"
                        )
                    self._conn.execute(
                        "UPDATE model_context_schema SET version = ?",
                        (self.SCHEMA_VERSION,),
                    )
                elif version == 2:
                    self._conn.execute(
                        "UPDATE model_context_schema SET version = ?",
                        (self.SCHEMA_VERSION,),
                    )
                if version < 4:
                    actual_usage = {
                        row["name"]
                        for row in self._conn.execute(
                            "PRAGMA table_info(model_context_usage)"
                        )
                    }
                    if "elapsed_ms" not in actual_usage:
                        self._conn.execute(
                            "ALTER TABLE model_context_usage "
                            "ADD COLUMN elapsed_ms REAL NOT NULL DEFAULT 0"
                        )
                    self._conn.execute(
                        "UPDATE model_context_schema SET version = ?",
                        (4,),
                    )
                if version < 5:
                    self._conn.execute(
                        "UPDATE model_context_schema SET version = ?",
                        (self.SCHEMA_VERSION,),
                    )
                if version < 7:
                    actual_compactions = {
                        row["name"]
                        for row in self._conn.execute(
                            "PRAGMA table_info(model_context_compactions)"
                        )
                    }
                    for column, definition in (
                        ("summary_prompt_tokens", "INTEGER NOT NULL DEFAULT 0"),
                        ("summary_completion_tokens", "INTEGER NOT NULL DEFAULT 0"),
                        ("elapsed_ms", "REAL NOT NULL DEFAULT 0"),
                        ("usage_json", "TEXT NOT NULL DEFAULT '{}'"),
                    ):
                        if column not in actual_compactions:
                            self._conn.execute(
                                "ALTER TABLE model_context_compactions "
                                f"ADD COLUMN {column} {definition}"
                            )
                    self._conn.execute(
                        "UPDATE model_context_schema SET version = ?",
                        (self.SCHEMA_VERSION,),
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
                        "operation",
                    },
                    "model_context_usage": {
                        "observation_id",
                        "chat_id",
                        "principal_id",
                        "task_correlation_id",
                        "kind",
                        "generation",
                        "provider",
                        "model",
                        "prompt_tokens",
                        "cache_hit_tokens",
                        "cache_miss_tokens",
                        "usage_present",
                        "source",
                        "turn_id",
                        "elapsed_ms",
                        "recorded_at",
                    },
                    "model_context_compactions": {
                        "operation_id",
                        "chat_id",
                        "principal_id",
                        "task_correlation_id",
                        "kind",
                        "generation",
                        "operation",
                        "status",
                        "source_event_ids",
                        "replacement_event_id",
                        "before_tokens",
                        "after_tokens",
                        "saved_tokens",
                        "reason",
                        "error",
                        "summary_prompt_tokens",
                        "summary_completion_tokens",
                        "elapsed_ms",
                        "usage_json",
                        "created_at",
                        "updated_at",
                    },
                    "model_context_incidents": {
                        "incident_id",
                        "chat_id",
                        "principal_id",
                        "task_correlation_id",
                        "kind",
                        "generation",
                        "incident_kind",
                        "provider",
                        "model",
                        "recovered",
                        "detail",
                        "elapsed_ms",
                        "recorded_at",
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
                self._repair_report = self._repair_locked(self._conn)
                self._conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS
                        idx_model_context_active_compaction
                    ON model_context_compactions
                        (chat_id, principal_id, task_correlation_id, kind)
                    WHERE status = 'started'
                    """)
                self._conn.commit()
                self._schema_status = "ready"
            except Exception:
                self._conn.close()
                self._conn = None
                self._schema_status = "error"
                raise
        return self._conn

    def _repair_locked(self, conn: sqlite3.Connection) -> dict[str, int | str]:
        abandoned = conn.execute(
            """
            UPDATE model_context_compactions
               SET status = 'abandoned', error = 'process restarted',
                   updated_at = ?
             WHERE status = 'started'
            """,
            (time.time(),),
        ).rowcount
        orphan_events = int(conn.execute("""
                SELECT COUNT(*)
                  FROM model_context_events AS events
             LEFT JOIN model_context_scopes AS scopes
                    ON scopes.chat_id = events.chat_id
                   AND scopes.principal_id = events.principal_id
                   AND scopes.task_correlation_id = events.task_correlation_id
                   AND scopes.kind = events.kind
                   AND scopes.generation = events.generation
                 WHERE scopes.chat_id IS NULL
                """).fetchone()[0])
        invalid_events = 0
        invalid_pairs = 0
        scope_rows = conn.execute("""
            SELECT chat_id, principal_id, task_correlation_id, kind, generation
              FROM model_context_scopes
            """).fetchall()
        for scope_row in scope_rows:
            rows = conn.execute(
                """
                SELECT * FROM model_context_events
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ? AND generation = ?
                 ORDER BY seq
                """,
                (
                    scope_row["chat_id"],
                    scope_row["principal_id"],
                    scope_row["task_correlation_id"],
                    scope_row["kind"],
                    scope_row["generation"],
                ),
            ).fetchall()
            events = []
            for row in rows:
                try:
                    events.append(self._row_event(row))
                except ModelContextInvariantError:
                    invalid_events += 1
            if len(events) != len(rows):
                continue
            try:
                self._validate_history(events)
            except ModelContextInvariantError:
                invalid_pairs += 1
        return {
            "status": "ready",
            "abandoned_compaction_count": int(abandoned),
            "orphan_event_count": orphan_events,
            "invalid_event_count": invalid_events,
            "invalid_pair_count": invalid_pairs,
            "fallback_count": invalid_events + invalid_pairs + orphan_events,
        }

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
            operation=row["operation"] if "operation" in row.keys() else "append",
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

    async def generation_compatible(
        self,
        scope: ModelContextScope,
        fingerprint: str,
        *,
        provider_identity: Optional[str] = None,
    ) -> bool:
        """Check prompt identity without changing generation or events."""
        current = await self.current_scope(scope)
        conn = await self._ensure_open()
        async with self._lock:
            row = conn.execute(
                """
                SELECT fingerprint FROM model_context_scopes
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ?
                """,
                self._scope_values(current),
            ).fetchone()
        if row is None or not row["fingerprint"]:
            return True
        existing_base, existing_provider = self._fingerprint_parts(row["fingerprint"])
        if fingerprint and existing_base and existing_base != fingerprint:
            return False
        return provider_identity is None or existing_provider == provider_identity

    async def snapshot(
        self,
        scope: ModelContextScope,
        *,
        max_events: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelContextSnapshot:
        event_limit = max_events or self._max_events
        token_limit = max_tokens or self._max_tokens
        current, events = await self._read_events(scope, limit=event_limit + 1)
        if len(events) > event_limit:
            raise ModelContextLimitError("model context event limit exceeded")
        estimated_tokens = sum(self._estimate_wire_tokens(event) for event in events)
        if estimated_tokens > token_limit:
            raise ModelContextLimitError("model context token limit exceeded")
        self._validate_history(events)
        return ModelContextSnapshot(current, tuple(events))

    async def record_provider_usage(
        self,
        scope: ModelContextScope,
        usage: Optional[dict[str, Any]],
        *,
        provider: str = "",
        model: str = "",
        source: str = "provider",
        turn_id: str = "",
        elapsed_ms: float = 0.0,
    ) -> ModelContextUsage:
        """Persist one provider observation for the scope generation."""
        current = await self.current_scope(scope)
        usage = usage or {}
        prompt_tokens = max(0, int(usage.get("prompt_tokens", 0) or 0))
        cache_hit_tokens = max(0, int(usage.get("prompt_cache_hit_tokens", 0) or 0))
        cache_miss_tokens = max(0, int(usage.get("prompt_cache_miss_tokens", 0) or 0))
        usage_present = (
            "prompt_cache_hit_tokens" in usage and "prompt_cache_miss_tokens" in usage
        )
        recorded_at = time.time()
        observation_id = (
            f"usage:{current.generation}:{turn_id or 'unknown'}:{time.time_ns()}"
        )
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                """
                INSERT INTO model_context_usage
                    (observation_id, chat_id, principal_id, task_correlation_id,
                     kind, generation, provider, model, prompt_tokens,
                     cache_hit_tokens, cache_miss_tokens, usage_present, source,
                     turn_id, elapsed_ms, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    *self._scope_values(current),
                    current.generation,
                    provider,
                    model,
                    prompt_tokens,
                    cache_hit_tokens,
                    cache_miss_tokens,
                    int(usage_present),
                    source,
                    turn_id,
                    max(0.0, float(elapsed_ms or 0.0)),
                    recorded_at,
                ),
            )
            conn.commit()
        return ModelContextUsage(
            scope=current,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            cache_hit_tokens=cache_hit_tokens,
            cache_miss_tokens=cache_miss_tokens,
            usage_present=usage_present,
            source=source,
            turn_id=turn_id,
            elapsed_ms=max(0.0, float(elapsed_ms or 0.0)),
            recorded_at=recorded_at,
        )

    async def latest_provider_usage(
        self,
        scope: ModelContextScope,
        *,
        provider: Optional[str] = None,
    ) -> Optional[ModelContextUsage]:
        current = await self.current_scope(scope)
        conn = await self._ensure_open()
        async with self._lock:
            query = """
                SELECT * FROM model_context_usage
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ? AND generation = ?
            """
            params: tuple[Any, ...] = (
                *self._scope_values(current),
                current.generation,
            )
            if provider:
                query += " AND provider = ?"
                params += (provider,)
            query += " ORDER BY recorded_at DESC LIMIT 1"
            row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        return ModelContextUsage(
            scope=current,
            provider=row["provider"],
            model=row["model"],
            prompt_tokens=int(row["prompt_tokens"]),
            cache_hit_tokens=int(row["cache_hit_tokens"]),
            cache_miss_tokens=int(row["cache_miss_tokens"]),
            usage_present=bool(row["usage_present"]),
            source=row["source"],
            turn_id=row["turn_id"],
            elapsed_ms=float(row["elapsed_ms"]),
            recorded_at=float(row["recorded_at"]),
        )

    async def record_incident(
        self,
        scope: ModelContextScope,
        incident_kind: str,
        *,
        provider: str = "",
        model: str = "",
        recovered: bool = False,
        detail: str = "",
        elapsed_ms: float = 0.0,
    ) -> ModelContextIncident:
        current = await self.current_scope(scope)
        recorded_at = time.time()
        incident = ModelContextIncident(
            scope=current,
            kind=incident_kind,
            provider=provider,
            model=model,
            recovered=bool(recovered),
            detail=str(detail)[:2000],
            elapsed_ms=max(0.0, float(elapsed_ms or 0.0)),
            recorded_at=recorded_at,
        )
        incident_id = f"incident:{current.generation}:{time.time_ns()}"
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                """
                INSERT INTO model_context_incidents
                    (incident_id, chat_id, principal_id, task_correlation_id,
                     kind, generation, incident_kind, provider, model, recovered,
                     detail, elapsed_ms, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    incident_id,
                    *self._scope_values(current),
                    current.generation,
                    incident.kind,
                    incident.provider,
                    incident.model,
                    int(incident.recovered),
                    incident.detail,
                    incident.elapsed_ms,
                    incident.recorded_at,
                ),
            )
            conn.commit()
        return incident

    async def repair(self) -> dict[str, int | str]:
        """Validate persisted events and recover interrupted compactions."""
        conn = await self._ensure_open()
        async with self._lock:
            self._repair_report = self._repair_locked(conn)
            conn.commit()
            return dict(self._repair_report)

    async def _read_events(
        self,
        scope: ModelContextScope,
        *,
        limit: Optional[int] = None,
    ) -> tuple[ModelContextScope, tuple[ModelContextEvent, ...]]:
        current = await self.current_scope(scope)
        conn = await self._ensure_open()
        async with self._lock:
            values = (*self._scope_values(current), current.generation)
            query = """
                SELECT * FROM model_context_events
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ? AND generation = ?
                 ORDER BY seq
            """
            params: tuple[Any, ...] = values
            if limit is not None:
                query += " LIMIT ?"
                params += (limit,)
            rows = conn.execute(query, params).fetchall()
        return current, tuple(self._row_event(row) for row in rows)

    @classmethod
    def _snapshot_tokens(cls, events: Sequence[ModelContextEvent]) -> int:
        return sum(cls._estimate_wire_tokens(event) for event in events)

    @staticmethod
    def _group_events_by_turn(
        events: Sequence[ModelContextEvent],
    ) -> list[tuple[str, list[ModelContextEvent]]]:
        groups: list[tuple[str, list[ModelContextEvent]]] = []
        indexes: dict[str, int] = {}
        for event in events:
            turn_id = event.source_turn_id or event.event_id
            index = indexes.get(turn_id)
            if index is None:
                indexes[turn_id] = len(groups)
                groups.append((turn_id, [event]))
            else:
                groups[index][1].append(event)
        return groups

    def _protected_turn_ids(
        self,
        events: Sequence[ModelContextEvent],
    ) -> set[str]:
        protected: set[str] = set()
        total = 0
        for turn_id, turn_events in reversed(self._group_events_by_turn(events)):
            protected.add(turn_id)
            total += self._snapshot_tokens(turn_events)
            if total >= self._compaction_keep_recent_tokens:
                break
        return protected

    def _snip_content(self, content: str) -> str:
        limit = self._compaction_snip_max_chars
        if len(content) <= limit:
            return content
        head = max(1, limit // 2 - 32)
        tail = max(1, limit - head - 64)
        omitted = len(content) - head - tail
        return (
            f"{content[:head]}\n"
            f"[tool result snipped: {omitted} characters omitted]\n"
            f"{content[-tail:]}"
        )

    def _local_replacements(
        self,
        events: Sequence[ModelContextEvent],
        *,
        force: bool = False,
    ) -> tuple[list[ModelContextEvent], set[str], str]:
        before_tokens = self._snapshot_tokens(events)
        tier1_limit = 0 if force else self._max_tokens * self._tier1_ratio
        tier2_limit = 0 if force else self._max_tokens * self._tier2_ratio
        if before_tokens < tier1_limit:
            return list(events), set(), ""
        protected = self._protected_turn_ids(events)
        replacements: list[ModelContextEvent] = []
        operations: set[str] = set()
        current_tokens = before_tokens
        for event in events:
            if (
                current_tokens < tier1_limit
                or event.source_turn_id in protected
                or event.role != "tool"
                or event.operation != "append"
            ):
                replacements.append(event)
                continue
            content = self._snip_content(event.content)
            if content == event.content:
                replacements.append(event)
                continue
            replacement = replace(
                event,
                event_id=f"snip:{event.event_id}",
                content=content,
                compacted=True,
                operation="snip",
            )
            replacements.append(replacement)
            operations.add("snip")
            current_tokens -= self._estimate_wire_tokens(event)
            current_tokens += self._estimate_wire_tokens(replacement)

        if current_tokens >= tier2_limit:
            for index, event in enumerate(replacements):
                if (
                    current_tokens < tier2_limit
                    or event.source_turn_id in protected
                    or event.role != "tool"
                    or event.operation == "prune"
                ):
                    continue
                replacement = replace(
                    event,
                    event_id=f"prune:{event.event_id}",
                    content="[tool result pruned to save context space]",
                    compacted=True,
                    operation="prune",
                )
                replacements[index] = replacement
                operations.add("prune")
                current_tokens -= self._estimate_wire_tokens(event)
                current_tokens += self._estimate_wire_tokens(replacement)
        return replacements, operations, "local waterline compaction"

    async def _replace_generation(
        self,
        scope: ModelContextScope,
        events: Sequence[ModelContextEvent],
        *,
        operation: str,
        reason: str,
        before_tokens: int,
        operation_id: Optional[str] = None,
        source_event_ids: Sequence[str] = (),
        replacement_event_id: str = "",
        expected_event_ids: Sequence[str] = (),
        usage: Optional[dict[str, Any]] = None,
        elapsed_ms: float = 0.0,
    ) -> ModelContextSnapshot:
        current = await self.current_scope(scope)
        if current.generation != scope.generation:
            raise ModelContextInvariantError("stale model context generation")
        new_scope = replace(current, generation=current.generation + 1)
        normalized = tuple(
            replace(event, scope=new_scope, seq=seq)
            for seq, event in enumerate(events, start=1)
        )
        self._validate_history(normalized)
        after_tokens = self._snapshot_tokens(normalized)
        usage = usage if isinstance(usage, dict) else {}
        summary_prompt_tokens = max(0, int(usage.get("prompt_tokens", 0) or 0))
        summary_completion_tokens = max(0, int(usage.get("completion_tokens", 0) or 0))
        elapsed_ms = max(0.0, float(elapsed_ms or 0.0))
        operation_id = operation_id or f"{operation}:{time.time_ns()}"
        conn = await self._ensure_open()
        values = self._scope_values(current)
        try:
            async with self._lock:
                scope_row = conn.execute(
                    """
                    SELECT generation FROM model_context_scopes
                     WHERE chat_id = ? AND principal_id = ?
                       AND task_correlation_id = ? AND kind = ?
                    """,
                    values,
                ).fetchone()
                actual_generation = (
                    int(scope_row["generation"]) if scope_row is not None else 0
                )
                if actual_generation != current.generation:
                    raise ModelContextConcurrentMutationError(
                        "model context generation changed during compaction"
                    )
                if expected_event_ids:
                    actual_event_ids = tuple(
                        row["event_id"]
                        for row in conn.execute(
                            """
                            SELECT event_id FROM model_context_events
                             WHERE chat_id = ? AND principal_id = ?
                               AND task_correlation_id = ? AND kind = ?
                               AND generation = ?
                             ORDER BY seq
                            """,
                            (*values, current.generation),
                        ).fetchall()
                    )
                    if actual_event_ids != tuple(expected_event_ids):
                        raise ModelContextConcurrentMutationError(
                            "model context changed during compaction"
                        )
                conn.execute(
                    """
                    DELETE FROM model_context_events
                     WHERE chat_id = ? AND principal_id = ?
                       AND task_correlation_id = ? AND kind = ? AND generation = ?
                    """,
                    (*values, current.generation),
                )
                for event in normalized:
                    self._insert_event_locked(conn, event)
                conn.execute(
                    """
                    UPDATE model_context_scopes
                       SET generation = ?, updated_at = ?
                     WHERE chat_id = ? AND principal_id = ?
                       AND task_correlation_id = ? AND kind = ?
                    """,
                    (new_scope.generation, time.time(), *values),
                )
                now = time.time()
                conn.execute(
                    """
                    INSERT INTO model_context_compactions
                        (operation_id, chat_id, principal_id, task_correlation_id,
                        kind, generation, operation, status, source_event_ids,
                        replacement_event_id,
                        before_tokens, after_tokens, saved_tokens, reason,
                        summary_prompt_tokens, summary_completion_tokens,
                        elapsed_ms, usage_json,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'committed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(operation_id) DO UPDATE SET
                        status = 'committed', generation = excluded.generation,
                        source_event_ids = excluded.source_event_ids,
                        replacement_event_id = excluded.replacement_event_id,
                        before_tokens = excluded.before_tokens,
                        after_tokens = excluded.after_tokens,
                        saved_tokens = excluded.saved_tokens,
                        reason = excluded.reason,
                        summary_prompt_tokens = excluded.summary_prompt_tokens,
                        summary_completion_tokens = excluded.summary_completion_tokens,
                        elapsed_ms = excluded.elapsed_ms,
                        usage_json = excluded.usage_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        operation_id,
                        *values,
                        new_scope.generation,
                        operation,
                        json.dumps(list(source_event_ids), ensure_ascii=False),
                        replacement_event_id,
                        before_tokens,
                        after_tokens,
                        max(0, before_tokens - after_tokens),
                        reason,
                        summary_prompt_tokens,
                        summary_completion_tokens,
                        elapsed_ms,
                        json.dumps(usage, ensure_ascii=False, sort_keys=True),
                        now,
                        now,
                    ),
                )
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        return ModelContextSnapshot(new_scope, normalized)

    async def begin_compaction(
        self,
        scope: ModelContextScope,
        *,
        operation: str,
        source_event_ids: Sequence[str] = (),
        reason: str = "",
    ) -> str:
        current = await self.current_scope(scope)
        operation_id = f"{operation}:{current.generation}:{time.time_ns()}"
        conn = await self._ensure_open()
        values = self._scope_values(current)
        now = time.time()
        async with self._lock:
            active = conn.execute(
                """
                SELECT operation_id FROM model_context_compactions
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ?
                   AND status = 'started'
                 LIMIT 1
                """,
                values,
            ).fetchone()
            if active is not None:
                raise ModelContextCompactionInProgressError(
                    f"compaction already running: {active['operation_id']}"
                )
            try:
                conn.execute(
                    """
                    INSERT INTO model_context_compactions
                        (operation_id, chat_id, principal_id, task_correlation_id,
                         kind, generation, operation, status, source_event_ids,
                         reason, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'started', ?, ?, ?, ?)
                    """,
                    (
                        operation_id,
                        *values,
                        current.generation,
                        operation,
                        json.dumps(list(source_event_ids), ensure_ascii=False),
                        reason,
                        now,
                        now,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise ModelContextCompactionInProgressError(
                    "compaction already running"
                ) from exc
        return operation_id

    async def fail_compaction(self, operation_id: str, error: str) -> None:
        if not operation_id:
            return
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute(
                """
                UPDATE model_context_compactions
                   SET status = 'failed', error = ?, updated_at = ?
                 WHERE operation_id = ? AND status = 'started'
                """,
                (str(error)[:2000], time.time(), operation_id),
            )
            conn.commit()

    async def compact_if_needed(
        self,
        scope: ModelContextScope,
        *,
        provider_usage: Optional[ModelContextUsage] = None,
        summary_factory: Optional[
            Callable[
                [Sequence[dict[str, Any]], int],
                Awaitable[
                    str
                    | tuple[str, Optional[dict[str, Any]]]
                    | tuple[str, Optional[dict[str, Any]], float]
                ],
            ]
        ] = None,
        force: bool = False,
    ) -> ModelContextCompressionResult:
        if provider_usage is None:
            provider_usage = await self.latest_provider_usage(scope)
        current, events = await self._read_events(scope)
        estimated_before = self._snapshot_tokens(events)
        observed_prompt_tokens = (
            provider_usage.prompt_tokens
            if provider_usage is not None and provider_usage.prompt_tokens > 0
            else None
        )
        before_tokens = observed_prompt_tokens or estimated_before
        waterline_source = "provider usage" if observed_prompt_tokens else "estimate"

        def _after_local_tokens(local_events: Sequence[ModelContextEvent]) -> int:
            estimated_after = self._snapshot_tokens(local_events)
            if observed_prompt_tokens is None:
                return estimated_after
            return max(
                0,
                observed_prompt_tokens - max(0, estimated_before - estimated_after),
            )

        if not self._compaction_enabled and not force:
            return ModelContextCompressionResult(
                current,
                ModelContextSnapshot(current, events),
                0,
                False,
                "none",
                before_tokens,
                before_tokens,
                0,
                f"disabled ({waterline_source})",
            )
        if not force and before_tokens < self._max_tokens * self._tier1_ratio:
            return ModelContextCompressionResult(
                current,
                ModelContextSnapshot(current, events),
                0,
                False,
                "none",
                before_tokens,
                before_tokens,
                0,
                f"below_tier1 ({waterline_source})",
            )

        local_events, operations, reason = self._local_replacements(events, force=force)
        local_changed = bool(operations)
        local_scope = current
        local_snapshot = ModelContextSnapshot(current, tuple(events))
        if local_changed:
            operation = "snip_prune" if len(operations) > 1 else next(iter(operations))
            local_source_event_ids = tuple(
                source_id
                for before, after in zip(events, local_events)
                if before.event_id != after.event_id
                for source_id in (before.source_event_ids or (before.event_id,))
            )
            try:
                local_snapshot = await self._replace_generation(
                    current,
                    local_events,
                    operation=operation,
                    reason=reason,
                    before_tokens=before_tokens,
                    source_event_ids=local_source_event_ids,
                    expected_event_ids=tuple(event.event_id for event in events),
                )
            except ModelContextConcurrentMutationError as exc:
                fresh_scope, fresh_events = await self._read_events(scope)
                return ModelContextCompressionResult(
                    fresh_scope,
                    ModelContextSnapshot(fresh_scope, fresh_events),
                    0,
                    False,
                    "none",
                    self._snapshot_tokens(fresh_events),
                    self._snapshot_tokens(fresh_events),
                    0,
                    str(exc),
                )
            local_scope = local_snapshot.scope

        after_local = _after_local_tokens(local_snapshot.events)
        if (not force and after_local < self._max_tokens * self._tier3_ratio) or (
            summary_factory is None
        ):
            tier = (
                3
                if after_local >= self._max_tokens * self._tier3_ratio
                else (
                    2
                    if local_changed
                    and after_local >= self._max_tokens * self._tier2_ratio
                    else 1
                )
            )
            return ModelContextCompressionResult(
                local_scope,
                local_snapshot,
                tier,
                local_changed,
                "snip_prune" if local_changed else "none",
                before_tokens,
                after_local,
                max(0, before_tokens - after_local),
                reason or f"local compaction only ({waterline_source})",
            )

        groups = self._group_events_by_turn(local_snapshot.events)
        protected = self._protected_turn_ids(local_snapshot.events)
        checkpoints = [
            event
            for event in local_snapshot.events
            if event.compacted and event.operation == "compact_replace"
        ]
        latest_checkpoint = max(checkpoints, key=lambda event: event.seq, default=None)
        selected = [
            (turn_id, turn_events)
            for turn_id, turn_events in groups
            if turn_id not in protected
            and (
                latest_checkpoint is None or turn_events[0].seq > latest_checkpoint.seq
            )
        ]
        if not selected:
            return ModelContextCompressionResult(
                local_scope,
                local_snapshot,
                2 if local_changed else 3,
                local_changed,
                "snip_prune" if local_changed else "none",
                before_tokens,
                after_local,
                max(0, before_tokens - after_local),
                f"no safe summary boundary ({waterline_source})",
            )
        source_turn_ids = tuple(turn_id for turn_id, _ in selected)
        source_events = tuple(
            event for _, turn_events in selected for event in turn_events
        )
        summary_input_events = (
            (latest_checkpoint,) if latest_checkpoint is not None else ()
        ) + source_events
        source_event_ids = tuple(
            source_id
            for event in source_events
            for source_id in (event.source_event_ids or (event.event_id,))
        )
        try:
            operation_id = await self.begin_compaction(
                local_scope,
                operation="compact_replace",
                source_event_ids=source_event_ids,
                reason="tier3 summary",
            )
        except ModelContextCompactionInProgressError as exc:
            fresh_scope, fresh_events = await self._read_events(local_scope)
            fresh_tokens = self._snapshot_tokens(fresh_events)
            return ModelContextCompressionResult(
                fresh_scope,
                ModelContextSnapshot(fresh_scope, fresh_events),
                3,
                False,
                "none",
                fresh_tokens,
                fresh_tokens,
                0,
                f"compaction already in progress: {exc}",
            )
        try:
            result = await summary_factory(
                [event.to_wire() for event in summary_input_events],
                self._compaction_max_summary_tokens,
            )
            if isinstance(result, tuple):
                summary = result[0] if result else ""
                usage = result[1] if len(result) > 1 else None
                summary_elapsed_ms = float(result[2] or 0.0) if len(result) > 2 else 0.0
            else:
                summary, usage = result, None
                summary_elapsed_ms = 0.0
            summary = (summary or "").strip()
            source_tokens = self._snapshot_tokens(source_events)
            if not summary or self._estimate_text_tokens(summary) >= source_tokens:
                await self.fail_compaction(
                    operation_id, "summary is empty or not smaller"
                )
                fresh_scope, fresh_events = await self._read_events(local_scope)
                return ModelContextCompressionResult(
                    fresh_scope,
                    ModelContextSnapshot(fresh_scope, fresh_events),
                    3,
                    local_changed,
                    "snip_prune" if local_changed else "none",
                    before_tokens,
                    after_local,
                    max(0, before_tokens - after_local),
                    "summary rejected",
                    usage,
                    summary_elapsed_ms,
                )
            compacted = await self.compact(
                local_scope,
                summary=summary,
                source_turn_ids=source_turn_ids,
                source_event_ids=source_event_ids,
                replacement_event_id=operation_id,
                operation_id=operation_id,
                expected_event_ids=tuple(
                    event.event_id for event in local_snapshot.events
                ),
                usage=usage,
                elapsed_ms=summary_elapsed_ms,
            )
            after_summary = self._snapshot_tokens(compacted.events)
            return ModelContextCompressionResult(
                compacted.scope,
                compacted,
                3,
                True,
                "compact_replace",
                before_tokens,
                after_summary,
                max(0, before_tokens - after_summary),
                "tier3 summary",
                usage,
                summary_elapsed_ms,
            )
        except asyncio.CancelledError:
            await self.fail_compaction(operation_id, "cancelled")
            raise
        except Exception as exc:
            await self.fail_compaction(operation_id, str(exc))
            fresh_scope, fresh_events = await self._read_events(local_scope)
            return ModelContextCompressionResult(
                fresh_scope,
                ModelContextSnapshot(fresh_scope, fresh_events),
                3,
                local_changed,
                "snip_prune" if local_changed else "none",
                before_tokens,
                after_local,
                max(0, before_tokens - after_local),
                "summary failed",
            )

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _insert_event_locked(
        conn: sqlite3.Connection, event: ModelContextEvent
    ) -> None:
        values = (
            *ModelContextTranscript._scope_values(event.scope),
            event.scope.generation,
            event.seq,
            event.event_id,
            event.role,
            event.content,
            event.source_turn_id,
            json.dumps(list(event.source_event_ids), ensure_ascii=False),
            event.tool_call_id,
            event.tool_name,
            json.dumps(list(event.tool_calls), ensure_ascii=False, sort_keys=True),
            event.reasoning_content,
            int(event.compacted),
            event.sender_id,
            event.timestamp or time.time(),
            event.operation,
        )
        conn.execute(
            """
            INSERT INTO model_context_events
                (chat_id, principal_id, task_correlation_id, kind, generation,
                 seq, event_id, role, content, source_turn_id, source_event_ids,
                 tool_call_id, tool_name, tool_calls, reasoning_content, compacted,
                 sender_id, timestamp, operation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )

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
            if len(rows) <= self._max_events and (
                self._compaction_enabled or estimated_tokens <= self._max_tokens
            ):
                return
            oldest_turn_id = next(
                (row["source_turn_id"] for row in rows if row["source_turn_id"]),
                None,
            )
            if oldest_turn_id is None:
                return
            removed = [
                self._row_event(row)
                for row in rows
                if row["source_turn_id"] == oldest_turn_id
            ]
            removed_tokens = sum(self._estimate_wire_tokens(event) for event in removed)
            source_event_ids = tuple(
                source_id
                for event in removed
                for source_id in (event.source_event_ids or (event.event_id,))
            )
            conn.execute(
                """
                DELETE FROM model_context_events
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ? AND generation = ?
                   AND source_turn_id = ?
                """,
                (*values, oldest_turn_id),
            )
            operation_id = f"event_prune:{scope.generation}:{time.time_ns()}"
            now = time.time()
            conn.execute(
                """
                INSERT INTO model_context_compactions
                    (operation_id, chat_id, principal_id, task_correlation_id,
                     kind, generation, operation, status, source_event_ids,
                     before_tokens, after_tokens, saved_tokens, reason,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'committed', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    operation_id,
                    *self._scope_values(scope),
                    scope.generation,
                    "event_prune",
                    json.dumps(list(source_event_ids), ensure_ascii=False),
                    estimated_tokens,
                    max(0, estimated_tokens - removed_tokens),
                    removed_tokens,
                    "max_events/max_tokens bound",
                    now,
                    now,
                ),
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
        operation_id: Optional[str] = None,
        expected_event_ids: Sequence[str] = (),
        usage: Optional[dict[str, Any]] = None,
        elapsed_ms: float = 0.0,
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
                operation="compact_replace",
            )
            remaining.append(summary_event)
            remaining.sort(key=lambda event: event.seq)
            self._validate_history(remaining)
        return await self._replace_generation(
            current,
            remaining,
            operation="compact_replace",
            reason="summary replacement",
            before_tokens=self._snapshot_tokens(events),
            operation_id=operation_id,
            replacement_event_id=f"compaction:{current.generation}:{replacement_event_id}",
            source_event_ids=source_event_ids,
            expected_event_ids=expected_event_ids
            or tuple(event.event_id for event in events),
            usage=usage,
            elapsed_ms=elapsed_ms,
        )

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
        if self._compaction_enabled:
            current, events = await self._read_events(current)
            return ModelContextSnapshot(current, events)
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
            conn.execute(
                """
                DELETE FROM model_context_usage
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ?
                """,
                values,
            )
            conn.execute(
                """
                DELETE FROM model_context_incidents
                 WHERE chat_id = ? AND principal_id = ?
                   AND task_correlation_id = ? AND kind = ?
                """,
                values,
            )
            conn.execute(
                """
                DELETE FROM model_context_compactions
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
            compactions = conn.execute(
                "SELECT status, COUNT(*) AS count "
                "FROM model_context_compactions GROUP BY status"
            ).fetchall()
            usage_count = int(
                conn.execute("SELECT COUNT(*) FROM model_context_usage").fetchone()[0]
            )
            usage_missing_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM model_context_usage WHERE usage_present = 0"
                ).fetchone()[0]
            )
            usage_totals = conn.execute("""
                SELECT COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(cache_hit_tokens), 0) AS cache_hit_tokens,
                       COALESCE(SUM(cache_miss_tokens), 0) AS cache_miss_tokens,
                       COALESCE(SUM(elapsed_ms), 0) AS elapsed_ms
                  FROM model_context_usage
                 WHERE usage_present = 1
                """).fetchone()
            compaction_totals = conn.execute("""
                SELECT COALESCE(SUM(summary_prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(summary_completion_tokens), 0) AS completion_tokens,
                       COALESCE(SUM(elapsed_ms), 0) AS elapsed_ms
                  FROM model_context_compactions
                 WHERE operation = 'compact_replace' AND status = 'committed'
                """).fetchone()
            overflow_count = int(conn.execute("""
                    SELECT COUNT(*) FROM model_context_incidents
                     WHERE incident_kind = 'context_overflow'
                    """).fetchone()[0])
            overflow_recovered_count = int(conn.execute("""
                    SELECT COUNT(*) FROM model_context_incidents
                     WHERE incident_kind = 'context_overflow' AND recovered = 1
                    """).fetchone()[0])
        compaction_counts = {
            str(row["status"]): int(row["count"]) for row in compactions
        }
        operation_counts = {
            str(row["operation"]): int(row["count"]) for row in conn.execute("""
                SELECT operation, COUNT(*) AS count
                  FROM model_context_compactions
                 WHERE status = 'committed'
                 GROUP BY operation
                """).fetchall()
        }
        return {
            "schema_version": self.SCHEMA_VERSION,
            "migration_status": self._schema_status,
            "scope_count": int(scopes),
            "event_count": int(events),
            "compaction_count": sum(compaction_counts.values()),
            "compaction_committed_count": compaction_counts.get("committed", 0),
            "compaction_failed_count": compaction_counts.get("failed", 0),
            "compaction_abandoned_count": compaction_counts.get("abandoned", 0),
            "compaction_snip_count": operation_counts.get("snip", 0),
            "compaction_prune_count": operation_counts.get("prune", 0),
            "compaction_event_prune_count": operation_counts.get("event_prune", 0),
            "usage_observation_count": usage_count,
            "usage_missing_count": usage_missing_count,
            "prompt_tokens": int(usage_totals["prompt_tokens"]),
            "cache_hit_tokens": int(usage_totals["cache_hit_tokens"]),
            "cache_miss_tokens": int(usage_totals["cache_miss_tokens"]),
            "cache_hit_rate": (
                round(
                    100
                    * int(usage_totals["cache_hit_tokens"])
                    / max(
                        1,
                        int(usage_totals["cache_hit_tokens"])
                        + int(usage_totals["cache_miss_tokens"]),
                    ),
                    1,
                )
                if int(usage_totals["cache_hit_tokens"])
                + int(usage_totals["cache_miss_tokens"])
                else 0.0
            ),
            "usage_elapsed_ms": round(float(usage_totals["elapsed_ms"]), 1),
            "summary_prompt_tokens": int(compaction_totals["prompt_tokens"]),
            "summary_completion_tokens": int(compaction_totals["completion_tokens"]),
            "summary_elapsed_ms": round(float(compaction_totals["elapsed_ms"]), 1),
            "overflow_count": overflow_count,
            "overflow_recovery_count": overflow_recovered_count,
            **self._repair_report,
        }

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
