"""Idempotent delivery preparation and settlement.

The ledger records delivery intent separately from transport success. A caller
must prepare a unique turn before sending, then settle it as sent or suppressed.
"""

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional, Union

from core.engine.ambient_delivery import (
    AmbientDeliveryDecision,
    decide_ambient_delivery,
)
from core.engine.conversation_event_log import (
    ConversationEvent,
    ConversationEventLog,
    EventKind,
    EventLogInvariantError,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeliveryReceipt:
    """Transport result kept separate from the local delivery state."""

    status: str
    logical_delivery_id: str = ""
    transport_id: str = ""
    platform_message_id: str = ""
    error_code: str = ""
    retryable: bool = False
    chunk_index: int = 0
    chunk_count: int = 1

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "failed", "unknown", "partial"}:
            raise ValueError(f"invalid delivery receipt status: {self.status}")
        if self.chunk_index < 0 or self.chunk_count < 1:
            raise ValueError("invalid delivery receipt chunk metadata")
        if self.chunk_index >= self.chunk_count:
            raise ValueError("delivery receipt chunk_index must be below chunk_count")


@dataclass(frozen=True)
class DeliveryRecord:
    key: str
    chat_id: str
    turn_id: str
    status: str
    reason: str
    reply_anchor_id: str
    content_hash: str
    content: str = ""
    transport_id: str = ""
    attempts: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    logical_delivery_id: str = ""
    receipt_status: str = ""
    platform_message_id: str = ""
    error_code: str = ""
    chunk_index: int = 0
    chunk_count: int = 1


@dataclass(frozen=True)
class DeliveryRecoveryResult:
    scanned: int = 0
    sent: int = 0
    retryable: int = 0
    failed: int = 0
    unknown: int = 0
    partial: int = 0


class DeliveryLedger:
    """Small SQLite ledger with idempotent keys and crash-recoverable states."""

    def __init__(self, path: str = "data/delivery_ledger.sqlite3"):
        self._path = path
        self._conn: Optional[sqlite3.Connection] = None
        import asyncio

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
                CREATE TABLE IF NOT EXISTS delivery_ledger (
                    delivery_key TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    reply_anchor_id TEXT NOT NULL DEFAULT '',
                    content_hash TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '',
                    transport_id TEXT NOT NULL DEFAULT '',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    receipt_status TEXT NOT NULL DEFAULT '',
                    logical_delivery_id TEXT NOT NULL DEFAULT '',
                    platform_message_id TEXT NOT NULL DEFAULT '',
                    error_code TEXT NOT NULL DEFAULT '',
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """)
            columns = {
                row[1]
                for row in self._conn.execute(
                    "PRAGMA table_info(delivery_ledger)"
                ).fetchall()
            }
            if "attempts" not in columns:
                self._conn.execute(
                    "ALTER TABLE delivery_ledger ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
                )
            if "content" not in columns:
                self._conn.execute(
                    "ALTER TABLE delivery_ledger ADD COLUMN content TEXT NOT NULL DEFAULT ''"
                )
            for column, definition in (
                ("receipt_status", "TEXT NOT NULL DEFAULT ''"),
                ("logical_delivery_id", "TEXT NOT NULL DEFAULT ''"),
                ("platform_message_id", "TEXT NOT NULL DEFAULT ''"),
                ("error_code", "TEXT NOT NULL DEFAULT ''"),
                ("chunk_index", "INTEGER NOT NULL DEFAULT 0"),
                ("chunk_count", "INTEGER NOT NULL DEFAULT 1"),
            ):
                if column not in columns:
                    self._conn.execute(
                        f"ALTER TABLE delivery_ledger ADD COLUMN {column} {definition}"
                    )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_delivery_chat ON delivery_ledger(chat_id, created_at)"
            )
            self._conn.commit()
        return self._conn

    @staticmethod
    def content_hash(content: str | None) -> str:
        return hashlib.sha256((content or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _record(row: sqlite3.Row) -> DeliveryRecord:
        return DeliveryRecord(
            key=row["delivery_key"],
            chat_id=row["chat_id"],
            turn_id=row["turn_id"],
            status=row["status"],
            reason=row["reason"],
            reply_anchor_id=row["reply_anchor_id"],
            content_hash=row["content_hash"],
            content=row["content"] if "content" in row.keys() else "",
            transport_id=row["transport_id"],
            attempts=int(row["attempts"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            logical_delivery_id=row["logical_delivery_id"],
            receipt_status=row["receipt_status"],
            platform_message_id=row["platform_message_id"],
            error_code=row["error_code"],
            chunk_index=int(row["chunk_index"]),
            chunk_count=int(row["chunk_count"]),
        )

    async def get(self, key: str) -> Optional[DeliveryRecord]:
        conn = await self._ensure_open()
        async with self._lock:
            row = conn.execute(
                "SELECT * FROM delivery_ledger WHERE delivery_key = ?", (key,)
            ).fetchone()
        return self._record(row) if row else None

    async def prepare(
        self,
        *,
        key: str,
        chat_id: str,
        turn_id: str,
        reason: str,
        reply_anchor_id: str,
        content_hash: str,
        status: str = "prepared",
        logical_delivery_id: str = "",
        content: str | None = None,
    ) -> DeliveryRecord:
        if status not in {"prepared", "suppressed"}:
            raise ValueError(f"invalid delivery preparation status: {status}")
        conn = await self._ensure_open()
        now = time.time()
        async with self._lock:
            conn.execute(
                """
                INSERT INTO delivery_ledger
                    (delivery_key, chat_id, turn_id, status, reason,
                     reply_anchor_id, content_hash, content, attempts, receipt_status,
                     logical_delivery_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, '', ?, ?, ?)
                ON CONFLICT(delivery_key) DO NOTHING
                """,
                (
                    key,
                    chat_id,
                    turn_id,
                    status,
                    reason,
                    reply_anchor_id,
                    content_hash,
                    content or "",
                    logical_delivery_id or key,
                    now,
                    now,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM delivery_ledger WHERE delivery_key = ?", (key,)
            ).fetchone()
        return self._record(row)

    async def settle(
        self,
        key: str,
        *,
        status: str,
        reason: str = "",
        transport_id: str = "",
        receipt_status: str = "",
        logical_delivery_id: str = "",
        platform_message_id: str = "",
        error_code: str = "",
        chunk_index: int = 0,
        chunk_count: int = 1,
    ) -> Optional[DeliveryRecord]:
        if status not in {
            "sent",
            "accepted",
            "failed",
            "unknown",
            "partial",
            "suppressed",
        }:
            raise ValueError(f"invalid delivery settlement status: {status}")
        conn = await self._ensure_open()
        now = time.time()
        async with self._lock:
            conn.execute(
                """
                UPDATE delivery_ledger
                       SET status = ?, reason = ?, transport_id = ?,
                       receipt_status = ?, logical_delivery_id = ?,
                       platform_message_id = ?,
                       error_code = ?, chunk_index = ?, chunk_count = ?,
                       updated_at = ?
                 WHERE delivery_key = ? AND status = 'prepared'
                """,
                (
                    status,
                    reason,
                    transport_id,
                    receipt_status,
                    logical_delivery_id,
                    platform_message_id,
                    error_code,
                    chunk_index,
                    chunk_count,
                    now,
                    key,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM delivery_ledger WHERE delivery_key = ?", (key,)
            ).fetchone()
        return self._record(row) if row else None

    async def stale_prepared(
        self,
        *,
        older_than: float,
        limit: int = 100,
        chat_id: str | None = None,
        key_prefix: str | None = None,
        now: float | None = None,
        retry_base_seconds: float = 30.0,
        max_attempts: int = 5,
    ) -> list[DeliveryRecord]:
        """List prepared rows old enough for transport recovery inspection."""
        conn = await self._ensure_open()
        current_time = time.time() if now is None else now
        async with self._lock:
            rows = conn.execute(
                """
                SELECT * FROM delivery_ledger
                 WHERE status = 'prepared'
                   AND updated_at < ?
                   AND (? IS NULL OR chat_id = ?)
                   AND (? IS NULL OR delivery_key LIKE ?)
                   AND attempts <= ?
                 ORDER BY updated_at
                 LIMIT ?
                """,
                (
                    float(older_than),
                    chat_id,
                    chat_id,
                    key_prefix,
                    f"{key_prefix}%" if key_prefix is not None else None,
                    max(1, max_attempts),
                    max(1, limit),
                ),
            ).fetchall()
        return [
            record
            for record in (self._record(row) for row in rows)
            if record.updated_at
            + retry_base_seconds * (2 ** max(0, record.attempts - 1))
            <= current_time
        ]

    async def note_retry(
        self,
        key: str,
        *,
        reason: str,
        receipt: DeliveryReceipt | None = None,
    ) -> Optional[DeliveryRecord]:
        """Record a retryable recovery failure without losing the prepared intent."""
        conn = await self._ensure_open()
        now = time.time()
        async with self._lock:
            conn.execute(
                """
                UPDATE delivery_ledger
                   SET reason = ?, attempts = attempts + 1,
                       transport_id = COALESCE(NULLIF(?, ''), transport_id),
                       receipt_status = COALESCE(NULLIF(?, ''), receipt_status),
                       logical_delivery_id = COALESCE(NULLIF(?, ''), logical_delivery_id),
                       platform_message_id = COALESCE(NULLIF(?, ''), platform_message_id),
                       error_code = COALESCE(NULLIF(?, ''), error_code),
                       chunk_index = ?, chunk_count = ?,
                       updated_at = ?
                 WHERE delivery_key = ? AND status = 'prepared'
                """,
                (
                    reason,
                    receipt.transport_id if receipt is not None else "",
                    receipt.status if receipt is not None else "",
                    receipt.logical_delivery_id if receipt is not None else "",
                    receipt.platform_message_id if receipt is not None else "",
                    receipt.error_code if receipt is not None else "",
                    receipt.chunk_index if receipt is not None else 0,
                    receipt.chunk_count if receipt is not None else 1,
                    now,
                    key,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM delivery_ledger WHERE delivery_key = ?", (key,)
            ).fetchone()
        return self._record(row) if row else None

    async def status_counts(self) -> dict[str, int]:
        """Return status counts for health checks without exposing message text."""
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM delivery_ledger GROUP BY status"
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


class DeliveryController:
    """Bind ambient policy decisions to durable, idempotent delivery records."""

    def __init__(
        self,
        ledger: DeliveryLedger,
        *,
        retry_base_seconds: float = 30.0,
        max_attempts: int = 5,
        timeline=None,
        event_log: Optional[ConversationEventLog] = None,
        audit_delivery: Optional[Callable[[str, str, str], Awaitable[None]]] = None,
    ):
        self.ledger = ledger
        self.timeline = timeline
        self.event_log = event_log
        self.retry_base_seconds = max(1.0, retry_base_seconds)
        self.max_attempts = max(1, max_attempts)
        self._audit_delivery = audit_delivery

    async def _audit(self, record: DeliveryRecord, status: str) -> None:
        if self._audit_delivery is not None:
            await self._audit_delivery(
                record.turn_id, record.logical_delivery_id or record.key, status
            )

    async def prepare_ambient(
        self,
        *,
        chat_id: str,
        turn_id: str,
        content: str | None,
        delivery_mode: str,
        tool_delivered: bool = False,
        reply_anchor_id: str = "",
    ) -> tuple[AmbientDeliveryDecision, Optional[DeliveryRecord]]:
        decision = decide_ambient_delivery(
            content,
            delivery_mode=delivery_mode,
            tool_delivered=tool_delivered,
            reply_anchor_id=reply_anchor_id,
        )
        key = f"ambient:{chat_id}:{turn_id}"
        record = await self.ledger.prepare(
            key=key,
            chat_id=chat_id,
            turn_id=turn_id,
            reason=decision.reason,
            reply_anchor_id=reply_anchor_id,
            content_hash=self.ledger.content_hash(content),
            status="prepared" if decision.should_deliver else "suppressed",
            logical_delivery_id=key,
            content=None,
        )
        await self._audit(record, record.status)
        if record.status != "prepared" and decision.should_deliver:
            decision = AmbientDeliveryDecision(
                False,
                reason="already_settled",
                reply_anchor_id=reply_anchor_id,
            )
        return decision, record

    async def recover_prepared(
        self,
        *,
        older_than: float,
        content_resolver: Callable[[DeliveryRecord], Awaitable[Optional[str]]],
        transport: Callable[
            [DeliveryRecord, str], Awaitable[Union[Optional[str], DeliveryReceipt]]
        ],
        limit: int = 100,
        chat_id: str | None = None,
        key_prefix: str | None = None,
        retry_base_seconds: float | None = None,
        max_attempts: int | None = None,
        allow_transport_retry: bool = False,
    ) -> DeliveryRecoveryResult:
        """Recover stale records without retrying transport by default."""
        effective_max_attempts = (
            self.max_attempts if max_attempts is None else max(1, max_attempts)
        )
        records = await self.ledger.stale_prepared(
            older_than=older_than,
            limit=limit,
            chat_id=chat_id,
            key_prefix=key_prefix,
            retry_base_seconds=(
                self.retry_base_seconds
                if retry_base_seconds is None
                else max(1.0, retry_base_seconds)
            ),
            max_attempts=(
                self.max_attempts if max_attempts is None else max(1, max_attempts)
            ),
        )
        result = DeliveryRecoveryResult(scanned=len(records))
        for record in records:
            if record.attempts >= effective_max_attempts:
                await self.ledger.settle(
                    record.key,
                    status="failed",
                    reason="recovery_attempt_limit",
                )
                result = DeliveryRecoveryResult(
                    result.scanned, result.sent, result.retryable, result.failed + 1
                )
                continue
            if not allow_transport_retry:
                settled = await self.ledger.settle(
                    record.key,
                    status="unknown",
                    reason="recovery_requires_idempotency",
                    receipt_status="unknown",
                    logical_delivery_id=record.logical_delivery_id or record.key,
                )
                if settled is not None:
                    await self._audit(settled, settled.status)
                result = DeliveryRecoveryResult(
                    result.scanned,
                    result.sent,
                    result.retryable,
                    result.failed,
                    result.unknown + 1,
                    result.partial,
                )
                continue
            try:
                content = await content_resolver(record)
            except Exception:
                await self.ledger.note_retry(
                    record.key, reason="recovery_resolver_error"
                )
                result = DeliveryRecoveryResult(
                    result.scanned, result.sent, result.retryable + 1, result.failed
                )
                continue
            if content is None:
                await self.ledger.settle(
                    record.key,
                    status="failed",
                    reason="recovery_content_unavailable",
                )
                result = DeliveryRecoveryResult(
                    result.scanned, result.sent, result.retryable, result.failed + 1
                )
                continue
            if self.ledger.content_hash(content) != record.content_hash:
                await self.ledger.settle(
                    record.key,
                    status="failed",
                    reason="recovery_content_hash_mismatch",
                )
                result = DeliveryRecoveryResult(
                    result.scanned, result.sent, result.retryable, result.failed + 1
                )
                continue
            try:
                transport_result = await transport(record, content)
            except Exception as exc:
                await self.settle_receipt(
                    record,
                    DeliveryReceipt(
                        status="unknown",
                        logical_delivery_id=record.logical_delivery_id or record.key,
                        error_code=type(exc).__name__,
                        retryable=False,
                    ),
                    content=content,
                )
                result = DeliveryRecoveryResult(
                    result.scanned,
                    result.sent,
                    result.retryable,
                    result.failed,
                    result.unknown + 1,
                    result.partial,
                )
                continue
            if isinstance(transport_result, DeliveryReceipt):
                if transport_result.status == "failed" and transport_result.retryable:
                    retried = await self.ledger.note_retry(
                        record.key,
                        reason=transport_result.error_code
                        or "transport_failed_retryable",
                        receipt=transport_result,
                    )
                    if retried is not None:
                        await self._audit(retried, retried.status)
                    result = DeliveryRecoveryResult(
                        result.scanned,
                        result.sent,
                        result.retryable + 1,
                        result.failed,
                        result.unknown,
                        result.partial,
                    )
                elif transport_result.status == "accepted":
                    await self.settle_receipt(record, transport_result, content=content)
                    result = DeliveryRecoveryResult(
                        result.scanned,
                        result.sent + 1,
                        result.retryable,
                        result.failed,
                        result.unknown,
                        result.partial,
                    )
                elif transport_result.status == "unknown":
                    await self.settle_receipt(record, transport_result, content=content)
                    result = DeliveryRecoveryResult(
                        result.scanned,
                        result.sent,
                        result.retryable,
                        result.failed,
                        result.unknown + 1,
                        result.partial,
                    )
                elif transport_result.status == "partial":
                    await self.settle_receipt(record, transport_result, content=content)
                    result = DeliveryRecoveryResult(
                        result.scanned,
                        result.sent,
                        result.retryable,
                        result.failed,
                        result.unknown,
                        result.partial + 1,
                    )
                elif transport_result.retryable:
                    await self.settle_receipt(record, transport_result, content=content)
                    result = DeliveryRecoveryResult(
                        result.scanned,
                        result.sent,
                        result.retryable + 1,
                        result.failed,
                        result.unknown,
                        result.partial,
                    )
                else:
                    await self.settle_receipt(record, transport_result, content=content)
                    result = DeliveryRecoveryResult(
                        result.scanned,
                        result.sent,
                        result.retryable,
                        result.failed + 1,
                        result.unknown,
                        result.partial,
                    )
                continue
            await self.mark_sent(
                record, transport_id=transport_result or "", content=content
            )
            result = DeliveryRecoveryResult(
                result.scanned,
                result.sent + 1,
                result.retryable,
                result.failed,
                result.unknown,
                result.partial,
            )
        return result

    async def deliver_text(
        self,
        *,
        delivery_id: str,
        chat_id: str,
        content: str,
        callback: Callable[..., Awaitable[object]],
        message_id: str = "",
        is_group: bool = False,
        reason: str = "external_reply",
        timeline_delivery_kind: str | None = "response",
    ) -> DeliveryReceipt:
        """Persist and settle a non-chat-consumer text delivery."""
        key = f"external:{delivery_id}"
        record = await self.ledger.prepare(
            key=key,
            chat_id=chat_id,
            turn_id=delivery_id,
            reason=reason,
            reply_anchor_id=message_id,
            content_hash=self.ledger.content_hash(content),
            logical_delivery_id=key,
        )
        await self._audit(record, record.status)
        if record.status != "prepared":
            status = {
                "sent": "accepted",
                "failed": "failed",
                "unknown": "unknown",
                "partial": "partial",
            }.get(record.status, "failed")
            return DeliveryReceipt(
                status=status,
                logical_delivery_id=record.logical_delivery_id or key,
                transport_id=record.transport_id,
                platform_message_id=record.platform_message_id,
                error_code=record.error_code,
                chunk_index=record.chunk_index,
                chunk_count=record.chunk_count,
            )
        try:
            receipt = await callback(
                chat_id=chat_id,
                content=content,
                message_id=message_id,
                is_group=is_group,
            )
        except Exception:
            receipt = DeliveryReceipt(
                status="failed",
                logical_delivery_id=key,
                error_code="transport_exception",
                retryable=True,
            )
        if not isinstance(receipt, DeliveryReceipt):
            receipt = DeliveryReceipt(
                status="accepted",
                logical_delivery_id=key,
            )
        await self.settle_receipt(
            record,
            receipt,
            content=content,
            delivery_kind=timeline_delivery_kind,
        )
        return receipt

    async def prepare_reply_delivery(
        self,
        *,
        chat_id: str,
        turn_id: str,
        sequence: int,
        content: str,
        reply_anchor_id: str = "",
    ) -> DeliveryRecord:
        key = f"reply:{chat_id}:{turn_id}:{sequence}"
        record = await self.ledger.prepare(
            key=key,
            chat_id=chat_id,
            turn_id=turn_id,
            reason="model_reply",
            reply_anchor_id=reply_anchor_id,
            content_hash=self.ledger.content_hash(content),
            logical_delivery_id=key,
        )
        await self._audit(record, record.status)
        return record

    async def prepare_tool_delivery(
        self,
        *,
        chat_id: str,
        turn_id: str,
        tool_name: str,
        tool_call_id: str,
        content: str,
        reply_anchor_id: str = "",
    ) -> DeliveryRecord:
        key = f"tool:{chat_id}:{turn_id}:{tool_name}:{tool_call_id}"
        record = await self.ledger.prepare(
            key=key,
            chat_id=chat_id,
            turn_id=turn_id,
            reason=f"tool:{tool_name}",
            reply_anchor_id=reply_anchor_id,
            content_hash=self.ledger.content_hash(content),
            logical_delivery_id=key,
            content=None,
        )
        await self._audit(record, record.status)
        return record

    async def settle_tool_delivery(
        self,
        record: DeliveryRecord,
        receipt: DeliveryReceipt | None,
        *,
        content: str | None = None,
    ) -> Optional[DeliveryRecord]:
        if receipt is None:
            settled = await self.ledger.settle(
                record.key,
                status="sent",
                reason="legacy_transport_callback",
                receipt_status="accepted",
                logical_delivery_id=record.logical_delivery_id,
            )
            if settled is not None:
                await self._audit(settled, settled.status)
                await self._append_delivery_receipt(
                    record,
                    DeliveryReceipt(
                        status="accepted",
                        logical_delivery_id=record.logical_delivery_id,
                    ),
                )
            if settled is not None and content is not None:
                await self._append_accepted_timeline(
                    record, content=content, delivery_kind="response"
                )
            return settled
        return await self.settle_receipt(record, receipt, content=content)

    async def _append_accepted_timeline(
        self, record: DeliveryRecord, *, content: str, delivery_kind: str
    ) -> None:
        if self.event_log is not None:
            try:
                await self.event_log.append_accepted_delivery(
                    chat_id=record.chat_id,
                    turn_id=record.turn_id,
                    delivery_id=record.logical_delivery_id or record.key,
                    content=content,
                    message_id=record.reply_anchor_id,
                    timestamp=record.updated_at,
                )
            except EventLogInvariantError as exc:
                try:
                    await self.event_log.append_late_delivery_event(
                        chat_id=record.chat_id,
                        original_turn_id=record.turn_id,
                        delivery_id=record.logical_delivery_id or record.key,
                        content=content,
                        message_id=record.reply_anchor_id,
                        delivery_kind=delivery_kind,
                        timestamp=record.updated_at,
                    )
                except Exception:
                    _log.warning(
                        "failed to record late accepted delivery lineage "
                        "chat=%s turn=%s delivery=%s",
                        record.chat_id[:12],
                        record.turn_id,
                        record.logical_delivery_id or record.key,
                        exc_info=True,
                    )
                _log.warning(
                    "accepted delivery arrived after terminal turn; recorded orphan event "
                    "chat=%s turn=%s delivery=%s: %s",
                    record.chat_id[:12],
                    record.turn_id,
                    record.logical_delivery_id or record.key,
                    exc,
                )
                return
            return
        if self.timeline is not None:
            await self.timeline.append_accepted_delivery(
                chat_id=record.chat_id,
                delivery_id=record.logical_delivery_id or record.key,
                content=content,
                delivery_kind=delivery_kind,
                message_id=record.reply_anchor_id,
            )

    async def _append_delivery_receipt(
        self, record: DeliveryRecord, receipt: DeliveryReceipt
    ) -> None:
        if self.event_log is None or not record.turn_id:
            return
        receipt_key = (
            receipt.transport_id
            or receipt.platform_message_id
            or receipt.error_code
            or receipt.status
        )
        try:
            await self.event_log.append_event(
                ConversationEvent(
                    chat_id=record.chat_id,
                    turn_id=record.turn_id,
                    event_id=f"receipt:{record.key}:{receipt.status}:{receipt_key}",
                    role="system",
                    kind=EventKind.DELIVERY_RECEIPT,
                    content=json.dumps(
                        {
                            "status": receipt.status,
                            "logical_delivery_id": receipt.logical_delivery_id
                            or record.logical_delivery_id,
                            "transport_id": receipt.transport_id,
                            "platform_message_id": receipt.platform_message_id,
                            "error_code": receipt.error_code,
                            "retryable": receipt.retryable,
                            "chunk_index": receipt.chunk_index,
                            "chunk_count": receipt.chunk_count,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    timestamp=record.updated_at,
                )
            )
        except Exception:
            _log.warning(
                "delivery receipt event append failed [%s..]",
                record.chat_id[:12],
                exc_info=True,
            )

    async def settle_receipt(
        self,
        record: DeliveryRecord,
        receipt: DeliveryReceipt,
        *,
        content: str | None = None,
        delivery_kind: str | None = "response",
    ) -> Optional[DeliveryRecord]:
        """Persist receipt and project accepted content to the timeline."""
        status = {
            "accepted": "sent",
            "failed": "failed",
            "unknown": "unknown",
            "partial": "partial",
        }[receipt.status]
        settled = await self.ledger.settle(
            record.key,
            status=status,
            reason=(
                receipt.error_code
                or (
                    record.reason
                    if receipt.status == "accepted"
                    else f"transport_{receipt.status}"
                )
            ),
            transport_id=receipt.transport_id,
            receipt_status=receipt.status,
            logical_delivery_id=(
                receipt.logical_delivery_id or record.logical_delivery_id or record.key
            ),
            platform_message_id=receipt.platform_message_id,
            error_code=receipt.error_code,
            chunk_index=receipt.chunk_index,
            chunk_count=receipt.chunk_count,
        )
        if settled is not None:
            await self._audit(settled, settled.status)
            await self._append_delivery_receipt(record, receipt)
        if (
            settled is not None
            and receipt.status == "accepted"
            and content is not None
            and delivery_kind is not None
        ):
            await self._append_accepted_timeline(
                record, content=content, delivery_kind=delivery_kind
            )
        return settled

    async def mark_sent(
        self,
        record: DeliveryRecord,
        *,
        transport_id: str = "",
        content: str | None = None,
    ) -> Optional[DeliveryRecord]:
        settled = await self.ledger.settle(
            record.key,
            status="sent",
            reason="transport_sent",
            receipt_status="accepted",
            logical_delivery_id=f"ambient:{record.chat_id}:{record.turn_id}",
            transport_id=transport_id,
        )
        if settled is not None:
            await self._append_delivery_receipt(
                record,
                DeliveryReceipt(
                    status="accepted",
                    logical_delivery_id=record.logical_delivery_id,
                    transport_id=transport_id,
                ),
            )
            if content is not None:
                await self._append_accepted_timeline(
                    record, content=content, delivery_kind="response"
                )
        return settled
