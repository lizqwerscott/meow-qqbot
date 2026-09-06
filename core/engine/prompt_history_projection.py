"""Bounded prompt history derived from the conversation event ledger."""

import asyncio
import hashlib
import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

from core.engine.conversation_event_log import ConversationEvent, ConversationEventLog


@dataclass(frozen=True)
class PromptHistorySnapshot:
    events: tuple[ConversationEvent, ...]
    cutoff_seq: int
    projection_version: int
    estimated_tokens: int
    truncated_event_ids: tuple[str, ...] = ()
    degraded_reason: str = ""


@dataclass(frozen=True)
class PromptProjectionRepairReport:
    chat_id: str
    cutoff_seq: int
    source_event_count: int
    visibility_event_count: int
    inserted_event_count: int
    operation_id: str


class PromptHistoryProjection:
    """Select complete recent turns behind one small prompt-facing interface."""

    PROJECTION_VERSION = 1

    @staticmethod
    def _is_prompt_source_event(event: ConversationEvent) -> bool:
        return event.session_kind in {
            "chat",
            "group",
            "private",
        } and not event.chat_id.startswith(("task:", "cron:", "heartbeat:"))

    def __init__(
        self,
        event_log: ConversationEventLog,
        *,
        max_tokens: int = 12000,
        max_turns: int = 32,
        metadata_path: str = "data/prompt_history_projection.sqlite3",
    ):
        if max_tokens < 1 or max_turns < 1:
            raise ValueError("prompt projection limits must be positive")
        self._event_log = event_log
        self._max_tokens = max_tokens
        self._max_turns = max_turns
        self._metadata_path = metadata_path
        self._metadata_conn: Optional[sqlite3.Connection] = None
        self._metadata_lock = asyncio.Lock()

    async def _ensure_metadata_open(self) -> sqlite3.Connection:
        if self._metadata_conn is not None:
            return self._metadata_conn
        async with self._metadata_lock:
            if self._metadata_conn is not None:
                return self._metadata_conn
            if self._metadata_path != ":memory:":
                Path(self._metadata_path).parent.mkdir(parents=True, exist_ok=True)
            self._metadata_conn = sqlite3.connect(
                self._metadata_path, check_same_thread=False
            )
            self._metadata_conn.row_factory = sqlite3.Row
            self._metadata_conn.executescript("""
                CREATE TABLE IF NOT EXISTS prompt_archive_operations (
                    operation_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    cutoff_seq INTEGER NOT NULL,
                    projection_version INTEGER NOT NULL,
                    source_hash TEXT NOT NULL,
                    ledger_source_hash TEXT NOT NULL DEFAULT '',
                    hidden_ids TEXT NOT NULL,
                    retained_ids TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS prompt_event_visibility (
                    chat_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    prompt_visible INTEGER NOT NULL,
                    storage_tier TEXT NOT NULL,
                    archive_batch_id TEXT NOT NULL DEFAULT '',
                    operation_id TEXT NOT NULL,
                    PRIMARY KEY (chat_id, event_id)
                );
                CREATE TABLE IF NOT EXISTS prompt_projection_watermarks (
                    chat_id TEXT PRIMARY KEY,
                    cutoff_seq INTEGER NOT NULL,
                    projection_version INTEGER NOT NULL,
                    source_hash TEXT NOT NULL,
                    ledger_source_hash TEXT NOT NULL DEFAULT '',
                    operation_id TEXT NOT NULL
                );
                """)
            for table in ("prompt_archive_operations", "prompt_projection_watermarks"):
                columns = {
                    str(row["name"])
                    for row in self._metadata_conn.execute(
                        f"PRAGMA table_info({table})"
                    ).fetchall()
                }
                if "ledger_source_hash" not in columns:
                    self._metadata_conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN ledger_source_hash "
                        "TEXT NOT NULL DEFAULT ''"
                    )
            self._metadata_conn.commit()
        return self._metadata_conn

    @staticmethod
    def _membership_hash(
        hidden_event_ids: Sequence[str], retained_event_ids: Sequence[str]
    ) -> str:
        payload = {
            "hidden": sorted(set(hidden_event_ids)),
            "retained": sorted(set(retained_event_ids)),
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _ledger_source_hash(event_ids: Sequence[str]) -> str:
        return hashlib.sha256(
            json.dumps(
                sorted(set(str(event_id) for event_id in event_ids if event_id)),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _restore_visibility_rows(
        conn: sqlite3.Connection,
        chat_id: str,
        operation_id: str,
        hidden_event_ids: Sequence[str],
        retained_event_ids: Sequence[str],
    ) -> None:
        for event_id in hidden_event_ids:
            conn.execute(
                """
                INSERT INTO prompt_event_visibility
                    (chat_id, event_id, prompt_visible, storage_tier,
                     archive_batch_id, operation_id)
                VALUES (?, ?, 0, 'cold', ?, ?)
                ON CONFLICT(chat_id, event_id) DO UPDATE SET
                    prompt_visible = 0,
                    storage_tier = 'cold',
                    archive_batch_id = excluded.archive_batch_id,
                    operation_id = excluded.operation_id
                """,
                (chat_id, str(event_id), operation_id, operation_id),
            )
        for event_id in retained_event_ids:
            conn.execute(
                """
                INSERT INTO prompt_event_visibility
                    (chat_id, event_id, prompt_visible, storage_tier,
                     archive_batch_id, operation_id)
                VALUES (?, ?, 1, 'hot', '', ?)
                ON CONFLICT(chat_id, event_id) DO UPDATE SET
                    prompt_visible = CASE
                        WHEN prompt_event_visibility.prompt_visible = 0 THEN 0
                        ELSE 1 END,
                    storage_tier = CASE
                        WHEN prompt_event_visibility.prompt_visible = 0
                        THEN 'cold' ELSE 'hot' END,
                    archive_batch_id = CASE
                        WHEN prompt_event_visibility.prompt_visible = 0
                        THEN prompt_event_visibility.archive_batch_id ELSE '' END,
                    operation_id = CASE
                        WHEN prompt_event_visibility.prompt_visible = 0
                        THEN prompt_event_visibility.operation_id
                        ELSE excluded.operation_id END
                """,
                (chat_id, str(event_id), operation_id),
            )

    async def repair_archive_operation(self, operation_id: str) -> bool:
        """Restore visibility rows for a persisted operation without CAS replay."""
        if not operation_id:
            return False
        conn = await self._ensure_metadata_open()
        async with self._metadata_lock:
            row = conn.execute(
                "SELECT * FROM prompt_archive_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                return False
            try:
                hidden = json.loads(row["hidden_ids"] or "[]")
                retained = json.loads(row["retained_ids"] or "[]")
            except (TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "persisted archive operation visibility is invalid"
                ) from exc
            if (
                not isinstance(hidden, list)
                or not isinstance(retained, list)
                or not all(isinstance(item, str) for item in (*hidden, *retained))
            ):
                raise RuntimeError("persisted archive operation visibility is invalid")
            if int(row["projection_version"]) != self.PROJECTION_VERSION:
                raise RuntimeError("unsupported persisted prompt projection version")
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._restore_visibility_rows(
                    conn, row["chat_id"], operation_id, hidden, retained
                )
                conn.execute(
                    """
                    INSERT INTO prompt_projection_watermarks
                        (chat_id, cutoff_seq, projection_version, source_hash,
                         ledger_source_hash, operation_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        cutoff_seq = MAX(
                            prompt_projection_watermarks.cutoff_seq,
                            excluded.cutoff_seq
                        ),
                        projection_version = excluded.projection_version,
                        source_hash = CASE
                            WHEN excluded.cutoff_seq > prompt_projection_watermarks.cutoff_seq
                            THEN excluded.source_hash
                            ELSE prompt_projection_watermarks.source_hash END,
                        operation_id = CASE
                            WHEN excluded.cutoff_seq > prompt_projection_watermarks.cutoff_seq
                            THEN excluded.operation_id
                            ELSE prompt_projection_watermarks.operation_id END,
                        ledger_source_hash = CASE
                            WHEN excluded.cutoff_seq > prompt_projection_watermarks.cutoff_seq
                            THEN excluded.ledger_source_hash
                            WHEN prompt_projection_watermarks.ledger_source_hash = ''
                            THEN excluded.ledger_source_hash
                            ELSE prompt_projection_watermarks.ledger_source_hash END
                    """,
                    (
                        row["chat_id"],
                        int(row["cutoff_seq"]),
                        int(row["projection_version"]),
                        row["source_hash"],
                        row["ledger_source_hash"] or "",
                        operation_id,
                    ),
                )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return True

    async def apply_archive_retention(
        self,
        chat_id: str,
        *,
        operation_id: str,
        hidden_event_ids: Sequence[str] = (),
        retained_event_ids: Sequence[str] = (),
        captured_cutoff_seq: Optional[int] = None,
        captured_projection_version: Optional[int] = None,
        source_hash: str = "",
        captured_source_hash: str = "",
    ) -> None:
        """Commit durable archive membership and prompt visibility atomically."""
        if not chat_id or not operation_id:
            raise ValueError("chat_id and operation_id are required")
        hidden = tuple(dict.fromkeys(str(item) for item in hidden_event_ids if item))
        retained = tuple(
            dict.fromkeys(str(item) for item in retained_event_ids if item)
        )
        overlap = set(hidden) & set(retained)
        if overlap:
            raise ValueError("hidden and retained event IDs overlap")
        cutoff = (
            int(captured_cutoff_seq)
            if captured_cutoff_seq is not None
            else await self._event_log.latest_event_seq(chat_id)
        )
        expected_version = (
            self.PROJECTION_VERSION
            if captured_projection_version is None
            else captured_projection_version
        )
        if expected_version != self.PROJECTION_VERSION:
            raise ValueError("unsupported prompt projection version")
        calculated_hash = self._membership_hash(hidden, retained)
        ledger_event_ids = await self._event_log.event_ids(
            chat_id,
            upto_seq=cutoff,
            include_internal=True,
        )
        calculated_ledger_hash = self._ledger_source_hash(ledger_event_ids)
        if source_hash and source_hash not in {
            calculated_hash,
            calculated_ledger_hash,
        }:
            raise ValueError("archive source hash mismatch")
        if captured_source_hash and captured_source_hash != calculated_ledger_hash:
            raise ValueError("archive ledger source hash mismatch")
        conn = await self._ensure_metadata_open()
        async with self._metadata_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT * FROM prompt_archive_operations WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["chat_id"] != chat_id
                        or existing["source_hash"] != calculated_hash
                    ):
                        raise ValueError("archive operation identity collision")
                    existing_ledger_hash = str(existing["ledger_source_hash"] or "")
                    if (
                        existing_ledger_hash
                        and existing_ledger_hash != calculated_ledger_hash
                    ):
                        raise RuntimeError("archive operation ledger source changed")
                    if not existing_ledger_hash:
                        conn.execute(
                            "UPDATE prompt_archive_operations "
                            "SET ledger_source_hash = ? WHERE operation_id = ?",
                            (calculated_ledger_hash, operation_id),
                        )
                    try:
                        raw_hidden = json.loads(existing["hidden_ids"] or "[]")
                        raw_retained = json.loads(existing["retained_ids"] or "[]")
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise RuntimeError(
                            "persisted archive operation visibility is invalid"
                        ) from exc
                    if (
                        not isinstance(raw_hidden, list)
                        or not isinstance(raw_retained, list)
                        or not all(
                            isinstance(item, str)
                            for item in (*raw_hidden, *raw_retained)
                        )
                    ):
                        raise RuntimeError(
                            "persisted archive operation visibility is invalid"
                        )
                    existing_hidden = tuple(raw_hidden)
                    existing_retained = tuple(raw_retained)
                    self._restore_visibility_rows(
                        conn,
                        chat_id,
                        operation_id,
                        existing_hidden,
                        existing_retained,
                    )
                    conn.execute(
                        """
                        INSERT INTO prompt_projection_watermarks
                            (chat_id, cutoff_seq, projection_version, source_hash,
                             ledger_source_hash, operation_id)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(chat_id) DO UPDATE SET
                            cutoff_seq = MAX(
                                prompt_projection_watermarks.cutoff_seq,
                                excluded.cutoff_seq
                            ),
                            projection_version = excluded.projection_version,
                            source_hash = CASE
                                WHEN excluded.cutoff_seq > prompt_projection_watermarks.cutoff_seq
                                THEN excluded.source_hash
                                ELSE prompt_projection_watermarks.source_hash END,
                            operation_id = CASE
                                WHEN excluded.cutoff_seq > prompt_projection_watermarks.cutoff_seq
                                THEN excluded.operation_id
                                ELSE prompt_projection_watermarks.operation_id END,
                            ledger_source_hash = CASE
                                WHEN excluded.cutoff_seq > prompt_projection_watermarks.cutoff_seq
                                THEN excluded.ledger_source_hash
                                WHEN prompt_projection_watermarks.ledger_source_hash = ''
                                THEN excluded.ledger_source_hash
                                ELSE prompt_projection_watermarks.ledger_source_hash END
                        """,
                        (
                            chat_id,
                            cutoff,
                            self.PROJECTION_VERSION,
                            existing["source_hash"],
                            calculated_ledger_hash,
                            operation_id,
                        ),
                    )
                    conn.commit()
                    return
                watermark = conn.execute(
                    "SELECT * FROM prompt_projection_watermarks WHERE chat_id = ?",
                    (chat_id,),
                ).fetchone()
                if watermark is not None:
                    if int(watermark["projection_version"]) != expected_version:
                        raise RuntimeError("prompt projection version changed")
                    if cutoff < int(watermark["cutoff_seq"]):
                        raise RuntimeError(
                            "stale archive projection watermark: "
                            f"{cutoff} < {watermark['cutoff_seq']}"
                        )
                    existing_ledger_hash = str(watermark["ledger_source_hash"] or "")
                    if (
                        cutoff == int(watermark["cutoff_seq"])
                        and existing_ledger_hash
                        and existing_ledger_hash != calculated_ledger_hash
                    ):
                        raise RuntimeError(
                            "archive projection source hash changed at watermark"
                        )
                    if (
                        cutoff == int(watermark["cutoff_seq"])
                        and str(watermark["source_hash"] or "")
                        and str(watermark["source_hash"]) != calculated_hash
                    ):
                        visibility_rows = conn.execute(
                            "SELECT event_id, prompt_visible "
                            "FROM prompt_event_visibility WHERE chat_id = ?",
                            (chat_id,),
                        ).fetchall()
                        existing_hidden_ids = {
                            str(row["event_id"])
                            for row in visibility_rows
                            if not int(row["prompt_visible"])
                        }
                        existing_visible_ids = {
                            str(row["event_id"])
                            for row in visibility_rows
                            if int(row["prompt_visible"])
                        }
                        newly_hidden = set(hidden) - existing_hidden_ids
                        monotonic_archive = (
                            bool(newly_hidden)
                            and existing_hidden_ids.issubset(set(hidden))
                            and not existing_hidden_ids.intersection(retained)
                            and newly_hidden.issubset(existing_visible_ids)
                        )
                        if not monotonic_archive:
                            raise RuntimeError(
                                "archive projection membership changed at watermark"
                            )
                unknown_event_ids = (set(hidden) | set(retained)) - set(
                    ledger_event_ids
                )
                if unknown_event_ids:
                    raise ValueError(
                        "archive visibility references unknown ledger events: "
                        f"{len(unknown_event_ids)}"
                    )
                conn.execute(
                    """
                    INSERT INTO prompt_archive_operations
                        (operation_id, chat_id, cutoff_seq, projection_version,
                         source_hash, ledger_source_hash, hidden_ids, retained_ids,
                         created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
                    """,
                    (
                        operation_id,
                        chat_id,
                        cutoff,
                        self.PROJECTION_VERSION,
                        calculated_hash,
                        calculated_ledger_hash,
                        json.dumps(hidden, ensure_ascii=False),
                        json.dumps(retained, ensure_ascii=False),
                    ),
                )
                for event_id in hidden:
                    conn.execute(
                        """
                        INSERT INTO prompt_event_visibility
                            (chat_id, event_id, prompt_visible, storage_tier,
                             archive_batch_id, operation_id)
                        VALUES (?, ?, 0, 'cold', ?, ?)
                        ON CONFLICT(chat_id, event_id) DO UPDATE SET
                            prompt_visible = 0,
                            storage_tier = 'cold',
                            archive_batch_id = excluded.archive_batch_id,
                            operation_id = excluded.operation_id
                        """,
                        (chat_id, event_id, operation_id, operation_id),
                    )
                for event_id in retained:
                    conn.execute(
                        """
                        INSERT INTO prompt_event_visibility
                            (chat_id, event_id, prompt_visible, storage_tier,
                             archive_batch_id, operation_id)
                        VALUES (?, ?, 1, 'hot', '', ?)
                        ON CONFLICT(chat_id, event_id) DO UPDATE SET
                            prompt_visible = CASE
                                WHEN prompt_event_visibility.prompt_visible = 0 THEN 0
                                ELSE 1
                            END,
                            storage_tier = CASE
                                WHEN prompt_event_visibility.prompt_visible = 0 THEN 'cold'
                                ELSE 'hot'
                            END,
                            archive_batch_id = CASE
                                WHEN prompt_event_visibility.prompt_visible = 0
                                THEN prompt_event_visibility.archive_batch_id
                                ELSE ''
                            END,
                            operation_id = excluded.operation_id
                        """,
                        (chat_id, event_id, operation_id),
                    )
                conn.execute(
                    """
                    INSERT INTO prompt_projection_watermarks
                        (chat_id, cutoff_seq, projection_version, source_hash,
                         ledger_source_hash, operation_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        cutoff_seq = MAX(
                            prompt_projection_watermarks.cutoff_seq,
                            excluded.cutoff_seq
                        ),
                        projection_version = excluded.projection_version,
                        source_hash = CASE
                            WHEN excluded.cutoff_seq > prompt_projection_watermarks.cutoff_seq
                            THEN excluded.source_hash
                            ELSE prompt_projection_watermarks.source_hash
                        END,
                        operation_id = CASE
                            WHEN excluded.cutoff_seq > prompt_projection_watermarks.cutoff_seq
                            THEN excluded.operation_id
                            ELSE prompt_projection_watermarks.operation_id
                        END,
                        ledger_source_hash = CASE
                            WHEN excluded.cutoff_seq > prompt_projection_watermarks.cutoff_seq
                            THEN excluded.ledger_source_hash
                            WHEN prompt_projection_watermarks.ledger_source_hash = ''
                            THEN excluded.ledger_source_hash
                            ELSE prompt_projection_watermarks.ledger_source_hash
                        END
                    """,
                    (
                        chat_id,
                        cutoff,
                        self.PROJECTION_VERSION,
                        calculated_hash,
                        calculated_ledger_hash,
                        operation_id,
                    ),
                )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    async def close(self) -> None:
        async with self._metadata_lock:
            if self._metadata_conn is not None:
                self._metadata_conn.close()
                self._metadata_conn = None

    async def hidden_event_ids(self, chat_id: str) -> frozenset[str]:
        """Return archived source IDs without loading their event bodies."""
        metadata = await self._ensure_metadata_open()
        async with self._metadata_lock:
            rows = metadata.execute(
                """
                SELECT event_id FROM prompt_event_visibility
                 WHERE chat_id = ? AND prompt_visible = 0
                """,
                (chat_id,),
            ).fetchall()
        return frozenset(str(row["event_id"]) for row in rows)

    async def visibility_event_ids(self, chat_id: str) -> frozenset[str]:
        """Return source IDs with an explicit prompt visibility row."""
        metadata = await self._ensure_metadata_open()
        async with self._metadata_lock:
            rows = metadata.execute(
                "SELECT event_id FROM prompt_event_visibility WHERE chat_id = ?",
                (chat_id,),
            ).fetchall()
        return frozenset(str(row["event_id"]) for row in rows)

    async def status(self, chat_id: str | None = None) -> dict[str, int]:
        """Return bounded visibility and watermark counters for rollout monitoring."""
        metadata = await self._ensure_metadata_open()
        if chat_id is not None:
            chat_ids = (str(chat_id),)
        else:
            chat_ids = tuple(await self._event_log.chat_ids())
        if not chat_ids:
            return {
                "chat_count": 0,
                "visibility_count": 0,
                "visible_count": 0,
                "hidden_count": 0,
                "watermark_count": 0,
                "projection_lag": 0,
            }
        visibility_count = visible_count = hidden_count = watermark_count = 0
        max_lag = 0
        watermark_by_chat: dict[str, int] = {}
        async with self._metadata_lock:
            for offset in range(0, len(chat_ids), 500):
                chunk = chat_ids[offset : offset + 500]
                placeholders = ", ".join("?" for _ in chunk)
                row = metadata.execute(
                    "SELECT COUNT(*) AS total, "
                    "COALESCE(SUM(prompt_visible = 1), 0) AS visible, "
                    "COALESCE(SUM(prompt_visible = 0), 0) AS hidden "
                    "FROM prompt_event_visibility WHERE chat_id IN ("
                    + placeholders
                    + ")",
                    chunk,
                ).fetchone()
                visibility_count += int(row["total"])
                visible_count += int(row["visible"])
                hidden_count += int(row["hidden"])
                watermark_count += int(
                    metadata.execute(
                        "SELECT COUNT(*) FROM prompt_projection_watermarks "
                        "WHERE chat_id IN (" + placeholders + ")",
                        chunk,
                    ).fetchone()[0]
                )
                watermark_rows = metadata.execute(
                    "SELECT chat_id, cutoff_seq FROM prompt_projection_watermarks "
                    "WHERE chat_id IN (" + placeholders + ")",
                    chunk,
                ).fetchall()
                watermark_by_chat.update(
                    {
                        str(item["chat_id"]): int(item["cutoff_seq"])
                        for item in watermark_rows
                    }
                )
        for item_chat_id in chat_ids:
            latest = await self._event_log.latest_event_seq(item_chat_id)
            max_lag = max(
                max_lag,
                max(0, latest - watermark_by_chat.get(item_chat_id, 0)),
            )
        return {
            "chat_count": len(chat_ids),
            "visibility_count": visibility_count,
            "visible_count": visible_count,
            "hidden_count": hidden_count,
            "watermark_count": watermark_count,
            "projection_lag": max_lag,
        }

    async def repair_chat(
        self,
        chat_id: str,
        *,
        upto_seq: Optional[int] = None,
    ) -> PromptProjectionRepairReport:
        """Ensure interactive ledger events have explicit retained visibility rows."""
        if not chat_id:
            raise ValueError("chat_id is required")
        cutoff = (
            int(upto_seq)
            if upto_seq is not None
            else await self._event_log.latest_event_seq(chat_id)
        )
        source_ids = await self._event_log.event_ids(
            chat_id,
            upto_seq=cutoff,
            include_internal=True,
            prompt_only=True,
        )
        existing_ids = await self.visibility_event_ids(chat_id)
        missing_ids = tuple(
            event_id for event_id in source_ids if event_id not in existing_ids
        )
        operation_payload = json.dumps(
            {"chat_id": chat_id, "event_ids": sorted(missing_ids)},
            ensure_ascii=False,
            sort_keys=True,
        )
        operation_id = (
            "prompt-repair:"
            + hashlib.sha256(operation_payload.encode("utf-8")).hexdigest()[:32]
        )
        if missing_ids:
            await self.apply_archive_retention(
                chat_id,
                operation_id=operation_id,
                retained_event_ids=missing_ids,
                captured_cutoff_seq=cutoff,
            )
        return PromptProjectionRepairReport(
            chat_id=chat_id,
            cutoff_seq=cutoff,
            source_event_count=len(source_ids),
            visibility_event_count=len(existing_ids) + len(missing_ids),
            inserted_event_count=len(missing_ids),
            operation_id=operation_id,
        )

    async def clear_chat(self, chat_id: str) -> None:
        metadata = await self._ensure_metadata_open()
        async with self._metadata_lock:
            metadata.execute(
                "DELETE FROM prompt_event_visibility WHERE chat_id = ?", (chat_id,)
            )
            metadata.execute(
                "DELETE FROM prompt_archive_operations WHERE chat_id = ?", (chat_id,)
            )
            metadata.execute(
                "DELETE FROM prompt_projection_watermarks WHERE chat_id = ?", (chat_id,)
            )
            metadata.commit()

    async def snapshot_for_prompt(
        self,
        chat_id: str,
        *,
        upto_seq: Optional[int] = None,
        current_turn_id: str = "",
    ) -> PromptHistorySnapshot:
        metadata = await self._ensure_metadata_open()
        async with self._metadata_lock:
            hidden_rows = metadata.execute(
                "SELECT event_id FROM prompt_event_visibility "
                "WHERE chat_id = ? AND prompt_visible = 0",
                (chat_id,),
            ).fetchall()
        hidden_ids = {str(row["event_id"]) for row in hidden_rows}
        turn_budgets, cutoff = await self._event_log.snapshot_turn_budgets(
            chat_id,
            upto_seq=upto_seq,
            include_internal=False,
            exclude_event_ids=tuple(hidden_ids),
            current_turn_id=current_turn_id,
        )

        selected_turns: list[str] = []
        used_tokens = 0
        degraded_reason = ""
        budget_by_turn = {item.turn.turn_id: item for item in turn_budgets}
        turn_order = [item.turn.turn_id for item in turn_budgets]
        if current_turn_id in budget_by_turn:
            current_tokens = budget_by_turn[current_turn_id].estimated_tokens
            if current_tokens <= self._max_tokens:
                selected_turns.append(current_turn_id)
                used_tokens = current_tokens
            else:
                selected_turns.append(current_turn_id)
                used_tokens = self._max_tokens
                degraded_reason = "current_turn_exceeds_budget"
        selected_historical_turns = 0
        for turn_id in turn_order:
            if turn_id == current_turn_id:
                continue
            turn_tokens = budget_by_turn[turn_id].estimated_tokens
            if selected_historical_turns >= self._max_turns:
                break
            if selected_turns and used_tokens + turn_tokens > self._max_tokens:
                break
            if not selected_turns and turn_tokens > self._max_tokens:
                degraded_reason = "latest_turn_exceeds_budget"
                continue
            selected_turns.append(turn_id)
            selected_historical_turns += 1
            used_tokens += turn_tokens

        selected_integrity = await self._event_log.validate_turns(
            selected_turns, chat_id=chat_id
        )
        invalid_historical = {
            turn_id
            for turn_id, report in selected_integrity.items()
            if not report.valid and turn_id != current_turn_id
        }
        if invalid_historical:
            selected_turns = [
                turn_id
                for turn_id in selected_turns
                if turn_id not in invalid_historical
            ]
            used_tokens = sum(
                budget_by_turn[turn_id].estimated_tokens for turn_id in selected_turns
            )
            selected_historical_turns = sum(
                turn_id != current_turn_id for turn_id in selected_turns
            )
            for turn_id in turn_order:
                if (
                    turn_id == current_turn_id
                    or turn_id in selected_turns
                    or turn_id in invalid_historical
                    or selected_historical_turns >= self._max_turns
                ):
                    continue
                turn_tokens = budget_by_turn[turn_id].estimated_tokens
                if selected_turns and used_tokens + turn_tokens > self._max_tokens:
                    break
                if not selected_turns and turn_tokens > self._max_tokens:
                    continue
                candidate_report = (
                    await self._event_log.validate_turns((turn_id,), chat_id=chat_id)
                ).get(turn_id)
                if candidate_report is None or not candidate_report.valid:
                    invalid_historical.add(turn_id)
                    continue
                selected_turns.append(turn_id)
                selected_historical_turns += 1
                used_tokens += turn_tokens
            degraded_reason = degraded_reason or "invalid_historical_turn_excluded"

        selected_ids = set(selected_turns)
        source = await self._event_log.snapshot_events(
            chat_id,
            upto_seq=upto_seq,
            include_internal=False,
            turn_ids=tuple(selected_ids),
        )
        source_events = tuple(
            event for event in source.events if event.event_id not in hidden_ids
        )
        if current_turn_id in selected_ids and current_turn_id in budget_by_turn:
            current_events = tuple(
                event for event in source_events if event.turn_id == current_turn_id
            )
            if budget_by_turn[current_turn_id].estimated_tokens > self._max_tokens:
                user_events = tuple(
                    event
                    for event in current_events
                    if event.kind.value == "user_message"
                )
                if user_events:
                    user_event = user_events[-1]
                    max_chars = self._max_tokens * 4
                    user_event = replace(
                        user_event,
                        content=user_event.content[:max_chars].rstrip() + "…",
                        token_count=self._max_tokens,
                    )
                    source_events = tuple(
                        event
                        for event in source_events
                        if event.turn_id != current_turn_id
                    ) + (user_event,)
                    source_events = tuple(
                        sorted(source_events, key=lambda event: event.event_seq)
                    )
        events = tuple(source_events)
        selected_event_ids = {item.event_id for item in events}
        truncated = tuple(
            event_id
            for event_id in await self._event_log.event_ids(
                chat_id, upto_seq=upto_seq, include_internal=False
            )
            if event_id not in selected_event_ids and event_id not in hidden_ids
        )
        return PromptHistorySnapshot(
            events=events,
            cutoff_seq=cutoff,
            projection_version=self.PROJECTION_VERSION,
            estimated_tokens=sum(event.token_count for event in events),
            truncated_event_ids=truncated,
            degraded_reason=degraded_reason,
        )

    async def snapshot_for_active(
        self,
        chat_id: str,
        *,
        upto_seq: Optional[int] = None,
    ) -> PromptHistorySnapshot:
        """Return the complete visible hot projection without prompt budgeting."""
        metadata = await self._ensure_metadata_open()
        async with self._metadata_lock:
            hidden_rows = metadata.execute(
                "SELECT event_id FROM prompt_event_visibility "
                "WHERE chat_id = ? AND prompt_visible = 0",
                (chat_id,),
            ).fetchall()
        hidden_ids = {str(row["event_id"]) for row in hidden_rows}
        source = await self._event_log.snapshot_events(
            chat_id,
            upto_seq=upto_seq,
            include_internal=False,
        )
        events = tuple(
            event for event in source.events if event.event_id not in hidden_ids
        )
        return PromptHistorySnapshot(
            events=events,
            cutoff_seq=source.cutoff_seq,
            projection_version=self.PROJECTION_VERSION,
            estimated_tokens=sum(event.token_count for event in events),
            truncated_event_ids=tuple(
                event.event_id
                for event in source.events
                if event.event_id in hidden_ids
            ),
        )
