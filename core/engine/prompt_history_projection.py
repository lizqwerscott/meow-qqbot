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


class PromptHistoryProjection:
    """Select complete recent turns behind one small prompt-facing interface."""

    PROJECTION_VERSION = 1

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
                    operation_id TEXT NOT NULL
                );
                """)
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
        if source_hash and source_hash != calculated_hash:
            raise ValueError("archive membership source hash mismatch")
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
                conn.execute(
                    """
                    INSERT INTO prompt_archive_operations
                        (operation_id, chat_id, cutoff_seq, projection_version,
                         source_hash, hidden_ids, retained_ids, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
                    """,
                    (
                        operation_id,
                        chat_id,
                        cutoff,
                        self.PROJECTION_VERSION,
                        calculated_hash,
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
                        (chat_id, cutoff_seq, projection_version, source_hash, operation_id)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        cutoff_seq = excluded.cutoff_seq,
                        projection_version = excluded.projection_version,
                        source_hash = excluded.source_hash,
                        operation_id = excluded.operation_id
                    """,
                    (
                        chat_id,
                        cutoff,
                        self.PROJECTION_VERSION,
                        calculated_hash,
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
        source_ids = set(
            await self._event_log.event_ids(
                chat_id, upto_seq=upto_seq, include_internal=False
            )
        )
        source = await self._event_log.snapshot_events(
            chat_id,
            upto_seq=upto_seq,
            include_internal=False,
            event_ids=tuple(sorted(source_ids - hidden_ids)),
        )
        source_events = source.events
        by_turn: dict[str, list[ConversationEvent]] = {}
        turn_order: list[str] = []
        for event in source_events:
            if event.turn_id not in by_turn:
                by_turn[event.turn_id] = []
                turn_order.append(event.turn_id)
            by_turn[event.turn_id].append(event)

        selected_turns: list[str] = []
        used_tokens = 0
        degraded_reason = ""
        if current_turn_id in by_turn:
            current_events = by_turn[current_turn_id]
            current_tokens = sum(event.token_count for event in current_events)
            if current_tokens <= self._max_tokens:
                selected_turns.append(current_turn_id)
                used_tokens = current_tokens
            else:
                user_events = [
                    event
                    for event in current_events
                    if event.kind.value == "user_message"
                ]
                if user_events:
                    user_event = user_events[-1]
                    max_chars = self._max_tokens * 4
                    if user_event.token_count > self._max_tokens:
                        user_event = replace(
                            user_event,
                            content=user_event.content[:max_chars].rstrip() + "…",
                            token_count=self._max_tokens,
                        )
                    selected_turns.append(current_turn_id)
                    used_tokens = user_event.token_count
                    current_events = [user_event]
                    by_turn[current_turn_id] = current_events
                degraded_reason = "current_turn_exceeds_budget"
        selected_historical_turns = 0
        for turn_id in reversed(turn_order):
            if turn_id == current_turn_id:
                continue
            turn_events = by_turn[turn_id]
            turn_tokens = sum(event.token_count for event in turn_events)
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

        selected_ids = set(selected_turns)
        events = tuple(
            event for event in source.events if event.turn_id in selected_ids
        )
        selected_event_ids = {item.event_id for item in events}
        truncated = tuple(
            event.event_id
            for event in source_events
            if event.event_id not in selected_event_ids
        )
        return PromptHistorySnapshot(
            events=events,
            cutoff_seq=source.cutoff_seq,
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
