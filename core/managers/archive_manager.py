"""ArchiveManager — 会话归档 + 自动摘要 + 上下文回放

消息驱动触发：每条消息前检查消息流中是否存在"今天之前"的消息
（按消息时间戳判断跨天，不依赖 last_activity），跨天则归档一次（同一天
内只归档一次，状态持久化跨重启）：
1. 仅将本次尚未归档的旧消息写入 .archived.<timestamp>
2. 旧消息中取最后 N 条 user/assistant 消息 → 写入 .md 摘要
3. 保留：今天的消息全部保留；按连续时间段的切点决定是否携带昨天尾段
4. 首次 build() 时注入归档摘要（仅一次，后续不再重复）
"""

import asyncio
import hashlib
import inspect
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set
from zoneinfo import ZoneInfo

from core.engine.archive_export import ArchiveExportResult, ArchiveJSONLExportAdapter
from core.engine.archive_index import ArchiveIndex, ArchiveTurnRecord
from core.engine.history_projection import merge_timeline_visible_events
from core.managers.archive_ledger import ArchiveLedger
from core.managers.archive_manifest import ArchiveManifestStore
from core.managers.chat_message import (
    ChatMessage,
    group_user_messages,
    strip_content_prefix,
)
from core.managers.context_store import ContextStore

_log = logging.getLogger(__name__)

_DEFAULT_SUMMARY_COUNT = 15
_DEFAULT_ARCHIVE_HOUR = 4
_DEFAULT_SUMMARY_DAYS = 2
_DEFAULT_RETENTION_DAYS = 30
_DEFAULT_REPLAY_GAP_SECONDS = 600
_ARCHIVE_STATE_ORDER = {
    "prepared": 0,
    "archive_written": 1,
    "active_written": 2,
    "summary_written": 3,
    "committed": 4,
}

# ── 工具函数 ──


def _format_archive_timestamp(
    t: Optional[float] = None, timezone_name: str = "Asia/Shanghai"
) -> str:
    dt = datetime.fromtimestamp(t or time.time(), tz=ZoneInfo(timezone_name))
    return dt.strftime("%Y-%m-%dT%H-%M-%S")


def _date_str(t: Optional[float] = None, timezone_name: str = "Asia/Shanghai") -> str:
    dt = datetime.fromtimestamp(t or time.time(), tz=ZoneInfo(timezone_name))
    return dt.strftime("%Y-%m-%d")


def _previous_date_str(
    t: Optional[float] = None, timezone_name: str = "Asia/Shanghai"
) -> str:
    dt = datetime.fromtimestamp(t or time.time(), tz=ZoneInfo(timezone_name))
    return (dt.date() - timedelta(days=1)).isoformat()


def _get_memory_dir(memory_root: str, chat_id: str) -> Path:
    return Path(memory_root) / chat_id


def _coerce_archive_path(value: Any) -> Optional[str]:
    if isinstance(value, (str, os.PathLike)):
        path = os.fspath(value)
        return path or None
    return None


# ── ArchiveResult ──


@dataclass(frozen=True)
class ArchiveBatchResult:
    """One source-date archive artifact within an archive operation."""

    batch_id: str
    partition_date: str
    archive_path: Optional[str] = None
    summary_path: Optional[str] = None
    message_count: int = 0
    event_count: int = 0
    unit_count: int = 0


@dataclass(frozen=True)
class ArchiveUnit:
    """Immutable logical unit used by archive partitioning and replay."""

    unit_id: str
    kind: str
    message_indices: tuple[int, ...]
    message_identities: tuple[str, ...]
    partition_time: float
    activity_end_time: float
    replayable: bool
    incomplete: bool = False


class ArchiveResult:
    """归档操作的返回信息。"""

    def __init__(
        self,
        chat_id: str,
        reason: str,
        archive_path: Optional[str] = None,
        summary_path: Optional[str] = None,
        replay_count: int = 0,
        operation_id: Optional[str] = None,
        batches: Optional[List[ArchiveBatchResult]] = None,
    ):
        self.chat_id = chat_id
        self.reason = reason
        self.archive_path = archive_path
        self.summary_path = summary_path
        self.replay_count = replay_count
        self.operation_id = operation_id
        self.batches = list(batches or [])

    @property
    def archive_paths(self) -> List[str]:
        if self.batches:
            return [batch.archive_path for batch in self.batches if batch.archive_path]
        return [self.archive_path] if self.archive_path else []

    @property
    def summary_paths(self) -> List[str]:
        if self.batches:
            return [batch.summary_path for batch in self.batches if batch.summary_path]
        return [self.summary_path] if self.summary_path else []


# ── ArchiveManager ──


class ArchiveManager:
    """会话归档管理器。"""

    def __init__(
        self,
        context_manager: Any,
        memory_dir: str = "data/archives/memory/",
        archive_hour: int = _DEFAULT_ARCHIVE_HOUR,
        replay_count: Optional[int] = None,
        replay_gap_seconds: int = _DEFAULT_REPLAY_GAP_SECONDS,
        summary_count: int = _DEFAULT_SUMMARY_COUNT,
        summary_days: int = _DEFAULT_SUMMARY_DAYS,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
        merge_window_seconds: int = 15,
        archive_ledger: Optional[ArchiveLedger] = None,
        timezone_name: str = "Asia/Shanghai",
        archive_index: Optional[ArchiveIndex] = None,
        hot_max_tokens: int = 12000,
        hot_max_turns: int = 32,
        hot_max_bytes: int = 4_000_000,
        hot_max_age_seconds: float = 7 * 86400,
        hot_low_water_ratio: float = 0.75,
    ):
        self._cm = context_manager
        self._timeline = None
        self._memory_dir = memory_dir
        self._archive_hour = archive_hour
        # replay_count 是旧配置的兼容参数；回放改为按完整时间段切分。
        self._legacy_replay_count = replay_count
        self._replay_gap_seconds = max(0, replay_gap_seconds)
        self._summary_count = summary_count
        self._summary_days = summary_days
        self._retention_days = retention_days
        self.merge_window_seconds = merge_window_seconds
        self._timezone_name = timezone_name
        self._timezone = ZoneInfo(timezone_name)
        self._hot_max_tokens = max(1, int(hot_max_tokens))
        self._hot_max_turns = max(1, int(hot_max_turns))
        self._hot_max_bytes = max(1, int(hot_max_bytes))
        self._hot_max_age_seconds = max(0.0, float(hot_max_age_seconds))
        self._hot_low_water_ratio = min(0.99, max(0.1, float(hot_low_water_ratio)))

        self._pending_injection: Set[str] = set()
        # chat_id → 上次自动归档的日期（防止同一天重复归档；持久化跨重启）
        self._last_daily_archive: Dict[str, str] = {}
        # chat_id → 当前 active history 中已写入旧 archive 的回放前缀指纹。
        # 下一次归档跳过此段，避免回放消息进入第二份 archive。
        self._replayed_prefix_keys: Dict[str, List[str]] = {}
        # 已知键集合用于区分新版空前缀与旧版仅记录归档日期的 state。
        self._replayed_prefix_known: Set[str] = set()
        self._daily_state_lock = threading.Lock()
        self._daily_state_path = Path(memory_dir).parent / "daily_archive_state.json"
        self._ledger = archive_ledger
        self._event_log = None
        self._prompt_projection = None
        self._summary_store = None
        self._model_context_transcript = None
        self._export_adapter: Optional[ArchiveJSONLExportAdapter] = None
        self._archive_index = archive_index
        self._legacy_ledger_seeded: Set[str] = set()
        self._event_archive_locks: Dict[str, asyncio.Lock] = {}
        self._session_lock_provider = None
        self._manifest_store = ArchiveManifestStore(str(Path(memory_dir).parent))
        self._load_daily_state()
        self.recover_incomplete_archives()

    def set_timeline(self, timeline: Any) -> None:
        self._timeline = timeline

    def set_event_log(
        self,
        event_log: Any,
        prompt_projection: Any = None,
        summary_store: Any = None,
    ) -> None:
        self._event_log = event_log
        self._prompt_projection = prompt_projection
        self._summary_store = summary_store
        if self._archive_index is None:
            self._archive_index = ArchiveIndex(
                str(Path(self._memory_dir).parent / "archive_index.sqlite3")
            )

    def set_archive_index(self, archive_index: ArchiveIndex) -> None:
        self._archive_index = archive_index

    def set_session_lock_provider(self, provider: Any) -> None:
        self._session_lock_provider = provider

    def set_summary_store(self, summary_store: Any) -> None:
        self._summary_store = summary_store

    def set_model_context_transcript(self, transcript: Any) -> None:
        self._model_context_transcript = transcript

    def set_export_adapter(self, export_adapter: ArchiveJSONLExportAdapter) -> None:
        self._export_adapter = export_adapter

    def _ensure_event_manifest(
        self, batch: Any, event_ids: Sequence[str]
    ) -> Dict[str, Any]:
        manifest = self._manifest_store.load(batch.operation_id)
        if manifest is not None:
            if manifest.get("kind") != "event_log":
                raise RuntimeError(
                    f"archive operation identity collision: {batch.operation_id}"
                )
            if (
                manifest.get("chat_id") != batch.chat_id
                or int(manifest.get("captured_cutoff_seq", -1))
                != int(batch.captured_cutoff_seq)
                or manifest.get("source_hash") != batch.source_hash
            ):
                raise RuntimeError(
                    f"archive operation identity mismatch: {batch.operation_id}"
                )
            manifest_event_ids = {
                str(event_id)
                for item in manifest.get("batches", [])
                for event_id in item.get("event_ids", [])
            }
            if manifest_event_ids != {str(event_id) for event_id in event_ids}:
                raise RuntimeError(
                    f"archive operation event identity mismatch: {batch.operation_id}"
                )
            return manifest
        partition_date = batch.source_dates[0] if batch.source_dates else "unknown"
        manifest = {
            "version": 2,
            "kind": "event_log",
            "operation_id": batch.operation_id,
            "chat_id": batch.chat_id,
            "state": "prepared",
            "phase": "prepared",
            "captured_cutoff_seq": batch.captured_cutoff_seq,
            "source_hash": batch.source_hash,
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "partition_date": partition_date,
                    "source_dates": list(batch.source_dates),
                    "event_ids": sorted(str(event_id) for event_id in event_ids),
                    "state": "prepared",
                }
            ],
        }
        self._manifest_store.write(manifest)
        return manifest

    def _write_event_manifest_phase(
        self, operation_id: str, phase: str, *, committed: bool = False
    ) -> None:
        manifest = self._manifest_store.load(operation_id)
        if manifest is None:
            raise RuntimeError(f"event archive manifest not found: {operation_id}")
        manifest["phase"] = phase
        if committed:
            manifest["state"] = "committed"
        self._manifest_store.write(manifest)

    @staticmethod
    def _event_hot_bytes(event: Any) -> int:
        payload = {
            "role": event.role,
            "kind": str(event.kind),
            "content": event.content,
            "tool_call_id": event.tool_call_id,
            "tool_name": event.tool_name,
            "tool_calls": event.tool_calls,
        }
        return len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))

    @staticmethod
    def _event_archive_operation_id(
        chat_id: str, cutoff_seq: int, turn_ids: List[str]
    ) -> str:
        payload = json.dumps(
            {"chat_id": chat_id, "cutoff_seq": cutoff_seq, "turn_ids": turn_ids},
            ensure_ascii=False,
            sort_keys=True,
        )
        return (
            "event-archive:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
        )

    async def _finish_event_archive_batch(self, batch: Any) -> ArchiveResult:
        event_log = self._event_log
        index = self._archive_index
        if event_log is None or index is None:
            raise RuntimeError("event archive dependencies are not configured")
        archived_ids = await index.event_ids(batch.batch_id)
        manifest = self._ensure_event_manifest(batch, tuple(sorted(archived_ids)))
        phase = str(manifest.get("phase", "prepared"))
        phase_order = {
            "prepared": 0,
            "event_partition_written": 1,
            "active_projection_written": 2,
            "summary_projection_written": 3,
            "prompt_visibility_written": 4,
            "model_context_rotated": 5,
            "ledger_committed": 6,
        }
        hidden_ids = (
            await self._prompt_projection.hidden_event_ids(batch.chat_id)
            if self._prompt_projection is not None
            else frozenset()
        )
        source_ids = set(
            await event_log.event_ids(
                batch.chat_id,
                upto_seq=batch.captured_cutoff_seq,
                include_internal=True,
            )
        )
        source = await event_log.snapshot_events(
            batch.chat_id,
            upto_seq=batch.captured_cutoff_seq,
            include_internal=True,
            event_ids=tuple(sorted((source_ids - set(hidden_ids)) | set(archived_ids))),
        )
        if phase_order.get(phase, 0) < phase_order["event_partition_written"]:
            self._write_event_manifest_phase(
                batch.operation_id, "event_partition_written"
            )
        if phase_order.get(phase, 0) < phase_order["active_projection_written"]:
            await self._prompt_projection.snapshot_for_active(
                batch.chat_id, upto_seq=batch.captured_cutoff_seq
            )
            self._write_event_manifest_phase(
                batch.operation_id, "active_projection_written"
            )
        archived_events = tuple(
            event for event in source.events if event.event_id in archived_ids
        )
        turn_ids = {event.turn_id for event in archived_events}
        for turn_id in turn_ids:
            integrity = await event_log.validate_turn(turn_id)
            if not integrity.valid:
                raise RuntimeError(
                    f"cannot archive invalid turn {turn_id}: {integrity.reason}"
                )
        summaries = ()
        if self._summary_store is not None:
            if phase_order.get(phase, 0) < phase_order["summary_projection_written"]:
                summaries = await self._summary_store.ensure_for_archived_events(
                    batch.chat_id,
                    tuple(sorted(archived_ids)),
                    archive_batch_id=batch.batch_id,
                )
            else:
                loaded_summaries = []
                for turn_id in sorted(turn_ids):
                    summary = await self._summary_store.get(batch.chat_id, turn_id)
                    if summary is not None:
                        loaded_summaries.append(summary)
                summaries = tuple(loaded_summaries)
            if len({summary.turn_id for summary in summaries}) < len(turn_ids):
                raise RuntimeError("deterministic turn summary is incomplete")
        if phase_order.get(phase, 0) < phase_order["summary_projection_written"]:
            self._write_event_manifest_phase(
                batch.operation_id, "summary_projection_written"
            )
        if (
            self._prompt_projection is not None
            and phase_order.get(phase, 0) < phase_order["prompt_visibility_written"]
        ):
            retained_ids = tuple(
                sorted(source_ids - set(archived_ids) - set(hidden_ids))
            )
            await self._prompt_projection.apply_archive_retention(
                batch.chat_id,
                operation_id=batch.operation_id,
                hidden_event_ids=tuple(sorted(set(archived_ids) & source_ids)),
                retained_event_ids=retained_ids,
                captured_cutoff_seq=batch.captured_cutoff_seq,
            )
            self._write_event_manifest_phase(
                batch.operation_id, "prompt_visibility_written"
            )
        transcript = self._model_context_transcript
        if (
            transcript is not None
            and archived_ids
            and phase_order.get(phase, 0) < phase_order["model_context_rotated"]
        ):
            scopes = await transcript.scopes_for_chat(batch.chat_id)
            for scope in scopes:
                scope_snapshot = await transcript.snapshot(scope)
                scope_source_ids = scope_snapshot.source_event_ids
                scope_summaries = tuple(
                    summary
                    for summary in summaries
                    if scope_source_ids.intersection(summary.coverage_event_ids)
                )
                scope_summary_source_event_ids = tuple(
                    summary.coverage_event_ids for summary in scope_summaries
                )
                await transcript.rotate_for_hidden_sources(
                    scope,
                    tuple(sorted(archived_ids)),
                    summary_texts=tuple(summary.text for summary in scope_summaries),
                    summary_source_event_ids=scope_summary_source_event_ids,
                    operation_id=f"{batch.operation_id}:model-context:{scope.key}",
                )
        if phase_order.get(phase, 0) < phase_order["model_context_rotated"]:
            self._write_event_manifest_phase(
                batch.operation_id, "model_context_rotated"
            )
        if phase_order.get(phase, 0) < phase_order["ledger_committed"]:
            await index.mark_state(batch.batch_id, "committed")
        export_adapter = self._export_adapter
        if export_adapter is not None:
            try:
                export = await export_adapter.export_batch(batch.batch_id)
            except Exception as exc:
                _log.warning(
                    "JSONL 归档导出失败，不影响核心 batch [%s..] batch=%s: %s",
                    batch.chat_id[:12],
                    batch.batch_id,
                    exc,
                )
                export = ArchiveExportResult(
                    batch_id=batch.batch_id,
                    status="failed",
                    error=str(exc)[:1000],
                )
            await index.record_export(
                batch.batch_id,
                status=export.status,
                path=export.path,
                content_hash=export.content_hash,
                manifest_hash=export.manifest_hash,
                error=export.error,
            )
        self._write_event_manifest_phase(
            batch.operation_id, "ledger_committed", committed=True
        )
        source_dates = batch.source_dates
        result_batch = ArchiveBatchResult(
            batch_id=batch.batch_id,
            partition_date=source_dates[0] if source_dates else "",
            archive_path=None,
            summary_path=None,
            message_count=len(archived_events),
            event_count=len(archived_ids),
            unit_count=len(turn_ids),
        )
        return ArchiveResult(
            chat_id=batch.chat_id,
            reason="retention",
            replay_count=0,
            operation_id=batch.operation_id,
            batches=[result_batch],
        )

    async def _recover_event_log_archives(self, chat_id: str) -> None:
        if self._archive_index is None:
            return
        batches = {
            batch.batch_id: batch
            for batch in await self._archive_index.list_pending(chat_id)
        }
        for manifest in self._manifest_store.load_pending(kind="event_log"):
            if manifest.get("chat_id") != chat_id:
                continue
            for item in manifest.get("batches", []):
                batch_id = str(item.get("batch_id") or "")
                if batch_id:
                    batch = await self._archive_index.get(batch_id)
                    if batch is not None:
                        batches[batch_id] = batch
        for batch in batches.values():
            try:
                await self._finish_event_archive_batch(batch)
            except Exception as exc:
                _log.warning(
                    "核心账本归档恢复失败 [%s..] batch=%s: %s",
                    chat_id[:12],
                    batch.batch_id,
                    exc,
                )

    async def _archive_event_log_if_needed(
        self, chat_id: str, is_group: bool, *, force: bool = False
    ) -> Optional[ArchiveResult]:
        del is_group
        event_log = self._event_log
        projection = self._prompt_projection
        index = self._archive_index
        if event_log is None or index is None:
            return None
        await self._recover_event_log_archives(chat_id)
        source = await event_log.snapshot_events(chat_id, include_internal=True)
        if not source.events:
            return None
        turns_snapshot = await event_log.snapshot_turns(chat_id, include_internal=True)
        events_by_turn: Dict[str, List[Any]] = {}
        for event in source.events:
            events_by_turn.setdefault(event.turn_id, []).append(event)
        hidden = (
            set(await projection.hidden_event_ids(chat_id))
            if projection is not None
            else set()
        )
        hot_turns = [
            turn
            for turn in turns_snapshot.turns
            if events_by_turn.get(turn.turn_id)
            and all(
                event.event_id not in hidden
                for event in events_by_turn.get(turn.turn_id, ())
            )
        ]
        if not hot_turns:
            return None
        hot_events = [event for event in source.events if event.event_id not in hidden]
        hot_tokens = sum(event.token_count for event in hot_events)
        hot_bytes = sum(self._event_hot_bytes(event) for event in hot_events)
        terminal_turns = []
        for turn in hot_turns:
            if not turn.is_terminal or str(turn.status) == "incomplete":
                continue
            integrity = await event_log.validate_turn(turn.turn_id)
            if integrity.valid:
                terminal_turns.append(turn)
            else:
                _log.warning(
                    "跳过不完整归档 turn [%s..] turn=%s reason=%s",
                    chat_id[:12],
                    turn.turn_id[:12],
                    integrity.reason,
                )
        terminal_turns.sort(key=lambda turn: turn.turn_sequence)
        if not terminal_turns:
            return None

        def turn_age(turn: Any) -> float:
            events = events_by_turn.get(turn.turn_id, ())
            timestamp = min(
                (event.timestamp for event in events), default=turn.updated_at
            )
            return max(0.0, time.time() - timestamp)

        capacity_exceeded = (
            hot_tokens > self._hot_max_tokens
            or len(hot_turns) > self._hot_max_turns
            or hot_bytes > self._hot_max_bytes
        )
        age_exceeded = turn_age(terminal_turns[0]) > self._hot_max_age_seconds
        if not force and not capacity_exceeded and not age_exceeded:
            return None

        low_tokens = max(1, int(self._hot_max_tokens * self._hot_low_water_ratio))
        low_turns = max(1, int(self._hot_max_turns * self._hot_low_water_ratio))
        low_bytes = max(1, int(self._hot_max_bytes * self._hot_low_water_ratio))
        selected_turns: List[Any] = []
        selected_ids: set[str] = set()
        remaining_turns = list(hot_turns)
        while terminal_turns:
            candidate = terminal_turns.pop(0)
            candidate_events = events_by_turn.get(candidate.turn_id, ())
            candidate_ids = {event.event_id for event in candidate_events}
            selected_turns.append(candidate)
            selected_ids.update(candidate_ids)
            remaining_turns = [
                turn for turn in remaining_turns if turn.turn_id != candidate.turn_id
            ]
            hot_tokens -= sum(
                event.token_count
                for event in candidate_events
                if event.event_id not in hidden
            )
            hot_bytes -= sum(
                self._event_hot_bytes(event)
                for event in candidate_events
                if event.event_id not in hidden
            )
            hot_turns_count = len(remaining_turns)
            next_age_exceeded = (
                bool(terminal_turns)
                and turn_age(terminal_turns[0]) > self._hot_max_age_seconds
            )
            if not force and (
                hot_tokens <= low_tokens
                and hot_turns_count <= low_turns
                and hot_bytes <= low_bytes
                and not next_age_exceeded
            ):
                break

        if not selected_turns:
            return None
        selected_events = [
            event for event in source.events if event.event_id in selected_ids
        ]
        selected_turn_ids = [turn.turn_id for turn in selected_turns]
        operation_id = self._event_archive_operation_id(
            chat_id, source.cutoff_seq, selected_turn_ids
        )
        batch_id = f"batch:{operation_id}"
        turn_records = [
            ArchiveTurnRecord(
                turn_id=turn.turn_id,
                turn_sequence=turn.turn_sequence,
                source_date=turn.source_date,
                event_count=len(events_by_turn.get(turn.turn_id, ())),
                estimated_tokens=sum(
                    event.token_count for event in events_by_turn.get(turn.turn_id, ())
                ),
            )
            for turn in selected_turns
        ]
        batch = await index.prepare_batch(
            batch_id=batch_id,
            operation_id=operation_id,
            chat_id=chat_id,
            captured_cutoff_seq=source.cutoff_seq,
            turn_records=turn_records,
            event_ids=[(event.event_id, event.turn_id) for event in selected_events],
        )
        return await self._finish_event_archive_batch(batch)

    async def _archive_event_log_serialized(
        self,
        chat_id: str,
        is_group: bool,
        *,
        force: bool = False,
        session_lock_held: bool = False,
    ) -> Optional[ArchiveResult]:
        async def run_with_archive_lock() -> Optional[ArchiveResult]:
            lock = self._event_archive_locks.setdefault(chat_id, asyncio.Lock())
            async with lock:
                return await self._archive_event_log_if_needed(
                    chat_id, is_group, force=force
                )

        if session_lock_held or self._session_lock_provider is None:
            return await run_with_archive_lock()
        session_lock = await self._session_lock_provider(chat_id)
        async with session_lock:
            return await run_with_archive_lock()

    async def _sync_prompt_projection(self, manifest: Dict[str, Any]) -> None:
        projection = self._prompt_projection
        event_log = self._event_log
        if projection is None or event_log is None:
            return
        chat_id = str(manifest.get("chat_id") or "")
        operation_id = str(manifest.get("operation_id") or "")
        if not chat_id or not operation_id:
            return

        def event_id_from_identity(identity: Any) -> str:
            value = str(identity or "")
            if value.startswith("timeline:"):
                return value.removeprefix("timeline:")
            if value.startswith("event:"):
                return value.removeprefix("event:")
            return value

        hidden_ids = {
            event_id_from_identity(record.get("event_id"))
            for batch in manifest.get("batches", [])
            for record in batch.get("records", [])
            if record.get("event_id")
        }
        retained_ids = {
            event_id_from_identity(identity)
            for identity in manifest.get("keep_identities", [])
            if identity
        }
        hidden_ids -= retained_ids
        if not hidden_ids and not retained_ids:
            return
        snapshot = await event_log.snapshot_events(chat_id, include_internal=True)
        source_ids = {event.event_id for event in snapshot.events}
        await projection.apply_archive_retention(
            chat_id,
            operation_id=operation_id,
            hidden_event_ids=tuple(sorted(hidden_ids & source_ids)),
            retained_event_ids=tuple(sorted(retained_ids & source_ids)),
            captured_cutoff_seq=snapshot.cutoff_seq,
        )
        summary_store = self._summary_store
        summaries = ()
        if summary_store is not None and hidden_ids:
            try:
                recovered_summaries = []
                for batch in manifest.get("batches", []):
                    batch_id = str(batch.get("batch_id") or operation_id)
                    batch_event_ids = tuple(
                        event_id_from_identity(record.get("event_id"))
                        for record in batch.get("records", [])
                        if record.get("event_id")
                    )
                    if not batch_event_ids:
                        continue
                    recovered_summaries.extend(
                        await summary_store.ensure_for_archived_events(
                            chat_id,
                            batch_event_ids,
                            archive_batch_id=batch_id,
                        )
                    )
                summaries = tuple(recovered_summaries)
            except Exception as exc:
                _log.warning(
                    "确定性 TurnSummary 生成失败 [%s..]: %s", chat_id[:12], exc
                )
        transcript = self._model_context_transcript
        if transcript is not None and hidden_ids:
            try:
                scopes = await transcript.scopes_for_chat(chat_id)
                for scope in scopes:
                    scope_snapshot = await transcript.snapshot(scope)
                    scope_source_ids = scope_snapshot.source_event_ids
                    scope_summaries = tuple(
                        item
                        for item in summaries
                        if scope_source_ids.intersection(item.coverage_event_ids)
                    )
                    await transcript.rotate_for_hidden_sources(
                        scope,
                        tuple(hidden_ids),
                        summary_texts=tuple(item.text for item in scope_summaries),
                        summary_source_event_ids=tuple(
                            item.coverage_event_ids for item in scope_summaries
                        ),
                        operation_id=f"{operation_id}:model-context:{scope.key}",
                    )
            except Exception as exc:
                _log.warning("模型上下文归档轮换失败 [%s..]: %s", chat_id[:12], exc)

    def get_archive_operation_status(self, chat_id: str) -> Dict[str, Any]:
        committed_batches = 0
        latest_batch = None
        if self._ledger is not None:
            committed_batches = self._ledger.committed_batch_count(chat_id)
            latest_batch = self._ledger.latest_committed_batch(chat_id)
        return {
            "pending_operations": self._manifest_store.pending_count(chat_id),
            "committed_batches": committed_batches,
            "latest_committed_batch": latest_batch,
        }

    async def get_archive_operation_status_async(self, chat_id: str) -> Dict[str, Any]:
        if self._archive_index is not None:
            batches = await self._archive_index.list_for_webui(chat_id)
            committed = [
                batch for batch in batches if batch.get("state") == "committed"
            ]
            return {
                "pending_operations": len(
                    [batch for batch in batches if batch.get("state") == "prepared"]
                ),
                "committed_batches": len(committed),
                "latest_committed_batch": committed[0] if committed else None,
            }
        return self.get_archive_operation_status(chat_id)

    def recover_incomplete_archives(self) -> int:
        """Resume pending archive manifests and return the recovered count."""
        recovered = 0
        for manifest in self._manifest_store.load_pending():
            try:
                self._recover_manifest(manifest)
                if manifest.get("reason") != "manual" and any(
                    batch.get("summary_path") for batch in manifest.get("batches", [])
                ):
                    self._pending_injection.add(str(manifest["chat_id"]))
                recovered += 1
            except Exception as exc:
                _log.error(
                    "归档恢复失败 [%s]: %s",
                    manifest.get("operation_id", "unknown"),
                    exc,
                )
        if recovered:
            _log.info("归档恢复完成: %d 个 operation", recovered)
        return recovered

    def _write_manifest_state(self, manifest: Dict[str, Any], state: str) -> None:
        current_state = str(manifest.get("state", "prepared"))
        if _ARCHIVE_STATE_ORDER[state] >= _ARCHIVE_STATE_ORDER[current_state]:
            manifest["state"] = state
        self._manifest_store.write(manifest)

    def _recover_manifest(self, manifest: Dict[str, Any]) -> None:
        chat_id = str(manifest["chat_id"])
        for batch in manifest.get("batches", []):
            records = list(batch.get("records", []))
            expected_hash = batch.get("records_hash")
            if self._ledger is not None:
                self._ledger.recover_batch(
                    batch["batch_id"], chat_id, records_hash=expected_hash
                )
            archive_path = batch.get("archive_path")
            if archive_path and Path(archive_path).is_file():
                actual_hash = self._records_hash(
                    self._store.read_archive(archive_path, 0)
                )
                if expected_hash and actual_hash != expected_hash:
                    raise RuntimeError(f"archive hash mismatch: {archive_path}")
            else:
                archive_path = self._find_archive_for_batch(
                    chat_id, batch["batch_id"], expected_hash
                )
                if archive_path is None and records:
                    archive_batch = getattr(self._store, "archive_batch", None)
                    if callable(archive_batch):
                        archive_path = archive_batch(
                            chat_id,
                            batch["batch_id"],
                            batch["partition_date"],
                            records,
                            expected_hash,
                        )
                    else:
                        archive_path = self._store.archive_messages(
                            chat_id, batch["archive_ts"], records
                        )
                archive_path = _coerce_archive_path(archive_path)
                if records and archive_path is None:
                    raise RuntimeError(
                        f"archive adapter did not persist batch {batch['batch_id']}"
                    )
                batch["archive_path"] = archive_path
            if _ARCHIVE_STATE_ORDER["archive_written"] >= _ARCHIVE_STATE_ORDER.get(
                str(batch.get("state", "prepared")), 0
            ):
                batch["state"] = "archive_written"
            self._write_manifest_state(manifest, "archive_written")

        keep_messages = self._recovery_keep_messages(manifest, chat_id)
        if keep_messages != manifest.get("keep_messages", []):
            manifest["keep_messages"] = keep_messages
            manifest["keep_messages_hash"] = self._records_hash(keep_messages)
            self._manifest_store.write(manifest)
        replace = getattr(self._store, "replace", None)
        if callable(replace):
            replace(chat_id, keep_messages)
        elif keep_messages:
            self._store.flush(chat_id, keep_messages)
        else:
            self._store.delete(chat_id)
        self._write_manifest_state(manifest, "active_written")

        for batch in manifest.get("batches", []):
            summary_text = batch.get("summary_text")
            if summary_text:
                summary_path = batch.get("summary_path")
                if summary_path and Path(summary_path).is_file():
                    existing = Path(summary_path).read_text(encoding="utf-8")
                    actual_hash = hashlib.sha256(existing.encode("utf-8")).hexdigest()
                    if batch.get("summary_hash") != actual_hash:
                        raise RuntimeError(f"summary hash mismatch: {summary_path}")
                else:
                    summary_path = self._write_memory_file_sync(
                        chat_id,
                        batch["partition_date"],
                        summary_text,
                        batch["batch_id"],
                    )
                batch["summary_path"] = summary_path
            if _ARCHIVE_STATE_ORDER["summary_written"] >= _ARCHIVE_STATE_ORDER.get(
                str(batch.get("state", "prepared")), 0
            ):
                batch["state"] = "summary_written"
        self._write_manifest_state(manifest, "summary_written")

        if self._ledger is not None:
            for batch in manifest.get("batches", []):
                if batch.get("archive_path"):
                    self._ledger.commit_membership(
                        batch["batch_id"],
                        chat_id,
                        batch.get("identities", []),
                        records_hash=batch.get("records_hash"),
                    )
        daily_archive_on = manifest.get("daily_archive_on")
        if daily_archive_on:
            self._last_daily_archive[chat_id] = str(daily_archive_on)
            self._write_daily_state()
        self._write_manifest_state(manifest, "committed")

    def _find_archive_for_batch(
        self, chat_id: str, batch_id: str, expected_hash: str
    ) -> Optional[str]:
        for archive in self._store.list_archives(chat_id):
            path = archive.get("path") if isinstance(archive, dict) else None
            if not path or batch_id not in Path(path).name:
                continue
            actual_hash = self._records_hash(self._store.read_archive(path, 0))
            if actual_hash != expected_hash:
                raise RuntimeError(f"archive batch hash mismatch: {batch_id}")
            return path
        return None

    @staticmethod
    def _recovery_message_key(record: dict) -> str:
        role = str(record.get("role", ""))
        message_id = record.get("message_id")
        if message_id:
            return f"{role}:id:{message_id}"
        if role == "tool" and record.get("tool_call_id"):
            return f"tool:call:{record['tool_call_id']}"
        content = record.get("raw_content", record.get("content", "")) or ""
        return f"{role}:legacy:{record.get('timestamp', 0)}:{content}"

    @classmethod
    def _storage_archive_identity(cls, record: dict) -> str:
        return cls._archive_identity(ChatMessage.from_dict(record))

    def _recovery_keep_messages(
        self, manifest: Dict[str, Any], chat_id: str
    ) -> List[dict]:
        expected = list(manifest.get("keep_messages", []))
        active_before = list(manifest.get("active_before_messages", []))
        current = self._store.load(chat_id) or []
        current_hash = self._records_hash(current)
        if current_hash in {
            manifest.get("active_before_hash"),
            manifest.get("keep_messages_hash"),
        }:
            return expected

        expected_identities = list(manifest.get("keep_identities", []))
        expected_by_key = {
            self._recovery_message_key(record): record for record in expected
        }
        expected_by_identity = {
            identity: record
            for identity, record in zip(expected_identities, expected)
            if identity
        }
        archived_identities = {
            identity
            for batch in manifest.get("batches", [])
            for identity in batch.get("identities", [])
            if identity
        }
        archived_keys = {
            key
            for batch in manifest.get("batches", [])
            for key in batch.get("message_keys", [])
            if key
        }
        active_before_keys = {
            self._recovery_message_key(record) for record in active_before
        }
        merged: List[dict] = []
        seen_keys: set[str] = set()
        for record in current:
            identity = self._storage_archive_identity(record)
            key = self._recovery_message_key(record)
            selected = expected_by_key.get(key) or expected_by_identity.get(identity)
            if selected is None:
                if identity in archived_identities or key in archived_keys:
                    continue
                if key in active_before_keys:
                    continue
                selected = record
            selected_key = self._recovery_message_key(selected)
            if selected_key in seen_keys:
                continue
            seen_keys.add(selected_key)
            merged.append(selected)

        for record in expected:
            key = self._recovery_message_key(record)
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(record)
        return merged

    @property
    def _store(self):
        return self._cm.store

    @property
    def replay_gap_seconds(self) -> int:
        """昨天原始消息回放的连续会话间隔阈值（秒）。"""
        return self._replay_gap_seconds

    @property
    def summary_count(self) -> int:
        """摘要取最近 N 条有效消息。"""
        return self._summary_count

    def _date(self, timestamp: Optional[float] = None) -> str:
        return _date_str(timestamp, self._timezone_name)

    def _previous_date(self, timestamp: Optional[float] = None) -> str:
        return _previous_date_str(timestamp, self._timezone_name)

    def _archive_timestamp(self, timestamp: Optional[float] = None) -> str:
        return _format_archive_timestamp(timestamp, self._timezone_name)

    # ── 同日归档状态持久化 ──

    def _load_daily_state(self) -> None:
        """恢复同日守卫和已归档回放前缀，兼容旧版 {chat_id: date} 格式。"""
        try:
            if not self._daily_state_path.is_file():
                return
            data = json.loads(self._daily_state_path.read_text(encoding="utf-8"))
            for chat_id, value in data.items():
                if isinstance(value, dict):
                    archived_on = value.get("archived_on")
                    if archived_on:
                        self._last_daily_archive[chat_id] = str(archived_on)
                    if "replayed_prefix_keys" in value:
                        replayed_keys = value.get("replayed_prefix_keys", [])
                        if isinstance(replayed_keys, list) and all(
                            isinstance(key, str) for key in replayed_keys
                        ):
                            self._replayed_prefix_keys[chat_id] = replayed_keys
                            self._replayed_prefix_known.add(chat_id)
                elif isinstance(value, str):
                    self._last_daily_archive[chat_id] = value
        except Exception as e:
            _log.warning("加载归档状态失败 %s: %s", self._daily_state_path, e)

    def _save_daily_state(self) -> None:
        try:
            self._write_daily_state()
        except Exception as e:
            _log.warning("保存归档状态失败 %s: %s", self._daily_state_path, e)

    def _write_daily_state(self) -> None:
        """Atomically persist the daily archive guard."""
        self._daily_state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._daily_state_lock:
            state = {
                chat_id: {
                    "archived_on": archived_on,
                    **(
                        {
                            "replayed_prefix_keys": self._replayed_prefix_keys.get(
                                chat_id, []
                            )
                        }
                        if chat_id in self._replayed_prefix_known
                        else {}
                    ),
                }
                for chat_id, archived_on in self._last_daily_archive.items()
            }
            temporary_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self._daily_state_path.parent,
                    prefix=f".{self._daily_state_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary_path = Path(handle.name)
                    json.dump(state, handle, ensure_ascii=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, self._daily_state_path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _operation_id(chat_id: str, reason: str, batch_ids: List[str]) -> str:
        payload = json.dumps(
            {"chat_id": chat_id, "reason": reason, "batch_ids": batch_ids},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return "archive-op:" + hashlib.sha256(payload).hexdigest()[:32]

    def _archive_result_from_manifest(self, manifest: Dict[str, Any]) -> ArchiveResult:
        batches = [
            ArchiveBatchResult(
                batch_id=batch["batch_id"],
                partition_date=batch["partition_date"],
                archive_path=batch.get("archive_path"),
                summary_path=batch.get("summary_path"),
                message_count=len(batch.get("records", [])),
                event_count=len(batch.get("identities", [])),
                unit_count=batch.get("unit_count", 0),
            )
            for batch in manifest.get("batches", [])
        ]
        return ArchiveResult(
            chat_id=str(manifest["chat_id"]),
            reason=str(manifest.get("reason", "daily")),
            archive_path=batches[0].archive_path if batches else None,
            summary_path=batches[0].summary_path if batches else None,
            replay_count=len(manifest.get("keep_messages", [])),
            operation_id=str(manifest["operation_id"]),
            batches=batches,
        )

    def _restore_context_from_manifest(
        self, ctx: Any, manifest: Dict[str, Any]
    ) -> None:
        keep_messages = [
            ChatMessage.from_dict(record)
            for record in manifest.get("keep_messages", [])
        ]
        ctx.set_messages(keep_messages)
        ctx.last_activity = time.time()

    @staticmethod
    def _message_key(message: Any) -> str:
        """生成跨重启稳定的消息指纹，用于识别已归档的回放前缀。"""
        return json.dumps(
            {
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp,
                "message_id": message.message_id,
                "sender_id": message.sender_id,
                "name": message.name,
                "tool_call_id": message.tool_call_id,
                "tool_name": message.tool_name,
                "tool_calls": message.tool_calls,
                "reasoning_content": message.reasoning_content,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _replayed_prefix_length(self, chat_id: str, messages: List[Any]) -> int:
        metadata_length = 0
        for message in messages:
            if not getattr(message, "replayed_from_batch_id", None):
                break
            if self._ledger is not None and not self._ledger.is_archived(
                chat_id, self._archive_identity(message)
            ):
                break
            metadata_length += 1
        if metadata_length:
            return metadata_length

        keys = self._replayed_prefix_keys.get(chat_id, [])
        if not keys and chat_id not in self._replayed_prefix_known:
            # v1 state 只保存了归档日。active history 中早于该日的连续前缀
            # 必然是上一轮留下的回放，首次读取时将其升级为 v2 指纹。
            last_archived_on = self._last_daily_archive.get(chat_id)
            if last_archived_on:
                legacy_prefix = []
                for message, unit_start in zip(
                    messages, self._unit_start_timestamps(messages)
                ):
                    if self._date(unit_start) >= last_archived_on:
                        break
                    legacy_prefix.append(message)
                if legacy_prefix:
                    keys = [self._message_key(message) for message in legacy_prefix]
                    self._replayed_prefix_keys[chat_id] = keys
                    _log.info(
                        "已迁移归档回放前缀 [%s..]: %d 条",
                        chat_id[:12],
                        len(keys),
                    )
            self._replayed_prefix_known.add(chat_id)

        if not keys:
            return 0

        # ChatContext 是有界 deque，后续消息可能将回放前缀的最早记录挤出。
        # active history 的开头仍与已知前缀的一个后缀相同，就必须继续跳过
        # 这部分，不能将剩余回放消息视为新的待归档历史。
        max_overlap = min(len(messages), len(keys))
        for length in range(max_overlap, 0, -1):
            actual_keys = [self._message_key(message) for message in messages[:length]]
            if actual_keys == keys[-length:]:
                if length != len(keys):
                    _log.info(
                        "回放前缀已截断 [%s..]: 保留 %d/%d 条已归档记录",
                        chat_id[:12],
                        length,
                        len(keys),
                    )
                return length

        _log.warning("回放前缀不匹配 [%s..]，按未归档消息处理", chat_id[:12])
        return 0

    # ── 公开方法 ──

    async def archive_if_stale(
        self, chat_id: str, is_group: bool, *, session_lock_held: bool = False
    ) -> Optional[ArchiveResult]:
        if self._event_log is not None and self._archive_index is not None:
            return await self._archive_event_log_serialized(
                chat_id, is_group, session_lock_held=session_lock_held
            )

        async def _do(ctx):
            now = time.time()
            today = self._date(now)

            wait_for_save = getattr(ctx, "wait_for_save_async", None)
            if callable(wait_for_save):
                pending_save = wait_for_save()
                if inspect.isawaitable(pending_save):
                    await pending_save

            pending = self._manifest_store.find_pending(chat_id, "daily")
            if pending is not None:
                self._recover_manifest(pending)
                await self._sync_prompt_projection(pending)
                self._restore_context_from_manifest(ctx, pending)
                result = self._archive_result_from_manifest(pending)
                if result.summary_path:
                    self._pending_injection.add(chat_id)
                return result

            history = ctx.get_history()
            merged_history = await self._merge_timeline_messages(chat_id, history)
            if len(merged_history) != len(history):
                ctx.set_messages(merged_history)
                history = merged_history
            if not history:
                return None
            if not self._crossed_day(history, today, self._timezone_name):
                return None

            # active history 可能以已归档的回放前缀开头。只有该前缀之后仍有
            # 今天之前的新消息时，才需要生成新的 archive 文件。
            replayed_prefix_length = self._replayed_prefix_length(chat_id, history)
            unit_start_times = self._unit_start_timestamps(history)
            has_unarchived_old_history = any(
                self._date(unit_start) < today
                for unit_start in unit_start_times[replayed_prefix_length:]
            )
            if not has_unarchived_old_history:
                return None

            # 同一天通常只会进入一次；但迟到消息可能在此前归档之后才写入
            # history，必须允许它作为新的旧单元在当天补归档。
            if self._last_daily_archive.get(chat_id) == today:
                _log.info("检测到迟到旧消息，追加同日归档 [%s..]", chat_id[:12])

            result = await self._do_archive(ctx, chat_id, is_group, "daily")
            return result

        return await self._cm._with_context_locked(chat_id, _do)

    @staticmethod
    def _tool_transaction_start_indices(messages: List[Any]) -> Dict[int, int]:
        """返回 tool result 下标到发起该调用的 assistant 下标的映射。

        此映射只用于分区和回放切点判断；ChatMessage.timestamp 始终保持原值。
        """
        call_owners: Dict[str, int] = {}
        for index, message in enumerate(messages):
            if message.role != "assistant" or not message.tool_calls:
                continue
            for call in message.tool_calls:
                if isinstance(call, dict) and call.get("id"):
                    call_owners[call["id"]] = index

        return {
            index: call_owners[message.tool_call_id]
            for index, message in enumerate(messages)
            if message.role == "tool" and message.tool_call_id in call_owners
        }

    def _build_archive_units(self, messages: List[Any]) -> List[ArchiveUnit]:
        """Build stable logical units while retaining original message order."""
        transaction_starts = self._tool_transaction_start_indices(messages)
        transaction_results: Dict[int, List[int]] = {}
        for result_index, owner_index in transaction_starts.items():
            transaction_results.setdefault(owner_index, []).append(result_index)
        result_call_ids = {
            message.tool_call_id
            for message in messages
            if message.role == "tool" and message.tool_call_id
        }
        incomplete_owners = {
            index
            for index, message in enumerate(messages)
            if message.role == "assistant"
            and message.tool_calls
            and any(
                isinstance(call, dict) and call.get("id") not in result_call_ids
                for call in message.tool_calls
            )
        }

        units: List[ArchiveUnit] = []
        consumed: Set[int] = set()
        for index, message in enumerate(messages):
            if index in consumed:
                continue
            if index in transaction_results:
                indices = tuple(sorted((index, *transaction_results[index])))
                kind = "tool_transaction"
            else:
                indices = (index,)
                kind = "message"
            consumed.update(indices)
            identities = [self._archive_identity(messages[item]) for item in indices]
            unit_id = (
                "unit-v1:"
                + hashlib.sha256(
                    json.dumps(identities, ensure_ascii=False).encode("utf-8")
                ).hexdigest()[:32]
            )
            partition_time = messages[index].timestamp
            activity_end_time = max(messages[item].timestamp for item in indices)
            replayable = any(self._is_replayable(messages[item]) for item in indices)
            units.append(
                ArchiveUnit(
                    unit_id=unit_id,
                    kind=kind,
                    message_indices=indices,
                    message_identities=tuple(identities),
                    partition_time=partition_time,
                    activity_end_time=activity_end_time,
                    replayable=replayable,
                    incomplete=index in incomplete_owners,
                )
            )
        return units

    @staticmethod
    def _records_hash(records: List[dict]) -> str:
        payload = "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _batch_records(
        messages: List[Any], identities: List[str], batch_id: str
    ) -> List[dict]:
        records = []
        for index, (message, identity) in enumerate(zip(messages, identities)):
            record = message.to_storage_dict()
            record["record_id"] = (
                "record-v1:"
                + hashlib.sha256(
                    f"{batch_id}:{index}:{identity}".encode("utf-8")
                ).hexdigest()[:32]
            )
            record["source_batch_id"] = batch_id
            records.append(record)
        return records

    @classmethod
    def _unit_start_timestamps(cls, messages: List[Any]) -> List[float]:
        """计算每条记录所属单元的开始时间，不修改记录本身的 timestamp。"""
        transaction_starts = cls._tool_transaction_start_indices(messages)
        return [
            messages[transaction_starts.get(index, index)].timestamp
            for index in range(len(messages))
        ]

    @classmethod
    def _unit_activity_end_timestamps(cls, messages: List[Any]) -> List[float]:
        transaction_starts = cls._tool_transaction_start_indices(messages)
        transaction_ends: Dict[int, float] = {}
        for index, message in enumerate(messages):
            owner = transaction_starts.get(index, index)
            transaction_ends[owner] = max(
                transaction_ends.get(owner, message.timestamp), message.timestamp
            )
        return [
            transaction_ends[transaction_starts.get(index, index)]
            for index in range(len(messages))
        ]

    @classmethod
    def _crossed_day(
        cls,
        messages: List[Any],
        today: str,
        timezone_name: str = "Asia/Shanghai",
    ) -> bool:
        """是否有按所属单元开始时间归属到今天之前的消息。"""
        return any(
            _date_str(unit_start, timezone_name) < today
            for unit_start in cls._unit_start_timestamps(messages)
        )

    def load_recent_summaries(self, chat_id: str) -> Optional[str]:
        mem_dir = _get_memory_dir(self._memory_dir, chat_id)
        if not mem_dir.is_dir():
            return None

        now = time.time()
        parts: List[str] = []
        for day_offset in range(self._summary_days):
            date = self._date(now - day_offset * 86400)
            day_files = sorted(
                [mem_dir / f"{date}.md"] + list(mem_dir.glob(f"{date}.*.md")),
                key=lambda path: path.name,
            )
            for day_file in day_files:
                if not day_file.is_file():
                    continue
                try:
                    text = day_file.read_text(encoding="utf-8").strip()
                    if text:
                        parts.append(f"{date}:\n{text}")
                except Exception as e:
                    _log.warning(
                        "读取归档摘要失败 [%s..] %s: %s",
                        chat_id[:12],
                        day_file.name,
                        e,
                    )

        return "\n\n---\n\n".join(parts) if parts else None

    async def load_recent_summaries_async(self, chat_id: str) -> Optional[str]:
        if self._summary_store is not None:
            eligible_batches = await self._prompt_summary_batch_ids(chat_id)
            selection = await self._summary_store.select_for_prompt(
                chat_id, eligible_archive_batch_ids=eligible_batches
            )
            if selection.text:
                return selection.text
            if self._event_log is not None and self._archive_index is not None:
                return None
        if self._event_log is not None and self._archive_index is not None:
            return None
        return await asyncio.to_thread(self.load_recent_summaries, chat_id)

    async def get_prompt_summaries_async(
        self, chat_id: str, *, covered_event_ids: Sequence[str] = ()
    ) -> Optional[str]:
        """Read a bounded, durable summary projection for prompt assembly."""
        summary_store = self._summary_store
        if summary_store is not None:
            eligible_batches = await self._prompt_summary_batch_ids(chat_id)
            selection = await summary_store.select_for_prompt(
                chat_id,
                covered_event_ids=covered_event_ids,
                eligible_archive_batch_ids=eligible_batches,
            )
            return selection.text or None
        return await self.load_recent_summaries_async(chat_id)

    async def get_prompt_summary_selection_async(
        self, chat_id: str, *, covered_event_ids: Sequence[str] = ()
    ) -> Any:
        """Return the bounded summary selection used for prompt diagnostics."""
        if self._summary_store is None:
            return type("SummarySelection", (), {"summaries": ()})()
        eligible_batches = await self._prompt_summary_batch_ids(chat_id)
        return await self._summary_store.select_for_prompt(
            chat_id,
            covered_event_ids=covered_event_ids,
            eligible_archive_batch_ids=eligible_batches,
        )

    async def _prompt_summary_batch_ids(self, chat_id: str) -> Optional[frozenset[str]]:
        if self._archive_index is None:
            return None
        batches = await self._archive_index.list_for_webui(chat_id)
        return frozenset(
            str(batch["batch_id"])
            for batch in batches
            if batch.get("state") in {"committed", "export_degraded", "soft_deleted"}
        )

    async def list_archive_batches_async(self, chat_id: str) -> list[dict[str, Any]]:
        if self._archive_index is None:
            return []
        return await self._archive_index.list_for_webui(chat_id)

    async def archive_batch_turns_async(self, batch_id: str) -> list[ArchiveTurnRecord]:
        if self._archive_index is None:
            return []
        return await self._archive_index.turns_for_batch(batch_id)

    async def archive_batch_event_ids_async(self, batch_id: str) -> frozenset[str]:
        if self._archive_index is None:
            return frozenset()
        return await self._archive_index.event_ids(batch_id)

    async def export_archives_async(
        self, chat_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Retry optional JSONL exports without changing archive membership."""
        adapter = self._export_adapter
        index = self._archive_index
        if adapter is None or index is None:
            return []
        results = await adapter.export_pending(chat_id)
        for result in results:
            await index.record_export(
                result.batch_id,
                status=result.status,
                path=result.path,
                content_hash=result.content_hash,
                manifest_hash=result.manifest_hash,
                error=result.error,
            )
        return [result.__dict__ for result in results]

    async def import_legacy_archives_async(self, chat_id: str) -> dict[str, Any]:
        """Import legacy archive files once into the ledger during migration."""
        if self._event_log is None or self._archive_index is None:
            return {
                "chat_id": chat_id,
                "archive_count": 0,
                "imported_event_count": 0,
                "imported_batch_count": 0,
                "error_count": 0,
                "status": "ledger_not_configured",
            }
        imported_events = 0
        imported_batches = 0
        error_count = 0
        for archive in self._store.list_archives(chat_id):
            path = archive.get("path") if isinstance(archive, dict) else None
            if not path:
                continue
            try:
                records = await asyncio.to_thread(self._store.read_archive, path, 0)
                before = await self._event_log.snapshot_events(
                    chat_id, include_internal=True
                )
                before_ids = {event.event_id for event in before.events}
                await self._event_log.repair_from_legacy_history(
                    chat_id,
                    records,
                    source_id=Path(path).name,
                )
                snapshot = await self._event_log.snapshot_events(
                    chat_id, include_internal=True
                )
                new_ids = {
                    event.event_id
                    for event in snapshot.events
                    if event.event_id not in before_ids
                }
                if not new_ids:
                    continue
                new_turn_ids = {
                    event.turn_id
                    for event in snapshot.events
                    if event.event_id in new_ids
                }
                for turn_id in new_turn_ids:
                    turn_report = await self._event_log.validate_turn(turn_id)
                    turn = next(
                        (
                            item
                            for item in (
                                await self._event_log.snapshot_turns(
                                    chat_id, include_internal=True
                                )
                            ).turns
                            if item.turn_id == turn_id
                        ),
                        None,
                    )
                    if turn is not None and not turn.is_terminal:
                        await self._event_log.append_turn_terminal(
                            chat_id=chat_id,
                            turn_id=turn_id,
                            status=("completed" if turn_report.valid else "incomplete"),
                        )
                snapshot = await self._event_log.snapshot_events(
                    chat_id, include_internal=True
                )
                imported_event_ids = {
                    event.event_id
                    for event in snapshot.events
                    if event.event_id in new_ids or event.turn_id in new_turn_ids
                }
                committed = await self._archive_index.committed_event_ids(chat_id)
                imported_event_ids -= set(committed)
                turn_records = []
                for turn_id in sorted(new_turn_ids):
                    turn = next(
                        item
                        for item in (
                            await self._event_log.snapshot_turns(
                                chat_id, include_internal=True
                            )
                        ).turns
                        if item.turn_id == turn_id
                    )
                    turn_events = tuple(
                        event
                        for event in snapshot.events
                        if event.turn_id == turn_id
                        and event.event_id in imported_event_ids
                    )
                    if not turn_events or not turn.is_terminal:
                        continue
                    report = await self._event_log.validate_turn(turn_id)
                    if not report.valid:
                        continue
                    turn_records.append(
                        ArchiveTurnRecord(
                            turn_id=turn_id,
                            turn_sequence=turn.turn_sequence,
                            source_date=turn.source_date,
                            event_count=len(turn_events),
                            estimated_tokens=sum(
                                event.token_count for event in turn_events
                            ),
                        )
                    )
                if not turn_records:
                    continue
                selected_turn_ids = {record.turn_id for record in turn_records}
                selected_event_ids = [
                    (event.event_id, event.turn_id)
                    for event in snapshot.events
                    if event.turn_id in selected_turn_ids
                    and event.event_id in imported_event_ids
                ]
                operation_id = (
                    "legacy-import:"
                    + hashlib.sha256(
                        f"{chat_id}:{Path(path).name}:{self._archive_index.membership_hash(item[0] for item in selected_event_ids)}".encode()
                    ).hexdigest()[:32]
                )
                batch = await self._archive_index.prepare_batch(
                    batch_id=f"batch:{operation_id}",
                    operation_id=operation_id,
                    chat_id=chat_id,
                    captured_cutoff_seq=snapshot.cutoff_seq,
                    turn_records=turn_records,
                    event_ids=selected_event_ids,
                )
                await self._finish_event_archive_batch(batch)
                imported_events += len(selected_event_ids)
                imported_batches += 1
            except Exception as exc:
                error_count += 1
                _log.warning(
                    "legacy archive import failed [%s..] %s: %s",
                    chat_id[:12],
                    Path(path).name,
                    exc,
                )
        return {
            "chat_id": chat_id,
            "archive_count": len(self._store.list_archives(chat_id)),
            "imported_event_count": imported_events,
            "imported_batch_count": imported_batches,
            "error_count": error_count,
            "status": "ok" if error_count == 0 else "degraded",
        }

    def consume_summary(self, chat_id: str) -> Optional[str]:
        if chat_id not in self._pending_injection:
            return None
        self._pending_injection.discard(chat_id)
        return self.load_recent_summaries(chat_id)

    async def consume_summary_async(self, chat_id: str) -> Optional[str]:
        if chat_id not in self._pending_injection:
            return None
        self._pending_injection.discard(chat_id)
        return await self.load_recent_summaries_async(chat_id)

    async def get_session_status_async(self, chat_id: str) -> Dict[str, Any]:
        if self._event_log is not None:
            summary = await self._event_log.session_summary(chat_id)
            batches = await self.list_archive_batches_async(chat_id)
            return {
                "message_count": summary["message_count"],
                "last_activity": summary["last_activity"],
                "archive_count": sum(
                    1 for batch in batches if batch.get("state") == "committed"
                ),
            }
        summary = None
        if self._timeline is not None:
            try:
                candidate = await self._timeline.session_summary(chat_id)
                if candidate.get("message_count", 0):
                    summary = candidate
            except Exception:
                summary = None
        if summary is not None:
            message_count = summary["message_count"]
            last_activity = summary["last_activity"]
        else:
            history = await self._cm.get_chat_history_async(chat_id)
            if self._timeline is not None:
                repair = getattr(self._timeline, "repair_from_legacy_history", None)
                if repair is not None:
                    await repair(chat_id, history)
                repaired = await self._timeline.session_summary(chat_id)
                if repaired.get("message_count", 0):
                    message_count = repaired["message_count"]
                    last_activity = repaired["last_activity"]
                else:
                    message_count = len(history)
                    last_activity = (
                        history[-1].get("timestamp", time.time()) if history else None
                    )
            else:
                message_count = len(history)
                last_activity = (
                    history[-1].get("timestamp", time.time()) if history else None
                )
        return {
            "message_count": message_count,
            "last_activity": last_activity,
            "archive_count": len(
                await asyncio.to_thread(
                    lambda: list(
                        (_get_memory_dir(self._memory_dir, chat_id)).glob("*.md")
                    )
                )
            ),
        }

    async def archive_manual(
        self, chat_id: str, is_group: Optional[bool] = None
    ) -> ArchiveResult:
        if self._event_log is not None and self._archive_index is not None:
            result = await self._archive_event_log_serialized(
                chat_id, bool(is_group), force=True
            )
            return result or ArchiveResult(chat_id=chat_id, reason="retention")

        if is_group is None:
            is_group = self._cm.get_chat_type(chat_id)
            if is_group is None:
                _log.warning("未记录聊天类型 [%s..]，按私聊归档", chat_id[:12])
                is_group = False

        async def _do(ctx):
            return await self._do_archive(ctx, chat_id, is_group, "manual")

        result = await self._cm._with_context_locked(chat_id, _do)
        if result is not None:
            # 手动归档同样可能保留已归档的回放前缀；重启后必须能识别它。
            self._save_daily_state()
        return result

    async def archive_snapshot(
        self, chat_id: str, is_group: Optional[bool] = None
    ) -> ArchiveResult:
        if self._event_log is not None and self._archive_index is not None:
            snapshot = await self._event_log.snapshot_events(
                chat_id, include_internal=False
            )
            hidden = (
                await self._prompt_projection.hidden_event_ids(chat_id)
                if self._prompt_projection is not None
                else frozenset()
            )
            active = tuple(
                event for event in snapshot.events if event.event_id not in hidden
            )
            operation_id = f"snapshot:{chat_id}:{snapshot.cutoff_seq}"
            return ArchiveResult(
                chat_id,
                "snapshot",
                replay_count=len(active),
                operation_id=operation_id,
                batches=(
                    [
                        ArchiveBatchResult(
                            batch_id=operation_id,
                            partition_date=active[0].source_date if active else "",
                            message_count=len(active),
                            event_count=len(active),
                            unit_count=len({event.turn_id for event in active}),
                        )
                    ]
                    if active
                    else []
                ),
            )

        async def _do(ctx):
            history = list(ctx.get_history())
            operation_id = f"snapshot-op:{uuid.uuid4().hex}"
            batch_id = f"snapshot-v1:{uuid.uuid4().hex}"
            if not history:
                return ArchiveResult(
                    chat_id,
                    "snapshot",
                    operation_id=operation_id,
                )
            records = [replace(message).to_storage_dict() for message in history]
            archive_path = await asyncio.to_thread(
                self._store.archive_messages,
                chat_id,
                batch_id,
                records,
            )
            archive_path = _coerce_archive_path(archive_path)
            if (
                records
                and archive_path is None
                and isinstance(self._store, ContextStore)
            ):
                raise RuntimeError(
                    f"archive adapter did not persist snapshot {batch_id}"
                )
            batch = ArchiveBatchResult(
                batch_id=batch_id,
                partition_date=self._date(history[0].timestamp),
                archive_path=archive_path,
                message_count=len(records),
            )
            return ArchiveResult(
                chat_id,
                "snapshot",
                archive_path=archive_path,
                replay_count=len(history),
                operation_id=operation_id,
                batches=[batch],
            )

        return await self._cm._with_context_locked(chat_id, _do)

    def cleanup_old_archives(self) -> int:
        if self._event_log is not None and self._archive_index is not None:
            return 0
        retention_seconds = self._retention_days * 86400
        removed = self._store.cleanup_stale_archives(retention_seconds)

        # 清理过期的 .md 摘要
        mem_root = Path(self._memory_dir)
        cutoff = time.time() - retention_seconds
        if mem_root.is_dir():
            for chat_dir in mem_root.iterdir():
                if not chat_dir.is_dir():
                    continue
                for f in chat_dir.iterdir():
                    if f.suffix == ".md":
                        try:
                            mtime = f.stat().st_mtime
                            if mtime < cutoff:
                                f.unlink()
                                removed += 1
                        except Exception as e:
                            _log.warning("清理摘要文件失败 %s: %s", f.name, e)

        # 清理过期的同日归档状态条目
        cutoff_date = self._date(time.time() - retention_seconds)
        stale_chats = [
            cid for cid, d in self._last_daily_archive.items() if d < cutoff_date
        ]
        if stale_chats:
            for cid in stale_chats:
                self._last_daily_archive.pop(cid, None)
                self._replayed_prefix_keys.pop(cid, None)
                self._replayed_prefix_known.discard(cid)
            self._save_daily_state()

        if removed:
            _log.info("归档清理完成: 移除了 %d 个文件", removed)
        return removed

    async def cleanup_old_archives_async(self) -> int:
        return await asyncio.to_thread(self.cleanup_old_archives)

    async def list_archives_async(self, chat_id: str) -> List[dict]:
        if self._archive_index is not None:
            return await self.list_archive_batches_async(chat_id)
        return await asyncio.to_thread(self._list_memory_files, chat_id)

    def _list_memory_files(self, chat_id: str) -> List[dict]:
        mem_dir = _get_memory_dir(self._memory_dir, chat_id)
        if not mem_dir.is_dir():
            return []
        return [
            {"path": str(path), "size": path.stat().st_size}
            for path in mem_dir.glob("*.md")
        ]

    # ── 内部方法（需在 per-chat 锁内调用） ──

    async def _do_archive(
        self, ctx: Any, chat_id: str, is_group: bool, reason: str
    ) -> ArchiveResult:
        store = self._store
        now = time.time()
        date = self._date(now)

        # ChatContext 可能还有旧 history 的异步保存任务。必须先等待，避免
        # 该任务在归档完成后将已归档消息重新追加到 active JSONL。
        wait_for_save = getattr(ctx, "wait_for_save_async", None)
        if callable(wait_for_save):
            pending_save = wait_for_save()
            if inspect.isawaitable(pending_save):
                await pending_save

        pending = self._manifest_store.find_pending(chat_id, reason)
        if pending is not None:
            self._recover_manifest(pending)
            await self._sync_prompt_projection(pending)
            self._restore_context_from_manifest(ctx, pending)
            return self._archive_result_from_manifest(pending)

        active_history = list(ctx.get_history())
        active_before_messages = [
            message.to_storage_dict() for message in active_history
        ]
        active_before_identities = [
            self._archive_identity(message) for message in active_history
        ]

        # 2. 收集消息。active history 的前缀可能是上一轮已进入 archive 的
        # 回放消息；它们仅用于上下文，不应再次写入新 archive。
        source_msgs = await self._merge_timeline_messages(chat_id, ctx.get_history())
        all_msgs = await self._apply_timeline_projection(chat_id, source_msgs)
        archive_units = self._build_archive_units(source_msgs)
        unit_by_index = {
            index: unit for unit in archive_units for index in unit.message_indices
        }
        unit_start_times = self._unit_start_timestamps(source_msgs)
        unit_activity_end_times = self._unit_activity_end_timestamps(source_msgs)
        replayed_prefix_length = self._replayed_prefix_length(chat_id, source_msgs)
        source_identities = [self._archive_identity(message) for message in source_msgs]
        archived_identities: set[str] = set()
        archived_batch_ids: Dict[str, str] = {}
        if self._ledger is not None:
            archived_identities = await asyncio.to_thread(
                self._ledger.archived_identities,
                chat_id,
                source_identities,
            )
            archived_batch_ids = await asyncio.to_thread(
                self._ledger.batch_ids_for_identities,
                chat_id,
                source_identities,
            )
        unarchived_indices = [
            index
            for index in range(replayed_prefix_length, len(source_msgs))
            if source_identities[index] not in archived_identities
        ]
        old_indices = [
            index
            for index in unarchived_indices
            if self._date(unit_start_times[index]) < date
            and not unit_by_index[index].incomplete
        ]
        incomplete_indices = {
            index for index in unarchived_indices if unit_by_index[index].incomplete
        }
        if incomplete_indices:
            _log.warning(
                "检测到未闭合工具事务，延迟归档 [%s..]: units=%d",
                chat_id[:12],
                len({unit_by_index[index].unit_id for index in incomplete_indices}),
            )
        old_history_indices = [
            index
            for index, unit_start in enumerate(unit_start_times)
            if self._date(unit_start) == self._previous_date(now)
        ]
        today_indices = [
            index
            for index in unarchived_indices
            if self._date(unit_start_times[index]) >= date
        ]
        old_msgs = [all_msgs[index] for index in old_indices]

        # 3. 今天及以后的单元全部保留。只允许回放昨天最后一个连续
        # 时间段；更早的历史仍会归档，但不能作为原始上下文进入今天。
        following_unit_start = (
            unit_start_times[today_indices[0]] if today_indices else None
        )
        replay_indices = self._select_replay_indices(
            source_msgs,
            old_history_indices,
            self._replay_gap_seconds,
            unit_start_times,
            following_unit_start,
            unit_activity_end_times,
        )

        base_keep_indices = (
            set(today_indices) | set(replay_indices) | incomplete_indices
        )
        # tool 调用与其全部结果是一个协议事务。跨日时，只要事务中任一消息
        # 必须保留，就将 assistant tool_calls 和每个对应 tool result 一起保留。
        keep_indices = self._close_tool_transactions(
            source_msgs, set(base_keep_indices)
        )
        # 仅因工具事务闭包而保留的旧消息尚不完整归档，留到整个事务都属于
        # 旧历史时再写入 archive；普通回放消息仍会在本次 archive 中保留副本。
        transaction_carried_indices = keep_indices - base_keep_indices
        archive_indices = [
            index for index in old_indices if index not in transaction_carried_indices
        ]
        archived_msgs = [all_msgs[index] for index in archive_indices]
        if not archive_indices:
            return None
        sorted_keep_indices = sorted(keep_indices)
        keep_msgs = [
            self._copy_for_history(all_msgs[index]) for index in sorted_keep_indices
        ]

        # 4. 每个来源日独立生成 archive 和摘要，延迟触发不会混合多天数据。
        partition_indices: Dict[str, List[int]] = {}
        for index in archive_indices:
            partition_date = self._date(unit_by_index[index].partition_time)
            partition_indices.setdefault(partition_date, []).append(index)
        batch_plans: List[Dict[str, Any]] = []
        for partition_date, indices in sorted(partition_indices.items()):
            partition_msgs = [all_msgs[index] for index in indices]
            batch_id = (
                "archive-v1:"
                + hashlib.sha256(
                    json.dumps(
                        {
                            "chat_id": chat_id,
                            "date": partition_date,
                            "reason": reason,
                            "identities": [
                                source_identities[index] for index in indices
                            ],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()[:32]
            )
            summary_text = self._format_summary_text(
                partition_msgs,
                self._summary_count,
                is_group,
                chat_id,
                partition_date,
            )
            records = self._batch_records(
                partition_msgs,
                [source_identities[index] for index in indices],
                batch_id,
            )
            batch_plans.append(
                {
                    "batch_id": batch_id,
                    "partition_date": partition_date,
                    "message_keys": [
                        self._recovery_message_key(message.to_storage_dict())
                        for message in partition_msgs
                    ],
                    "identities": [source_identities[index] for index in indices],
                    "unit_count": len(
                        {unit_by_index[index].unit_id for index in indices}
                    ),
                    "archive_ts": f"{partition_date}.{batch_id}",
                    "records": records,
                    "records_hash": self._records_hash(records),
                    "summary_text": summary_text,
                    "summary_hash": (
                        hashlib.sha256(summary_text.encode("utf-8")).hexdigest()
                        if summary_text
                        else None
                    ),
                    "archive_path": None,
                    "summary_path": None,
                    "state": "prepared",
                }
            )

        batch_by_identity = {
            identity: batch["batch_id"]
            for batch in batch_plans
            for identity in batch["identities"]
        }
        for position, index in enumerate(sorted_keep_indices):
            if index not in replay_indices:
                continue
            batch_id = archived_batch_ids.get(
                source_identities[index]
            ) or batch_by_identity.get(source_identities[index])
            if batch_id:
                keep_msgs[position].replayed_from_batch_id = batch_id

        operation_id = self._operation_id(
            chat_id, reason, [batch["batch_id"] for batch in batch_plans]
        )
        keep_records = [message.to_storage_dict() for message in keep_msgs]
        manifest: Dict[str, Any] = {
            "version": 1,
            "operation_id": operation_id,
            "chat_id": chat_id,
            "reason": reason,
            "state": "prepared",
            "active_before_messages": active_before_messages,
            "active_before_identities": active_before_identities,
            "active_before_hash": self._records_hash(active_before_messages),
            "keep_messages": keep_records,
            "keep_identities": [
                source_identities[index] for index in sorted_keep_indices
            ],
            "keep_messages_hash": self._records_hash(keep_records),
            "batches": batch_plans,
            "incomplete_units": sorted(
                {unit_by_index[index].unit_id for index in incomplete_indices}
            ),
        }
        if reason == "daily":
            manifest["daily_archive_on"] = date
        existing_manifest = await asyncio.to_thread(
            self._manifest_store.load, operation_id
        )
        if existing_manifest and existing_manifest.get("state") == "committed":
            return None
        if existing_manifest and existing_manifest.get("state") != "committed":
            await asyncio.to_thread(self._recover_manifest, existing_manifest)
            await self._sync_prompt_projection(existing_manifest)
            self._restore_context_from_manifest(ctx, existing_manifest)
            result = self._archive_result_from_manifest(existing_manifest)
            if result.summary_path and reason != "manual":
                self._pending_injection.add(chat_id)
            return result
        if len(source_msgs) != len(active_history):
            ctx.set_messages(source_msgs)
        if batch_plans:
            await asyncio.to_thread(self._manifest_store.write, manifest)

        await asyncio.to_thread(
            store.flush,
            chat_id,
            [m.to_storage_dict() for m in active_history],
        )

        batch_results: List[ArchiveBatchResult] = []
        for batch in batch_plans:
            if self._ledger is not None:
                await asyncio.to_thread(
                    self._ledger.prepare_batch,
                    batch["batch_id"],
                    chat_id,
                    batch["records_hash"],
                )
            archive_batch = getattr(store, "archive_batch", None)
            if callable(archive_batch):
                archive_path = await asyncio.to_thread(
                    archive_batch,
                    chat_id,
                    batch["batch_id"],
                    batch["partition_date"],
                    batch["records"],
                    batch["records_hash"],
                )
            else:
                archive_path = await asyncio.to_thread(
                    self._find_archive_for_batch,
                    chat_id,
                    batch["batch_id"],
                    batch["records_hash"],
                )
                if archive_path is None:
                    archive_path = await asyncio.to_thread(
                        store.archive_messages,
                        chat_id,
                        f"{batch['partition_date']}.{batch['batch_id']}",
                        batch["records"],
                    )
            archive_path = _coerce_archive_path(archive_path)
            if (
                batch["records"]
                and archive_path is None
                and isinstance(store, ContextStore)
            ):
                raise RuntimeError(
                    f"archive adapter did not persist batch {batch['batch_id']}"
                )
            batch["archive_path"] = archive_path
            batch["state"] = "archive_written"
            manifest["state"] = "archive_written"
            await asyncio.to_thread(self._manifest_store.write, manifest)
            batch_results.append(
                ArchiveBatchResult(
                    batch_id=batch["batch_id"],
                    partition_date=batch["partition_date"],
                    archive_path=archive_path,
                    message_count=len(batch["records"]),
                    event_count=len(batch["identities"]),
                    unit_count=batch["unit_count"],
                )
            )

        # 5. 保留新历史（今天全部 + 昨天最后一个连续时间段），并持久化其中
        # 已归档的回放前缀。该前缀在下一天只参与上下文，不会第二次写入 archive。
        ctx.set_messages(keep_msgs)
        archived_keep_indices = set(range(replayed_prefix_length)) | set(
            archive_indices
        )
        ctx.last_activity = time.time()

        # 6. 写入新数据（后台线程）。没有任何保留消息时必须删除 active
        # history；具体存储由 ContextStore adapter 决定。
        await asyncio.to_thread(
            getattr(store, "replace", store.flush),
            chat_id,
            [m.to_storage_dict() for m in keep_msgs],
        )
        if batch_plans:
            manifest["state"] = "active_written"
            await asyncio.to_thread(self._manifest_store.write, manifest)

        # 7. 摘要写入在 active 稳定后进行，重复执行时复用批次文件。
        for batch in batch_plans:
            if batch["summary_text"]:
                batch["summary_path"] = await self._write_memory_file(
                    chat_id,
                    batch["partition_date"],
                    batch["summary_text"],
                    batch["batch_id"],
                )
            batch["state"] = "summary_written"
        if batch_plans:
            manifest["state"] = "summary_written"
            await asyncio.to_thread(self._manifest_store.write, manifest)

        # 8. ledger 最后提交 membership，确保恢复时可以区分 active 与 archive。
        for index, batch in enumerate(batch_plans):
            if self._ledger is not None and batch["archive_path"]:
                await asyncio.to_thread(
                    self._ledger.commit_membership,
                    batch["batch_id"],
                    chat_id,
                    batch["identities"],
                    batch["records_hash"],
                )
            batch_results[index] = ArchiveBatchResult(
                batch_id=batch["batch_id"],
                partition_date=batch["partition_date"],
                archive_path=batch["archive_path"],
                summary_path=batch["summary_path"],
                message_count=len(batch["records"]),
                event_count=len(batch["identities"]),
                unit_count=batch["unit_count"],
            )
            _log.info(
                "归档 batch 已提交 [%s..]: operation=%s batch=%s partition=%s "
                "events=%d units=%d records=%d",
                chat_id[:12],
                operation_id,
                batch["batch_id"],
                batch["partition_date"],
                len(batch["identities"]),
                batch["unit_count"],
                len(batch["records"]),
            )
        if batch_plans:
            await self._sync_prompt_projection(manifest)
            if reason == "daily":
                self._last_daily_archive[chat_id] = date
                self._write_daily_state()
            manifest["state"] = "committed"
            await asyncio.to_thread(self._manifest_store.write, manifest)

        _log.info(
            "归档完成 [%s..]: reason=%s keep=%d (replay=%d+今天%d) summary=%s",
            chat_id[:12],
            reason,
            len(keep_msgs),
            len(replay_indices),
            len(today_indices),
            ",".join(batch.summary_path or "无" for batch in batch_results) or "无",
        )

        result = ArchiveResult(
            chat_id=chat_id,
            reason=reason,
            archive_path=batch_results[0].archive_path if batch_results else None,
            summary_path=batch_results[0].summary_path if batch_results else None,
            replay_count=len(keep_msgs),
            operation_id=operation_id,
            batches=batch_results,
        )

        if (
            batch_results
            and any(batch.summary_path for batch in batch_results)
            and reason != "manual"
        ):
            self._pending_injection.add(chat_id)

        return result

    async def _merge_timeline_messages(
        self, chat_id: str, messages: List[Any]
    ) -> List[Any]:
        """Materialize timeline-only visible events into legacy history.

        The legacy history remains the owner of assistant/tool protocol order,
        but accepted visible deliveries and user admissions may already exist
        only in the timeline during the migration. Materializing those events
        before partitioning makes archive and replay operate on one ordered
        visible projection without treating a tool-bearing assistant message as
        the final visible response.
        """
        if self._timeline is None:
            return messages
        repair = getattr(self._timeline, "repair_from_legacy_history", None)
        if callable(repair) and messages:
            try:
                legacy_history = [
                    (
                        message.to_storage_dict()
                        if hasattr(message, "to_storage_dict")
                        else (
                            message.to_dict()
                            if hasattr(message, "to_dict")
                            else message
                        )
                    )
                    for message in messages
                ]
                await repair(chat_id, legacy_history)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log.warning("归档回填 timeline 失败 [%s..]: %s", chat_id[:12], exc)
        snapshot = getattr(self._timeline, "snapshot", None)
        if not callable(snapshot):
            return messages
        try:
            events = await snapshot(chat_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning("归档合并 timeline 失败 [%s..]: %s", chat_id[:12], exc)
            return messages
        if not events:
            return messages

        if self._ledger is not None and chat_id not in self._legacy_ledger_seeded:
            if await self._seed_legacy_membership(chat_id, events):
                self._legacy_ledger_seeded.add(chat_id)
        archived_event_ids: set[str] = set()
        if self._ledger is not None:
            identities = [
                f"timeline:{event.event_id}"
                for event in events
                if getattr(event, "event_id", None)
            ]
            archived_event_ids = await asyncio.to_thread(
                self._ledger.archived_identities, chat_id, identities
            )

        return merge_timeline_visible_events(
            messages,
            events,
            skip_event_ids={
                identity.removeprefix("timeline:") for identity in archived_event_ids
            },
        )

    @staticmethod
    def _archive_identity(message: Any) -> str:
        event_id = getattr(message, "event_id", None)
        if event_id:
            if str(event_id).startswith(("user:", "delivery:", "legacy:")):
                return f"timeline:{event_id}"
            return f"event:{event_id}"
        if getattr(message, "role", None) == "user" and message.message_id:
            return f"legacy:user:{message.message_id}"
        if getattr(message, "role", None) == "tool" and message.tool_call_id:
            return f"legacy:tool:{message.tool_call_id}"
        if message.message_id:
            return f"legacy:{message.role}:{message.message_id}"
        record_id = getattr(message, "record_id", None)
        if record_id:
            return f"legacy:record:{record_id}"
        payload = (
            message.to_storage_dict()
            if hasattr(message, "to_storage_dict")
            else message
        )
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode(
                "utf-8"
            )
        ).hexdigest()
        return f"legacy:v1:{digest}"

    @staticmethod
    def _legacy_event_match(record: dict, events: Any) -> Optional[str]:
        role = record.get("role")
        if role not in {"user", "assistant"} or record.get("tool_calls"):
            return None
        candidates = [event for event in events if event.role == role]
        message_id = record.get("message_id")
        if message_id:
            candidates = [
                event for event in candidates if event.message_id == message_id
            ]
        else:
            content = record.get("raw_content", record.get("content", "")) or ""
            if role == "user":
                content = strip_content_prefix(content)
            candidates = [event for event in candidates if event.content == content]
            timestamp = record.get("timestamp")
            if timestamp is not None:
                candidates = [
                    event
                    for event in candidates
                    if abs(float(event.timestamp) - float(timestamp)) <= 1.0
                ]
        if len(candidates) != 1 or not getattr(candidates[0], "event_id", None):
            return None
        return f"timeline:{candidates[0].event_id}"

    async def _seed_legacy_membership(self, chat_id: str, events: Any) -> bool:
        """Best-effortly associate pre-lineage archives with timeline events."""
        if self._ledger is None:
            return True
        try:
            archives = await asyncio.to_thread(self._store.list_archives, chat_id)
            if not isinstance(archives, list):
                return True
            ambiguous: List[dict] = []
            duplicate_archives: List[dict] = []
            for archive in archives:
                path = archive.get("path") if isinstance(archive, dict) else None
                if not path:
                    continue
                if "snapshot-v1:" in Path(path).name:
                    continue
                records = await asyncio.to_thread(self._store.read_archive, path, 0)
                identities = []
                for record_index, record in enumerate(records):
                    identity = (
                        f"timeline:{record['event_id']}"
                        if record.get("event_id")
                        and not str(record["event_id"]).startswith("timeline:")
                        else record.get("event_id")
                        or self._legacy_event_match(record, events)
                    )
                    if identity:
                        identities.append(identity)
                    else:
                        ambiguous.append(
                            {
                                "archive_path": path,
                                "record_index": record_index,
                                "reason": "no_unique_timeline_match",
                            }
                        )
                if identities:
                    batch_id = (
                        "legacy-import:"
                        + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:32]
                    )
                    archived = await asyncio.to_thread(
                        self._ledger.archived_identities, chat_id, identities
                    )
                    duplicate_identities = [
                        identity for identity in identities if identity in archived
                    ]
                    if duplicate_identities:
                        duplicate_archives.append(
                            {
                                "archive_path": path,
                                "duplicate_identities": duplicate_identities,
                            }
                        )
                    identities = [
                        identity for identity in identities if identity not in archived
                    ]
                    if not identities:
                        continue
                    await asyncio.to_thread(
                        self._ledger.commit_membership,
                        batch_id,
                        chat_id,
                        identities,
                    )
            if ambiguous or duplicate_archives:
                self._write_legacy_audit(chat_id, ambiguous, duplicate_archives)
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning("初始化归档账本失败 [%s..]: %s", chat_id[:12], exc)
            return False

    def _write_legacy_audit(
        self,
        chat_id: str,
        ambiguous: List[dict],
        duplicate_archives: Optional[List[dict]] = None,
    ) -> None:
        audit_root = Path(self._memory_dir).parent / "archive_audit"
        audit_root.mkdir(parents=True, exist_ok=True)
        safe_chat_id = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in chat_id
        )
        path = audit_root / f"{safe_chat_id}.json"
        payload = {
            "chat_id": chat_id,
            "ambiguous_count": len(ambiguous),
            "ambiguous_records": ambiguous,
            "duplicate_archive_count": len(duplicate_archives or []),
            "duplicate_archives": duplicate_archives or [],
        }
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=audit_root,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    async def _apply_timeline_projection(
        self, chat_id: str, messages: List[Any]
    ) -> List[Any]:
        """Overlay accepted visible text without rewriting protocol messages."""
        if self._timeline is None:
            return messages
        snapshot = getattr(self._timeline, "snapshot", None)
        if not callable(snapshot):
            return messages
        try:
            events = await snapshot(chat_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning("归档读取 timeline 投影失败 [%s..]: %s", chat_id[:12], exc)
            return messages
        visible = {
            (event.role, event.message_id): event.content
            for event in events
            if event.role in {"user", "assistant"}
            and event.message_id
            and event.content
        }
        if not visible:
            return messages
        projected: List[Any] = []
        for message in messages:
            if message.role == "assistant" and message.tool_calls:
                projected.append(message)
                continue
            content = visible.get((message.role, message.message_id))
            projected.append(
                replace(message, content=content) if content is not None else message
            )
        return projected

    # ── 消息提取 ──

    def _is_replayable(self, msg: Any) -> bool:
        """消息是否参与回放/摘要（过滤 tool、工具调用、表情、system、空内容）。

        回放与摘要共用同一套谓词，避免过滤逻辑漂移。
        """
        if msg.role == "tool":
            return False
        if msg.role == "assistant" and msg.tool_calls:
            return False
        if (
            msg.role == "assistant"
            and msg.content
            and "[助手发送了一个表情]" in msg.content
        ):
            return False
        if msg.sender_id == "system":
            return False
        content = msg.content or ""
        if msg.role == "user":
            content = strip_content_prefix(content)
        return bool(content.strip())

    def _select_replay_indices(
        self,
        messages: List[Any],
        candidate_indices: List[int],
        gap_seconds: int,
        unit_start_times: Optional[List[float]] = None,
        following_unit_start: Optional[float] = None,
        unit_activity_end_times: Optional[List[float]] = None,
    ) -> List[int]:
        """选择最后一个完整连续时间段中的普通消息及完整工具事务。

        ``gap_seconds`` 是两个相邻消息所属单元的最大间隔。超过该间隔才允许
        在两段对话之间切开；不再按固定条数截断尾部。工具调用与结果即使
        本身不参与摘要，也必须作为同一个协议单元一起回放。
        """
        if gap_seconds <= 0 or not candidate_indices:
            return []
        unit_start_times = unit_start_times or self._unit_start_timestamps(messages)
        unit_activity_end_times = (
            unit_activity_end_times or self._unit_activity_end_timestamps(messages)
        )
        if (
            following_unit_start is not None
            and following_unit_start - unit_activity_end_times[candidate_indices[-1]]
            > gap_seconds
        ):
            return []

        segment_start = len(candidate_indices) - 1
        for position in range(len(candidate_indices) - 1, 0, -1):
            previous = candidate_indices[position - 1]
            current = candidate_indices[position]
            if (
                unit_start_times[current] - unit_activity_end_times[previous]
                > gap_seconds
            ):
                break
            segment_start = position - 1

        segment_indices = candidate_indices[segment_start:]
        if not any(self._is_replayable(messages[index]) for index in segment_indices):
            return []

        selected = {
            index for index in segment_indices if self._is_replayable(messages[index])
        }
        # 将段内工具事务作为原子单元加入；闭包会补齐跨段边界的配对记录。
        selected.update(
            index
            for index in segment_indices
            if messages[index].role == "tool"
            or (messages[index].role == "assistant" and messages[index].tool_calls)
        )
        return sorted(self._close_tool_transactions(messages, selected))

    def _close_tool_transactions(
        self, messages: List[Any], keep_indices: Set[int]
    ) -> Set[int]:
        """扩展保留集，使 tool 调用与对应结果不可被跨日切分。"""
        call_owners: Dict[str, int] = {}
        call_results: Dict[str, List[int]] = {}
        assistant_call_ids: Dict[int, Set[str]] = {}

        for index, message in enumerate(messages):
            if message.role == "assistant" and message.tool_calls:
                call_ids = {
                    call.get("id")
                    for call in message.tool_calls
                    if isinstance(call, dict) and call.get("id")
                }
                if call_ids:
                    assistant_call_ids[index] = call_ids
                    for call_id in call_ids:
                        call_owners[call_id] = index
            elif message.role == "tool" and message.tool_call_id:
                call_results.setdefault(message.tool_call_id, []).append(index)

        pending = list(keep_indices)
        while pending:
            index = pending.pop()
            message = messages[index]

            if message.role == "tool" and message.tool_call_id:
                owner = call_owners.get(message.tool_call_id)
                if owner is not None and owner not in keep_indices:
                    keep_indices.add(owner)
                    pending.append(owner)

            for call_id in assistant_call_ids.get(index, set()):
                for result_index in call_results.get(call_id, []):
                    if result_index not in keep_indices:
                        keep_indices.add(result_index)
                        pending.append(result_index)

        return keep_indices

    @staticmethod
    def _copy_for_history(msg: Any) -> ChatMessage:
        """复制保留消息，恢复历史时去除旧 JSONL 的 user 内容前缀。"""
        content = msg.content or ""
        if msg.role == "user":
            content = strip_content_prefix(content)
        return ChatMessage(
            role=msg.role,
            content=content,
            timestamp=msg.timestamp,
            message_id=msg.message_id,
            sender_id=msg.sender_id,
            name=msg.name,
            tool_call_id=msg.tool_call_id,
            tool_name=msg.tool_name,
            tool_calls=msg.tool_calls,
            reasoning_content=msg.reasoning_content,
            event_id=getattr(msg, "event_id", None),
            record_id=None,
            source_batch_id=None,
            replayed_from_batch_id=None,
        )

    def _extract_replay_messages(
        self, messages: List[Any], gap_seconds: Optional[int] = None
    ) -> List[Any]:
        indices = self._select_replay_indices(
            messages,
            list(range(len(messages))),
            self._replay_gap_seconds if gap_seconds is None else gap_seconds,
        )
        return [self._copy_for_history(messages[index]) for index in indices]

    def _format_summary_text(
        self,
        messages: List[Any],
        count: int,
        is_group: bool,
        chat_id: str,
        date: str,
    ) -> Optional[str]:
        # 1. 正向收集有效消息（与回放共用同一套过滤谓词，避免漂移）
        selected: List[ChatMessage] = []
        for msg in messages:
            if not self._is_replayable(msg):
                continue
            selected.append(msg)

        # 取最后 count 条
        selected = selected[-count:]
        if not selected:
            return None

        # 2. 分组合并
        groups = group_user_messages(selected)

        lines: List[str] = []
        for group in groups:
            _build_summary_group(lines, group, self.merge_window_seconds)

        chat_type = "群聊" if is_group else "私聊"
        short_id = chat_id[:16] + "…" if len(chat_id) > 16 else chat_id

        parts = [
            f"# Session: {date}",
            "",
            f"- **Chat**: {short_id}",
            f"- **Type**: {chat_type}",
            f"- **Messages**: {len(lines)}",
            "",
            "## 对话记录",
            "",
        ]
        parts.extend(lines)
        return "\n".join(parts)

    async def _write_memory_file(
        self, chat_id: str, date: str, text: str, batch_id: Optional[str] = None
    ) -> Optional[str]:
        try:
            file_path = await asyncio.to_thread(
                self._write_memory_file_sync, chat_id, date, text, batch_id
            )
            _log.info(
                "归档摘要已写入 [%s..] %s (%d 字符)",
                chat_id[:12],
                Path(file_path).name,
                len(text),
            )
            return file_path
        except Exception as e:
            _log.error(
                "写入归档摘要失败 [%s..]: %s",
                chat_id[:12],
                e,
            )
            raise

    def _write_memory_file_sync(
        self, chat_id: str, date: str, text: str, batch_id: Optional[str] = None
    ) -> str:
        mem_dir = _get_memory_dir(self._memory_dir, chat_id)
        mem_dir.mkdir(parents=True, exist_ok=True)
        suffix = f".{batch_id}" if batch_id else ""
        file_path = mem_dir / f"{date}{suffix}.md"
        if file_path.exists():
            existing = file_path.read_text(encoding="utf-8")
            if existing == text:
                return str(file_path)
            raise RuntimeError(
                f"summary batch already exists with different content: {file_path}"
            )
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=mem_dir,
                prefix=f".{file_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, file_path)
            return str(file_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _build_summary_group(
    lines: List[str], group: List[ChatMessage], window_seconds: int
) -> None:
    """将合并分组格式化为一行或多行，追加到 lines。"""
    first = group[0]

    if first.role != "user":
        content = first.content or ""
        lines.append(f"猫猫: {content}")
        return

    display = first.name or first.sender_id or "未知"
    content_parts: List[str] = []
    prev_ts = first.timestamp

    for msg in group:
        raw = strip_content_prefix(msg.content or "").strip()
        if not raw:
            continue

        if content_parts:
            gap = msg.timestamp - prev_ts
            if gap > window_seconds:
                ts_marker = time.strftime("[%H:%M:%S]", time.localtime(msg.timestamp))
                content_parts.append(ts_marker)

        content_parts.append(raw)
        prev_ts = msg.timestamp

    if content_parts:
        joined = "\n".join(content_parts)
        lines.append(f"{display}: {joined}")
