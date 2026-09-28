"""ArchiveManager — 事件账本归档 + 摘要投影

归档由 `ConversationEventLog` 的完整 terminal turn 驱动：按 token/turn/字节/最大年龄
的 retention 选出可归档 turn，写入 `ArchiveIndex` 批次，再提交 prompt visibility、
摘要投影与模型上下文轮换。所有归档读取都走账本投影，不再从 JSONL active/archive
文件读取。
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from core.engine.archive_export import ArchiveExportResult, ArchiveJSONLExportAdapter
from core.engine.archive_index import ArchiveIndex, ArchiveTurnRecord
from core.engine.conversation_event_log import RepairRevision, TurnStatus
from core.managers.archive_manifest import ArchiveManifestStore

_log = logging.getLogger(__name__)

_DEFAULT_SUMMARY_COUNT = 15
_DEFAULT_SUMMARY_DAYS = 2
_DEFAULT_RETENTION_DAYS = 30
_DEFAULT_REPLAY_GAP_SECONDS = 600


# ── ArchiveManager ──


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
        skipped_turns: Optional[List[Dict[str, Any]]] = None,
    ):
        self.chat_id = chat_id
        self.reason = reason
        self.archive_path = archive_path
        self.summary_path = summary_path
        self.replay_count = replay_count
        self.operation_id = operation_id
        self.batches = list(batches or [])
        self.skipped_turns = list(skipped_turns or [])

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
        replay_gap_seconds: int = _DEFAULT_REPLAY_GAP_SECONDS,
        summary_count: int = _DEFAULT_SUMMARY_COUNT,
        summary_days: int = _DEFAULT_SUMMARY_DAYS,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
        merge_window_seconds: int = 15,
        timezone_name: str = "Asia/Shanghai",
        archive_index: Optional[ArchiveIndex] = None,
        hot_max_tokens: int = 12000,
        hot_max_turns: int = 32,
        hot_max_bytes: int = 4_000_000,
        hot_max_age_seconds: float = 7 * 86400,
        hot_low_water_ratio: float = 0.75,
    ):
        self._cm = context_manager
        self._memory_dir = memory_dir
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

        self._event_log = None
        self._prompt_projection = None
        self._summary_store = None
        self._model_context_transcript = None
        self._export_adapter: Optional[ArchiveJSONLExportAdapter] = None
        self._archive_index = archive_index
        self._event_archive_locks: Dict[str, asyncio.Lock] = {}
        self._session_lock_provider = None
        self._manifest_store = ArchiveManifestStore(str(Path(memory_dir).parent))

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
        if not archived_ids:
            raise RuntimeError("cannot finish empty archive batch")
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
        archived_source = await event_log.snapshot_events(
            batch.chat_id,
            upto_seq=batch.captured_cutoff_seq,
            include_internal=True,
            event_ids=tuple(sorted(archived_ids)),
        )
        archived_events = archived_source.events
        unknown_archived_ids = set(archived_ids) - source_ids
        if unknown_archived_ids:
            raise RuntimeError(
                "cannot archive events missing from ledger: "
                f"{len(unknown_archived_ids)} events"
            )
        if phase_order.get(phase, 0) < phase_order["event_partition_written"]:
            self._write_event_manifest_phase(
                batch.operation_id, "event_partition_written"
            )
        if phase_order.get(phase, 0) < phase_order["active_projection_written"]:
            self._write_event_manifest_phase(
                batch.operation_id, "active_projection_written"
            )
        turn_ids = {event.turn_id for event in archived_events}
        full_turn_source = await event_log.snapshot_events(
            batch.chat_id,
            include_internal=True,
            turn_ids=tuple(sorted(turn_ids)),
        )
        batch_turns = await index.turns_for_batch(batch.batch_id)
        records_by_turn = {record.turn_id: record for record in batch_turns}
        if set(records_by_turn) != turn_ids:
            raise RuntimeError("archive turn metadata does not match event membership")
        for turn_id in turn_ids:
            turn_source_ids = {
                event.event_id
                for event in full_turn_source.events
                if event.turn_id == turn_id
            }
            archived_turn_ids = {
                event.event_id for event in archived_events if event.turn_id == turn_id
            }
            if archived_turn_ids != turn_source_ids:
                raise RuntimeError(
                    "cannot archive partial turn "
                    f"{turn_id}: {len(archived_turn_ids)}/{len(turn_source_ids)} events"
                )
            if records_by_turn[turn_id].event_count != len(turn_source_ids):
                raise RuntimeError(
                    "archive turn metadata event count mismatch " f"for {turn_id}"
                )
            integrity = await event_log.validate_turn(turn_id, chat_id=batch.chat_id)
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
        if self._prompt_projection is None:
            return
        try:
            hidden_projection_ids = set(
                await self._prompt_projection.hidden_event_ids(chat_id)
            )
            committed = await self._archive_index.list_for_webui(
                chat_id, state="committed"
            )
            for item in committed:
                try:
                    batch = await self._archive_index.get(str(item["batch_id"]))
                    if batch is None:
                        continue
                    event_ids = await self._archive_index.event_ids(batch.batch_id)
                    if not event_ids:
                        continue
                    if set(event_ids).issubset(hidden_projection_ids):
                        continue
                    repair_operation = getattr(
                        self._prompt_projection, "repair_archive_operation", None
                    )
                    if callable(repair_operation) and await repair_operation(
                        batch.operation_id
                    ):
                        hidden_projection_ids.update(event_ids)
                        continue
                    await self._prompt_projection.apply_archive_retention(
                        chat_id,
                        operation_id=f"{batch.operation_id}:projection-reconcile",
                        hidden_event_ids=tuple(sorted(event_ids)),
                        captured_cutoff_seq=batch.captured_cutoff_seq,
                    )
                    hidden_projection_ids.update(event_ids)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    _log.warning(
                        "单个已提交归档投影修复失败 [%s..] batch=%s: %s",
                        chat_id[:12],
                        item.get("batch_id", ""),
                        exc,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning("已提交归档投影修复失败 [%s..]: %s", chat_id[:12], exc)

    async def recover_event_log_archives_async(
        self, chat_ids: Optional[Sequence[str]] = None
    ) -> int:
        """Recover pending ledger archive operations before serving requests."""
        if self._event_log is None or self._archive_index is None:
            return 0
        if isinstance(chat_ids, str):
            selected = (chat_ids,)
        elif chat_ids is not None:
            selected = tuple(
                dict.fromkeys(str(chat_id) for chat_id in chat_ids if chat_id)
            )
        else:
            selected = tuple(await self._event_log.chat_ids())
        recovered = 0
        for chat_id in selected:
            try:
                await self._recover_event_log_archives(chat_id)
                recovered += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log.warning("启动期账本归档恢复失败 [%s..]: %s", chat_id[:12], exc)
        return recovered

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
        all_budgets, cutoff_seq = await event_log.snapshot_turn_budgets(
            chat_id,
            include_internal=True,
            include_nonterminal=True,
        )
        if not all_budgets:
            return None
        hidden = (
            set(await projection.hidden_event_ids(chat_id))
            if projection is not None
            else set()
        )
        visible_budgets, _ = await event_log.snapshot_turn_budgets(
            chat_id,
            upto_seq=cutoff_seq,
            include_internal=True,
            exclude_event_ids=tuple(hidden),
            include_nonterminal=True,
        )
        visible_by_turn = {budget.turn.turn_id: budget for budget in visible_budgets}
        hot_budgets = [
            budget
            for budget in all_budgets
            if budget.event_count > 0
            and visible_by_turn.get(budget.turn.turn_id) is not None
            and visible_by_turn[budget.turn.turn_id].event_count == budget.event_count
        ]
        if not hot_budgets:
            return None
        hot_budgets.sort(key=lambda budget: budget.turn.turn_sequence)
        hot_tokens = sum(budget.estimated_tokens for budget in hot_budgets)
        hot_bytes = sum(budget.estimated_bytes for budget in hot_budgets)
        terminal_budgets = [
            budget
            for budget in hot_budgets
            if budget.turn.is_terminal
            and budget.turn.status is not TurnStatus.INCOMPLETE
        ]
        if not terminal_budgets:
            return None

        def turn_age(budget: Any) -> float:
            return max(0.0, time.time() - budget.oldest_timestamp)

        capacity_exceeded = (
            hot_tokens > self._hot_max_tokens
            or len(hot_budgets) > self._hot_max_turns
            or hot_bytes > self._hot_max_bytes
        )
        age_exceeded = force or capacity_exceeded

        low_tokens = max(1, int(self._hot_max_tokens * self._hot_low_water_ratio))
        low_turns = max(1, int(self._hot_max_turns * self._hot_low_water_ratio))
        low_bytes = max(1, int(self._hot_max_bytes * self._hot_low_water_ratio))
        selected_turns: List[Any] = []
        selected_events_by_turn: Dict[str, tuple[Any, ...]] = {}
        remaining_turns = [budget.turn for budget in hot_budgets]
        while terminal_budgets:
            candidate_budget = terminal_budgets.pop(0)
            candidate = candidate_budget.turn
            candidate_source = await event_log.snapshot_events(
                chat_id,
                upto_seq=cutoff_seq,
                include_internal=True,
                turn_ids=(candidate.turn_id,),
            )
            candidate_events = candidate_source.events
            if len(candidate_events) != candidate_budget.event_count:
                _log.warning(
                    "跳过事件数漂移归档 turn [%s..] turn=%s expected=%s actual=%s",
                    chat_id[:12],
                    candidate.turn_id[:12],
                    candidate_budget.event_count,
                    len(candidate_events),
                )
                continue
            integrity = await event_log.validate_turn(
                candidate.turn_id, chat_id=chat_id
            )
            if not integrity.valid:
                _log.warning(
                    "跳过不完整归档 turn [%s..] turn=%s reason=%s",
                    chat_id[:12],
                    candidate.turn_id[:12],
                    integrity.reason,
                )
                continue
            if not age_exceeded:
                if turn_age(candidate_budget) <= self._hot_max_age_seconds:
                    return None
                age_exceeded = True
            selected_turns.append(candidate)
            selected_events_by_turn[candidate.turn_id] = candidate_events
            remaining_turns = [
                turn for turn in remaining_turns if turn.turn_id != candidate.turn_id
            ]
            hot_tokens -= candidate_budget.estimated_tokens
            hot_bytes -= candidate_budget.estimated_bytes
            hot_turns_count = len(remaining_turns)
            next_age_exceeded = (
                bool(terminal_budgets)
                and turn_age(terminal_budgets[0]) > self._hot_max_age_seconds
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
            event
            for turn in selected_turns
            for event in selected_events_by_turn[turn.turn_id]
        ]
        selected_turn_ids = [turn.turn_id for turn in selected_turns]
        operation_id = self._event_archive_operation_id(
            chat_id, cutoff_seq, selected_turn_ids
        )
        batch_id = f"batch:{operation_id}"
        turn_records = [
            ArchiveTurnRecord(
                turn_id=turn.turn_id,
                turn_sequence=turn.turn_sequence,
                source_date=turn.source_date,
                event_count=len(selected_events_by_turn[turn.turn_id]),
                estimated_tokens=next(
                    budget.estimated_tokens
                    for budget in hot_budgets
                    if budget.turn.turn_id == turn.turn_id
                ),
                turn_kind=turn.turn_kind.value,
            )
            for turn in selected_turns
        ]
        batch = await index.prepare_batch(
            batch_id=batch_id,
            operation_id=operation_id,
            chat_id=chat_id,
            captured_cutoff_seq=cutoff_seq,
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
        captured_cutoff = manifest.get("captured_cutoff_seq")
        cutoff_seq = (
            int(captured_cutoff)
            if captured_cutoff is not None
            else await event_log.latest_event_seq(chat_id)
        )
        source_ids = set(
            await event_log.event_ids(
                chat_id,
                upto_seq=cutoff_seq,
                include_internal=True,
            )
        )
        await projection.apply_archive_retention(
            chat_id,
            operation_id=operation_id,
            hidden_event_ids=tuple(sorted(hidden_ids & source_ids)),
            retained_event_ids=tuple(sorted(retained_ids & source_ids)),
            captured_cutoff_seq=cutoff_seq,
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
        return {
            "pending_operations": self._manifest_store.pending_count(chat_id),
            "committed_batches": 0,
            "latest_committed_batch": None,
        }

    async def clear_pending_operations_async(self, chat_id: str) -> int:
        """Discard resumable archive work for an explicitly cleared chat."""
        return await asyncio.to_thread(self._manifest_store.clear_pending, chat_id)

    @property
    def replay_gap_seconds(self) -> int:
        """昨天原始消息回放的连续会话间隔阈值（秒）。"""
        return self._replay_gap_seconds

    @property
    def summary_count(self) -> int:
        """摘要取最近 N 条有效消息。"""
        return self._summary_count

    async def archive_if_stale(
        self, chat_id: str, is_group: bool, *, session_lock_held: bool = False
    ) -> Optional[ArchiveResult]:
        if self._event_log is not None and self._archive_index is not None:
            return await self._archive_event_log_serialized(
                chat_id, is_group, session_lock_held=session_lock_held
            )
        return None

    async def load_recent_summaries_async(self, chat_id: str) -> Optional[str]:
        if self._summary_store is None:
            return None
        eligible_batches = await self._prompt_summary_batch_ids(chat_id)
        selection = await self._summary_store.select_for_prompt(
            chat_id, eligible_archive_batch_ids=eligible_batches
        )
        return selection.text or None

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

    async def get_event_integrity_async(self, chat_id: str) -> dict[str, Any]:
        """Return bounded turn-integrity diagnostics for an administrator."""
        if self._event_log is None:
            return {
                "turn_count": 0,
                "invalid_turn_count": 0,
                "invalid_reasons": {},
                "error": "event_log_not_configured",
            }
        summary = await self._event_log.integrity_summary(chat_id)
        turns = await self._event_log.snapshot_turns(chat_id, include_internal=True)
        reports = await self._event_log.validate_turns(
            [turn.turn_id for turn in turns.turns], chat_id=chat_id
        )
        summary["invalid_turns"] = [
            {
                "turn_id": report.turn_id,
                "status": report.status,
                "event_count": report.event_count,
                "reason": report.reason or "invalid_turn",
                "missing_tool_result_ids": list(report.missing_tool_result_ids),
                "duplicate_tool_result_ids": list(report.duplicate_tool_result_ids),
            }
            for report in reports.values()
            if not report.valid
        ]
        return summary

    async def record_turn_repair_revision_async(
        self,
        chat_id: str,
        turn_id: str,
        revision_id: str,
        reason: str,
        *,
        operator: str = "",
    ) -> RepairRevision:
        """Append an administrator repair note without rewriting ledger facts."""
        if self._event_log is None:
            raise RuntimeError("event log is not configured")
        return await self._event_log.append_repair_revision(
            chat_id=chat_id,
            original_turn_id=turn_id,
            revision_id=revision_id,
            reason=reason,
            operator=operator,
        )

    async def get_turn_repair_revisions_async(
        self, chat_id: str, turn_id: str = ""
    ) -> tuple[RepairRevision, ...]:
        """Read bounded administrator repair notes for a chat or turn."""
        if self._event_log is None:
            return ()
        return await self._event_log.repair_revisions(chat_id, original_turn_id=turn_id)

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
        return {"message_count": 0, "last_activity": None, "archive_count": 0}

    async def archive_manual(
        self, chat_id: str, is_group: Optional[bool] = None
    ) -> ArchiveResult:
        if self._event_log is not None and self._archive_index is not None:
            result = await self._archive_event_log_serialized(
                chat_id, bool(is_group), force=True
            )
            return result or ArchiveResult(chat_id=chat_id, reason="retention")
        return ArchiveResult(chat_id=chat_id, reason="retention")

    async def repair_event_log_archives(
        self,
        chat_id: str,
        *,
        before_date: str,
        captured_cutoff_seq: Optional[int] = None,
    ) -> ArchiveResult:
        """Repair missing archive projections for complete turns before a date."""
        if self._session_lock_provider is None:
            return await self._repair_event_log_archives_unlocked(
                chat_id,
                before_date=before_date,
                captured_cutoff_seq=captured_cutoff_seq,
            )
        session_lock = await self._session_lock_provider(chat_id)
        async with session_lock:
            archive_lock = self._event_archive_locks.setdefault(chat_id, asyncio.Lock())
            async with archive_lock:
                return await self._repair_event_log_archives_unlocked(
                    chat_id,
                    before_date=before_date,
                    captured_cutoff_seq=captured_cutoff_seq,
                )

    async def _repair_event_log_archives_unlocked(
        self,
        chat_id: str,
        *,
        before_date: str,
        captured_cutoff_seq: Optional[int] = None,
    ) -> ArchiveResult:
        """Repair missing archive projections for complete turns before a date."""
        event_log = self._event_log
        index = self._archive_index
        if event_log is None or index is None:
            raise RuntimeError("event archive dependencies are not configured")
        if not chat_id or not before_date:
            raise ValueError("chat_id and before_date are required")
        try:
            datetime.strptime(before_date, "%Y-%m-%d")
        except (TypeError, ValueError) as exc:
            raise ValueError("before_date must be YYYY-MM-DD") from exc
        await self._recover_event_log_archives(chat_id)
        cutoff_seq = (
            int(captured_cutoff_seq)
            if captured_cutoff_seq is not None
            else await event_log.latest_event_seq(chat_id)
        )
        event_identities = await event_log.event_identities(
            chat_id,
            upto_seq=cutoff_seq,
            include_internal=True,
        )
        committed_ids = set(await index.committed_event_ids(chat_id))
        turns_snapshot = await event_log.snapshot_turns(
            chat_id,
            upto_turn_sequence=None,
            include_internal=True,
        )
        turns_by_id = {turn.turn_id: turn for turn in turns_snapshot.turns}
        event_ids_by_turn: Dict[str, List[str]] = {}
        for event_id, turn_id in event_identities:
            event_ids_by_turn.setdefault(turn_id, []).append(event_id)
        repair_scope_ids = {
            event_id
            for event_id, turn_id in event_identities
            if turns_by_id.get(turn_id) is not None
            and turns_by_id[turn_id].source_date < before_date
        }
        visible_metadata_ids = (
            set(await self._prompt_projection.visibility_event_ids(chat_id))
            if self._prompt_projection is not None
            else set()
        )
        missing_ids = repair_scope_ids - visible_metadata_ids
        hidden_ids = (
            set(await self._prompt_projection.hidden_event_ids(chat_id))
            if self._prompt_projection is not None
            else set()
        )
        selected_turns = []
        skipped_turns: List[Dict[str, Any]] = []
        for turn in turns_snapshot.turns:
            if turn.source_date >= before_date:
                continue
            turn_event_ids = event_ids_by_turn.get(turn.turn_id, ())
            if not turn_event_ids:
                continue
            turn_ids = set(turn_event_ids)
            candidate_ids = turn_ids - committed_ids - hidden_ids
            if not candidate_ids:
                continue
            if not turn.is_terminal:
                skipped_turns.append(
                    {
                        "turn_id": turn.turn_id,
                        "turn_sequence": turn.turn_sequence,
                        "source_date": turn.source_date,
                        "reason": "incomplete_turn",
                    }
                )
                continue
            if turn.status is TurnStatus.INCOMPLETE:
                skipped_turns.append(
                    {
                        "turn_id": turn.turn_id,
                        "turn_sequence": turn.turn_sequence,
                        "source_date": turn.source_date,
                        "reason": "incomplete_turn",
                    }
                )
                continue
            if turn.ended_seq > cutoff_seq or len(turn_event_ids) != turn.event_count:
                skipped_turns.append(
                    {
                        "turn_id": turn.turn_id,
                        "turn_sequence": turn.turn_sequence,
                        "source_date": turn.source_date,
                        "reason": "cutoff_splits_turn",
                    }
                )
                continue
            membership_ids = turn_ids & (committed_ids | hidden_ids)
            if membership_ids:
                if membership_ids != turn_ids:
                    skipped_turns.append(
                        {
                            "turn_id": turn.turn_id,
                            "turn_sequence": turn.turn_sequence,
                            "source_date": turn.source_date,
                            "reason": "partial_archive_membership",
                        }
                    )
                continue
            selected_turns.append(turn)
        selected_turns.sort(key=lambda turn: turn.turn_sequence)
        validate_many = getattr(event_log, "validate_turns", None)
        if callable(validate_many) and selected_turns:
            integrity_reports = await validate_many(
                [turn.turn_id for turn in selected_turns], chat_id=chat_id
            )
        else:
            integrity_reports = {
                turn.turn_id: await event_log.validate_turn(
                    turn.turn_id, chat_id=chat_id
                )
                for turn in selected_turns
            }
        valid_turns = []
        for turn in selected_turns:
            integrity = integrity_reports[turn.turn_id]
            if integrity.valid:
                valid_turns.append(turn)
            else:
                skipped_turns.append(
                    {
                        "turn_id": turn.turn_id,
                        "turn_sequence": turn.turn_sequence,
                        "source_date": turn.source_date,
                        "reason": integrity.reason or "invalid_turn",
                    }
                )
        selected_turns = valid_turns
        selected_events_by_turn: Dict[str, tuple[Any, ...]] = {}
        if selected_turns:
            selected_source = await event_log.snapshot_events(
                chat_id,
                upto_seq=cutoff_seq,
                include_internal=True,
                turn_ids=tuple(turn.turn_id for turn in selected_turns),
            )
            selected_events_by_turn = {}
            for event in selected_source.events:
                selected_events_by_turn.setdefault(event.turn_id, ())
                selected_events_by_turn[event.turn_id] += (event,)
        repair_operation_payload = json.dumps(
            {
                "chat_id": chat_id,
                "before_date": before_date,
                "event_ids": tuple(sorted(repair_scope_ids)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        repair_operation_id = (
            "event-repair-visibility:"
            + hashlib.sha256(repair_operation_payload.encode("utf-8")).hexdigest()[:32]
        )
        if not selected_turns and not missing_ids:
            return ArchiveResult(
                chat_id=chat_id,
                reason="repair",
                operation_id=repair_operation_id,
                skipped_turns=skipped_turns,
            )
        if selected_turns:
            selected_ids = {
                event.event_id
                for turn in selected_turns
                for event in selected_events_by_turn.get(turn.turn_id, ())
            }
            operation_payload = json.dumps(
                {"chat_id": chat_id, "event_ids": sorted(selected_ids)},
                ensure_ascii=False,
                sort_keys=True,
            )
            operation_id = (
                "event-repair-archive:"
                + hashlib.sha256(operation_payload.encode("utf-8")).hexdigest()[:32]
            )
            batch = await index.prepare_batch(
                batch_id=f"batch:{operation_id}",
                operation_id=operation_id,
                chat_id=chat_id,
                captured_cutoff_seq=cutoff_seq,
                turn_records=[
                    ArchiveTurnRecord(
                        turn_id=turn.turn_id,
                        turn_sequence=turn.turn_sequence,
                        source_date=turn.source_date,
                        event_count=len(selected_events_by_turn.get(turn.turn_id, ())),
                        estimated_tokens=sum(
                            event.token_count
                            for event in selected_events_by_turn.get(turn.turn_id, ())
                        ),
                        turn_kind=turn.turn_kind.value,
                    )
                    for turn in selected_turns
                ],
                event_ids=[
                    (event.event_id, event.turn_id)
                    for event in (
                        event
                        for turn in selected_turns
                        for event in selected_events_by_turn.get(turn.turn_id, ())
                    )
                    if event.event_id in selected_ids
                ],
            )
            result = await self._finish_event_archive_batch(batch)
            visible_after = (
                set(await self._prompt_projection.visibility_event_ids(chat_id))
                if self._prompt_projection is not None
                else set()
            )
            remaining_hidden_ids = tuple(
                sorted(
                    (missing_ids & committed_ids)
                    - hidden_ids
                    - selected_ids
                    - visible_after
                )
            )
            remaining_retained_ids = tuple(
                sorted(
                    missing_ids
                    - committed_ids
                    - hidden_ids
                    - selected_ids
                    - visible_after
                )
            )
            if self._prompt_projection is not None and (
                remaining_hidden_ids or remaining_retained_ids
            ):
                await self._prompt_projection.apply_archive_retention(
                    chat_id,
                    operation_id=repair_operation_id,
                    hidden_event_ids=remaining_hidden_ids,
                    retained_event_ids=remaining_retained_ids,
                    captured_cutoff_seq=cutoff_seq,
                )
            result.reason = "repair"
            result.skipped_turns.extend(skipped_turns)
            return result

        retained_ids = tuple(sorted(repair_scope_ids - hidden_ids - committed_ids))
        hidden_ids_to_repair = tuple(sorted((missing_ids & committed_ids) - hidden_ids))
        operation_payload = json.dumps(
            {
                "chat_id": chat_id,
                "before_date": before_date,
                "event_ids": retained_ids,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        operation_id = (
            "event-repair-visibility:"
            + hashlib.sha256(operation_payload.encode("utf-8")).hexdigest()[:32]
        )
        if self._prompt_projection is not None and (
            hidden_ids_to_repair or retained_ids
        ):
            await self._prompt_projection.apply_archive_retention(
                chat_id,
                operation_id=operation_id,
                hidden_event_ids=hidden_ids_to_repair,
                retained_event_ids=retained_ids,
                captured_cutoff_seq=cutoff_seq,
            )
        return ArchiveResult(
            chat_id=chat_id,
            reason="repair",
            operation_id=operation_id,
            skipped_turns=skipped_turns,
        )

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
        return ArchiveResult(chat_id=chat_id, reason="retention")

    async def cleanup_old_archives_async(self) -> int:
        """Retention cleanup is owned by the event-log archive index."""
        return 0

    async def list_archives_async(self, chat_id: str) -> List[dict]:
        return await self.list_archive_batches_async(chat_id)
