"""Durable summaries derived from complete conversation turns.

The event ledger remains the source of truth.  This module stores only a
versioned, bounded projection that can safely be selected for a future prompt.
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Optional, Sequence

from core.ai.fallback_runner import FallbackRunner
from core.engine.conversation_event_log import (
    ConversationEvent,
    ConversationEventLog,
    ConversationTurn,
    TurnStatus,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnSummary:
    chat_id: str
    turn_id: str
    turn_sequence: int
    source_date: str
    status: str
    deterministic_text: str
    archive_batch_id: str = ""
    semantic_text: str = ""
    coverage_event_ids: tuple[str, ...] = ()
    coverage_start_seq: int = 0
    coverage_end_seq: int = 0
    source_hash: str = ""
    revision: int = 1
    semantic_model: str = ""
    semantic_error: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def text(self) -> str:
        return self.semantic_text or self.deterministic_text

    @property
    def semantic_ready(self) -> bool:
        return bool(self.semantic_text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "turn_id": self.turn_id,
            "turn_sequence": self.turn_sequence,
            "source_date": self.source_date,
            "status": self.status,
            "deterministic_text": self.deterministic_text,
            "archive_batch_id": self.archive_batch_id,
            "semantic_text": self.semantic_text,
            "text": self.text,
            "coverage_event_ids": list(self.coverage_event_ids),
            "coverage_start_seq": self.coverage_start_seq,
            "coverage_end_seq": self.coverage_end_seq,
            "source_hash": self.source_hash,
            "revision": self.revision,
            "semantic_ready": self.semantic_ready,
            "semantic_model": self.semantic_model,
            "semantic_error": self.semantic_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class SummarySelection:
    summaries: tuple[TurnSummary, ...]
    estimated_tokens: int
    skipped_count: int = 0
    rollup_text: str = ""
    rollup_source_count: int = 0

    @property
    def text(self) -> str:
        selected_text = "\n\n---\n\n".join(
            f"[{summary.source_date} · turn {summary.turn_sequence}]\n{summary.text}"
            for summary in self.summaries
        )
        if not self.rollup_text:
            return selected_text
        if not selected_text:
            return self.rollup_text
        return f"{self.rollup_text}\n\n---\n\n{selected_text}"


class TurnSummaryStore:
    """Build and read summaries without making them another history source."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        event_log: ConversationEventLog,
        *,
        path: str = "data/turn_summaries.sqlite3",
        max_prompt_tokens: int = 1800,
        max_prompt_summaries: int = 12,
        max_summary_batches: int = 12,
        merge_strategy: str = "rollup",
        semantic_enabled: bool = False,
        semantic_group: str = "summary",
        model_registry: Any = None,
        semantic_max_tokens: int = 500,
    ) -> None:
        if max_prompt_tokens < 1 or max_prompt_summaries < 1 or max_summary_batches < 1:
            raise ValueError("summary prompt limits must be positive")
        self._event_log = event_log
        self._path = path
        self._max_prompt_tokens = int(max_prompt_tokens)
        self._max_prompt_summaries = int(max_prompt_summaries)
        self._max_summary_batches = int(max_summary_batches)
        self._merge_strategy = str(merge_strategy or "rollup").lower()
        if self._merge_strategy not in {"none", "rollup"}:
            raise ValueError("unsupported summary merge strategy")
        self._semantic_enabled = bool(semantic_enabled)
        self._semantic_group = str(semantic_group or "summary")
        self._model_registry = model_registry
        self._semantic_max_tokens = max(1, int(semantic_max_tokens))
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._semantic_tasks: set[asyncio.Task] = set()

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
                CREATE TABLE IF NOT EXISTS turn_summary_schema (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turn_summaries (
                    chat_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    turn_sequence INTEGER NOT NULL,
                    source_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    deterministic_text TEXT NOT NULL,
                    archive_batch_id TEXT NOT NULL DEFAULT '',
                    semantic_text TEXT NOT NULL DEFAULT '',
                    coverage_event_ids TEXT NOT NULL,
                    coverage_start_seq INTEGER NOT NULL,
                    coverage_end_seq INTEGER NOT NULL,
                    source_hash TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    semantic_model TEXT NOT NULL DEFAULT '',
                    semantic_error TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (chat_id, turn_id)
                );
                CREATE TABLE IF NOT EXISTS turn_summary_jobs (
                    job_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL
                );
                """)
            columns = {
                row["name"]
                for row in self._conn.execute(
                    "PRAGMA table_info(turn_summaries)"
                ).fetchall()
            }
            if "archive_batch_id" not in columns:
                self._conn.execute(
                    "ALTER TABLE turn_summaries ADD COLUMN archive_batch_id TEXT NOT NULL DEFAULT ''"
                )
            row = self._conn.execute(
                "SELECT version FROM turn_summary_schema LIMIT 1"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO turn_summary_schema(version) VALUES (?)",
                    (self.SCHEMA_VERSION,),
                )
            elif int(row["version"]) < self.SCHEMA_VERSION:
                self._conn.execute(
                    "UPDATE turn_summary_schema SET version = ?", (self.SCHEMA_VERSION,)
                )
            elif int(row["version"]) != self.SCHEMA_VERSION:
                raise RuntimeError("unsupported turn summary schema version")
            self._conn.commit()
        return self._conn

    @staticmethod
    def _event_hash(events: Sequence[ConversationEvent]) -> str:
        payload = [
            {
                "event_id": event.event_id,
                "event_seq": event.event_seq,
                "turn_sequence": event.turn_sequence,
                "role": event.role,
                "kind": str(event.kind),
                "content": event.content,
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "tool_calls": list(event.tool_calls),
                "reasoning_content": event.reasoning_content,
                "terminal_status": event.terminal_status,
                "timestamp": event.timestamp,
            }
            for event in events
        ]
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _clip(text: str, limit: int = 700) -> str:
        text = " ".join(str(text or "").split())
        return text if len(text) <= limit else text[:limit] + "…"

    @classmethod
    def _deterministic_text(
        cls, turn: ConversationTurn, events: Sequence[ConversationEvent]
    ) -> str:
        lines = [
            f"Turn {turn.turn_sequence}（{turn.source_date}，{turn.status.value}）"
        ]
        for event in events:
            if event.kind.value == "user_message":
                lines.append(f"用户：{cls._clip(event.content)}")
            elif event.kind.value == "assistant_tool_call":
                names = ", ".join(
                    str(
                        call.get("function", {}).get("name")
                        or call.get("name")
                        or "工具"
                    )
                    for call in event.tool_calls
                )
                lines.append(f"助手调用工具：{names or event.tool_name or '工具'}")
            elif event.kind.value == "tool_result":
                lines.append(
                    f"工具结果（{event.tool_name or event.tool_call_id}）：{cls._clip(event.content, 400)}"
                )
            elif event.kind.value == "accepted_delivery":
                lines.append(f"助手：{cls._clip(event.content)}")
        if len(lines) == 1:
            lines.append("该 turn 没有可见文本内容，仅保留协议/状态记录。")
        return "\n".join(lines)

    @classmethod
    def _parse_semantic_candidate(cls, candidate: str, summary: TurnSummary) -> str:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError("semantic summary must be JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("semantic summary must be an object")
        allowed = {
            "summary",
            "facts",
            "decisions",
            "todos",
            "tool_effects",
            "unresolved",
            "source_turn_ids",
            "source_event_ids",
        }
        if set(payload) - allowed:
            raise ValueError("semantic summary contains unknown fields")
        source_turn_ids = payload.get("source_turn_ids")
        if not isinstance(source_turn_ids, list) or set(source_turn_ids) != {
            summary.turn_id
        }:
            raise ValueError("semantic summary source_turn_ids is invalid")
        source_event_ids = payload.get("source_event_ids", [])
        if not isinstance(source_event_ids, list) or not set(source_event_ids).issubset(
            summary.coverage_event_ids
        ):
            raise ValueError("semantic summary source_event_ids is invalid")
        sections = []
        title = str(payload.get("summary") or "").strip()
        if not title:
            raise ValueError("semantic summary summary is empty")
        sections.append(f"摘要：{cls._clip(title, 1200)}")
        labels = (
            ("facts", "事实"),
            ("decisions", "决定"),
            ("todos", "待办"),
            ("tool_effects", "工具效果"),
            ("unresolved", "未解决"),
        )
        for key, label in labels:
            values = payload.get(key, [])
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise ValueError(f"semantic summary {key} is invalid")
            if values:
                sections.append(
                    f"{label}：" + "；".join(cls._clip(value, 500) for value in values)
                )
        text = "\n".join(sections)
        if len(text) > 4000 or any(
            ord(char) < 32 and char not in "\n\t" for char in text
        ):
            raise ValueError("semantic summary exceeds safety limits")
        return text

    @staticmethod
    def _from_row(row: sqlite3.Row) -> TurnSummary:
        return TurnSummary(
            chat_id=row["chat_id"],
            turn_id=row["turn_id"],
            turn_sequence=int(row["turn_sequence"]),
            source_date=row["source_date"],
            status=row["status"],
            deterministic_text=row["deterministic_text"],
            archive_batch_id=row["archive_batch_id"],
            semantic_text=row["semantic_text"],
            coverage_event_ids=tuple(json.loads(row["coverage_event_ids"] or "[]")),
            coverage_start_seq=int(row["coverage_start_seq"]),
            coverage_end_seq=int(row["coverage_end_seq"]),
            source_hash=row["source_hash"],
            revision=int(row["revision"]),
            semantic_model=row["semantic_model"],
            semantic_error=row["semantic_error"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    async def ensure_turn_summary(
        self,
        chat_id: str,
        turn_id: str,
        *,
        archived_event_ids: Sequence[str] = (),
        archive_batch_id: str = "",
    ) -> Optional[TurnSummary]:
        events_snapshot = await self._event_log.snapshot_events(
            chat_id, include_internal=True
        )
        events = tuple(
            event for event in events_snapshot.events if event.turn_id == turn_id
        )
        if not events:
            return None
        turns = await self._event_log.snapshot_turns(chat_id, include_internal=True)
        turn = next((item for item in turns.turns if item.turn_id == turn_id), None)
        if turn is None or not turn.is_terminal or turn.status is TurnStatus.INCOMPLETE:
            return None
        report = await self._event_log.validate_turn(turn_id)
        if not report.valid:
            return None
        archived = set(archived_event_ids)
        if archived and not archived.intersection(event.event_id for event in events):
            return None
        source_hash = self._event_hash(events)
        deterministic = self._deterministic_text(turn, events)
        now = time.time()
        conn = await self._ensure_open()
        async with self._lock:
            existing_row = conn.execute(
                "SELECT * FROM turn_summaries WHERE chat_id = ? AND turn_id = ?",
                (chat_id, turn_id),
            ).fetchone()
            if existing_row is not None:
                existing = self._from_row(existing_row)
                if existing.source_hash == source_hash:
                    if archive_batch_id and not existing.archive_batch_id:
                        conn.execute(
                            "UPDATE turn_summaries SET archive_batch_id = ? "
                            "WHERE chat_id = ? AND turn_id = ?",
                            (archive_batch_id, chat_id, turn_id),
                        )
                        conn.commit()
                        return TurnSummary(
                            **{
                                **existing.__dict__,
                                "archive_batch_id": archive_batch_id,
                            }
                        )
                    return existing
                revision = existing.revision + 1
                created_at = existing.created_at
            else:
                revision = 1
                created_at = now
            summary = TurnSummary(
                chat_id=chat_id,
                turn_id=turn_id,
                turn_sequence=turn.turn_sequence,
                source_date=turn.source_date,
                status=turn.status.value,
                deterministic_text=deterministic,
                archive_batch_id=archive_batch_id,
                semantic_text="",
                coverage_event_ids=tuple(event.event_id for event in events),
                coverage_start_seq=min(event.event_seq for event in events),
                coverage_end_seq=max(event.event_seq for event in events),
                source_hash=source_hash,
                revision=revision,
                created_at=created_at,
                updated_at=now,
            )
            conn.execute(
                """
                INSERT INTO turn_summaries
                    (chat_id, turn_id, turn_sequence, source_date, status,
                     deterministic_text, archive_batch_id, semantic_text, coverage_event_ids,
                     coverage_start_seq, coverage_end_seq, source_hash, revision,
                     semantic_model, semantic_error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, '', '', ?, ?)
                ON CONFLICT(chat_id, turn_id) DO UPDATE SET
                    turn_sequence = excluded.turn_sequence,
                    source_date = excluded.source_date,
                    status = excluded.status,
                    deterministic_text = excluded.deterministic_text,
                    archive_batch_id = CASE
                        WHEN turn_summaries.archive_batch_id = ''
                            THEN excluded.archive_batch_id
                        ELSE turn_summaries.archive_batch_id
                    END,
                    semantic_text = '',
                    coverage_event_ids = excluded.coverage_event_ids,
                    coverage_start_seq = excluded.coverage_start_seq,
                    coverage_end_seq = excluded.coverage_end_seq,
                    source_hash = excluded.source_hash,
                    revision = excluded.revision,
                    semantic_model = '',
                    semantic_error = '',
                    updated_at = excluded.updated_at
                """,
                (
                    summary.chat_id,
                    summary.turn_id,
                    summary.turn_sequence,
                    summary.source_date,
                    summary.status,
                    summary.deterministic_text,
                    summary.archive_batch_id,
                    json.dumps(summary.coverage_event_ids, ensure_ascii=False),
                    summary.coverage_start_seq,
                    summary.coverage_end_seq,
                    summary.source_hash,
                    summary.revision,
                    summary.created_at,
                    summary.updated_at,
                ),
            )
            conn.commit()
        return summary

    async def ensure_for_archived_events(
        self,
        chat_id: str,
        archived_event_ids: Sequence[str],
        *,
        archive_batch_id: str = "",
    ) -> tuple[TurnSummary, ...]:
        archived = set(archived_event_ids)
        snapshot = await self._event_log.snapshot_events(chat_id, include_internal=True)
        turn_ids = []
        for event in snapshot.events:
            if event.event_id in archived and event.turn_id not in turn_ids:
                turn_ids.append(event.turn_id)
        result = []
        for turn_id in turn_ids:
            summary = await self.ensure_turn_summary(
                chat_id,
                turn_id,
                archived_event_ids=archived_event_ids,
                archive_batch_id=archive_batch_id,
            )
            if summary is not None:
                result.append(summary)
        if self._semantic_enabled:
            for summary in result:
                if summary.semantic_ready:
                    continue
                await self._set_job_status(summary, "pending")
                task = asyncio.create_task(self.enhance_summary(summary))
                self._semantic_tasks.add(task)
                task.add_done_callback(self._semantic_tasks.discard)
        return tuple(result)

    async def _set_job_status(
        self, summary: TurnSummary, status: str, *, error: str = ""
    ) -> None:
        conn = await self._ensure_open()
        job_id = f"semantic:{summary.chat_id}:{summary.turn_id}:{summary.revision}"
        async with self._lock:
            conn.execute(
                """
                INSERT INTO turn_summary_jobs
                    (job_id, chat_id, turn_id, revision, status, attempts,
                     last_error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    attempts = CASE WHEN excluded.status = 'running'
                                    THEN turn_summary_jobs.attempts + 1
                                    ELSE turn_summary_jobs.attempts END,
                    last_error = excluded.last_error,
                    updated_at = excluded.updated_at
                """,
                (
                    job_id,
                    summary.chat_id,
                    summary.turn_id,
                    summary.revision,
                    status,
                    1 if status == "running" else 0,
                    error[:1000],
                    time.time(),
                ),
            )
            conn.commit()

    async def get(self, chat_id: str, turn_id: str) -> Optional[TurnSummary]:
        conn = await self._ensure_open()
        async with self._lock:
            row = conn.execute(
                "SELECT * FROM turn_summaries WHERE chat_id = ? AND turn_id = ?",
                (chat_id, turn_id),
            ).fetchone()
        return self._from_row(row) if row is not None else None

    async def select_for_prompt(
        self,
        chat_id: str,
        *,
        max_tokens: Optional[int] = None,
        max_summaries: Optional[int] = None,
        max_summary_batches: Optional[int] = None,
        merge_strategy: Optional[str] = None,
        covered_event_ids: Sequence[str] = (),
        eligible_archive_batch_ids: Optional[Collection[str]] = None,
    ) -> SummarySelection:
        token_limit = int(self._max_prompt_tokens if max_tokens is None else max_tokens)
        count_limit = int(
            self._max_prompt_summaries if max_summaries is None else max_summaries
        )
        batch_limit = int(
            self._max_summary_batches
            if max_summary_batches is None
            else max_summary_batches
        )
        if token_limit < 1 or count_limit < 1:
            raise ValueError("summary prompt limits must be positive")
        strategy = str(merge_strategy or self._merge_strategy).lower()
        if batch_limit < 1:
            raise ValueError("max_summary_batches must be positive")
        if strategy not in {"none", "rollup"}:
            raise ValueError("unsupported summary merge strategy")
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                """
                SELECT * FROM turn_summaries
                 WHERE chat_id = ?
                 ORDER BY coverage_end_seq DESC, turn_sequence DESC
                """,
                (chat_id,),
            ).fetchall()
        candidates = []
        covered = {str(event_id) for event_id in covered_event_ids if event_id}
        eligible_batches = (
            {str(batch_id) for batch_id in eligible_archive_batch_ids if batch_id}
            if eligible_archive_batch_ids is not None
            else None
        )
        for row in rows:
            summary = self._from_row(row)
            if (
                eligible_batches is not None
                and summary.archive_batch_id not in eligible_batches
            ):
                continue
            if not covered.intersection(summary.coverage_event_ids):
                candidates.append(summary)
        groups: dict[str, list[TurnSummary]] = {}
        group_order: list[str] = []
        for summary in candidates:
            group_id = summary.archive_batch_id or f"turn:{summary.turn_id}"
            if group_id not in groups:
                groups[group_id] = []
                group_order.append(group_id)
            groups[group_id].append(summary)

        selected = []
        used = 0
        skipped = 0
        rollup_candidates = []
        for group_index, group_id in enumerate(group_order):
            group = groups[group_id]
            if group_index >= batch_limit:
                rollup_candidates.extend(group)
                skipped += len(group)
                continue
            for summary in group:
                cost = max(1, len(summary.text) // 4)
                if len(selected) >= count_limit or used + cost > token_limit:
                    skipped += 1
                    rollup_candidates.append(summary)
                    continue
                selected.append(summary)
                used += cost
        selected.sort(key=lambda item: item.turn_sequence)
        rollup_text = ""
        rollup_tokens = 0
        if strategy == "rollup" and rollup_candidates:
            remaining_tokens = max(0, token_limit - used)
            max_chars = remaining_tokens * 4
            if max_chars > 0:
                rollup_lines = [
                    f"[{summary.source_date} · turn {summary.turn_sequence}] {summary.text}"
                    for summary in reversed(rollup_candidates)
                ]
                rollup_text = "较早归档摘要（roll-up，仅作参考）：\n" + "\n".join(
                    rollup_lines
                )
                if len(rollup_text) > max_chars:
                    rollup_text = rollup_text[:max_chars].rstrip() + "…"
                rollup_tokens = max(1, len(rollup_text) // 4)
                used += rollup_tokens
        return SummarySelection(
            tuple(selected),
            used,
            skipped,
            rollup_text=rollup_text,
            rollup_source_count=len(rollup_candidates),
        )

    async def enhance_summary(self, summary: TurnSummary) -> Optional[TurnSummary]:
        """Optionally enhance one deterministic summary through a model group."""
        if not self._semantic_enabled:
            return summary
        current = await self.get(summary.chat_id, summary.turn_id)
        if current is None or current.revision != summary.revision:
            return current
        await self._set_job_status(summary, "running")
        try:
            if self._model_registry is None:
                raise RuntimeError("semantic model registry is unavailable")
            chain = self._model_registry.get_chain(self._semantic_group)
            if not chain:
                raise RuntimeError(
                    f"semantic model group is unavailable: {self._semantic_group}"
                )
            runner = FallbackRunner(self._model_registry, chain)
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "你是对话摘要增强器。只根据给定的确定性摘要生成 JSON，"
                        "不得添加推测。字段必须为 summary、facts、decisions、todos、"
                        "tool_effects、unresolved、source_turn_ids、source_event_ids；"
                        "source_turn_ids 必须只包含给定 turn_id。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "turn_id": summary.turn_id,
                            "coverage_event_ids": list(summary.coverage_event_ids),
                            "deterministic_summary": summary.deterministic_text,
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            result = await runner.run(
                lambda service, _name: service.chat_completion(
                    messages=prompt, max_tokens=self._semantic_max_tokens
                )
            )
            candidate = result.message if result.ok else ""
            text = candidate.strip() if isinstance(candidate, str) else ""
            if not text:
                raise RuntimeError("semantic summary returned empty")
            text = self._parse_semantic_candidate(text, summary)
            model_name = result.model_name or ""
            conn = await self._ensure_open()
            async with self._lock:
                conn.execute(
                    """
                    UPDATE turn_summaries
                       SET semantic_text = ?, semantic_model = ?,
                           semantic_error = '', updated_at = ?
                     WHERE chat_id = ? AND turn_id = ? AND revision = ?
                    """,
                    (
                        text,
                        model_name,
                        time.time(),
                        summary.chat_id,
                        summary.turn_id,
                        summary.revision,
                    ),
                )
                conn.commit()
            await self._set_job_status(summary, "succeeded")
            return await self.get(summary.chat_id, summary.turn_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning(
                "语义摘要增强失败 [%s/%s]: %s", summary.chat_id, summary.turn_id, exc
            )
            conn = await self._ensure_open()
            async with self._lock:
                conn.execute(
                    """
                    UPDATE turn_summaries
                       SET semantic_error = ?, updated_at = ?
                     WHERE chat_id = ? AND turn_id = ? AND revision = ?
                    """,
                    (
                        str(exc)[:1000],
                        time.time(),
                        summary.chat_id,
                        summary.turn_id,
                        summary.revision,
                    ),
                )
                conn.commit()
            await self._set_job_status(summary, "failed", error=str(exc))
            return summary

    async def list_for_webui(
        self, chat_id: str, *, limit: int = 200, offset: int = 0
    ) -> list[TurnSummary]:
        conn = await self._ensure_open()
        async with self._lock:
            rows = conn.execute(
                """
                SELECT * FROM turn_summaries
                 WHERE chat_id = ?
                 ORDER BY turn_sequence DESC LIMIT ? OFFSET ?
                """,
                (chat_id, max(1, int(limit)), max(0, int(offset))),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    async def status(self, chat_id: Optional[str] = None) -> dict[str, int]:
        conn = await self._ensure_open()
        async with self._lock:
            if chat_id:
                count = conn.execute(
                    "SELECT COUNT(*) FROM turn_summaries WHERE chat_id = ?",
                    (chat_id,),
                ).fetchone()[0]
            else:
                count = conn.execute("SELECT COUNT(*) FROM turn_summaries").fetchone()[
                    0
                ]
            jobs = conn.execute(
                "SELECT status, COUNT(*) AS count FROM turn_summary_jobs "
                + ("WHERE chat_id = ? " if chat_id else "")
                + "GROUP BY status",
                (chat_id,) if chat_id else (),
            ).fetchall()
        result = {"summary_count": int(count)}
        result.update(
            {f"semantic_{row['status']}_count": int(row["count"]) for row in jobs}
        )
        return result

    async def count_for_webui(self, chat_id: str) -> int:
        conn = await self._ensure_open()
        async with self._lock:
            row = conn.execute(
                "SELECT COUNT(*) FROM turn_summaries WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        return int(row[0])

    async def clear_chat(self, chat_id: str) -> None:
        conn = await self._ensure_open()
        async with self._lock:
            conn.execute("DELETE FROM turn_summary_jobs WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM turn_summaries WHERE chat_id = ?", (chat_id,))
            conn.commit()

    async def close(self) -> None:
        tasks = tuple(self._semantic_tasks)
        self._semantic_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
