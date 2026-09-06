"""Optional JSONL cold-backup export for ledger archive batches.

The export is deliberately one-way.  Archive reads, prompt projections and
recovery continue to use the event ledger and archive index when this adapter
is disabled or unavailable.
"""

import asyncio
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.engine.archive_index import ArchiveIndex
from core.engine.conversation_event_log import ConversationEventLog


@dataclass(frozen=True)
class ArchiveExportResult:
    batch_id: str
    status: str
    path: str = ""
    event_count: int = 0
    content_hash: str = ""
    manifest_hash: str = ""
    error: str = ""


class ArchiveJSONLExportAdapter:
    """Export committed archive batches without becoming a read dependency."""

    VERSION = 1

    def __init__(
        self,
        event_log: ConversationEventLog,
        archive_index: ArchiveIndex,
        root_dir: str = "data/archives/export",
        *,
        enabled: bool = False,
    ) -> None:
        self._event_log = event_log
        self._archive_index = archive_index
        self._root = Path(root_dir)
        self._enabled = bool(enabled)
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def _safe_component(value: str) -> str:
        if value and all(char.isalnum() or char in "-_." for char in value):
            return value
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _json_line(value: dict[str, Any]) -> bytes:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")

    async def export_batch(self, batch_id: str) -> ArchiveExportResult:
        if not self._enabled:
            return ArchiveExportResult(batch_id, "disabled")
        lock = self._locks.setdefault(batch_id, asyncio.Lock())
        async with lock:
            return await self._export_locked(batch_id)

    async def _export_locked(self, batch_id: str) -> ArchiveExportResult:
        batch = await self._archive_index.get(batch_id)
        if batch is None:
            return ArchiveExportResult(batch_id, "failed", error="batch_not_found")
        if batch.state not in {"committed", "export_degraded", "soft_deleted"}:
            return ArchiveExportResult(
                batch_id, "deferred", error="batch_not_committed"
            )

        event_ids = await self._archive_index.event_ids(batch_id)
        snapshot = await self._event_log.snapshot_events(
            batch.chat_id,
            upto_seq=batch.captured_cutoff_seq,
            include_internal=True,
            event_ids=tuple(sorted(event_ids)),
        )
        events = tuple(
            event for event in snapshot.events if event.event_id in event_ids
        )
        if len(events) != len(event_ids):
            return ArchiveExportResult(
                batch_id,
                "failed",
                error="ledger_event_missing",
                event_count=len(events),
            )

        manifest = {
            "record_type": "archive_export_manifest",
            "export_version": self.VERSION,
            "batch_id": batch.batch_id,
            "operation_id": batch.operation_id,
            "chat_id": batch.chat_id,
            "captured_cutoff_seq": batch.captured_cutoff_seq,
            "source_hash": batch.source_hash,
            "event_count": len(events),
            "event_ids": [event.event_id for event in events],
        }
        manifest_hash = hashlib.sha256(self._json_line(manifest)).hexdigest()
        manifest["manifest_hash"] = manifest_hash
        lines = [self._json_line(manifest)]
        lines.extend(
            self._json_line(
                {
                    "record_type": "event",
                    "export_version": self.VERSION,
                    "event": event.to_history_dict(),
                }
            )
            for event in events
        )
        content = b"".join(lines)
        content_hash = hashlib.sha256(content).hexdigest()
        path = (
            self._root
            / self._safe_component(batch.chat_id)
            / f"{self._safe_component(batch.batch_id)}.jsonl"
        )
        temporary_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        except Exception as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            return ArchiveExportResult(
                batch_id,
                "failed",
                path=str(path),
                event_count=len(events),
                content_hash=content_hash,
                manifest_hash=manifest_hash,
                error=str(exc)[:1000],
            )
        return ArchiveExportResult(
            batch_id,
            "exported",
            path=str(path),
            event_count=len(events),
            content_hash=content_hash,
            manifest_hash=manifest_hash,
        )

    async def export_pending(
        self, chat_id: str | None = None
    ) -> list[ArchiveExportResult]:
        batches = await self._archive_index.list_for_webui(chat_id) if chat_id else []
        if chat_id is None:
            chat_ids = await self._archive_index.chat_ids()
            batches = [
                batch
                for current_chat_id in chat_ids
                for batch in await self._archive_index.list_for_webui(current_chat_id)
            ]
        results = []
        for batch in batches:
            export = batch.get("export_status", "")
            if export == "exported":
                continue
            results.append(await self.export_batch(str(batch["batch_id"])))
        return results
