"""Batch-scoped lazy media context construction for provider turns."""

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Iterable

from core.managers.session_manager import PendingInbound
from core.media.models import MediaTurnContext
from core.message import InputMessage, ResourceMeta


@dataclass(frozen=True)
class MediaCapabilityReceipt:
    """Observable outcome for one batch-scoped media capability request."""

    capability: str = "prepare_for_ai"
    capability_version: str = "media-v1"
    status: str = "accepted"
    cache_state: str = "miss"
    resource_count: int = 0
    skipped_resource_count: int = 0
    input_chars: int = 0
    output_chars: int = 0
    download_bytes: int = 0
    timeout_seconds: float = 0.0
    elapsed_ms: int = 0
    error_code: str = ""


@dataclass(frozen=True)
class BatchMediaLimits:
    """Hard limits applied after admission and before media analysis."""

    max_resources: int = 8
    max_chars: int = 12000
    max_download_bytes: int = 100 * 1024 * 1024
    capability_timeout_seconds: float = 120.0


@dataclass(frozen=True)
class BatchMediaContext:
    """The media context and stable resource references consumed by one turn."""

    turn_id: str
    resource_uris: tuple[str, ...]
    context: MediaTurnContext
    receipt: MediaCapabilityReceipt = field(default_factory=MediaCapabilityReceipt)

    def as_text(self) -> str:
        return self.context.as_text()


class BatchMediaContextBuilder:
    """Build media context only after a batch has been admitted for a provider turn.

    A turn may retry or await the same builder concurrently. The immutable batch
    fingerprint makes both cases share one preparation task without turning the
    ingress path into a media-analysis path.
    """

    def __init__(
        self,
        media_service,
        *,
        max_cached_turns: int = 256,
        limits: BatchMediaLimits | None = None,
        capability_version: str = "media-v1",
    ):
        self.media_service = media_service
        self._max_cached_turns = max(1, max_cached_turns)
        self.limits = limits or BatchMediaLimits()
        self.capability_version = capability_version
        self._tasks: OrderedDict[tuple[str, str, tuple[str, ...]], asyncio.Task] = (
            OrderedDict()
        )
        self._lock = asyncio.Lock()

    @staticmethod
    def _resource_key(resource: ResourceMeta) -> str:
        return (
            resource.media_uri
            or resource.hash
            or resource.media_id
            or resource.resource_id
        )

    @classmethod
    def _deduplicate_resources(
        cls, resources: Iterable[ResourceMeta]
    ) -> list[ResourceMeta]:
        unique: list[ResourceMeta] = []
        seen: set[tuple[str, str]] = set()
        for resource in resources:
            key = cls._resource_key(resource)
            identity = (resource.resource_type, key)
            if key and identity in seen:
                continue
            if key:
                seen.add(identity)
            unique.append(resource)
        return unique

    def _batch_message(
        self, turn_id: str, items: tuple[PendingInbound, ...]
    ) -> tuple[InputMessage, tuple[str, ...], int, int, int, int]:
        if not items:
            raise ValueError("media batch cannot be empty")
        last = items[-1].message
        if any(item.message.chat_id != last.chat_id for item in items):
            raise ValueError("media batch must belong to one chat")
        seen: set[tuple[str, str]] = set()
        used_bytes = 0
        skipped = 0
        selected_count = 0

        def select(candidates: Iterable[ResourceMeta]) -> list[ResourceMeta]:
            nonlocal selected_count, used_bytes, skipped
            selected: list[ResourceMeta] = []
            for resource in candidates:
                key = self._resource_key(resource)
                identity = (resource.resource_type, key)
                if key and identity in seen:
                    continue
                try:
                    size = max(0, int(resource.size or 0))
                except (TypeError, ValueError, OverflowError):
                    size = 0
                if selected_count >= self.limits.max_resources or (
                    used_bytes + size > self.limits.max_download_bytes
                ):
                    skipped += 1
                    continue
                if key:
                    seen.add(identity)
                selected_count += 1
                used_bytes += size
                selected.append(resource)
            return selected

        resources = select(
            resource for item in items for resource in item.message.resources
        )
        replied_resources = select(
            resource for item in items for resource in item.message.replied_resources
        )
        resource_uris = tuple(
            resource.media_uri
            for resource in (*resources, *replied_resources)
            if resource.media_uri
        )
        input_chars = sum(
            len(item.prepared_content or item.message.content or "") for item in items
        )
        return (
            InputMessage(
                id=turn_id,
                sender_id=last.sender_id,
                chat_id=last.chat_id,
                content=last.content,
                is_group=last.is_group,
                resources=resources,
                replied_resources=replied_resources,
            ),
            resource_uris,
            len(resources) + len(replied_resources),
            skipped,
            input_chars,
            used_bytes,
        )

    async def build(
        self, *, turn_id: str, items: tuple[PendingInbound, ...]
    ) -> BatchMediaContext:
        (
            message,
            resource_uris,
            resource_count,
            skipped_resource_count,
            input_chars,
            download_bytes,
        ) = self._batch_message(turn_id, items)
        cache_key = (turn_id, self.capability_version, resource_uris)
        async with self._lock:
            task = self._tasks.get(cache_key)
            if task is None:
                cache_state = "miss"
                task = asyncio.create_task(
                    self._prepare(
                        turn_id,
                        resource_uris,
                        message,
                        resource_count=resource_count,
                        skipped_resource_count=skipped_resource_count,
                        input_chars=input_chars,
                        download_bytes=download_bytes,
                        cache_state=cache_state,
                    )
                )
                self._tasks[cache_key] = task
            else:
                cache_state = "hit" if task.done() else "inflight"
            self._tasks.move_to_end(cache_key)
            while len(self._tasks) > self._max_cached_turns:
                _, evicted = self._tasks.popitem(last=False)
                if not evicted.done():
                    evicted.cancel()
        try:
            result = await asyncio.shield(task)
            if result.receipt.cache_state != cache_state:
                return replace(
                    result,
                    receipt=replace(result.receipt, cache_state=cache_state),
                )
            return result
        except BaseException:
            async with self._lock:
                if self._tasks.get(cache_key) is task:
                    self._tasks.pop(cache_key, None)
            raise

    async def _prepare(
        self,
        turn_id: str,
        resource_uris: tuple[str, ...],
        message: InputMessage,
        *,
        resource_count: int,
        skipped_resource_count: int,
        input_chars: int,
        download_bytes: int,
        cache_state: str,
    ) -> BatchMediaContext:
        started = time.monotonic()
        status = "accepted"
        error_code = ""
        if self.media_service is None:
            context = MediaTurnContext()
        else:
            try:
                context = await asyncio.wait_for(
                    self.media_service.prepare_for_ai(message),
                    timeout=self.limits.capability_timeout_seconds,
                )
            except asyncio.TimeoutError:
                context = MediaTurnContext()
                status = "timeout"
                error_code = "CAPABILITY_TIMEOUT"
            except Exception as exc:
                context = MediaTurnContext()
                status = "failed"
                error_code = type(exc).__name__
        output_chars = len(context.as_text())
        if output_chars > self.limits.max_chars:
            context = MediaTurnContext(
                current_blocks=(context.as_text()[: self.limits.max_chars],)
            )
            output_chars = len(context.as_text())
            status = "truncated" if status == "accepted" else status
        receipt = MediaCapabilityReceipt(
            capability_version=self.capability_version,
            status=status,
            cache_state=cache_state,
            resource_count=resource_count,
            skipped_resource_count=skipped_resource_count,
            input_chars=input_chars,
            output_chars=output_chars,
            download_bytes=download_bytes,
            timeout_seconds=self.limits.capability_timeout_seconds,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            error_code=error_code,
        )
        return BatchMediaContext(turn_id, resource_uris, context, receipt)
