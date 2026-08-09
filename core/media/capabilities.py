import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol

from core.media.models import MediaRecord

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CapabilityResult:
    content: str
    provider: str
    model: str
    cached: bool = False


class MediaCapabilityProvider(Protocol):
    name: str
    model_name: str

    async def execute(self, record: MediaRecord, **kwargs: Any) -> str: ...


class MediaCapabilityTimeoutError(RuntimeError):
    pass


class MediaCapability:
    """Runs a media capability through its configured provider fallback chain."""

    def __init__(
        self,
        *,
        name: str,
        resource_types: set[str],
        max_bytes: int,
        timeout: float,
        concurrency: int,
        providers: list[MediaCapabilityProvider],
        cache_size: int = 200,
    ):
        self.name = name
        self.resource_types = frozenset(resource_types)
        self.max_bytes = max(1, int(max_bytes))
        self.timeout = max(1, float(timeout))
        self.providers = tuple(providers)
        self._semaphore = asyncio.Semaphore(max(1, int(concurrency)))
        self._cache: OrderedDict[tuple[str, ...], CapabilityResult] = OrderedDict()
        self._cache_size = max(0, int(cache_size))
        self._tasks: dict[tuple[str, ...], asyncio.Task[CapabilityResult]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.providers)

    def supports(self, record: MediaRecord) -> bool:
        return (
            record.resource_type in self.resource_types
            and record.size <= self.max_bytes
        )

    async def execute(
        self,
        record: MediaRecord,
        *,
        cache_key: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> CapabilityResult:
        if not self.supports(record):
            raise ValueError(f"{self.name} does not support this media")

        if cache_key is None:
            return await self._execute(record, **kwargs)

        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return CapabilityResult(
                cached.content, cached.provider, cached.model, cached=True
            )

        task = self._tasks.get(cache_key)
        if task is None:
            task = asyncio.create_task(self._execute(record, **kwargs))
            self._tasks[cache_key] = task
            task.add_done_callback(
                lambda completed: (
                    self._tasks.pop(cache_key, None)
                    if self._tasks.get(cache_key) is completed
                    else None
                )
            )
        try:
            result = await asyncio.shield(task)
        finally:
            if task.done():
                self._tasks.pop(cache_key, None)

        if self._cache_size:
            self._cache[cache_key] = result
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result

    async def _execute(self, record: MediaRecord, **kwargs: Any) -> CapabilityResult:
        errors: list[Exception] = []
        for provider in self.providers:
            started = time.monotonic()
            try:
                async with self._semaphore:
                    content = await asyncio.wait_for(
                        provider.execute(record, **kwargs), timeout=self.timeout
                    )
                content = str(content or "").strip()
                if not content:
                    raise ValueError("provider returned empty content")
                result = CapabilityResult(
                    content,
                    provider.name,
                    getattr(provider, "model_name", ""),
                )
                _log.info(
                    "媒体能力完成 capability=%s provider=%s model=%s media_id=%s elapsed_ms=%d",
                    self.name,
                    result.provider,
                    result.model,
                    record.media_id,
                    (time.monotonic() - started) * 1000,
                )
                return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors.append(exc)
                _log.warning(
                    "媒体能力失败 capability=%s provider=%s model=%s media_id=%s elapsed_ms=%d error=%s detail=%.500s",
                    self.name,
                    provider.name,
                    getattr(provider, "model_name", ""),
                    record.media_id,
                    (time.monotonic() - started) * 1000,
                    type(exc).__name__,
                    str(exc),
                    exc_info=True,
                )
        if errors and all(isinstance(error, asyncio.TimeoutError) for error in errors):
            raise MediaCapabilityTimeoutError(f"{self.name} providers timed out")
        raise RuntimeError(f"{self.name} providers failed") from errors[-1]
