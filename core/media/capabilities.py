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


def _has_dependency_error(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ImportError, ModuleNotFoundError)):
            return True
        message = str(current).lower()
        if "未安装" in message or "missing dependency" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def safe_error_category(error: BaseException) -> str:
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    if status in {401, 403}:
        return "unauthorized"
    if status == 429:
        return "rate_limited"
    if status is not None and status >= 500:
        return "server_error"
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    if error.__class__.__module__.startswith("httpx"):
        return "network_error"
    if _has_dependency_error(error):
        return "dependency_missing"
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return "dependency_missing"
    if str(error).strip().lower() == "unavailable":
        return "unavailable"
    return "failed"


class MediaProviderHealth:
    """Tracks transient cooling and permanent provider unavailability."""

    def __init__(self, *, cooldown_seconds: float = 60.0):
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._cooldown_until: dict[str, float] = {}
        self._unavailable: set[str] = set()
        self._last_error_category: dict[str, str] = {}

    def available(self, provider_id: str) -> bool:
        if provider_id in self._unavailable:
            return False
        return time.monotonic() >= self._cooldown_until.get(provider_id, 0.0)

    def mark_success(self, provider_id: str) -> None:
        self._cooldown_until.pop(provider_id, None)
        self._last_error_category.pop(provider_id, None)

    def mark_failure(self, provider_id: str, error: BaseException) -> None:
        self._last_error_category[provider_id] = safe_error_category(error)
        status = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        if status in {401, 403, 400} or _has_dependency_error(error):
            self._unavailable.add(provider_id)
            return
        if status in {408, 409, 425, 429} or (status is not None and status >= 500):
            self._cooldown_until[provider_id] = (
                time.monotonic() + self.cooldown_seconds
            )
            return
        if isinstance(
            error, (asyncio.TimeoutError, OSError)
        ) or error.__class__.__module__.startswith("httpx"):
            self._cooldown_until[provider_id] = (
                time.monotonic() + self.cooldown_seconds
            )

    def snapshot(self) -> dict[str, str]:
        now = time.monotonic()
        result = {provider_id: "unavailable" for provider_id in self._unavailable}
        result.update(
            {
                provider_id: "cooldown"
                for provider_id, until in self._cooldown_until.items()
                if until > now and provider_id not in self._unavailable
            }
        )
        return result

    def details(self) -> dict[str, dict[str, str]]:
        details = {
            provider_id: {
                "status": status,
                "last_error_category": self._last_error_category.get(provider_id, ""),
            }
            for provider_id, status in self.snapshot().items()
        }
        for provider_id, category in self._last_error_category.items():
            details.setdefault(
                provider_id,
                {"status": "available", "last_error_category": category},
            )
        return details


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
        total_timeout: float | None = None,
        health: MediaProviderHealth | None = None,
    ):
        self.name = name
        self.resource_types = frozenset(resource_types)
        self.max_bytes = max(1, int(max_bytes))
        self.timeout = max(1, float(timeout))
        self.total_timeout = (
            None if total_timeout is None else max(0.1, float(total_timeout))
        )
        self.providers = tuple(providers)
        self._semaphore = asyncio.Semaphore(max(1, int(concurrency)))
        self.health = health or MediaProviderHealth()
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

    async def preload(self) -> None:
        """Best-effort preload; an adapter may expose ``preload`` and ``required``."""
        for provider in self.providers:
            preload = getattr(provider, "preload", None)
            if preload is None:
                continue
            try:
                await preload()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                provider_id = getattr(provider, "provider_id", provider.name)
                self.health.mark_failure(provider_id, exc)
                _log.warning(
                    "媒体 provider 预加载失败 capability=%s provider=%s error=%s",
                    self.name,
                    provider_id,
                    type(exc).__name__,
                )
                if getattr(provider, "required", False):
                    raise

    async def _execute(self, record: MediaRecord, **kwargs: Any) -> CapabilityResult:
        errors: list[Exception] = []
        deadline = (
            None
            if self.total_timeout is None
            else asyncio.get_running_loop().time() + self.total_timeout
        )
        for provider in self.providers:
            provider_id = getattr(provider, "provider_id", provider.name)
            if not self.health.available(provider_id):
                _log.info(
                    "媒体 provider 跳过 capability=%s provider=%s state=%s",
                    self.name,
                    provider_id,
                    self.health.snapshot().get(provider_id, "unknown"),

                )
                continue
            remaining = (
                None
                if deadline is None
                else deadline - asyncio.get_running_loop().time()
            )
            if remaining is not None and remaining <= 0:
                errors.append(asyncio.TimeoutError())
                break
            provider_timeout = getattr(provider, "timeout_seconds", None)
            if provider_timeout is None:
                provider_timeout = self.timeout
            try:
                provider_timeout = max(0.1, float(provider_timeout))
            except (TypeError, ValueError, OverflowError):
                provider_timeout = self.timeout
            timeout = min(self.timeout, provider_timeout)
            if remaining is not None:
                timeout = min(timeout, remaining)
            started = time.monotonic()
            try:
                async with self._semaphore:
                    content = await asyncio.wait_for(
                        provider.execute(record, **kwargs), timeout=timeout
                    )
                content = str(content or "").strip()
                if not content:
                    raise ValueError("provider returned empty content")
                self.health.mark_success(provider_id)
                result = CapabilityResult(
                    content,
                    provider_id,
                    getattr(provider, "model_name", ""),
                )
                _log.info(
                    "媒体能力完成 capability=%s provider=%s model=%s media_id=%s elapsed_ms=%d",
                    self.name,
                    provider_id,
                    result.model,
                    record.media_id,
                    (time.monotonic() - started) * 1000,
                )
                return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors.append(exc)
                self.health.mark_failure(provider_id, exc)
                _log.warning(
                    "媒体能力失败 capability=%s provider=%s model=%s media_id=%s elapsed_ms=%d detail=%s",
                    self.name,
                    provider_id,
                    getattr(provider, "model_name", ""),
                    record.media_id,
                    (time.monotonic() - started) * 1000,
                    safe_error_category(exc),
                    exc_info=False,
                )
        if errors and all(isinstance(error, asyncio.TimeoutError) for error in errors):
            raise MediaCapabilityTimeoutError(f"{self.name} providers timed out")
        raise RuntimeError(f"{self.name} providers failed") from (
            errors[-1] if errors else None
        )
