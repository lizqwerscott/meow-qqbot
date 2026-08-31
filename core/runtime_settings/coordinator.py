"""Atomic, revisioned installation of runtime domain snapshots."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping

from core.runtime_settings.engagement import (
    EngagementSettingsAdapter,
    EngagementSnapshot,
)
from core.runtime_settings.store import RuntimeSettingsConflict, RuntimeSettingsStore

_log = logging.getLogger(__name__)


class RuntimeSettingsDegraded(RuntimeError):
    """Raised when a runtime snapshot cannot be safely restored."""


class RuntimeSettingsCoordinator:
    """Serialize validation, live installation, persistence, and rollback."""

    def __init__(self, store: RuntimeSettingsStore, adapter: EngagementSettingsAdapter):
        self._store = store
        self._adapter = adapter
        self._lock = asyncio.Lock()
        self._snapshot: EngagementSnapshot | None = None
        self._degraded = False
        self._degraded_reason: str | None = None

    async def initialize(self) -> EngagementSnapshot:
        async with self._lock:
            record = await self._store.get("engagement")
            self._snapshot = await self._adapter.initialize(record)
            return self._snapshot

    def snapshot(self) -> EngagementSnapshot:
        if self._snapshot is None:
            raise RuntimeError("runtime settings coordinator is not initialized")
        return self._snapshot

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def degraded_reason(self) -> str | None:
        return self._degraded_reason

    async def _audit_failure(
        self,
        *,
        action: str,
        fields: list[str],
        previous_revision: int,
        new_revision: int,
        failure_class: str,
        source: str,
        remote_ip: str | None,
        expected_revision: int | None = None,
        actual_revision: int | None = None,
    ) -> None:
        change: dict[str, Any] = {"action": action, "fields": fields}
        if expected_revision is not None:
            change["expected_revision"] = expected_revision
        if actual_revision is not None:
            change["actual_revision"] = actual_revision
        try:
            await self._store.append_audit(
                "engagement",
                previous_revision=previous_revision,
                new_revision=new_revision,
                change=change,
                source=source,
                remote_ip=remote_ip,
                outcome="failure",
                failure_class=failure_class,
            )
        except Exception:
            _log.exception("runtime settings failure audit could not be persisted")

    async def _restore_or_degrade(
        self,
        previous: EngagementSnapshot,
        *,
        action: str,
        fields: list[str],
        source: str,
        remote_ip: str | None,
        failure_revision: int,
    ) -> None:
        try:
            await self._adapter.restore(previous)
        except Exception as exc:
            self._degraded = True
            self._degraded_reason = "rollback_failed"
            await self._audit_failure(
                action=action,
                fields=fields,
                previous_revision=previous.revision,
                new_revision=failure_revision,
                failure_class="rollback_failed",
                source=source,
                remote_ip=remote_ip,
            )
            try:
                await self._adapter.fail_closed()
            except Exception:
                _log.exception("runtime settings fail-closed installation failed")
            raise RuntimeSettingsDegraded(
                "runtime settings rollback failed; engagement execution is degraded"
            ) from exc

    async def update(
        self,
        *,
        expected_revision: int,
        patch: Mapping[str, Any],
        source: str = "webui",
        remote_ip: str | None = None,
    ) -> EngagementSnapshot:
        async with self._lock:
            if self._degraded:
                raise RuntimeSettingsDegraded(
                    f"runtime settings are degraded: {self._degraded_reason}"
                )
            current = await self._store.get("engagement")
            if current.revision != expected_revision:
                fields = sorted(str(field) for field in patch)
                await self._audit_failure(
                    action="update",
                    fields=fields,
                    previous_revision=current.revision,
                    new_revision=current.revision,
                    failure_class="conflict",
                    source=source,
                    remote_ip=remote_ip,
                    expected_revision=expected_revision,
                    actual_revision=current.revision,
                )
                raise RuntimeSettingsConflict(
                    "engagement", expected_revision, current.revision
                )
            previous = self.snapshot()
            fields = sorted(str(field) for field in patch)
            try:
                candidate = await self._adapter.build(
                    current.overrides,
                    revision=current.revision + 1,
                    patch=patch,
                )
            except Exception:
                await self._audit_failure(
                    action="update",
                    fields=fields,
                    previous_revision=current.revision,
                    new_revision=current.revision + 1,
                    failure_class="rejected",
                    source=source,
                    remote_ip=remote_ip,
                )
                raise
            try:
                await self._adapter.install(candidate)
            except Exception:
                await self._audit_failure(
                    action="update",
                    fields=fields,
                    previous_revision=current.revision,
                    new_revision=candidate.revision,
                    failure_class="install_failed",
                    source=source,
                    remote_ip=remote_ip,
                )
                await self._restore_or_degrade(
                    previous,
                    action="update",
                    fields=fields,
                    source=source,
                    remote_ip=remote_ip,
                    failure_revision=candidate.revision,
                )
                raise
            try:
                await self._store.commit(
                    "engagement",
                    expected_revision=expected_revision,
                    overrides=candidate.overrides,
                    source=source,
                    remote_ip=remote_ip,
                    change={"action": "update", "fields": sorted(patch)},
                )
            except RuntimeSettingsConflict as exc:
                await self._audit_failure(
                    action="update",
                    fields=fields,
                    previous_revision=current.revision,
                    new_revision=candidate.revision,
                    failure_class="conflict",
                    source=source,
                    remote_ip=remote_ip,
                    expected_revision=expected_revision,
                    actual_revision=exc.actual,
                )
                await self._restore_or_degrade(
                    previous,
                    action="update",
                    fields=fields,
                    source=source,
                    remote_ip=remote_ip,
                    failure_revision=candidate.revision,
                )
                raise
            except Exception:
                await self._audit_failure(
                    action="update",
                    fields=fields,
                    previous_revision=current.revision,
                    new_revision=candidate.revision,
                    failure_class="persist_failed",
                    source=source,
                    remote_ip=remote_ip,
                )
                await self._restore_or_degrade(
                    previous,
                    action="update",
                    fields=fields,
                    source=source,
                    remote_ip=remote_ip,
                    failure_revision=candidate.revision,
                )
                raise
            self._snapshot = candidate
            return candidate

    async def clear(
        self,
        *,
        expected_revision: int,
        key: str | None = None,
        source: str = "webui",
        remote_ip: str | None = None,
    ) -> EngagementSnapshot:
        async with self._lock:
            if self._degraded:
                raise RuntimeSettingsDegraded(
                    f"runtime settings are degraded: {self._degraded_reason}"
                )
            record = await self._store.get("engagement")
            if key is not None and key not in self._adapter_runtime_fields:
                await self._audit_failure(
                    action="clear",
                    fields=[key],
                    previous_revision=record.revision,
                    new_revision=record.revision,
                    failure_class="rejected",
                    source=source,
                    remote_ip=remote_ip,
                )
                raise ValueError(f"unknown runtime setting: {key}")
            if record.revision != expected_revision:
                fields = sorted(record.overrides) if key is None else [key]
                await self._audit_failure(
                    action="clear",
                    fields=fields,
                    previous_revision=record.revision,
                    new_revision=record.revision,
                    failure_class="conflict",
                    source=source,
                    remote_ip=remote_ip,
                    expected_revision=expected_revision,
                    actual_revision=record.revision,
                )
                raise RuntimeSettingsConflict(
                    "engagement", expected_revision, record.revision
                )
            overrides = dict(record.overrides)
            if key is None:
                overrides.clear()
                fields = sorted(record.overrides)
            else:
                overrides.pop(key, None)
                fields = [key]
            previous = self.snapshot()
            try:
                candidate = await self._adapter.build(
                    overrides, revision=record.revision + 1
                )
            except Exception:
                await self._audit_failure(
                    action="clear",
                    fields=fields,
                    previous_revision=record.revision,
                    new_revision=record.revision + 1,
                    failure_class="rejected",
                    source=source,
                    remote_ip=remote_ip,
                )
                raise
            try:
                await self._adapter.install(candidate)
            except Exception:
                await self._audit_failure(
                    action="clear",
                    fields=fields,
                    previous_revision=record.revision,
                    new_revision=candidate.revision,
                    failure_class="install_failed",
                    source=source,
                    remote_ip=remote_ip,
                )
                await self._restore_or_degrade(
                    previous,
                    action="clear",
                    fields=fields,
                    source=source,
                    remote_ip=remote_ip,
                    failure_revision=candidate.revision,
                )
                raise
            try:
                await self._store.commit(
                    "engagement",
                    expected_revision=expected_revision,
                    overrides=overrides,
                    source=source,
                    remote_ip=remote_ip,
                    change={"action": "clear", "fields": fields},
                )
            except RuntimeSettingsConflict as exc:
                await self._audit_failure(
                    action="clear",
                    fields=fields,
                    previous_revision=record.revision,
                    new_revision=candidate.revision,
                    failure_class="conflict",
                    source=source,
                    remote_ip=remote_ip,
                    expected_revision=expected_revision,
                    actual_revision=exc.actual,
                )
                await self._restore_or_degrade(
                    previous,
                    action="clear",
                    fields=fields,
                    source=source,
                    remote_ip=remote_ip,
                    failure_revision=candidate.revision,
                )
                raise
            except Exception:
                await self._audit_failure(
                    action="clear",
                    fields=fields,
                    previous_revision=record.revision,
                    new_revision=candidate.revision,
                    failure_class="persist_failed",
                    source=source,
                    remote_ip=remote_ip,
                )
                await self._restore_or_degrade(
                    previous,
                    action="clear",
                    fields=fields,
                    source=source,
                    remote_ip=remote_ip,
                    failure_revision=candidate.revision,
                )
                raise
            self._snapshot = candidate
            return candidate

    @property
    def _adapter_runtime_fields(self) -> tuple[str, ...]:
        from core.runtime_settings.engagement import ENGAGEMENT_RUNTIME_FIELDS

        return ENGAGEMENT_RUNTIME_FIELDS

    async def targets(self):
        return await self._adapter.list_targets()

    async def verify_target(self, chat_id: str):
        async with self._lock:
            return await self._adapter.verify_target(chat_id)

    async def remove_target(self, chat_id: str):
        async with self._lock:
            if self._degraded:
                raise RuntimeSettingsDegraded(
                    f"runtime settings are degraded: {self._degraded_reason}"
                )
            return await self._adapter.remove_target(chat_id)

    async def audit(self, *, limit: int = 50, before_id: int | None = None):
        return await self._store.list_audit(
            "engagement", limit=limit, before_id=before_id
        )

    async def close(self) -> None:
        await self._store.close()
