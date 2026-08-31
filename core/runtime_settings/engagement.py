"""Strict validation and application model for group engagement settings."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Awaitable, Callable, Mapping, Protocol

from core.engine.engagement_config import EngagementConfig, normalize_engagement_config
from core.runtime_settings.store import (
    EngagementTarget,
    RuntimeSettingsRecord,
    RuntimeSettingsStore,
)

ENGAGEMENT_RUNTIME_FIELDS = (
    "group_ambient_mode",
    "group_ambient_active_chats",
    "group_ambient_idle_ms",
    "group_ambient_cooldown_seconds",
    "group_ambient_quiet_cooldown_seconds",
    "group_ambient_window_seconds",
    "group_ambient_max_turns_per_window",
    "group_ambient_max_age_seconds",
    "group_ambient_min_messages",
    "group_ambient_allow_single_question",
    "group_ambient_allow_single_media",
    "group_ambient_quote",
    "group_ambient_stale_quote_seconds",
)

_CHAT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class TargetStatus(StrEnum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REMOVED = "removed"


class GroupTargetVerifier(Protocol):
    async def verify(self, chat_id: str) -> bool:
        """Return whether the bot currently belongs to the target group."""


class UnavailableGroupTargetVerifier:
    async def verify(self, chat_id: str) -> bool:
        return False


class ObservedGroupTargetVerifier:
    """Use a received group event as a conservative membership proof."""

    def __init__(self):
        self._observed: set[str] = set()

    def observe(self, chat_id: str) -> None:
        self._observed.add(chat_id)

    async def verify(self, chat_id: str) -> bool:
        return chat_id in self._observed


class InMemoryGroupTargetVerifier:
    """Deterministic verifier adapter for tests and local development."""

    def __init__(self, verified: set[str] | None = None):
        self.verified = set(verified or ())

    async def verify(self, chat_id: str) -> bool:
        return chat_id in self.verified


@dataclass(frozen=True)
class EngagementSnapshot:
    config: EngagementConfig
    revision: int
    overrides: dict[str, Any]
    targets: tuple[EngagementTarget, ...] = ()
    capability_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        values = {key: getattr(self.config, key) for key in ENGAGEMENT_RUNTIME_FIELDS}
        values["group_ambient_active_chats"] = list(
            values["group_ambient_active_chats"]
        )
        return {
            "revision": self.revision,
            "overrides": dict(self.overrides),
            "effective": values,
            "capability_enabled": self.capability_enabled,
            "targets": [
                {
                    "chat_id": target.chat_id,
                    "verification_status": target.verification_status,
                    "first_observed_at": target.first_observed_at,
                    "last_observed_at": target.last_observed_at,
                    "verified_at": target.verified_at,
                }
                for target in self.targets
            ],
        }


def _error(field: str, message: str) -> ValueError:
    return ValueError(f"{field}: {message}")


def _validate_number(
    field: str,
    value: Any,
    *,
    minimum: int | float,
    maximum: int | float,
    integer: bool = False,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(field, "must be a number")
    if value < minimum or value > maximum:
        raise _error(field, f"must be between {minimum} and {maximum}")
    if integer and int(value) != value:
        raise _error(field, "must be an integer")
    return int(value) if integer else float(value)


def _validate_chat_ids(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise _error(field, "must be an array of IDs")
    if len(value) > 100:
        raise _error(field, "at most 100 targets are allowed")
    result = []
    for chat_id in value:
        if not isinstance(chat_id, str) or not _CHAT_ID_RE.fullmatch(chat_id):
            raise _error(field, "contains an invalid group ID")
        if chat_id in result:
            raise _error(field, "contains duplicate group IDs")
        result.append(chat_id)
    return tuple(result)


def _validate_bool(field: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise _error(field, "must be a boolean")
    return value


def validate_engagement_patch(
    base: Mapping[str, Any],
    current_override: Mapping[str, Any],
    patch: Mapping[str, Any],
    *,
    capability_enabled: bool = True,
    verified_targets: set[str] | None = None,
) -> tuple[dict[str, Any], EngagementConfig]:
    unknown = set(patch) - set(ENGAGEMENT_RUNTIME_FIELDS)
    if unknown:
        raise _error("settings", f"unknown fields: {', '.join(sorted(unknown))}")
    persisted_unknown = set(current_override) - set(ENGAGEMENT_RUNTIME_FIELDS)
    if persisted_unknown:
        raise _error(
            "settings",
            f"unknown persisted fields: {', '.join(sorted(persisted_unknown))}",
        )
    merged_override = dict(current_override)
    merged_override.update(patch)
    merged = dict(base)
    merged.update(merged_override)
    clean: dict[str, Any] = {}
    ambient_mode = merged.get("group_ambient_mode", "off")
    if ambient_mode not in {"off", "shadow", "active"}:
        raise _error("group_ambient_mode", "must be off, shadow, or active")
    clean["group_ambient_mode"] = ambient_mode
    clean["group_ambient_active_chats"] = _validate_chat_ids(
        merged.get("group_ambient_active_chats", ()),
        "group_ambient_active_chats",
    )
    clean["group_ambient_idle_ms"] = _validate_number(
        "group_ambient_idle_ms",
        merged.get("group_ambient_idle_ms", 1000),
        minimum=0,
        maximum=120000,
        integer=True,
    )
    clean["group_ambient_cooldown_seconds"] = _validate_number(
        "group_ambient_cooldown_seconds",
        merged.get("group_ambient_cooldown_seconds", 30.0),
        minimum=0,
        maximum=86400,
    )
    clean["group_ambient_quiet_cooldown_seconds"] = _validate_number(
        "group_ambient_quiet_cooldown_seconds",
        merged.get("group_ambient_quiet_cooldown_seconds", 10.0),
        minimum=0,
        maximum=86400,
    )
    clean["group_ambient_window_seconds"] = _validate_number(
        "group_ambient_window_seconds",
        merged.get("group_ambient_window_seconds", 300.0),
        minimum=1,
        maximum=604800,
    )
    clean["group_ambient_max_turns_per_window"] = _validate_number(
        "group_ambient_max_turns_per_window",
        merged.get("group_ambient_max_turns_per_window", 4),
        minimum=1,
        maximum=1000,
        integer=True,
    )
    clean["group_ambient_max_age_seconds"] = _validate_number(
        "group_ambient_max_age_seconds",
        merged.get("group_ambient_max_age_seconds", 600.0),
        minimum=1,
        maximum=604800,
    )
    clean["group_ambient_min_messages"] = _validate_number(
        "group_ambient_min_messages",
        merged.get("group_ambient_min_messages", 2),
        minimum=1,
        maximum=100,
        integer=True,
    )
    for field, default in (
        ("group_ambient_allow_single_question", True),
        ("group_ambient_allow_single_media", False),
        ("group_ambient_quote", False),
    ):
        clean[field] = _validate_bool(field, merged.get(field, default))
    clean["group_ambient_stale_quote_seconds"] = _validate_number(
        "group_ambient_stale_quote_seconds",
        merged.get("group_ambient_stale_quote_seconds", 120.0),
        minimum=0,
        maximum=604800,
    )
    if ambient_mode in {"shadow", "active"} and not clean["group_ambient_active_chats"]:
        raise _error("group_ambient_active_chats", "is required when enabled")
    if ambient_mode in {"shadow", "active"} and not capability_enabled:
        raise _error("group_ambient_mode", "mode routing capability is disabled")
    if ambient_mode == "active":
        missing = set(clean["group_ambient_active_chats"]) - set(verified_targets or ())
        if missing:
            raise _error(
                "group_ambient_active_chats",
                f"unverified targets: {', '.join(sorted(missing))}",
            )

    effective = dict(merged)
    effective.update(clean)
    return (
        {
            key: value
            for key, value in merged_override.items()
            if key in ENGAGEMENT_RUNTIME_FIELDS
        },
        normalize_engagement_config(effective),
    )


class EngagementSettingsAdapter:
    """Translate one runtime domain into an immutable engagement snapshot."""

    def __init__(
        self,
        base: Mapping[str, Any],
        store: RuntimeSettingsStore,
        *,
        capability_enabled: bool,
        verifier: GroupTargetVerifier | None = None,
        install: Callable[[EngagementConfig], Awaitable[None] | None] | None = None,
    ):
        self._base = dict(base)
        self._store = store
        self._capability_enabled = capability_enabled
        self._verifier = verifier or UnavailableGroupTargetVerifier()
        self._install = install
        self._snapshot: EngagementSnapshot | None = None

    @property
    def snapshot(self) -> EngagementSnapshot:
        if self._snapshot is None:
            raise RuntimeError("engagement settings adapter is not initialized")
        return self._snapshot

    def set_install(
        self, install: Callable[[EngagementConfig], Awaitable[None] | None]
    ) -> None:
        self._install = install

    async def _target_records(
        self, chat_ids: tuple[str, ...]
    ) -> tuple[EngagementTarget, ...]:
        records = []
        for chat_id in chat_ids:
            record = await self._store.ensure_target(chat_id)
            records.append(record)
        return tuple(records)

    async def build(
        self,
        overrides: Mapping[str, Any],
        *,
        revision: int,
        patch: Mapping[str, Any] | None = None,
    ) -> EngagementSnapshot:
        patch = patch or {}
        existing_targets = await self._store.list_targets()
        verified_targets = {
            target.chat_id
            for target in existing_targets
            if target.verification_status == TargetStatus.VERIFIED
        }
        next_overrides, config = validate_engagement_patch(
            self._base,
            overrides,
            patch,
            capability_enabled=self._capability_enabled,
            verified_targets=verified_targets,
        )
        target_ids = tuple(dict.fromkeys(config.group_ambient_active_chats))
        targets = await self._target_records(target_ids)
        return EngagementSnapshot(
            config=config,
            revision=revision,
            overrides=next_overrides,
            targets=targets,
            capability_enabled=self._capability_enabled,
        )

    async def initialize(self, record: RuntimeSettingsRecord) -> EngagementSnapshot:
        if record.overrides:
            snapshot = await self.build(record.overrides, revision=record.revision)
        else:
            config = normalize_engagement_config(self._base)
            target_ids = tuple(dict.fromkeys(config.group_ambient_active_chats))
            targets = await self._target_records(target_ids)
            snapshot = EngagementSnapshot(
                config=config,
                revision=record.revision,
                overrides={},
                targets=targets,
                capability_enabled=self._capability_enabled,
            )
        self._snapshot = snapshot
        return snapshot

    async def install(self, snapshot: EngagementSnapshot) -> None:
        if self._install is not None:
            result = self._install(snapshot.config)
            if inspect.isawaitable(result):
                await result
        self._snapshot = snapshot

    async def restore(self, snapshot: EngagementSnapshot) -> None:
        await self.install(snapshot)

    async def fail_closed(self) -> None:
        current_revision = self._snapshot.revision if self._snapshot else 0
        snapshot = await self.build(
            {},
            revision=current_revision,
            patch={
                "group_ambient_mode": "off",
                "group_ambient_active_chats": [],
            },
        )
        await self.install(snapshot)

    async def verify_target(self, chat_id: str) -> EngagementTarget:
        if not _CHAT_ID_RE.fullmatch(chat_id):
            raise ValueError("invalid group ID")
        await self._store.ensure_target(chat_id)
        if not await self._verifier.verify(chat_id):
            return await self._store.set_target_status(chat_id, TargetStatus.UNVERIFIED)
        return await self._store.set_target_status(chat_id, TargetStatus.VERIFIED)

    async def remove_target(self, chat_id: str) -> EngagementTarget:
        if not _CHAT_ID_RE.fullmatch(chat_id):
            raise ValueError("invalid group ID")
        return await self._store.set_target_status(chat_id, TargetStatus.REMOVED)

    async def list_targets(self) -> list[EngagementTarget]:
        return await self._store.list_targets()
