"""Runtime configuration with validated, revisioned domain snapshots."""

from core.runtime_settings.coordinator import (
    RuntimeSettingsConflict,
    RuntimeSettingsCoordinator,
    RuntimeSettingsDegraded,
)
from core.runtime_settings.engagement import (
    ENGAGEMENT_RUNTIME_FIELDS,
    EngagementSettingsAdapter,
    EngagementSnapshot,
    GroupTargetVerifier,
    InMemoryGroupTargetVerifier,
    ObservedGroupTargetVerifier,
    TargetStatus,
    UnavailableGroupTargetVerifier,
    validate_engagement_patch,
)
from core.runtime_settings.store import (
    AuditRecord,
    EngagementTarget,
    RuntimeSettingsError,
    RuntimeSettingsRecord,
    RuntimeSettingsStore,
)

__all__ = [
    "AuditRecord",
    "ENGAGEMENT_RUNTIME_FIELDS",
    "EngagementSettingsAdapter",
    "EngagementSnapshot",
    "EngagementTarget",
    "GroupTargetVerifier",
    "InMemoryGroupTargetVerifier",
    "ObservedGroupTargetVerifier",
    "RuntimeSettingsConflict",
    "RuntimeSettingsError",
    "RuntimeSettingsCoordinator",
    "RuntimeSettingsDegraded",
    "RuntimeSettingsRecord",
    "RuntimeSettingsStore",
    "TargetStatus",
    "UnavailableGroupTargetVerifier",
    "validate_engagement_patch",
]
