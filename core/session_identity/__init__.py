"""Canonical conversation identities and channel delivery targets."""

from .adapters import (
    ApprovalPrompt,
    ChannelAdapter,
    ChannelCapabilities,
    ChannelRegistry,
    InboundEnvelope,
    QQAdapter,
)
from .catalog import DeliveryTargetCatalog, KnownTarget, TargetFilters, TargetPrincipal
from .identity import (
    ChatType,
    ConversationRef,
    DeliveryTarget,
    DeliveryValidation,
    SessionKind,
    build_chat_session_key,
    build_internal_session_key,
    build_workspace_slug,
    parse_chat_session_key,
    require_delivery_target,
)
from .registry import SessionIdentityRegistry
from .resolver import SessionIdentityResolver

__all__ = [
    "ChannelAdapter",
    "ChannelCapabilities",
    "ChannelRegistry",
    "ApprovalPrompt",
    "ChatType",
    "ConversationRef",
    "DeliveryTarget",
    "DeliveryTargetCatalog",
    "DeliveryValidation",
    "InboundEnvelope",
    "KnownTarget",
    "TargetFilters",
    "TargetPrincipal",
    "QQAdapter",
    "SessionIdentityRegistry",
    "SessionIdentityResolver",
    "SessionKind",
    "build_chat_session_key",
    "build_internal_session_key",
    "build_workspace_slug",
    "parse_chat_session_key",
    "require_delivery_target",
]
