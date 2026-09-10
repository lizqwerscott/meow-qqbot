"""Value objects and key rules for canonical session identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote, unquote

ChatType = Literal["group", "direct"]
SessionKind = Literal["chat", "cron", "heartbeat", "work_plan"]


def _normalize_segment(value: str, field_name: str) -> str:
    value = str(value).strip().lower()
    if not value or ":" in value or "/" in value or "\x00" in value:
        raise ValueError(f"invalid {field_name}")
    return value


def encode_target_id(target_id: str) -> str:
    target_id = str(target_id)
    if not target_id or "\x00" in target_id:
        raise ValueError("target_id must be a non-empty string without NUL")
    return quote(target_id, safe="-._~")


def decode_target_id(encoded_target_id: str) -> str:
    decoded = unquote(encoded_target_id)
    if not decoded or "\x00" in decoded:
        raise ValueError("invalid encoded target_id")
    return decoded


def build_workspace_slug(target: DeliveryTarget) -> str:
    """Return the V1 path-safe workspace segment for a chat target."""
    prefix = "groups" if target.chat_type == "group" else "private"
    return f"{prefix}/v1-{encode_target_id(target.target_id)}"


@dataclass(frozen=True, slots=True)
class DeliveryTarget:
    channel: str
    account_id: str
    chat_type: ChatType
    target_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel", _normalize_segment(self.channel, "channel"))
        object.__setattr__(
            self, "account_id", _normalize_segment(self.account_id, "account_id")
        )
        if self.chat_type not in {"group", "direct"}:
            raise ValueError("chat_type must be 'group' or 'direct'")
        object.__setattr__(self, "target_id", str(self.target_id))
        encode_target_id(self.target_id)

    @property
    def catalog_key(self) -> tuple[str, str, str, str]:
        return (self.channel, self.account_id, self.chat_type, self.target_id)


@dataclass(frozen=True, slots=True)
class DeliveryValidation:
    ok: bool
    reason: str = ""
    target: DeliveryTarget | None = None


@dataclass(frozen=True, slots=True)
class ConversationRef:
    session_key: str
    target: DeliveryTarget | None
    session_kind: SessionKind
    legacy_session_keys: tuple[str, ...] = ()
    hindsight_document_id: str = ""

    def __post_init__(self) -> None:
        if not self.session_key or "\x00" in self.session_key:
            raise ValueError("session_key must be non-empty")
        if self.session_kind not in {"chat", "cron", "heartbeat", "work_plan"}:
            raise ValueError("invalid session_kind")
        if self.session_kind == "chat" and self.target is None:
            raise ValueError("chat sessions require a delivery target")
        if self.session_kind != "chat" and self.target is not None:
            raise ValueError("internal sessions cannot have a delivery target")


def build_chat_session_key(
    target: DeliveryTarget,
    *,
    agent_id: str = "main",
) -> str:
    agent_id = _normalize_segment(agent_id, "agent_id")
    return ":".join(
        (
            "agent",
            agent_id,
            target.channel,
            target.account_id,
            target.chat_type,
            encode_target_id(target.target_id),
        )
    )


def build_internal_session_key(
    kind: Literal["cron", "heartbeat", "work_plan", "work-plan"],
    *segments: str,
    agent_id: str = "main",
) -> str:
    agent_id = _normalize_segment(agent_id, "agent_id")
    kind = _normalize_segment(kind, "kind")
    if kind not in {"cron", "heartbeat", "work_plan", "work-plan"}:
        raise ValueError("invalid internal session kind")
    if not segments:
        raise ValueError("internal session key requires at least one segment")
    encoded = []
    for segment in segments:
        segment = str(segment)
        if not segment or "\x00" in segment:
            raise ValueError("internal session segment must be non-empty")
        encoded.append(quote(segment, safe="-._~"))
    return ":".join(("agent", agent_id, kind, *encoded))


def parse_chat_session_key(session_key: str) -> DeliveryTarget:
    parts = str(session_key).split(":")
    if len(parts) != 6 or parts[0] != "agent" or not parts[1]:
        raise ValueError("not a canonical chat session key")
    _, _, channel, account_id, chat_type, encoded_target_id = parts
    if chat_type not in {"group", "direct"}:
        raise ValueError("not a canonical chat session key")
    return DeliveryTarget(
        channel=channel,
        account_id=account_id,
        chat_type=chat_type,
        target_id=decode_target_id(encoded_target_id),
    )


def require_delivery_target(target: DeliveryTarget) -> DeliveryTarget:
    if not isinstance(target, DeliveryTarget):
        raise TypeError("channel adapters accept DeliveryTarget, not session keys")
    return target
