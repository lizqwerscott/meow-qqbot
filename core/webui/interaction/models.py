"""Stable DTOs for interactive conversations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class WebUiSession:
    session_id: str
    session_key: str
    operator_id: str
    title: str
    mode: str
    created_at: float
    updated_at: float
    read_only: bool = False
    channel: str = "webui"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WebUiSessionPage:
    items: tuple[WebUiSession, ...]
    has_more: bool
    next_cursor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [session.to_dict() for session in self.items],
            "has_more": self.has_more,
            "next_cursor": self.next_cursor,
        }


@dataclass(frozen=True, slots=True)
class SubmissionReceipt:
    session_id: str
    turn_id: str
    request_id: str
    accepted: bool
    duplicate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StreamEvent:
    event_id: str
    session_id: str
    turn_id: str
    event_type: str
    sequence: int
    occurred_at: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "type": self.event_type,
            "sequence": self.sequence,
            "occurred_at": self.occurred_at,
            "payload": dict(self.payload),
        }
