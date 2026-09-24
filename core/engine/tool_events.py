"""Channel-neutral tool lifecycle events."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass(frozen=True, slots=True)
class ToolLifecycleEvent:
    """One model tool call's execution lifecycle update.

    ``metadata`` carries the payload a channel needs for live tool cards
    (arguments, result, resources); each channel owns redaction and truncation.
    """

    event_type: str
    session_id: str
    turn_id: str
    tool_call_id: str
    tool_name: str
    status: str
    occurred_at: float = field(default_factory=time.time)
    elapsed_ms: int | None = None
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


ToolEventCallback = Callable[[ToolLifecycleEvent], Awaitable[None]]
