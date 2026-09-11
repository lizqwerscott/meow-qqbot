"""Bounded in-process event stream with replay and resync semantics."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from uuid import uuid4

from .models import StreamEvent


class EventHub:
    """Publish ordered session events and expose a reconnectable stream.

    The hub is deliberately bounded. Durable conversation history remains the
    source for resync after process restarts or event eviction.
    """

    def __init__(self, *, max_events_per_session: int = 1000) -> None:
        self._max_events = max(1, int(max_events_per_session))
        self._events: dict[str, deque[StreamEvent]] = defaultdict(
            lambda: deque(maxlen=self._max_events)
        )
        self._sequences: dict[str, int] = defaultdict(int)
        self._condition = asyncio.Condition()

    async def publish(
        self,
        session_id: str,
        event_type: str,
        *,
        turn_id: str = "",
        payload: dict | None = None,
    ) -> StreamEvent:
        async with self._condition:
            sequence = self._sequences[session_id] + 1
            event = StreamEvent(
                event_id=f"webui-event-{uuid4().hex}",
                session_id=session_id,
                turn_id=turn_id,
                event_type=event_type,
                sequence=sequence,
                occurred_at=time.time(),
                payload=dict(payload or {}),
            )
            self._sequences[session_id] = sequence
            self._events[session_id].append(event)
            self._condition.notify_all()
            return event

    async def has_turn_events(self, session_id: str, turn_id: str) -> bool:
        async with self._condition:
            return any(event.turn_id == turn_id for event in self._events[session_id])

    async def stream(
        self, session_id: str, *, after_event_id: str = ""
    ) -> AsyncIterator[StreamEvent | None]:
        """Yield events; ``None`` is a keep-alive marker for SSE."""
        last_sequence = 0
        pending: list[StreamEvent | None] = []
        initialized = False
        while True:
            async with self._condition:
                events = self._events[session_id]
                if not initialized:
                    initialized = True
                    if after_event_id:
                        positions = [
                            index
                            for index, event in enumerate(events)
                            if event.event_id == after_event_id
                        ]
                        if not positions:
                            pending.append(
                                StreamEvent(
                                    event_id=f"resync-{uuid4().hex}",
                                    session_id=session_id,
                                    turn_id="",
                                    event_type="resync_required",
                                    sequence=self._sequences[session_id],
                                    occurred_at=time.time(),
                                    payload={"reason": "event_not_available"},
                                )
                            )
                            last_sequence = self._sequences[session_id]
                        else:
                            last_sequence = events[positions[-1]].sequence
                for event in events:
                    if event.sequence > last_sequence:
                        pending.append(event)
                if pending:
                    last_sequence = max(
                        (event.sequence for event in pending if event is not None),
                        default=last_sequence,
                    )
                if not pending:
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=15.0)
                    except asyncio.TimeoutError:
                        pending.append(None)
            while pending:
                yield pending.pop(0)
