"""Thread-safe in-memory queue for human-readable system events.

Each session has one queue. A wake claims an immutable lease before building
its prompt; successful execution commits that lease, while failures release it.
"""

import logging
import re
import threading
import time
import uuid
from asyncio import current_task
from dataclasses import dataclass, field
from typing import Optional

_log = logging.getLogger(__name__)
_SYSTEM_PREFIX_RE = re.compile(r"^\s*System:\s*", re.IGNORECASE | re.MULTILINE)
MAX_EVENTS = 20


@dataclass(frozen=True)
class SystemEvent:
    text: str
    ts: float
    context_key: Optional[str] = None
    heartbeat_only: bool = False


@dataclass(frozen=True)
class SystemEventLease:
    token: str
    session_key: str
    events: tuple[SystemEvent, ...]
    owner_task_id: int | None = None


@dataclass
class _SessionQueue:
    queue: list[SystemEvent] = field(default_factory=list)
    _seen: set[tuple[str, Optional[str]]] = field(default_factory=set)


class SystemEventBusy(RuntimeError):
    """The session already has an in-flight system-event wake."""


class SystemEventQueue:
    def __init__(self):
        self._queues: dict[str, _SessionQueue] = {}
        self._leases: dict[str, SystemEventLease] = {}
        self._lock = threading.RLock()

    def enqueue(
        self,
        session_key: str,
        text: str,
        context_key: Optional[str] = None,
        *,
        replace: bool = False,
        heartbeat_only: bool = False,
    ) -> bool:
        if not session_key or not session_key.strip():
            _log.warning("system events require a non-empty session_key")
            return False
        cleaned = self._sanitize(text)
        if not cleaned:
            return False
        with self._lock:
            queue = self._get_or_create(session_key)
            if replace and context_key:
                return self._replace_in_queue(
                    queue, cleaned, context_key, heartbeat_only
                )
            dedup_key = (cleaned, context_key)
            if dedup_key in queue._seen:
                return False
            queue.queue.append(
                SystemEvent(
                    text=cleaned,
                    ts=time.time(),
                    context_key=context_key,
                    heartbeat_only=heartbeat_only,
                )
            )
            queue._seen.add(dedup_key)
            if len(queue.queue) > MAX_EVENTS:
                self._remove_from_seen(queue, queue.queue.pop(0))
        _log.debug("system event enqueued [%s..]: %s", session_key[:12], cleaned[:60])
        return True

    def claim_snapshot(self, session_key: str) -> SystemEventLease | None:
        """Claim one batch; a second wake cannot overwrite the first batch."""
        with self._lock:
            if session_key in self._leases:
                raise SystemEventBusy(session_key)
            queue = self._queues.get(session_key)
            if not queue or not queue.queue:
                return None
            try:
                owner_task_id = id(current_task())
            except RuntimeError:
                owner_task_id = None
            lease = SystemEventLease(
                token=uuid.uuid4().hex,
                session_key=session_key,
                events=tuple(queue.queue),
                owner_task_id=owner_task_id,
            )
            self._leases[session_key] = lease
            return lease

    def commit_snapshot(self, lease: SystemEventLease) -> None:
        """Remove only the claimed events and retain events enqueued meanwhile."""
        with self._lock:
            if self._leases.get(lease.session_key) is not lease:
                return
            queue = self._queues.get(lease.session_key)
            if queue:
                keys = {id(event) for event in lease.events}
                queue.queue = [event for event in queue.queue if id(event) not in keys]
                queue._seen = {(event.text, event.context_key) for event in queue.queue}
                if not queue.queue:
                    self._queues.pop(lease.session_key, None)
            self._leases.pop(lease.session_key, None)
            _log.debug(
                "consumed snapshot %d events [%s..]",
                len(lease.events),
                lease.session_key[:12],
            )

    def release_snapshot(self, lease: SystemEventLease) -> None:
        """Release a failed wake without dropping its events."""
        with self._lock:
            if self._leases.get(lease.session_key) is lease:
                self._leases.pop(lease.session_key, None)

    def release_snapshot_for_session(self, session_key: str) -> None:
        with self._lock:
            lease = self._leases.get(session_key)
        if lease:
            self.release_snapshot(lease)

    def snapshot_is_claimed(self, session_key: str) -> bool:
        with self._lock:
            return session_key in self._leases

    def peek_and_snapshot(self, session_key: str) -> list[SystemEvent]:
        """兼容旧调用：领取并返回当前不可变事件批次。"""
        lease = self.claim_snapshot(session_key)
        return list(lease.events) if lease else []

    def consume_snapshot(self, session_key: str) -> None:
        """兼容旧调用：成功执行后提交当前 lease。"""
        with self._lock:
            lease = self._leases.get(session_key)
            if lease is None:
                return
            try:
                task_id = id(current_task())
            except RuntimeError:
                task_id = None
            if lease.owner_task_id is not None and lease.owner_task_id != task_id:
                return
        if lease:
            self.commit_snapshot(lease)

    def drain(self, session_key: str) -> list[SystemEvent]:
        with self._lock:
            queue = self._queues.get(session_key)
            if not queue or not queue.queue:
                return []
            out = list(queue.queue)
            queue.queue.clear()
            queue._seen.clear()
            self._queues.pop(session_key, None)
            self._leases.pop(session_key, None)
            return out

    def peek(self, session_key: str) -> list[SystemEvent]:
        with self._lock:
            queue = self._queues.get(session_key)
            return list(queue.queue) if queue else []

    def has_events(self, session_key: str) -> bool:
        with self._lock:
            queue = self._queues.get(session_key)
            return bool(queue and queue.queue)

    def drain_non_heartbeat(
        self,
        session_key: str,
        expected_events: Optional[list[SystemEvent]] = None,
    ) -> list[SystemEvent]:
        """Drain ambient events observed by a user turn.

        When ``expected_events`` is supplied, events enqueued after the turn
        snapshot remain queued for a later wake.
        """
        with self._lock:
            queue = self._queues.get(session_key)
            if not queue:
                return []
            if session_key in self._leases:
                return []
            expected_ids = (
                {id(event) for event in expected_events}
                if expected_events is not None
                else None
            )
            removed = [
                event
                for event in queue.queue
                if not event.heartbeat_only
                and (expected_ids is None or id(event) in expected_ids)
            ]
            removed_ids = {id(event) for event in removed}
            kept = [event for event in queue.queue if id(event) not in removed_ids]
            queue.queue = kept
            queue._seen = {(event.text, event.context_key) for event in kept}
            if not queue.queue:
                self._queues.pop(session_key, None)
            return removed

    def peek_non_heartbeat(self, session_key: str) -> list[SystemEvent]:
        with self._lock:
            queue = self._queues.get(session_key)
            return (
                [event for event in queue.queue if not event.heartbeat_only]
                if queue
                else []
            )

    def clear(self, session_key: str) -> None:
        with self._lock:
            self._queues.pop(session_key, None)
            self._leases.pop(session_key, None)

    def clear_all(self) -> None:
        with self._lock:
            self._queues.clear()
            self._leases.clear()

    @staticmethod
    def _sanitize(text: str) -> str:
        return _SYSTEM_PREFIX_RE.sub("", text).strip()

    def _get_or_create(self, session_key: str) -> _SessionQueue:
        return self._queues.setdefault(session_key, _SessionQueue())

    def _replace_in_queue(
        self,
        queue: _SessionQueue,
        text: str,
        context_key: str,
        heartbeat_only: bool,
    ) -> bool:
        for index, event in enumerate(queue.queue):
            if event.context_key != context_key:
                continue
            if event.text == text:
                return False
            queue._seen.discard((event.text, event.context_key))
            queue.queue[index] = SystemEvent(
                text=text,
                ts=time.time(),
                context_key=context_key,
                heartbeat_only=heartbeat_only,
            )
            queue._seen.add((text, context_key))
            return True
        return self._append_replacement(queue, text, context_key, heartbeat_only)

    @staticmethod
    def _append_replacement(
        queue: _SessionQueue,
        text: str,
        context_key: str,
        heartbeat_only: bool,
    ) -> bool:
        key = (text, context_key)
        if key in queue._seen:
            return False
        queue.queue.append(SystemEvent(text, time.time(), context_key, heartbeat_only))
        queue._seen.add(key)
        if len(queue.queue) > MAX_EVENTS:
            removed = queue.queue.pop(0)
            queue._seen.discard((removed.text, removed.context_key))
        return True

    @staticmethod
    def _remove_from_seen(queue: _SessionQueue, event: SystemEvent) -> None:
        queue._seen.discard((event.text, event.context_key))
