"""Session inboxes and turn locks for per-chat serial processing."""

import asyncio
import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Generic, Literal, Optional, Set, TypeVar

from core.message import InputMessage

_log = logging.getLogger(__name__)

T = TypeVar("T")
MessageState = Literal["accepted", "admitted", "dropped", "failed"]


@dataclass(frozen=True)
class PendingInbound:
    """Immutable inbound envelope kept out of shared chat history until admission."""

    message: InputMessage
    prepared_content: str
    dispatch_mode: Literal["agent", "passive"]
    enqueued_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class InboxEnqueueResult(Generic[T]):
    accepted: bool
    should_start_consumer: bool
    consumer_token: Optional[int] = None
    dropped: Optional[T] = None


@dataclass
class InboxLease(Generic[T]):
    """A FIFO batch temporarily owned by one consumer or steering drain."""

    chat_id: str
    items: tuple[T, ...]
    _committed: int = 0
    _active: bool = True

    @property
    def uncommitted(self) -> tuple[T, ...]:
        return self.items[self._committed :]


class SessionTaskManager:
    """Manage FIFO inbox leases, consumers, and turn locks per chat session."""

    def __init__(self, max_inbox_size: int = 256):
        self._inboxes: Dict[str, Deque[PendingInbound]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._running: Dict[str, int] = {}
        self._next_consumer_token = 0
        self._steering_active: Set[str] = set()
        self._leased_counts: Dict[str, int] = {}
        self._states: OrderedDict[tuple[str, str], MessageState] = OrderedDict()
        self._max_inbox_size = max_inbox_size
        self._lock = asyncio.Lock()

    async def get_lock(self, chat_id: str) -> asyncio.Lock:
        async with self._lock:
            return self._locks.setdefault(chat_id, asyncio.Lock())

    def _set_state(self, chat_id: str, message_id: str, state: MessageState) -> None:
        key = (chat_id, message_id)
        self._states[key] = state
        self._states.move_to_end(key)
        while len(self._states) > self._max_inbox_size * 4:
            self._states.popitem(last=False)

    def _enqueue_locked(
        self, chat_id: str, pending: PendingInbound
    ) -> InboxEnqueueResult[PendingInbound]:
        inbox = self._inboxes.setdefault(chat_id, deque())
        leased_count = self._leased_counts.get(chat_id, 0)
        if len(inbox) + leased_count >= self._max_inbox_size:
            if inbox:
                dropped = inbox.popleft()
            else:
                self._set_state(chat_id, pending.message.id, "dropped")
                return InboxEnqueueResult(False, False, dropped=pending)
        else:
            dropped = None
        inbox.append(pending)
        self._set_state(chat_id, pending.message.id, "accepted")
        if dropped is not None:
            self._set_state(chat_id, dropped.message.id, "dropped")
        should_start = chat_id not in self._running
        consumer_token = None
        if should_start:
            self._next_consumer_token += 1
            consumer_token = self._next_consumer_token
            self._running[chat_id] = consumer_token
        return InboxEnqueueResult(True, should_start, consumer_token, dropped)

    async def enqueue_and_claim_consumer(
        self, chat_id: str, pending: PendingInbound
    ) -> InboxEnqueueResult[PendingInbound]:
        """Enqueue one message and atomically decide whether a consumer must start."""
        async with self._lock:
            return self._enqueue_locked(chat_id, pending)

    async def enqueue_with_dispatch_mode(
        self, chat_id: str, pending: PendingInbound, *, triggers_ai: bool
    ) -> InboxEnqueueResult[PendingInbound]:
        """Classify and enqueue inbound work atomically without pre-activating turns."""
        async with self._lock:
            dispatch_mode: Literal["agent", "passive"] = (
                "agent"
                if triggers_ai or chat_id in self._steering_active
                else "passive"
            )
            if pending.dispatch_mode != dispatch_mode:
                pending = PendingInbound(
                    pending.message,
                    pending.prepared_content,
                    dispatch_mode,
                    pending.enqueued_at,
                )
            return self._enqueue_locked(chat_id, pending)

    async def is_steering_active(self, chat_id: str) -> bool:
        async with self._lock:
            return chat_id in self._steering_active

    def _reserve_lease_items(self, chat_id: str, count: int) -> None:
        self._leased_counts[chat_id] = self._leased_counts.get(chat_id, 0) + count

    def _release_lease_items(self, chat_id: str, count: int) -> None:
        remaining = self._leased_counts.get(chat_id, 0) - count
        if remaining > 0:
            self._leased_counts[chat_id] = remaining
        else:
            self._leased_counts.pop(chat_id, None)

    async def claim_next_for_consumer(
        self, chat_id: str, consumer_token: int
    ) -> Optional[InboxLease[PendingInbound]]:
        async with self._lock:
            if self._running.get(chat_id) != consumer_token:
                return None
            inbox = self._inboxes.get(chat_id)
            if not inbox:
                return None
            pending = inbox.popleft()
            if pending.dispatch_mode == "agent":
                self._steering_active.add(chat_id)
            self._reserve_lease_items(chat_id, 1)
            return InboxLease(chat_id=chat_id, items=(pending,))

    async def claim_pending_for_steer(
        self, chat_id: str
    ) -> Optional[InboxLease[PendingInbound]]:
        """Claim pending messages only while this session has an active consumer."""
        async with self._lock:
            inbox = self._inboxes.get(chat_id)
            if not inbox or chat_id not in self._running:
                return None
            items = tuple(inbox)
            inbox.clear()
            self._reserve_lease_items(chat_id, len(items))
            return InboxLease(chat_id=chat_id, items=items)

    async def commit(self, lease: InboxLease[T], item: T) -> None:
        """Commit the next leased item after its local history admission succeeds."""
        async with self._lock:
            if not lease._active:
                raise RuntimeError("cannot commit an inactive inbox lease")
            if (
                lease._committed >= len(lease.items)
                or lease.items[lease._committed] is not item
            ):
                raise RuntimeError("inbox lease commit is out of order")
            lease._committed += 1
            self._release_lease_items(lease.chat_id, 1)
            self._set_state(lease.chat_id, item.message.id, "admitted")
            if lease._committed == len(lease.items):
                lease._active = False

    async def fail(
        self, lease: InboxLease[PendingInbound], item: PendingInbound
    ) -> None:
        """Commit a permanently failed item without leaving an orphaned inbox entry."""
        async with self._lock:
            if not lease._active:
                raise RuntimeError("cannot fail an inactive inbox lease")
            if (
                lease._committed >= len(lease.items)
                or lease.items[lease._committed] is not item
            ):
                raise RuntimeError("inbox lease failure is out of order")
            self._set_state(lease.chat_id, item.message.id, "failed")
            lease._committed += 1
            self._release_lease_items(lease.chat_id, 1)
            if lease._committed == len(lease.items):
                lease._active = False

    def get_message_state(
        self, chat_id: str, message_id: str
    ) -> Optional[MessageState]:
        return self._states.get((chat_id, message_id))

    async def requeue_front(self, lease: InboxLease[T]) -> int:
        """Restore only the uncommitted tail, preserving its original FIFO order."""
        async with self._lock:
            if not lease._active:
                return 0
            remaining = lease.uncommitted
            inbox = self._inboxes.setdefault(lease.chat_id, deque())
            for item in reversed(remaining):
                inbox.appendleft(item)
            self._release_lease_items(lease.chat_id, len(remaining))
            lease._active = False
            return len(remaining)

    async def handoff_consumer(
        self, chat_id: str, consumer_token: int
    ) -> Optional[int]:
        """Release this owner and return a replacement token if inbox work remains."""
        async with self._lock:
            if self._running.get(chat_id) != consumer_token:
                return None
            inbox = self._inboxes.get(chat_id)
            self._running.pop(chat_id, None)
            if not inbox:
                self._steering_active.discard(chat_id)
                return None
            self._next_consumer_token += 1
            replacement_token = self._next_consumer_token
            self._running[chat_id] = replacement_token
            return replacement_token

    async def release_consumer_if_idle(self, chat_id: str, consumer_token: int) -> bool:
        """Release this consumer only when its inbox is empty."""
        async with self._lock:
            if self._running.get(chat_id) != consumer_token:
                return True
            inbox = self._inboxes.get(chat_id)
            if inbox:
                return False
            self._running.pop(chat_id, None)
            self._steering_active.discard(chat_id)
            return True

    def get_queue_sizes(self) -> Dict[str, int]:
        return {
            chat_id: len(inbox) for chat_id, inbox in self._inboxes.items() if inbox
        }

    def has_active_consumer(self, chat_id: str) -> bool:
        return chat_id in self._running

    async def cleanup_session(self, chat_id: str) -> None:
        async with self._lock:
            self._steering_active.discard(chat_id)
            self._inboxes.pop(chat_id, None)
            self._leased_counts.pop(chat_id, None)
            self._locks.pop(chat_id, None)
            self._running.pop(chat_id, None)

    async def cleanup_all(self) -> None:
        async with self._lock:
            self._steering_active.clear()
            self._inboxes.clear()
            self._leased_counts.clear()
            self._locks.clear()
            self._running.clear()
