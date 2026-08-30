"""Session inboxes and turn locks for per-chat serial processing."""

import asyncio
import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Deque, Dict, Generic, Literal, Optional, Set, TypeVar

from core.message import InputMessage

_log = logging.getLogger(__name__)

T = TypeVar("T")
MessageState = Literal["accepted", "admitted", "dropped", "failed"]


class AdmissionOrigin(StrEnum):
    """Trusted source classification for admission side effects."""

    USER_MESSAGE = "user_message"
    INTERNAL_CONTROL = "internal_control"


class InboundIntent(StrEnum):
    """Immutable routing intent assigned before an item enters a session inbox."""

    DIRECT_TASK = "direct_task"
    PRIVATE_CONVERSATION = "private_conversation"
    GROUP_AMBIENT = "group_ambient"


@dataclass(frozen=True)
class ModeRoutingMetadata:
    """Frozen capability-route metadata; it does not replace ``InboundIntent``."""

    mode: Literal["chat", "agent"]
    capability_profile: str
    reason_code: str
    policy_version: str
    scheduler_revision: int
    work_plan_hint: str | None = None


@dataclass(frozen=True)
class PendingInbound:
    """Immutable inbound envelope kept out of shared chat history until admission."""

    message: InputMessage
    prepared_content: str
    intent: InboundIntent
    origin: AdmissionOrigin
    enqueued_at: float = field(default_factory=time.time)
    resource_refs: tuple[str, ...] = ()
    mode_routing: ModeRoutingMetadata | None = None

    def __post_init__(self) -> None:
        # Read compatibility for pre-intent inbox rows and old test fixtures.
        if isinstance(self.intent, str):
            legacy = {
                "agent": InboundIntent.PRIVATE_CONVERSATION,
                "passive": InboundIntent.GROUP_AMBIENT,
            }
            normalized = legacy.get(self.intent)
            if normalized is None:
                normalized = InboundIntent(self.intent)
            object.__setattr__(self, "intent", normalized)

    @property
    def dispatch_mode(self) -> Literal["agent", "passive"]:
        """Deprecated compatibility view; scheduling uses ``intent``."""
        return "passive" if self.intent is InboundIntent.GROUP_AMBIENT else "agent"


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
            dropped = next(
                (item for item in inbox if item.intent is InboundIntent.GROUP_AMBIENT),
                None,
            )
            if dropped is not None:
                inbox.remove(dropped)
            elif pending.intent is InboundIntent.GROUP_AMBIENT:
                self._set_state(chat_id, pending.message.id, "dropped")
                return InboxEnqueueResult(False, False, dropped=pending)
            else:
                # Do not silently discard explicit or private user work. A future
                # durable inbox may turn this into backpressure/retry handling.
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
        """Deprecated wrapper; it never rewrites the immutable inbound intent."""
        return await self.enqueue_and_claim_consumer(chat_id, pending)

    async def is_steering_active(self, chat_id: str) -> bool:
        """Deprecated compatibility query for callers migrating to turn state."""
        async with self._lock:
            return bool(self._leased_counts.get(chat_id))

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
            self._reserve_lease_items(chat_id, 1)
            return InboxLease(chat_id=chat_id, items=(pending,))

    async def peek_next_for_consumer(
        self, chat_id: str, consumer_token: int
    ) -> Optional[PendingInbound]:
        """Read the FIFO head without reserving it for a consumer."""
        return await self.peek_pending_for_consumer(chat_id, consumer_token, offset=0)

    async def peek_pending_for_consumer(
        self, chat_id: str, consumer_token: int, *, offset: int = 0
    ) -> Optional[PendingInbound]:
        """Read an unleased FIFO item without changing its reservation state."""
        if offset < 0:
            raise ValueError("offset must be non-negative")
        async with self._lock:
            if self._running.get(chat_id) != consumer_token:
                return None
            inbox = self._inboxes.get(chat_id)
            if not inbox or offset >= len(inbox):
                return None
            return inbox[offset]

    async def claim_pending_for_steer(
        self,
        chat_id: str,
        *,
        intent: Optional[InboundIntent] = None,
        principal_id: Optional[str] = None,
        candidate_filter: Optional[Callable[[PendingInbound], bool]] = None,
        max_items: Optional[int] = None,
        max_chars: Optional[int] = None,
        skip_head: bool = False,
    ) -> Optional[InboxLease[PendingInbound]]:
        """Claim a contiguous same-intent prefix, optionally after the FIFO head."""
        async with self._lock:
            inbox = self._inboxes.get(chat_id)
            if not inbox or chat_id not in self._running:
                return None
            start_index = 1 if skip_head else 0
            if len(inbox) <= start_index:
                return None
            candidate = inbox[start_index]
            if intent is None:
                intent = candidate.intent
            if candidate.intent is not intent:
                return None
            if principal_id and candidate.message.sender_id != principal_id:
                return None
            if candidate_filter is not None and not candidate_filter(candidate):
                return None
            items: list[PendingInbound] = []
            total_chars = 0
            while len(inbox) > start_index and inbox[start_index].intent is intent:
                candidate = inbox[start_index]
                if principal_id and candidate.message.sender_id != principal_id:
                    break
                if candidate_filter is not None and not candidate_filter(candidate):
                    break
                if max_items is not None and len(items) >= max_items:
                    break
                candidate_chars = len(candidate.prepared_content)
                if (
                    max_chars is not None
                    and items
                    and total_chars + candidate_chars > max_chars
                ):
                    break
                if skip_head:
                    inbox.rotate(-1)
                    items.append(inbox.popleft())
                    inbox.rotate(1)
                else:
                    items.append(inbox.popleft())
                total_chars += candidate_chars
            if not items:
                return None
            self._reserve_lease_items(chat_id, len(items))
            return InboxLease(chat_id=chat_id, items=tuple(items))

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
            return True

    async def has_pending_intent(self, chat_id: str, intent: InboundIntent) -> bool:
        """Return whether an unleased inbox item has the requested intent."""
        async with self._lock:
            return any(
                pending.intent is intent for pending in self._inboxes.get(chat_id, ())
            )

    async def claim_existing_consumers(
        self, chat_ids: Set[str]
    ) -> list[tuple[str, int]]:
        """Claim consumers for preserved inboxes after an engine restart."""
        async with self._lock:
            claims: list[tuple[str, int]] = []
            for chat_id in sorted(chat_ids):
                if chat_id in self._running or not self._inboxes.get(chat_id):
                    continue
                self._next_consumer_token += 1
                token = self._next_consumer_token
                self._running[chat_id] = token
                claims.append((chat_id, token))
            return claims

    def get_queue_sizes(self) -> Dict[str, int]:
        return {
            chat_id: len(inbox) for chat_id, inbox in self._inboxes.items() if inbox
        }

    def has_active_consumer(self, chat_id: str) -> bool:
        return chat_id in self._running

    def is_consumer_owner(self, chat_id: str, consumer_token: int) -> bool:
        """Return whether a token still owns the session consumer slot."""
        return self._running.get(chat_id) == consumer_token

    async def cleanup_session(self, chat_id: str) -> None:
        async with self._lock:
            self._inboxes.pop(chat_id, None)
            self._leased_counts.pop(chat_id, None)
            self._locks.pop(chat_id, None)
            self._running.pop(chat_id, None)

    async def cleanup_all(self, *, preserve_inboxes: bool = False) -> None:
        async with self._lock:
            if not preserve_inboxes:
                self._inboxes.clear()
            self._leased_counts.clear()
            self._locks.clear()
            self._running.clear()
