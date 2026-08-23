"""Conversation-level scheduling over the session inbox primitive.

The session manager owns storage and leases. This module owns the meaning of a
piece of work returned to the consumer, including queue revision and batching.
"""

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional

from core.engine.turn_state import TurnPhase, TurnState, TurnStateError
from core.managers.session_manager import (
    InboundIntent,
    InboxEnqueueResult,
    InboxLease,
    PendingInbound,
    SessionTaskManager,
)
from core.tasks.task_state_store import TaskStateStore


@dataclass(frozen=True)
class ScheduledWork:
    """Leased same-intent items with one immutable scheduling snapshot."""

    chat_id: str
    owner_token: int
    queue_revision: int
    leases: tuple[InboxLease[PendingInbound], ...]
    passive_admission_only: bool = False

    @property
    def lease(self) -> InboxLease[PendingInbound]:
        """Compatibility view for the first lease in a scheduled batch."""
        return self.leases[0]

    @property
    def items(self) -> tuple[PendingInbound, ...]:
        return tuple(item for lease in self.leases for item in lease.items)

    @property
    def pending(self) -> PendingInbound:
        return self.items[0]

    @property
    def intent(self) -> InboundIntent:
        return self.pending.intent


class StaleScheduledWork(RuntimeError):
    """Raised when a work item is used by a consumer that no longer owns it."""


class ConversationScheduler:
    """Own conversation work selection while delegating FIFO storage to the inbox."""

    def __init__(
        self,
        session_manager: SessionTaskManager,
        *,
        collect_idle_ms: int = 0,
        collect_max_wait_ms: int = 0,
        collect_max_messages: int = 8,
        collect_max_chars: int = 6000,
        ambient_collect_idle_ms: int = 0,
        direct_task_collaboration_enabled: bool = False,
        user_role: Optional[Callable[[str], str]] = None,
        role_at_least: Optional[Callable[[str, str], bool]] = None,
        task_state_store: Optional[TaskStateStore] = None,
    ):
        self.session_manager = session_manager
        self._collect_idle = max(0, collect_idle_ms) / 1000
        self._collect_max_wait = max(0, collect_max_wait_ms) / 1000
        self._collect_max_messages = max(1, collect_max_messages)
        self._collect_max_chars = max(1, collect_max_chars)
        self._ambient_collect_idle = max(0, ambient_collect_idle_ms) / 1000
        self._direct_task_collaboration_enabled = direct_task_collaboration_enabled
        self._user_role = user_role
        self._role_at_least = role_at_least
        self._task_state_store = task_state_store
        self._queue_revisions: dict[str, int] = {}
        self._turn_states: dict[str, TurnState] = {}
        self._selection_locks: dict[str, asyncio.Lock] = {}
        self._selection_locks_guard = asyncio.Lock()
        self._revision_condition = asyncio.Condition()

    async def _selection_lock(self, chat_id: str) -> asyncio.Lock:
        async with self._selection_locks_guard:
            return self._selection_locks.setdefault(chat_id, asyncio.Lock())

    def _revision(self, chat_id: str) -> int:
        return self._queue_revisions.get(chat_id, 0)

    def _bump(self, chat_id: str) -> int:
        revision = self._revision(chat_id) + 1
        self._queue_revisions[chat_id] = revision
        return revision

    async def _cancel_active_ambient_for_direct_task(self, chat_id: str) -> None:
        """Cancel an ambient turn before a newly admitted direct task can run.

        Provider and tool calls already in flight are allowed to return, but their
        captured cancellation generation will fail the scheduler execution gate.
        """
        for turn_id, state in tuple(self._turn_states.items()):
            if (
                state.chat_id != chat_id
                or state.intent is not InboundIntent.GROUP_AMBIENT
                or state.phase is not TurnPhase.ACTIVE
            ):
                continue
            cancelled = state.transition(
                TurnPhase.CANCELLED, expected_revision=state.revision
            )
            self._turn_states[turn_id] = cancelled
            if self._task_state_store is not None:
                await self._task_state_store.put(cancelled)

    async def enqueue(
        self, chat_id: str, pending: PendingInbound
    ) -> InboxEnqueueResult[PendingInbound]:
        """Append an item and wake quiet collectors after advancing revision."""
        async with self._revision_condition:
            result = await self.session_manager.enqueue_and_claim_consumer(
                chat_id, pending
            )
            if result.accepted:
                self._bump(chat_id)
                if pending.intent is InboundIntent.DIRECT_TASK:
                    await self._cancel_active_ambient_for_direct_task(chat_id)
                self._revision_condition.notify_all()
        return result

    async def _wait_for_queue_change(
        self, chat_id: str, revision: int, timeout: float
    ) -> bool:
        """Wait for a new accepted inbox item without losing an enqueue race."""
        if timeout <= 0:
            return False
        async with self._revision_condition:
            if self._revision(chat_id) != revision:
                return True
            try:
                await asyncio.wait_for(
                    self._revision_condition.wait_for(
                        lambda: self._revision(chat_id) != revision
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return False
            return True

    async def _collect_followups(
        self,
        chat_id: str,
        *,
        owner_token: int,
        intent: InboundIntent,
        first: PendingInbound,
    ) -> tuple[tuple[InboxLease[PendingInbound], ...], bool]:
        """Collect a stable same-intent prefix or passive-admit an ambient prefix.

        A direct task arriving while an ambient batch is settling wins priority.
        Earlier ambient messages are still returned first so the consumer can admit
        them in FIFO order without running the ambient engagement gate.
        """
        collect_idle = (
            self._ambient_collect_idle
            if intent is InboundIntent.GROUP_AMBIENT
            else self._collect_idle
        )
        if not collect_idle:
            return (), False

        deadline = asyncio.get_running_loop().time() + (
            self._collect_max_wait or collect_idle
        )
        revision = self._revision(chat_id)
        while True:
            if not self.session_manager.is_consumer_owner(chat_id, owner_token):
                return (), False
            if (
                intent is InboundIntent.GROUP_AMBIENT
                and await self.session_manager.has_pending_intent(
                    chat_id, InboundIntent.DIRECT_TASK
                )
            ):
                async with self._revision_condition:
                    if not self.session_manager.is_consumer_owner(chat_id, owner_token):
                        return (), False
                    followup = await self.session_manager.claim_pending_for_steer(
                        chat_id,
                        intent=intent,
                        max_items=self._collect_max_messages - 1,
                        max_chars=self._collect_max_chars - len(first.prepared_content),
                        skip_head=True,
                    )
                return ((followup,) if followup is not None else ()), True

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            changed = await self._wait_for_queue_change(
                chat_id, revision, min(collect_idle, remaining)
            )
            if not changed:
                break
            revision = self._revision(chat_id)
            next_pending = await self.session_manager.peek_pending_for_consumer(
                chat_id, owner_token, offset=1
            )
            if next_pending is not None and next_pending.intent is not intent:
                break

        remaining_chars = self._collect_max_chars - len(first.prepared_content)
        async with self._revision_condition:
            if not self.session_manager.is_consumer_owner(chat_id, owner_token):
                return (), False
            followup = await self.session_manager.claim_pending_for_steer(
                chat_id,
                intent=intent,
                max_items=self._collect_max_messages - 1,
                max_chars=remaining_chars,
                skip_head=True,
            )
        return ((followup,) if followup is not None else ()), False

    async def next_work(
        self, chat_id: str, *, owner_token: int
    ) -> Optional[ScheduledWork]:
        """Return the next FIFO work after a lease-free quiet reservation."""
        selection_lock = await self._selection_lock(chat_id)
        async with selection_lock:
            first = await self.session_manager.peek_next_for_consumer(
                chat_id, owner_token
            )
            if first is None:
                return None

            intent = first.intent
            followup_leases, passive_admission_only = await self._collect_followups(
                chat_id,
                owner_token=owner_token,
                intent=intent,
                first=first,
            )
            async with self._revision_condition:
                lease = await self.session_manager.claim_next_for_consumer(
                    chat_id, owner_token
                )
                if lease is None:
                    return None
                if lease.items[0] is not first:
                    raise StaleScheduledWork(
                        f"FIFO head changed during reservation: {chat_id}"
                    )
                if (
                    intent is InboundIntent.GROUP_AMBIENT
                    and await self.session_manager.has_pending_intent(
                        chat_id, InboundIntent.DIRECT_TASK
                    )
                ):
                    passive_admission_only = True
            return ScheduledWork(
                chat_id=chat_id,
                owner_token=owner_token,
                queue_revision=self._revision(chat_id),
                leases=(lease, *followup_leases),
                passive_admission_only=passive_admission_only,
            )

    async def validate_owner(self, work: ScheduledWork) -> None:
        """Fail closed if a replacement consumer tries to use old work."""
        async with self._revision_condition:
            self._validate_owner_locked(work)

    def _validate_owner_locked(self, work: ScheduledWork) -> None:
        """Validate ownership while the scheduler lifecycle lock is held."""
        if not self.session_manager.is_consumer_owner(work.chat_id, work.owner_token):
            raise StaleScheduledWork(
                f"consumer owner changed: {work.chat_id}:{work.owner_token}"
            )

    async def commit(self, work: ScheduledWork, item: PendingInbound) -> None:
        async with self._revision_condition:
            self._validate_owner_locked(work)
            for lease in work.leases:
                if any(existing is item for existing in lease.items):
                    await self.session_manager.commit(lease, item)
                    return
            raise RuntimeError("item does not belong to scheduled work")

    async def fail(self, work: ScheduledWork, item: PendingInbound) -> None:
        async with self._revision_condition:
            self._validate_owner_locked(work)
            for lease in work.leases:
                if any(existing is item for existing in lease.items):
                    await self.session_manager.fail(lease, item)
                    return
            raise RuntimeError("item does not belong to scheduled work")

    async def requeue_front(self, work: ScheduledWork) -> int:
        async with self._revision_condition:
            self._validate_owner_locked(work)
            count = 0
            for lease in reversed(work.leases):
                count += await self.session_manager.requeue_front(lease)
            if count:
                self._bump(work.chat_id)
                self._revision_condition.notify_all()
            return count

    async def release_consumer_if_idle(self, chat_id: str, owner_token: int) -> bool:
        async with self._revision_condition:
            released = await self.session_manager.release_consumer_if_idle(
                chat_id, owner_token
            )
            if released:
                self._bump(chat_id)
                self._revision_condition.notify_all()
            return released

    async def handoff_consumer(self, chat_id: str, owner_token: int) -> Optional[int]:
        async with self._revision_condition:
            replacement = await self.session_manager.handoff_consumer(
                chat_id, owner_token
            )
            self._bump(chat_id)
            self._revision_condition.notify_all()
            return replacement

    async def start_turn(
        self,
        work: ScheduledWork,
        *,
        turn_id: str,
        principal_id: str,
    ) -> TurnState:
        """Register one active turn from a leased work snapshot exactly once."""
        if not turn_id:
            raise ValueError("turn_id is required")
        async with self._revision_condition:
            self._validate_owner_locked(work)
            if turn_id in self._turn_states:
                raise TurnStateError(f"turn already exists: {turn_id}")
            state = TurnState(
                turn_id=turn_id,
                chat_id=work.chat_id,
                intent=work.intent,
                principal_id=principal_id,
                principal_role=(
                    self._user_role(principal_id) if self._user_role is not None else ""
                ),
                queue_revision=work.queue_revision,
                task_anchor_message_id=turn_id,
                task_correlation_id=work.pending.message.task_correlation_id,
            )
            if (
                work.intent is InboundIntent.GROUP_AMBIENT
                and await self.session_manager.has_pending_intent(
                    work.chat_id, InboundIntent.DIRECT_TASK
                )
            ):
                state = state.transition(
                    TurnPhase.CANCELLED, expected_revision=state.revision
                )
            self._turn_states[turn_id] = state
            if self._task_state_store is not None:
                await self._task_state_store.put(state)
            return state

    async def get_turn(self, turn_id: str) -> Optional[TurnState]:
        async with self._revision_condition:
            return self._turn_states.get(turn_id)

    async def claim_steer(self, turn_id: str) -> Optional[InboxLease[PendingInbound]]:
        """Lease same-principal or explicitly authorized collaborative follow-ups."""
        async with self._revision_condition:
            state = self._turn_states.get(turn_id)
            if (
                state is None
                or state.phase is not TurnPhase.ACTIVE
                or not self._principal_role_is_current(state)
            ):
                return None

            def allows(candidate: PendingInbound) -> bool:
                sender_id = candidate.message.sender_id
                if sender_id == state.principal_id:
                    return True
                anchor_matches = (
                    candidate.message.replied_message_id == state.task_anchor_message_id
                )
                correlation_matches = (
                    bool(state.task_correlation_id)
                    and candidate.message.task_correlation_id
                    == state.task_correlation_id
                )
                if (
                    not self._direct_task_collaboration_enabled
                    or state.intent is not InboundIntent.DIRECT_TASK
                    or candidate.intent is not InboundIntent.DIRECT_TASK
                    or not (anchor_matches or correlation_matches)
                    or (
                        candidate.message.replied_author_id
                        and candidate.message.replied_author_id != state.principal_id
                    )
                    or self._user_role is None
                    or self._role_at_least is None
                ):
                    return False
                return self._role_at_least(
                    self._user_role(sender_id), self._user_role(state.principal_id)
                )

            return await self.session_manager.claim_pending_for_steer(
                state.chat_id, intent=state.intent, candidate_filter=allows
            )

    async def allows_model_context_inheritance(
        self,
        turn_id: str,
        *,
        chat_id: str,
        principal_id: str,
        task_correlation_id: str,
        reply_to: str,
        capabilities,
    ) -> bool:
        """Authorize a direct-task model context while its turn is active."""
        async with self._revision_condition:
            state = self._turn_states.get(turn_id)
            if (
                state is None
                or state.phase is not TurnPhase.ACTIVE
                or state.intent is not InboundIntent.DIRECT_TASK
                or state.chat_id != chat_id
                or state.principal_id != principal_id
                or state.task_correlation_id != task_correlation_id
                or not task_correlation_id
                or not state.principal_role
                or self._user_role is None
                or self._role_at_least is None
                or not self._principal_role_is_current(state)
            ):
                return False
            if capabilities is None:
                return False
            return (
                capabilities.intent is InboundIntent.DIRECT_TASK
                and capabilities.chat_id == chat_id
                and capabilities.sender_id == principal_id
                and capabilities.reply_to == reply_to
            )

    async def commit_steer(
        self,
        turn_id: str,
        lease: InboxLease[PendingInbound],
        item: PendingInbound,
    ) -> None:
        """Commit a steer only while its owning turn remains active."""
        async with self._revision_condition:
            state = self._turn_states.get(turn_id)
            if (
                state is None
                or state.phase is not TurnPhase.ACTIVE
                or not self._principal_role_is_current(state)
            ):
                raise StaleScheduledWork(f"steer turn is no longer active: {turn_id}")
            await self.session_manager.commit(lease, item)

    async def transition_turn(
        self,
        turn_id: str,
        *,
        expected_revision: int,
        phase: TurnPhase,
        approval_plan_id: str | None = None,
    ) -> TurnState:
        """Apply one compare-and-swap lifecycle transition owned by the scheduler."""
        async with self._revision_condition:
            current = self._turn_states.get(turn_id)
            if current is None:
                raise TurnStateError(f"unknown turn: {turn_id}")
            updated = current.transition(
                phase,
                expected_revision=expected_revision,
                approval_plan_id=approval_plan_id,
            )
            self._turn_states[turn_id] = updated
            if self._task_state_store is not None:
                await self._task_state_store.put(updated)
            return updated

    async def is_turn_active(self, turn_id: str) -> bool:
        """Return whether a turn may begin another externally visible boundary."""
        async with self._revision_condition:
            state = self._turn_states.get(turn_id)
            return state is not None and state.phase is TurnPhase.ACTIVE

    def _principal_role_is_current(self, state: TurnState) -> bool:
        if (
            not state.principal_role
            or self._user_role is None
            or self._role_at_least is None
        ):
            return True
        return self._role_at_least(
            self._user_role(state.principal_id), state.principal_role
        )

    async def is_turn_execution_allowed(
        self, turn_id: str, cancellation_generation: int
    ) -> bool:
        """Validate the captured cancellation generation at a side-effect boundary."""
        async with self._revision_condition:
            state = self._turn_states.get(turn_id)
            return (
                state is not None
                and state.phase is TurnPhase.ACTIVE
                and state.cancellation_generation == cancellation_generation
                and self._principal_role_is_current(state)
            )

    async def is_turn_delivery_allowed(
        self, turn_id: str, cancellation_generation: int
    ) -> bool:
        """Allow the already-classified final delivery during FINALIZING only."""
        async with self._revision_condition:
            state = self._turn_states.get(turn_id)
            return (
                state is not None
                and state.phase is TurnPhase.FINALIZING
                and state.cancellation_generation == cancellation_generation
                and self._principal_role_is_current(state)
            )

    async def drop_turn(self, turn_id: str) -> None:
        """Remove a terminal turn after its durable trace is committed."""
        async with self._revision_condition:
            state = self._turn_states.get(turn_id)
            if state is None:
                return
            if state.phase not in {TurnPhase.COMPLETED, TurnPhase.CANCELLED}:
                raise TurnStateError(f"cannot drop nonterminal turn: {state.phase}")
            self._turn_states.pop(turn_id, None)

    def revision(self, chat_id: str) -> int:
        """Return the latest revision without exposing inbox internals."""
        return self._revision(chat_id)
