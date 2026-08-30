import asyncio

import pytest

from core.engine.conversation_scheduler import ConversationScheduler, StaleScheduledWork
from core.engine.turn_state import TurnPhase, TurnStateError
from core.managers.session_manager import (
    AdmissionOrigin,
    InboundIntent,
    PendingInbound,
    SessionTaskManager,
)
from core.message import InputMessage


def _pending() -> PendingInbound:
    return PendingInbound(
        InputMessage("message", "principal", "chat", "request", True),
        "request",
        InboundIntent.DIRECT_TASK,
        AdmissionOrigin.USER_MESSAGE,
    )


def _ambient_pending() -> PendingInbound:
    return PendingInbound(
        InputMessage("ambient", "speaker", "chat", "ambient-1", True),
        "ambient",
        InboundIntent.GROUP_AMBIENT,
        AdmissionOrigin.USER_MESSAGE,
    )


@pytest.mark.asyncio
async def test_scheduler_turn_state_freezes_identity_and_approval_plan():
    scheduler = ConversationScheduler(SessionTaskManager())
    enqueued = await scheduler.enqueue("chat", _pending())
    work = await scheduler.next_work("chat", owner_token=enqueued.consumer_token)
    assert work is not None

    active = await scheduler.start_turn(
        work, turn_id="turn-1", principal_id="principal"
    )
    assert await scheduler.is_turn_active("turn-1")
    awaiting = await scheduler.transition_turn(
        "turn-1",
        expected_revision=active.revision,
        phase=TurnPhase.AWAITING_APPROVAL,
        approval_plan_id="plan-1",
    )
    assert not await scheduler.is_turn_active("turn-1")
    resumed = await scheduler.transition_turn(
        "turn-1",
        expected_revision=awaiting.revision,
        phase=TurnPhase.ACTIVE,
    )

    assert active.queue_revision == work.queue_revision
    assert await scheduler.is_turn_active("turn-1")
    assert awaiting.approval_plan_id == "plan-1"
    assert resumed.approval_plan_id == "plan-1"
    assert not await scheduler.is_turn_delivery_allowed(
        resumed.turn_id, resumed.cancellation_generation
    )
    finalizing = await scheduler.transition_turn(
        resumed.turn_id,
        expected_revision=resumed.revision,
        phase=TurnPhase.FINALIZING,
    )
    assert await scheduler.is_turn_delivery_allowed(
        finalizing.turn_id, finalizing.cancellation_generation
    )
    assert not await scheduler.is_turn_execution_allowed(
        finalizing.turn_id, finalizing.cancellation_generation
    )


@pytest.mark.asyncio
async def test_scheduler_turn_waiting_is_durable_and_can_finalize():
    scheduler = ConversationScheduler(SessionTaskManager())
    enqueued = await scheduler.enqueue("chat", _pending())
    work = await scheduler.next_work("chat", owner_token=enqueued.consumer_token)
    assert work is not None
    active = await scheduler.start_turn(
        work, turn_id="turn-wait", principal_id="principal"
    )

    waiting = await scheduler.transition_turn(
        "turn-wait",
        expected_revision=active.revision,
        phase=TurnPhase.WAITING,
        wait_reason="quiet period",
        wait_deadline=123.0,
    )
    assert waiting.phase is TurnPhase.WAITING
    assert not await scheduler.is_turn_execution_allowed(
        waiting.turn_id, waiting.cancellation_generation
    )
    finalizing = await scheduler.transition_turn(
        "turn-wait", expected_revision=waiting.revision, phase=TurnPhase.FINALIZING
    )
    completed = await scheduler.transition_turn(
        "turn-wait", expected_revision=finalizing.revision, phase=TurnPhase.COMPLETED
    )
    await scheduler.drop_turn(completed.turn_id)

    session = SessionTaskManager()
    roles = {"principal": "trusted", "collaborator": "trusted"}
    scheduler = ConversationScheduler(
        session,
        direct_task_collaboration_enabled=True,
        user_role=roles.__getitem__,
        role_at_least=lambda candidate, required: candidate == required,
    )
    first = await scheduler.enqueue("chat", _pending())
    work = await scheduler.next_work("chat", owner_token=first.consumer_token)
    assert work is not None
    await scheduler.start_turn(work, turn_id="anchor", principal_id="principal")

    follow_up = PendingInbound(
        InputMessage(
            "follow up",
            "collaborator",
            "chat",
            "request",
            True,
            replied_message_id="anchor",
        ),
        "follow up",
        InboundIntent.DIRECT_TASK,
        AdmissionOrigin.USER_MESSAGE,
    )
    await scheduler.enqueue("chat", follow_up)

    lease = await scheduler.claim_steer("anchor")

    assert lease is not None
    assert lease.items == (follow_up,)


@pytest.mark.asyncio
async def test_scheduler_cross_principal_steer_accepts_matching_task_correlation():
    session = SessionTaskManager()
    roles = {"principal": "trusted", "collaborator": "trusted"}
    scheduler = ConversationScheduler(
        session,
        direct_task_collaboration_enabled=True,
        user_role=roles.__getitem__,
        role_at_least=lambda candidate, required: candidate == required,
    )
    initial = PendingInbound(
        InputMessage(
            "initial",
            "principal",
            "chat",
            "request",
            True,
            task_correlation_id="task:stable-1",
        ),
        "request",
        InboundIntent.DIRECT_TASK,
        AdmissionOrigin.USER_MESSAGE,
    )
    first = await scheduler.enqueue("chat", initial)
    work = await scheduler.next_work("chat", owner_token=first.consumer_token)
    assert work is not None
    await scheduler.start_turn(work, turn_id="anchor", principal_id="principal")

    follow_up = PendingInbound(
        InputMessage(
            "follow up",
            "collaborator",
            "chat",
            "continue",
            True,
            task_correlation_id="task:stable-1",
        ),
        "continue",
        InboundIntent.DIRECT_TASK,
        AdmissionOrigin.USER_MESSAGE,
    )
    await scheduler.enqueue("chat", follow_up)

    lease = await scheduler.claim_steer("anchor")

    assert lease is not None
    assert lease.items == (follow_up,)


@pytest.mark.asyncio
async def test_scheduler_cross_principal_steer_rejects_wrong_reply_author():
    session = SessionTaskManager()
    roles = {"principal": "trusted", "collaborator": "trusted"}
    scheduler = ConversationScheduler(
        session,
        direct_task_collaboration_enabled=True,
        user_role=roles.__getitem__,
        role_at_least=lambda candidate, required: candidate == required,
    )
    first = await scheduler.enqueue("chat", _pending())
    work = await scheduler.next_work("chat", owner_token=first.consumer_token)
    assert work is not None
    await scheduler.start_turn(work, turn_id="anchor", principal_id="principal")
    follow_up = PendingInbound(
        InputMessage(
            "follow up",
            "collaborator",
            "chat",
            "request",
            True,
            replied_message_id="anchor",
            replied_author_id="unrelated",
        ),
        "follow up",
        InboundIntent.DIRECT_TASK,
        AdmissionOrigin.USER_MESSAGE,
    )
    await scheduler.enqueue("chat", follow_up)

    assert await scheduler.claim_steer("anchor") is None


@pytest.mark.asyncio
async def test_scheduler_cross_principal_steer_rejects_missing_anchor_by_default():
    session = SessionTaskManager()
    scheduler = ConversationScheduler(session)
    first = await scheduler.enqueue("chat", _pending())
    work = await scheduler.next_work("chat", owner_token=first.consumer_token)
    assert work is not None
    await scheduler.start_turn(work, turn_id="anchor", principal_id="principal")
    follow_up = PendingInbound(
        InputMessage("follow up", "other", "chat", "request", True),
        "follow up",
        InboundIntent.DIRECT_TASK,
        AdmissionOrigin.USER_MESSAGE,
    )
    await scheduler.enqueue("chat", follow_up)

    assert await scheduler.claim_steer("anchor") is None


@pytest.mark.asyncio
async def test_scheduler_claim_steer_rejects_already_cancelled_turn():
    scheduler = ConversationScheduler(SessionTaskManager())
    first = await scheduler.enqueue("chat", _pending())
    work = await scheduler.next_work("chat", owner_token=first.consumer_token)
    assert work is not None
    active = await scheduler.start_turn(
        work, turn_id="anchor", principal_id="principal"
    )
    await scheduler.transition_turn(
        "anchor",
        expected_revision=active.revision,
        phase=TurnPhase.CANCELLED,
    )
    await scheduler.enqueue("chat", _pending())

    assert await scheduler.claim_steer("anchor") is None


@pytest.mark.asyncio
async def test_scheduler_commit_steer_rechecks_turn_after_claim():
    scheduler = ConversationScheduler(SessionTaskManager())
    first = await scheduler.enqueue("chat", _pending())
    work = await scheduler.next_work("chat", owner_token=first.consumer_token)
    assert work is not None
    active = await scheduler.start_turn(
        work, turn_id="anchor", principal_id="principal"
    )
    follow_up = _pending()
    await scheduler.enqueue("chat", follow_up)
    lease = await scheduler.claim_steer("anchor")
    assert lease is not None

    await scheduler.transition_turn(
        "anchor",
        expected_revision=active.revision,
        phase=TurnPhase.CANCELLED,
    )

    with pytest.raises(StaleScheduledWork, match="no longer active"):
        await scheduler.commit_steer("anchor", lease, follow_up)


@pytest.mark.asyncio
async def test_scheduler_turn_state_rejects_stale_and_illegal_transitions():
    scheduler = ConversationScheduler(SessionTaskManager())
    enqueued = await scheduler.enqueue("chat", _pending())
    work = await scheduler.next_work("chat", owner_token=enqueued.consumer_token)
    assert work is not None
    active = await scheduler.start_turn(
        work, turn_id="turn-1", principal_id="principal"
    )

    with pytest.raises(TurnStateError, match="stale turn revision"):
        await scheduler.transition_turn(
            "turn-1",
            expected_revision=active.revision + 1,
            phase=TurnPhase.FINALIZING,
        )


@pytest.mark.asyncio
async def test_scheduler_turn_allows_a_new_approval_after_previous_resolution():
    scheduler = ConversationScheduler(SessionTaskManager())
    enqueued = await scheduler.enqueue("chat", _pending())
    work = await scheduler.next_work("chat", owner_token=enqueued.consumer_token)
    assert work is not None
    active = await scheduler.start_turn(
        work, turn_id="turn-1", principal_id="principal"
    )

    awaiting = await scheduler.transition_turn(
        "turn-1",
        expected_revision=active.revision,
        phase=TurnPhase.AWAITING_APPROVAL,
        approval_plan_id="plan-1",
    )
    resumed = await scheduler.transition_turn(
        "turn-1",
        expected_revision=awaiting.revision,
        phase=TurnPhase.ACTIVE,
    )
    second = await scheduler.transition_turn(
        "turn-1",
        expected_revision=resumed.revision,
        phase=TurnPhase.AWAITING_APPROVAL,
        approval_plan_id="plan-2",
    )

    assert second.approval_plan_id == "plan-2"


@pytest.mark.asyncio
async def test_scheduler_execution_gate_rejects_principal_role_downgrade():
    roles = {"principal": "trusted"}
    scheduler = ConversationScheduler(
        SessionTaskManager(),
        user_role=roles.__getitem__,
        role_at_least=lambda candidate, required: candidate == required,
    )
    enqueued = await scheduler.enqueue("chat", _pending())
    work = await scheduler.next_work("chat", owner_token=enqueued.consumer_token)
    assert work is not None
    active = await scheduler.start_turn(
        work, turn_id="turn-1", principal_id="principal"
    )

    assert await scheduler.is_turn_execution_allowed(
        active.turn_id, active.cancellation_generation
    )
    roles["principal"] = "default"
    assert not await scheduler.is_turn_execution_allowed(
        active.turn_id, active.cancellation_generation
    )
    roles["principal"] = "trusted"
    assert await scheduler.is_turn_execution_allowed(
        active.turn_id, active.cancellation_generation
    )

    scheduler = ConversationScheduler(SessionTaskManager())
    enqueued = await scheduler.enqueue("chat", _pending())
    work = await scheduler.next_work("chat", owner_token=enqueued.consumer_token)
    assert work is not None
    active = await scheduler.start_turn(
        work, turn_id="turn-1", principal_id="principal"
    )

    assert await scheduler.is_turn_execution_allowed("turn-1", 0)
    cancelled = await scheduler.transition_turn(
        "turn-1",
        expected_revision=active.revision,
        phase=TurnPhase.CANCELLED,
    )

    assert cancelled.cancellation_generation == 1
    assert not await scheduler.is_turn_execution_allowed("turn-1", 0)
    assert not await scheduler.is_turn_execution_allowed("turn-1", 1)


@pytest.mark.asyncio
async def test_scheduler_steer_gate_rejects_principal_role_downgrade():
    roles = {"principal": "trusted"}
    scheduler = ConversationScheduler(
        SessionTaskManager(),
        user_role=roles.__getitem__,
        role_at_least=lambda candidate, required: candidate == required,
    )
    enqueued = await scheduler.enqueue("chat", _pending())
    work = await scheduler.next_work("chat", owner_token=enqueued.consumer_token)
    assert work is not None
    await scheduler.start_turn(work, turn_id="turn-1", principal_id="principal")

    roles["principal"] = "default"

    assert await scheduler.claim_steer("turn-1") is None


@pytest.mark.asyncio
async def test_direct_task_preempts_ambient_leased_before_turn_start():
    scheduler = ConversationScheduler(SessionTaskManager())
    ambient = await scheduler.enqueue("chat", _ambient_pending())
    work = await scheduler.next_work("chat", owner_token=ambient.consumer_token)
    assert work is not None

    direct = await scheduler.enqueue("chat", _pending())
    preempted = await scheduler.start_turn(
        work, turn_id="ambient-turn", principal_id="speaker"
    )

    assert direct.accepted
    assert preempted.phase is TurnPhase.CANCELLED
    assert preempted.cancellation_generation == 1
    assert not await scheduler.is_turn_execution_allowed(
        preempted.turn_id, preempted.cancellation_generation
    )


@pytest.mark.asyncio
async def test_direct_task_cancels_active_ambient_turn_at_next_safety_boundary():
    scheduler = ConversationScheduler(SessionTaskManager())
    ambient = await scheduler.enqueue("chat", _ambient_pending())
    work = await scheduler.next_work("chat", owner_token=ambient.consumer_token)
    assert work is not None
    active = await scheduler.start_turn(
        work, turn_id="ambient-turn", principal_id="speaker"
    )

    direct = await scheduler.enqueue("chat", _pending())

    cancelled = await scheduler.get_turn(active.turn_id)
    assert direct.accepted
    assert cancelled is not None
    assert cancelled.phase is TurnPhase.CANCELLED
    assert cancelled.cancellation_generation == active.cancellation_generation + 1
    assert not await scheduler.is_turn_execution_allowed(
        active.turn_id, active.cancellation_generation
    )


@pytest.mark.asyncio
async def test_scheduler_turn_state_only_drops_terminal_turns():
    scheduler = ConversationScheduler(SessionTaskManager())
    enqueued = await scheduler.enqueue("chat", _pending())
    work = await scheduler.next_work("chat", owner_token=enqueued.consumer_token)
    assert work is not None
    active = await scheduler.start_turn(
        work, turn_id="turn-1", principal_id="principal"
    )

    with pytest.raises(TurnStateError, match="nonterminal"):
        await scheduler.drop_turn("turn-1")
    finalizing = await scheduler.transition_turn(
        "turn-1",
        expected_revision=active.revision,
        phase=TurnPhase.FINALIZING,
    )
    completed = await scheduler.transition_turn(
        "turn-1",
        expected_revision=finalizing.revision,
        phase=TurnPhase.COMPLETED,
    )
    await scheduler.drop_turn(completed.turn_id)

    assert await scheduler.get_turn("turn-1") is None


@pytest.mark.asyncio
async def test_wait_for_intent_queue_change_ignores_unmatched_messages():
    scheduler = ConversationScheduler(SessionTaskManager())
    await scheduler.enqueue("chat", _ambient_pending())
    revision = scheduler._revision("chat")

    wait = asyncio.create_task(
        scheduler.wait_for_intent_queue_change(
            "chat",
            since_revision=revision,
            timeout=0.1,
            intents=frozenset({InboundIntent.DIRECT_TASK}),
        )
    )
    await scheduler.enqueue("chat", _ambient_pending())
    assert not await wait

    wait = asyncio.create_task(
        scheduler.wait_for_intent_queue_change(
            "chat",
            since_revision=scheduler._revision("chat"),
            timeout=1,
            intents=frozenset({InboundIntent.DIRECT_TASK}),
        )
    )
    await scheduler.enqueue("chat", _pending())
    assert await wait
