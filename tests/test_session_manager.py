import asyncio

import pytest

from core.managers.session_manager import (
    AdmissionOrigin,
    InboundIntent,
    PendingInbound,
    SessionTaskManager,
)
from core.message import InputMessage


@pytest.fixture
def mgr():
    return SessionTaskManager()


def pending(
    message_id: str, intent: InboundIntent = InboundIntent.PRIVATE_CONVERSATION
) -> PendingInbound:
    return PendingInbound(
        InputMessage(message_id, "user", "chat_001", message_id, False),
        message_id,
        intent,
        AdmissionOrigin.USER_MESSAGE,
    )


@pytest.mark.asyncio
async def test_get_lock_creates_on_demand(mgr):
    assert await mgr.get_lock("chat_001") is not None


@pytest.mark.asyncio
async def test_enqueue_starts_only_one_consumer(mgr):
    first = await mgr.enqueue_and_claim_consumer("chat_001", pending("first"))
    second = await mgr.enqueue_and_claim_consumer("chat_001", pending("second"))

    assert first.should_start_consumer is True
    assert second.should_start_consumer is False
    assert mgr.has_active_consumer("chat_001") is True


@pytest.mark.asyncio
async def test_steering_active_is_scoped_to_session(mgr):
    assert await mgr.is_steering_active("chat_001") is False
    agent = PendingInbound(
        InputMessage("first", "user", "chat_001", "first", False),
        "first",
        "agent",
        AdmissionOrigin.USER_MESSAGE,
    )
    enqueued = await mgr.enqueue_with_dispatch_mode("chat_001", agent, triggers_ai=True)
    lease = await mgr.claim_next_for_consumer("chat_001", enqueued.consumer_token)

    assert await mgr.is_steering_active("chat_001") is True
    assert await mgr.is_steering_active("chat_002") is False
    await mgr.commit(lease, lease.items[0])
    assert await mgr.release_consumer_if_idle("chat_001", enqueued.consumer_token)
    assert await mgr.is_steering_active("chat_001") is False


@pytest.mark.asyncio
async def test_agent_turn_activates_only_when_consumer_claims_it(mgr):
    first = PendingInbound(
        InputMessage("first", "user", "chat_001", "first", False),
        "first",
        "agent",
        AdmissionOrigin.USER_MESSAGE,
    )
    followup = PendingInbound(
        InputMessage("followup", "user", "chat_001", "followup", True),
        "followup",
        "passive",
        AdmissionOrigin.USER_MESSAGE,
    )
    enqueued = await mgr.enqueue_and_claim_consumer("chat_001", first)
    assert await mgr.is_steering_active("chat_001") is False
    await mgr.enqueue_and_claim_consumer("chat_001", followup)

    first_lease = await mgr.claim_next_for_consumer("chat_001", enqueued.consumer_token)
    assert first_lease.items[0].intent is InboundIntent.PRIVATE_CONVERSATION
    assert first_lease.items[0].origin is AdmissionOrigin.USER_MESSAGE
    assert await mgr.is_steering_active("chat_001") is True
    await mgr.commit(first_lease, first_lease.items[0])

    followup_lease = await mgr.claim_next_for_consumer(
        "chat_001", enqueued.consumer_token
    )
    assert followup_lease.items[0].intent is InboundIntent.GROUP_AMBIENT


@pytest.mark.asyncio
async def test_idle_release_deactivates_steering_session(mgr):
    agent = PendingInbound(
        InputMessage("first", "user", "chat_001", "first", False),
        "first",
        "agent",
        AdmissionOrigin.USER_MESSAGE,
    )
    enqueued = await mgr.enqueue_with_dispatch_mode("chat_001", agent, triggers_ai=True)
    lease = await mgr.claim_next_for_consumer("chat_001", enqueued.consumer_token)
    await mgr.commit(lease, lease.items[0])

    assert await mgr.release_consumer_if_idle("chat_001", enqueued.consumer_token)
    assert await mgr.is_steering_active("chat_001") is False


@pytest.mark.asyncio
async def test_steering_requeue_reserves_lease_capacity(mgr):
    mgr = SessionTaskManager(max_inbox_size=2)
    first = await mgr.enqueue_and_claim_consumer("chat_001", pending("one"))
    await mgr.enqueue_and_claim_consumer("chat_001", pending("two"))
    lease = await mgr.claim_pending_for_steer("chat_001")

    result = await mgr.enqueue_and_claim_consumer("chat_001", pending("three"))

    assert result.accepted is False
    assert result.dropped.message.id == "three"
    assert await mgr.requeue_front(lease) == 2
    claimed = await mgr.claim_pending_for_steer("chat_001")
    assert [item.message.id for item in claimed.items] == ["one", "two"]


@pytest.mark.asyncio
async def test_release_consumer_is_atomic_with_idle_check(mgr):
    enqueued = await mgr.enqueue_and_claim_consumer("chat_001", pending("first"))
    assert not await mgr.release_consumer_if_idle("chat_001", enqueued.consumer_token)

    lease = await mgr.claim_next_for_consumer("chat_001", enqueued.consumer_token)
    await mgr.commit(lease, lease.items[0])
    assert await mgr.release_consumer_if_idle("chat_001", enqueued.consumer_token)
    assert mgr.has_active_consumer("chat_001") is False


@pytest.mark.asyncio
async def test_stale_consumer_cannot_revoke_replacement_owner(mgr):
    first = await mgr.enqueue_and_claim_consumer("chat_001", pending("first"))
    first_lease = await mgr.claim_next_for_consumer("chat_001", first.consumer_token)
    await mgr.commit(first_lease, first_lease.items[0])
    assert await mgr.release_consumer_if_idle("chat_001", first.consumer_token)

    replacement = await mgr.enqueue_and_claim_consumer("chat_001", pending("next"))
    assert replacement.should_start_consumer is True
    assert await mgr.handoff_consumer("chat_001", first.consumer_token) is None
    assert mgr.has_active_consumer("chat_001") is True

    replacement_lease = await mgr.claim_next_for_consumer(
        "chat_001", replacement.consumer_token
    )
    assert replacement_lease.items[0].message.id == "next"


@pytest.mark.asyncio
async def test_overflow_drops_oldest_unadmitted_message():
    mgr = SessionTaskManager(max_inbox_size=2)
    await mgr.enqueue_and_claim_consumer(
        "chat_001", pending("one", InboundIntent.GROUP_AMBIENT)
    )
    await mgr.enqueue_and_claim_consumer(
        "chat_001", pending("two", InboundIntent.GROUP_AMBIENT)
    )
    result = await mgr.enqueue_and_claim_consumer(
        "chat_001", pending("three", InboundIntent.GROUP_AMBIENT)
    )

    assert result.dropped.message.id == "one"
    lease = await mgr.claim_pending_for_steer(
        "chat_001", intent=InboundIntent.GROUP_AMBIENT
    )
    assert [item.message.id for item in lease.items] == ["two", "three"]


@pytest.mark.asyncio
async def test_cleanup_session(mgr):
    await mgr.enqueue_and_claim_consumer("chat_001", pending("one"))
    await mgr.cleanup_session("chat_001")
    assert mgr.has_active_consumer("chat_001") is False
    assert mgr.get_queue_sizes() == {}


def test_get_queue_sizes(mgr):
    async def setup():
        await mgr.enqueue_and_claim_consumer("chat_001", pending("one"))
        assert mgr.get_queue_sizes() == {"chat_001": 1}

    asyncio.run(setup())


@pytest.mark.asyncio
async def test_steering_cannot_cross_an_ambient_intent_boundary(mgr):
    first = await mgr.enqueue_and_claim_consumer(
        "chat_001", pending("first", InboundIntent.DIRECT_TASK)
    )
    await mgr.enqueue_and_claim_consumer(
        "chat_001", pending("ambient", InboundIntent.GROUP_AMBIENT)
    )
    await mgr.enqueue_and_claim_consumer(
        "chat_001", pending("second", InboundIntent.DIRECT_TASK)
    )

    current = await mgr.claim_next_for_consumer("chat_001", first.consumer_token)
    await mgr.commit(current, current.items[0])

    assert (
        await mgr.claim_pending_for_steer("chat_001", intent=InboundIntent.DIRECT_TASK)
        is None
    )


@pytest.mark.asyncio
async def test_steering_claim_keeps_cross_principal_followups_in_inbox(mgr):
    initial = PendingInbound(
        InputMessage("initial", "owner", "chat_001", "initial", True),
        "initial",
        InboundIntent.DIRECT_TASK,
        AdmissionOrigin.USER_MESSAGE,
    )
    enqueued = await mgr.enqueue_and_claim_consumer("chat_001", initial)
    current = await mgr.claim_next_for_consumer("chat_001", enqueued.consumer_token)
    await mgr.commit(current, current.items[0])

    for message_id, sender_id in (("other", "other-user"), ("owner", "owner")):
        await mgr.enqueue_and_claim_consumer(
            "chat_001",
            PendingInbound(
                InputMessage(message_id, sender_id, "chat_001", message_id, True),
                message_id,
                InboundIntent.DIRECT_TASK,
                AdmissionOrigin.USER_MESSAGE,
            ),
        )

    assert (
        await mgr.claim_pending_for_steer(
            "chat_001",
            intent=InboundIntent.DIRECT_TASK,
            principal_id="owner",
        )
        is None
    )
    lease = await mgr.claim_pending_for_steer(
        "chat_001", intent=InboundIntent.DIRECT_TASK
    )
    assert [item.message.sender_id for item in lease.items] == ["other-user", "owner"]
