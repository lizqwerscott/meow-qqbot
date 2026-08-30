import asyncio

import pytest

from core.engine.conversation_scheduler import (
    ConversationScheduler,
    StaleScheduledWork,
)
from core.managers.session_manager import (
    AdmissionOrigin,
    InboundIntent,
    PendingInbound,
    SessionTaskManager,
)
from core.message import InputMessage


def pending(message_id: str, intent: InboundIntent) -> PendingInbound:
    return PendingInbound(
        InputMessage(
            message_id, "user", "chat", message_id, intent is InboundIntent.DIRECT_TASK
        ),
        message_id,
        intent,
        AdmissionOrigin.USER_MESSAGE,
    )


@pytest.mark.asyncio
async def test_scheduler_returns_intent_and_revision_snapshot():
    scheduler = ConversationScheduler(SessionTaskManager())
    first = await scheduler.enqueue("chat", pending("first", InboundIntent.DIRECT_TASK))
    await scheduler.enqueue("chat", pending("ambient", InboundIntent.GROUP_AMBIENT))

    work = await scheduler.next_work("chat", owner_token=first.consumer_token)

    assert work is not None
    assert work.intent is InboundIntent.DIRECT_TASK
    assert work.queue_revision == 2
    await scheduler.commit(work, work.pending)

    ambient = await scheduler.next_work("chat", owner_token=first.consumer_token)
    assert ambient is not None
    assert ambient.intent is InboundIntent.GROUP_AMBIENT
    assert ambient.queue_revision == 2


@pytest.mark.asyncio
async def test_wait_for_queue_change_wakes_on_new_inbound():
    scheduler = ConversationScheduler(SessionTaskManager())
    waiting = asyncio.create_task(
        scheduler.wait_for_queue_change("chat", since_revision=0, timeout=1)
    )
    await asyncio.sleep(0)
    await scheduler.enqueue("chat", pending("wake", InboundIntent.PRIVATE_CONVERSATION))
    assert await waiting is True

    scheduler = ConversationScheduler(
        SessionTaskManager(),
        collect_idle_ms=10,
        collect_max_wait_ms=20,
        collect_max_messages=3,
        collect_max_chars=100,
    )
    first = await scheduler.enqueue(
        "chat", pending("first", InboundIntent.PRIVATE_CONVERSATION)
    )
    selecting = asyncio.create_task(
        scheduler.next_work("chat", owner_token=first.consumer_token)
    )
    await asyncio.sleep(0)
    await scheduler.enqueue(
        "chat", pending("second", InboundIntent.PRIVATE_CONVERSATION)
    )

    work = await selecting

    assert work is not None
    assert [item.message.id for item in work.items] == ["first", "second"]
    for item in work.items:
        await scheduler.commit(work, item)


@pytest.mark.asyncio
async def test_quiet_reservation_does_not_hold_inbox_lease():
    session = SessionTaskManager()
    scheduler = ConversationScheduler(
        session,
        collect_idle_ms=30,
        collect_max_wait_ms=50,
    )
    first = await scheduler.enqueue(
        "chat", pending("first", InboundIntent.PRIVATE_CONVERSATION)
    )
    selecting = asyncio.create_task(
        scheduler.next_work("chat", owner_token=first.consumer_token)
    )
    await asyncio.sleep(0)

    assert session._leased_counts.get("chat", 0) == 0
    await selecting


@pytest.mark.asyncio
async def test_handoff_wakes_and_invalidates_quiet_reservation():
    session = SessionTaskManager()
    scheduler = ConversationScheduler(
        session,
        collect_idle_ms=500,
        collect_max_wait_ms=1000,
    )
    first = await scheduler.enqueue(
        "chat", pending("first", InboundIntent.PRIVATE_CONVERSATION)
    )
    selecting = asyncio.create_task(
        scheduler.next_work("chat", owner_token=first.consumer_token)
    )
    await asyncio.sleep(0)
    replacement = await scheduler.handoff_consumer("chat", first.consumer_token)

    assert replacement is not None
    assert await asyncio.wait_for(selecting, timeout=0.1) is None
    replacement_work = await scheduler.next_work("chat", owner_token=replacement)
    assert replacement_work is not None
    assert replacement_work.pending.message.id == "first"
    await scheduler.commit(replacement_work, replacement_work.pending)


@pytest.mark.asyncio
async def test_scheduler_serializes_concurrent_quiet_collectors_per_chat():
    scheduler = ConversationScheduler(
        SessionTaskManager(),
        collect_idle_ms=20,
        collect_max_wait_ms=40,
        collect_max_messages=3,
    )
    first = await scheduler.enqueue(
        "chat", pending("first", InboundIntent.PRIVATE_CONVERSATION)
    )
    await scheduler.enqueue(
        "chat", pending("second", InboundIntent.PRIVATE_CONVERSATION)
    )

    first_select, second_select = await asyncio.gather(
        scheduler.next_work("chat", owner_token=first.consumer_token),
        scheduler.next_work("chat", owner_token=first.consumer_token),
    )

    assert first_select is not None
    assert [item.message.id for item in first_select.items] == ["first", "second"]
    assert second_select is None
    for item in first_select.items:
        await scheduler.commit(first_select, item)


@pytest.mark.asyncio
async def test_scheduler_collects_ambient_batch_without_crossing_direct_boundary():
    scheduler = ConversationScheduler(
        SessionTaskManager(),
        ambient_collect_idle_ms=10,
        collect_max_wait_ms=20,
        collect_max_messages=8,
        collect_max_chars=100,
    )
    first = await scheduler.enqueue(
        "chat", pending("ambient-1", InboundIntent.GROUP_AMBIENT)
    )
    selecting = asyncio.create_task(
        scheduler.next_work("chat", owner_token=first.consumer_token)
    )
    await asyncio.sleep(0)
    await scheduler.enqueue("chat", pending("ambient-2", InboundIntent.GROUP_AMBIENT))
    await scheduler.enqueue("chat", pending("direct", InboundIntent.DIRECT_TASK))

    work = await selecting

    assert work is not None
    assert [item.message.id for item in work.items] == ["ambient-1", "ambient-2"]
    assert work.passive_admission_only is True
    for item in work.items:
        await scheduler.commit(work, item)
    direct = await scheduler.next_work("chat", owner_token=first.consumer_token)
    assert direct is not None
    assert [item.message.id for item in direct.items] == ["direct"]
    await scheduler.commit(direct, direct.pending)


@pytest.mark.asyncio
async def test_direct_preempts_ambient_quiet_wait_without_leasing_the_direct_task():
    scheduler = ConversationScheduler(
        SessionTaskManager(),
        ambient_collect_idle_ms=500,
        collect_max_wait_ms=1000,
    )
    first = await scheduler.enqueue(
        "chat", pending("ambient-1", InboundIntent.GROUP_AMBIENT)
    )
    selecting = asyncio.create_task(
        scheduler.next_work("chat", owner_token=first.consumer_token)
    )
    await asyncio.sleep(0)
    await scheduler.enqueue("chat", pending("direct", InboundIntent.DIRECT_TASK))

    work = await asyncio.wait_for(selecting, timeout=0.1)

    assert work is not None
    assert [item.message.id for item in work.items] == ["ambient-1"]
    assert work.passive_admission_only is True
    await scheduler.commit(work, work.pending)

    direct = await scheduler.next_work("chat", owner_token=first.consumer_token)
    assert direct is not None
    assert [item.message.id for item in direct.items] == ["direct"]
    assert direct.passive_admission_only is False
    await scheduler.commit(direct, direct.pending)


@pytest.mark.asyncio
async def test_same_intent_enqueue_resets_quiet_idle_until_max_wait():
    scheduler = ConversationScheduler(
        SessionTaskManager(),
        collect_idle_ms=30,
        collect_max_wait_ms=100,
        collect_max_messages=8,
    )
    first = await scheduler.enqueue(
        "chat", pending("first", InboundIntent.PRIVATE_CONVERSATION)
    )
    selecting = asyncio.create_task(
        scheduler.next_work("chat", owner_token=first.consumer_token)
    )
    await asyncio.sleep(0.01)
    await scheduler.enqueue(
        "chat", pending("second", InboundIntent.PRIVATE_CONVERSATION)
    )
    await asyncio.sleep(0.01)
    await scheduler.enqueue(
        "chat", pending("third", InboundIntent.PRIVATE_CONVERSATION)
    )

    work = await selecting

    assert work is not None
    assert [item.message.id for item in work.items] == ["first", "second", "third"]
    for item in work.items:
        await scheduler.commit(work, item)


@pytest.mark.asyncio
async def test_different_intent_freezes_quiet_batch_immediately():
    scheduler = ConversationScheduler(
        SessionTaskManager(),
        collect_idle_ms=500,
        collect_max_wait_ms=1000,
    )
    first = await scheduler.enqueue(
        "chat", pending("private", InboundIntent.PRIVATE_CONVERSATION)
    )
    selecting = asyncio.create_task(
        scheduler.next_work("chat", owner_token=first.consumer_token)
    )
    await asyncio.sleep(0)
    await scheduler.enqueue("chat", pending("direct", InboundIntent.DIRECT_TASK))

    work = await asyncio.wait_for(selecting, timeout=0.1)

    assert work is not None
    assert [item.message.id for item in work.items] == ["private"]
    await scheduler.commit(work, work.pending)
    scheduler = ConversationScheduler(SessionTaskManager())
    enqueued = await scheduler.enqueue(
        "chat", pending("first", InboundIntent.DIRECT_TASK)
    )
    work = await scheduler.next_work("chat", owner_token=enqueued.consumer_token)
    assert work is not None

    await scheduler.handoff_consumer("chat", enqueued.consumer_token)

    with pytest.raises(StaleScheduledWork):
        await scheduler.commit(work, work.pending)


@pytest.mark.asyncio
async def test_rejected_enqueue_does_not_advance_revision():
    scheduler = ConversationScheduler(SessionTaskManager(max_inbox_size=1))
    accepted = await scheduler.enqueue(
        "chat", pending("first", InboundIntent.DIRECT_TASK)
    )
    rejected = await scheduler.enqueue(
        "chat", pending("second", InboundIntent.DIRECT_TASK)
    )

    assert accepted.accepted is True
    assert rejected.accepted is False
    assert scheduler.revision("chat") == 1


@pytest.mark.asyncio
async def test_handoff_cannot_race_scheduled_work_commit():
    scheduler = ConversationScheduler(SessionTaskManager())
    enqueued = await scheduler.enqueue(
        "chat", pending("first", InboundIntent.DIRECT_TASK)
    )
    work = await scheduler.next_work("chat", owner_token=enqueued.consumer_token)
    assert work is not None

    await scheduler.handoff_consumer("chat", enqueued.consumer_token)

    with pytest.raises(StaleScheduledWork):
        await scheduler.commit(work, work.pending)
