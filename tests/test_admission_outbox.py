import asyncio

import pytest

from core.engine.admission_outbox import AdmissionOutbox
from core.engine.system_events import SystemEventBusy, SystemEventQueue


@pytest.mark.asyncio
async def test_outbox_retries_failed_effect_and_survives_reopen(tmp_path):
    path = tmp_path / "outbox.sqlite3"
    first = AdmissionOutbox(str(path))
    await first.prepare("chat", "message", {"chat_id": "chat", "value": "x"})
    await first.mark_ready("chat", "message")
    calls = []

    async def fail_once(payload):
        calls.append(payload)
        return len(calls) > 1

    await first.process({"hindsight": fail_once})
    assert await first.pending_count() == 1
    await first.close()

    second = AdmissionOutbox(str(path))
    await second.process({"hindsight": fail_once})
    assert await second.pending_count() == 0
    assert len(calls) == 2
    await second.close()


@pytest.mark.asyncio
async def test_outbox_recovers_prepared_after_local_admission(tmp_path):
    outbox = AdmissionOutbox(str(tmp_path / "outbox.sqlite3"))
    await outbox.prepare("chat", "message", {"chat_id": "chat"})
    recovered = await outbox.recover_prepared(
        lambda chat_id, message_id: asyncio.sleep(0, result=True)
    )
    assert recovered == 1
    assert await outbox.pending_count() == 2
    await outbox.close()


@pytest.mark.asyncio
async def test_outbox_leaves_prepared_rows_for_admission_in_progress(tmp_path):
    outbox = AdmissionOutbox(str(tmp_path / "outbox.sqlite3"))
    await outbox.prepare("chat", "message", {"chat_id": "chat"})

    recovered = await outbox.recover_prepared(
        lambda chat_id, message_id: asyncio.sleep(0, result=None)
    )

    assert recovered == 0
    assert await outbox.pending_count() == 2
    await outbox.close()


@pytest.mark.asyncio
async def test_outbox_handlers_receive_stable_effect_keys(tmp_path):
    outbox = AdmissionOutbox(str(tmp_path / "outbox.sqlite3"))
    await outbox.prepare("chat", "message", {"value": "x"})
    await outbox.mark_ready("chat", "message")
    keys = []

    async def handler(payload):
        keys.append(payload["idempotency_key"])
        return True

    await outbox.process({"hindsight": handler, "learner": handler})

    assert keys == [
        "admission:chat:message:hindsight",
        "admission:chat:message:learner",
    ]
    await outbox.close()


@pytest.mark.asyncio
async def test_outbox_keeps_success_record_for_duplicate_prepare(tmp_path):
    outbox = AdmissionOutbox(str(tmp_path / "outbox.sqlite3"))
    assert await outbox.prepare("chat", "message", {"value": "x"}) is True
    await outbox.mark_ready("chat", "message")

    async def handler(payload):
        return True

    await outbox.process({"hindsight": handler, "learner": handler})

    assert await outbox.prepare("chat", "message", {"value": "x"}) is False
    assert await outbox.pending_count() == 0
    await outbox.close()


@pytest.mark.asyncio
async def test_outbox_prunes_expired_success_records(tmp_path):
    outbox = AdmissionOutbox(
        str(tmp_path / "outbox.sqlite3"), succeeded_retention_seconds=0
    )
    assert await outbox.prepare("chat", "message", {"value": "x"}) is True
    await outbox.mark_ready("chat", "message")

    async def handler(payload):
        return True

    await outbox.process({"hindsight": handler, "learner": handler})

    assert await outbox.prepare("chat", "message", {"value": "x"}) is True
    await outbox.close()


@pytest.mark.asyncio
async def test_outbox_releases_all_claims_when_processing_is_cancelled(tmp_path):
    outbox = AdmissionOutbox(str(tmp_path / "outbox.sqlite3"))
    await outbox.prepare("chat", "message", {"value": "x"})
    await outbox.mark_ready("chat", "message")
    started = asyncio.Event()

    async def handler(payload):
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(outbox.process({"hindsight": handler}))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await outbox.pending_count() == 2
    await outbox.close()


def test_system_event_lease_does_not_overlap_or_drop_new_events():
    events = SystemEventQueue()
    events.enqueue("chat", "first", "one")
    first = events.claim_snapshot("chat")
    assert first is not None
    with pytest.raises(SystemEventBusy):
        events.claim_snapshot("chat")

    events.enqueue("chat", "second", "two")
    events.drain_non_heartbeat("chat")
    assert [event.text for event in events.peek("chat")] == ["first", "second"]

    events.commit_snapshot(first)
    assert [event.text for event in events.peek("chat")] == ["second"]


def test_system_event_failed_wake_releases_lease():
    events = SystemEventQueue()
    events.enqueue("chat", "first", "one")
    lease = events.claim_snapshot("chat")
    assert lease is not None
    events.release_snapshot(lease)
    retry = events.claim_snapshot("chat")
    assert retry is not None
    assert [event.text for event in retry.events] == ["first"]


def test_system_event_replace_during_wake_is_not_lost():
    events = SystemEventQueue()
    events.enqueue("chat", "old", "job")
    lease = events.claim_snapshot("chat")
    assert lease is not None
    events.enqueue("chat", "new", "job", replace=True)
    events.commit_snapshot(lease)
    assert [event.text for event in events.peek("chat")] == ["new"]


def test_system_event_drain_keeps_events_added_after_user_turn_snapshot():
    events = SystemEventQueue()
    events.enqueue("chat", "first", "first")
    observed = events.peek_non_heartbeat("chat")
    events.enqueue("chat", "second", "second")

    events.drain_non_heartbeat("chat", expected_events=observed)

    assert [event.text for event in events.peek("chat")] == ["second"]


@pytest.mark.asyncio
async def test_session_manager_can_resume_preserved_inbox():
    from core.managers.session_manager import PendingInbound, SessionTaskManager
    from core.message import InputMessage

    manager = SessionTaskManager()
    pending = PendingInbound(
        InputMessage("message", "user", "chat", "hello", False),
        "hello",
        "agent",
    )
    await manager.enqueue_and_claim_consumer("chat", pending)
    await manager.cleanup_all(preserve_inboxes=True)

    claims = await manager.claim_existing_consumers({"chat"})

    assert len(claims) == 1
    lease = await manager.claim_next_for_consumer("chat", claims[0][1])
    assert lease.items[0].message.id == "message"


@pytest.mark.asyncio
async def test_outbox_serializes_concurrent_processing_in_admission_order(tmp_path):
    outbox = AdmissionOutbox(str(tmp_path / "outbox.sqlite3"))
    await outbox.prepare("chat", "z-first", {"message": "z-first"})
    await outbox.mark_ready("chat", "z-first")
    await outbox.prepare("chat", "a-second", {"message": "a-second"})
    await outbox.mark_ready("chat", "a-second")
    calls = []

    async def handler(payload):
        calls.append(payload["message"])
        await asyncio.sleep(0)
        return True

    await asyncio.gather(
        outbox.process({"hindsight": handler, "learner": handler}, limit=2),
        outbox.process({"hindsight": handler, "learner": handler}, limit=2),
    )

    assert calls == ["z-first", "z-first", "a-second", "a-second"]
    assert await outbox.pending_count() == 0
    await outbox.close()
