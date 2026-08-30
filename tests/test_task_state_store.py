import json

import pytest

from core.engine.turn_state import TurnPhase, TurnState
from core.managers.session_manager import InboundIntent
from core.tasks.task_state_store import TaskStateStore


@pytest.mark.asyncio
async def test_task_state_store_persists_and_interrupts_active_turn(tmp_path):
    store = TaskStateStore(str(tmp_path))
    active = TurnState(
        turn_id="turn-1",
        chat_id="chat-1",
        intent=InboundIntent.DIRECT_TASK,
        principal_id="principal",
        queue_revision=3,
        task_anchor_message_id="anchor-1",
    )
    await store.put(active)

    recovered = TaskStateStore(str(tmp_path))
    interrupted = await recovered.mark_interrupted_on_restart()

    assert [state.turn_id for state in interrupted] == ["turn-1"]
    state = recovered.get("turn-1")
    assert state is not None
    assert state.phase == TurnPhase.CANCELLED.value
    assert state.cancellation_generation == 1
    assert state.interrupted_by_restart is True
    assert state.task_anchor_message_id == "anchor-1"

    await recovered.record_delivery("turn-1", "delivery-1", "prepared")
    await recovered.record_delivery("turn-1", "delivery-1", "sent")
    persisted = TaskStateStore(str(tmp_path)).get("turn-1")
    assert persisted is not None
    assert persisted.delivery_ids == ("delivery-1",)
    assert persisted.last_delivery_status == "sent"


@pytest.mark.asyncio
async def test_task_state_store_leaves_terminal_turn_unchanged(tmp_path):
    store = TaskStateStore(str(tmp_path))
    active = TurnState(
        turn_id="turn-1",
        chat_id="chat-1",
        intent=InboundIntent.DIRECT_TASK,
        principal_id="principal",
        queue_revision=3,
    )
    terminal = active.transition(TurnPhase.FINALIZING, expected_revision=0).transition(
        TurnPhase.COMPLETED, expected_revision=1
    )
    await store.put(terminal)

    assert await TaskStateStore(str(tmp_path)).mark_interrupted_on_restart() == []


@pytest.mark.asyncio
async def test_task_state_store_terminates_expired_wait_without_replaying_it(tmp_path):
    store = TaskStateStore(str(tmp_path))
    active = TurnState(
        turn_id="expired-wait",
        chat_id="chat-1",
        intent=InboundIntent.PRIVATE_CONVERSATION,
        principal_id="principal",
        queue_revision=3,
    )
    waiting = active.transition(
        TurnPhase.WAITING,
        expected_revision=active.revision,
        wait_reason="more input",
        wait_deadline=10.0,
    )
    await store.put(waiting)

    expired = await store.expire_waiting_turns(now=10.0)

    assert [state.turn_id for state in expired] == ["expired-wait"]
    current = store.get("expired-wait")
    assert current is not None
    assert current.phase == TurnPhase.CANCELLED.value
    assert current.cancellation_generation == 1
    assert (
        await store.claim_waiting_recoveries(
            chat_id="chat-1",
            principal_id="principal",
            intent=InboundIntent.PRIVATE_CONVERSATION.value,
        )
        == []
    )


@pytest.mark.asyncio
async def test_task_state_store_retains_waiting_turn_until_matching_recovery(
    tmp_path,
):
    store = TaskStateStore(str(tmp_path))
    active = TurnState(
        turn_id="wait-1",
        chat_id="chat-1",
        intent=InboundIntent.PRIVATE_CONVERSATION,
        principal_id="principal",
        queue_revision=3,
    )
    waiting = active.transition(
        TurnPhase.WAITING,
        expected_revision=active.revision,
        wait_reason="more input",
        wait_deadline=9999999999.0,
    )
    await store.put(waiting)

    recovered = TaskStateStore(str(tmp_path))
    assert await recovered.mark_interrupted_on_restart() == []
    assert recovered.get("wait-1").phase == TurnPhase.WAITING.value
    assert (
        await recovered.claim_waiting_recoveries(
            chat_id="chat-1",
            principal_id="other",
            intent=InboundIntent.PRIVATE_CONVERSATION.value,
        )
        == []
    )
    assert (
        await recovered.claim_waiting_recoveries(
            chat_id="chat-1",
            principal_id="principal",
            intent=InboundIntent.DIRECT_TASK.value,
        )
        == []
    )

    claimed = await recovered.claim_waiting_recoveries(
        chat_id="chat-1",
        principal_id="principal",
        intent=InboundIntent.PRIVATE_CONVERSATION.value,
    )
    assert [state.turn_id for state in claimed] == ["wait-1"]
    assert recovered.get("wait-1").phase == TurnPhase.CANCELLED.value


@pytest.mark.asyncio
async def test_task_state_store_skips_invalid_legacy_records_without_losing_valid_state(
    tmp_path,
):
    path = tmp_path / "turn_states.json"
    path.write_text(
        json.dumps(
            [
                {
                    "turn_id": "valid",
                    "chat_id": "chat-1",
                    "intent": "direct_task",
                    "principal_id": "principal",
                    "queue_revision": 1,
                    "phase": "completed",
                    "revision": 2,
                    "cancellation_generation": 0,
                    "approval_plan_id": "",
                    "task_anchor_message_id": "anchor",
                    "updated_at": 1.0,
                    "unknown_future_field": "ignored",
                },
                {"turn_id": "missing-phase"},
                {"turn_id": "bad-phase", "phase": "running"},
            ]
        ),
        encoding="utf-8",
    )

    store = TaskStateStore(str(tmp_path))

    state = store.get("valid")
    assert state is not None
    assert state.phase == TurnPhase.COMPLETED.value
    assert store.get("missing-phase") is None
    assert store.get("bad-phase") is None
