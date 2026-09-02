import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.engine.conversation_event_log import (
    ConversationEvent,
    ConversationEventLog,
    EventKind,
)
from core.engine.turn_summary import TurnSummaryStore


async def _append_completed_turn(event_log):
    await event_log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="message-1", content="问题"
    )
    await event_log.append_accepted_delivery(
        chat_id="chat", turn_id="turn-1", delivery_id="delivery-1", content="答案"
    )
    await event_log.append_turn_terminal(chat_id="chat", turn_id="turn-1")


@pytest.mark.asyncio
async def test_turn_summary_uses_complete_terminal_turn_and_is_idempotent(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    store = TurnSummaryStore(
        log,
        path=str(tmp_path / "summaries.sqlite3"),
        max_prompt_tokens=1000,
    )
    await log.append_user_message(
        chat_id="chat",
        turn_id="turn-1",
        message_id="message-1",
        content="请执行任务",
        timestamp=1,
    )
    await log.append_event(
        ConversationEvent(
            chat_id="chat",
            turn_id="turn-1",
            event_id="assistant:call",
            role="assistant",
            kind=EventKind.ASSISTANT_TOOL_CALL,
            tool_calls=({"id": "call-1", "function": {"name": "run_task"}},),
            timestamp=2,
        )
    )
    await log.append_event(
        ConversationEvent(
            chat_id="chat",
            turn_id="turn-1",
            event_id="tool:result",
            role="tool",
            kind=EventKind.TOOL_RESULT,
            tool_call_id="call-1",
            tool_name="run_task",
            content="任务完成",
            timestamp=3,
        )
    )
    await log.append_accepted_delivery(
        chat_id="chat",
        turn_id="turn-1",
        delivery_id="delivery-1",
        content="已经完成",
        timestamp=4,
    )
    await log.append_turn_terminal(chat_id="chat", turn_id="turn-1", timestamp=5)

    first = await store.ensure_for_archived_events(
        "chat", ["user:message-1", "delivery:delivery-1"]
    )
    second = await store.ensure_for_archived_events(
        "chat", ["user:message-1", "delivery:delivery-1"]
    )

    assert len(first) == 1
    assert second[0].revision == first[0].revision == 1
    assert "run_task" in first[0].text
    assert "任务完成" in first[0].text
    selected = await store.select_for_prompt("chat")
    assert selected.summaries[0].turn_id == "turn-1"
    await store.close()
    await log.close()


@pytest.mark.asyncio
async def test_summary_selection_limits_batches_and_rolls_up_older_turns(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    store = TurnSummaryStore(
        log,
        path=str(tmp_path / "summaries.sqlite3"),
        max_prompt_tokens=5000,
        max_summary_batches=1,
    )
    for index in range(3):
        turn_id = f"turn-{index}"
        await log.append_user_message(
            chat_id="chat",
            turn_id=turn_id,
            message_id=f"message-{index}",
            content=f"问题 {index}",
        )
        await log.append_accepted_delivery(
            chat_id="chat",
            turn_id=turn_id,
            delivery_id=f"delivery-{index}",
            content=f"回答 {index}",
        )
        await log.append_turn_terminal(chat_id="chat", turn_id=turn_id)
        await store.ensure_for_archived_events(
            "chat",
            [f"user:message-{index}"],
            archive_batch_id=f"batch-{index}",
        )

    selection = await store.select_for_prompt("chat")

    assert {summary.archive_batch_id for summary in selection.summaries} == {"batch-2"}
    assert selection.rollup_source_count == 2
    assert "问题 0" in selection.text
    assert selection.estimated_tokens <= 5000
    await store.close()
    await log.close()


@pytest.mark.asyncio
async def test_incomplete_turn_does_not_get_summary(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    store = TurnSummaryStore(log, path=str(tmp_path / "summaries.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="message-1", content="未完成"
    )
    await log.append_turn_terminal(
        chat_id="chat", turn_id="turn-1", status="incomplete"
    )

    assert await store.ensure_for_archived_events("chat", ["user:message-1"]) == ()
    assert await store.list_for_webui("chat") == []
    await store.close()
    await log.close()


@pytest.mark.asyncio
async def test_semantic_summary_without_model_group_fails_job_and_keeps_deterministic(
    tmp_path,
):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    store = TurnSummaryStore(
        log,
        path=str(tmp_path / "summaries.sqlite3"),
        semantic_enabled=True,
        model_registry=None,
    )
    await _append_completed_turn(log)

    await store.ensure_for_archived_events("chat", ["user:message-1"])
    await asyncio.gather(*tuple(store._semantic_tasks))

    summary = await store.get("chat", "turn-1")
    assert summary is not None
    assert summary.semantic_text == ""
    assert "registry" in summary.semantic_error
    assert (await store.status("chat"))["semantic_failed_count"] == 1
    await store.close()
    await log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate",
    [
        "not json",
        '{"summary":"坏引用","source_turn_ids":["turn-1"],'
        '"source_event_ids":["unknown"]}',
    ],
)
async def test_invalid_semantic_candidate_keeps_deterministic_summary(
    tmp_path, candidate
):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    service = MagicMock()
    service.chat_completion = AsyncMock(return_value=(candidate, None))
    registry = MagicMock()
    registry.get_chain.return_value = ["provider/model"]
    registry.resolve_model_chain = AsyncMock(return_value=("provider/model", service))
    registry.cooldown_manager.record_success = AsyncMock()
    registry.cooldown_manager.record_failure = AsyncMock()
    store = TurnSummaryStore(
        log,
        path=str(tmp_path / "summaries.sqlite3"),
        semantic_enabled=True,
        model_registry=registry,
    )
    await _append_completed_turn(log)

    await store.ensure_for_archived_events("chat", ["user:message-1"])
    await asyncio.gather(*tuple(store._semantic_tasks))

    summary = await store.get("chat", "turn-1")
    assert summary is not None
    assert summary.semantic_text == ""
    assert summary.deterministic_text.startswith("Turn 1")
    assert summary.semantic_error
    assert (await store.status("chat"))["semantic_failed_count"] == 1
    await store.close()
    await log.close()


@pytest.mark.asyncio
async def test_summary_selection_never_exceeds_explicit_token_budget(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    store = TurnSummaryStore(
        log, path=str(tmp_path / "summaries.sqlite3"), max_prompt_tokens=1000
    )
    await _append_completed_turn(log)
    await store.ensure_for_archived_events("chat", ["user:message-1"])

    selection = await store.select_for_prompt("chat", max_tokens=1)

    assert selection.estimated_tokens <= 1
    await store.close()
    await log.close()
