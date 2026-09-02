import pytest

from core.engine.conversation_event_log import ConversationEventLog
from core.engine.prompt_history_projection import PromptHistoryProjection


@pytest.mark.asyncio
async def test_prompt_projection_keeps_complete_recent_turns_with_budget(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    for index in range(3):
        turn_id = f"turn-{index}"
        await log.append_user_message(
            chat_id="chat",
            turn_id=turn_id,
            message_id=f"message-{index}",
            content=f"问题内容很长很长 {index}",
            timestamp=index + 1,
            session_kind="group",
        )
        await log.append_accepted_delivery(
            chat_id="chat",
            turn_id=turn_id,
            delivery_id=f"delivery-{index}",
            content=f"回答内容很长很长 {index}",
            timestamp=index + 1.5,
            session_kind="group",
        )
        await log.append_turn_terminal(
            chat_id="chat", turn_id=turn_id, timestamp=index + 2
        )

    snapshot = await PromptHistoryProjection(
        log, max_tokens=5, max_turns=10
    ).snapshot_for_prompt("chat")

    assert [event.turn_id for event in snapshot.events] == ["turn-2", "turn-2"]
    assert snapshot.truncated_event_ids
    assert snapshot.estimated_tokens <= 5


@pytest.mark.asyncio
async def test_prompt_projection_protects_current_turn_from_recent_window_limit(
    tmp_path,
):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    for turn_id, message_id in (("turn-1", "m1"), ("turn-2", "m2")):
        await log.append_user_message(
            chat_id="chat",
            turn_id=turn_id,
            message_id=message_id,
            content=turn_id,
        )

    snapshot = await PromptHistoryProjection(
        log, max_tokens=100, max_turns=1
    ).snapshot_for_prompt("chat", current_turn_id="turn-1")

    assert {event.turn_id for event in snapshot.events} == {"turn-1", "turn-2"}


@pytest.mark.asyncio
async def test_archive_retention_hides_events_idempotently_without_deleting_them(
    tmp_path,
):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat",
        turn_id="turn-1",
        message_id="m1",
        content="旧消息内容",
    )
    await log.append_user_message(
        chat_id="chat",
        turn_id="turn-2",
        message_id="m2",
        content="新消息内容",
    )
    projection = PromptHistoryProjection(
        log, max_tokens=100, metadata_path=str(tmp_path / "projection.sqlite3")
    )

    await projection.apply_archive_retention(
        "chat",
        operation_id="archive-1",
        hidden_event_ids=("user:m1",),
    )
    await projection.apply_archive_retention(
        "chat",
        operation_id="archive-1",
        hidden_event_ids=("user:m1",),
    )

    prompt = await projection.snapshot_for_prompt("chat")
    full = await log.snapshot_events("chat")

    assert [event.event_id for event in prompt.events] == ["user:m2"]
    assert [event.event_id for event in full.events] == ["user:m1", "user:m2"]


@pytest.mark.asyncio
async def test_stale_retained_projection_cannot_resurrect_hidden_event(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat",
        turn_id="turn-1",
        message_id="m1",
        content="old",
    )
    projection = PromptHistoryProjection(
        log, metadata_path=str(tmp_path / "projection.sqlite3")
    )

    await projection.apply_archive_retention(
        "chat", operation_id="archive-1", hidden_event_ids=("user:m1",)
    )
    await projection.apply_archive_retention(
        "chat", operation_id="stale-operation", retained_event_ids=("user:m1",)
    )

    assert await projection.hidden_event_ids("chat") == {"user:m1"}


@pytest.mark.asyncio
async def test_archive_retention_rejects_older_projection_watermark(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="m1", content="one"
    )
    await log.append_user_message(
        chat_id="chat", turn_id="turn-2", message_id="m2", content="two"
    )
    projection = PromptHistoryProjection(
        log, metadata_path=str(tmp_path / "projection.sqlite3")
    )

    await projection.apply_archive_retention(
        "chat",
        operation_id="archive-1",
        captured_cutoff_seq=2,
        hidden_event_ids=("user:m1",),
    )

    with pytest.raises(RuntimeError, match="stale archive projection watermark"):
        await projection.apply_archive_retention(
            "chat",
            operation_id="archive-stale",
            captured_cutoff_seq=1,
            retained_event_ids=("user:m2",),
        )


@pytest.mark.asyncio
async def test_active_projection_hides_archived_events_without_budgeting(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    for turn_id, message_id in (("turn-1", "m1"), ("turn-2", "m2")):
        await log.append_user_message(
            chat_id="chat", turn_id=turn_id, message_id=message_id, content=message_id
        )
    projection = PromptHistoryProjection(
        log, max_tokens=1, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    await projection.apply_archive_retention(
        "chat", operation_id="archive-1", hidden_event_ids=("user:m1",)
    )

    active = await projection.snapshot_for_active("chat")

    assert [event.event_id for event in active.events] == ["user:m2"]
    assert active.truncated_event_ids == ("user:m1",)
