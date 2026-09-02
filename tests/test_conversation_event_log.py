from datetime import datetime, timezone

import pytest

from core.engine.conversation_event_log import (
    ConversationEvent,
    ConversationEventLog,
    EventKind,
    EventLogInvariantError,
)


@pytest.mark.asyncio
async def test_event_log_keeps_one_ordered_turn_and_validates_tool_pairing(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))

    await log.append_user_message(
        chat_id="chat",
        turn_id="turn-1",
        message_id="message-1",
        content="查一下天气",
        timestamp=1,
    )
    await log.append_event(
        ConversationEvent(
            chat_id="chat",
            turn_id="turn-1",
            event_id="assistant:1",
            role="assistant",
            kind=EventKind.ASSISTANT_TOOL_CALL,
            tool_calls=({"id": "call-1", "function": {"name": "weather"}},),
            timestamp=2,
        )
    )
    await log.append_event(
        ConversationEvent(
            chat_id="chat",
            turn_id="turn-1",
            event_id="tool:call-1",
            role="tool",
            kind=EventKind.TOOL_RESULT,
            tool_call_id="call-1",
            tool_name="weather",
            content="晴天",
            timestamp=3,
        )
    )
    await log.append_accepted_delivery(
        chat_id="chat",
        turn_id="turn-1",
        delivery_id="delivery-1",
        content="今天晴天",
        timestamp=4,
    )
    await log.append_event(
        ConversationEvent(
            chat_id="chat",
            turn_id="turn-1",
            event_id="terminal:1",
            role="system",
            kind=EventKind.TURN_TERMINAL,
            terminal_status="completed",
            timestamp=5,
        )
    )

    visible = await log.snapshot_events("chat")
    complete = await log.snapshot_events("chat", include_internal=True)
    report = await log.validate_turn("turn-1")
    turns = await log.snapshot_turns("chat", include_internal=True)

    assert [event.kind for event in visible.events] == [
        EventKind.USER_MESSAGE,
        EventKind.ACCEPTED_DELIVERY,
    ]
    assert [event.event_id for event in complete.events] == [
        "user:message-1",
        "assistant:1",
        "tool:call-1",
        "delivery:delivery-1",
        "terminal:1",
    ]
    assert report.valid is True
    assert report.tool_call_ids == ("call-1",)
    assert report.tool_result_ids == ("call-1",)
    assert turns.turns[0].status == "completed"
    assert turns.turns[0].event_count == 5

    assert await log.protocol_wire("turn-1") == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1", "function": {"name": "weather"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "晴天"},
    ]


@pytest.mark.asyncio
async def test_event_log_materializes_tool_free_delivery_without_protocol_history(
    tmp_path,
):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))

    await log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="message-1", content="你好"
    )
    await log.append_accepted_delivery(
        chat_id="chat", turn_id="turn-1", delivery_id="delivery-1", content="你好呀"
    )
    await log.append_turn_terminal(chat_id="chat", turn_id="turn-1")

    assert [event["content"] for event in await log.history("chat")] == [
        "你好",
        "你好呀",
    ]
    assert await log.protocol_snapshot("turn-1") == ()


@pytest.mark.asyncio
async def test_event_log_is_idempotent_but_rejects_identity_collision(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    event = ConversationEvent(
        chat_id="chat",
        turn_id="turn-1",
        event_id="user:message-1",
        role="user",
        kind=EventKind.USER_MESSAGE,
        content="原始内容",
        timestamp=1,
    )

    first = await log.append_event(event)
    duplicate = await log.append_event(event)

    assert duplicate.event_seq == first.event_seq
    with pytest.raises(EventLogInvariantError, match="identity collision"):
        await log.append_event(
            ConversationEvent(
                chat_id="chat",
                turn_id="turn-1",
                event_id="user:message-1",
                role="user",
                kind=EventKind.USER_MESSAGE,
                content="被篡改内容",
                timestamp=1,
            )
        )


@pytest.mark.asyncio
async def test_event_log_retry_without_timestamp_is_idempotent(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    event = ConversationEvent(
        chat_id="chat",
        turn_id="turn-1",
        event_id="tool:call-1",
        role="tool",
        kind=EventKind.TOOL_RESULT,
        tool_call_id="call-1",
        content="result",
    )

    first = await log.append_event(event)
    second = await log.append_event(event)

    assert second.event_seq == first.event_seq


@pytest.mark.asyncio
async def test_terminal_turn_rejects_late_event_and_reports_missing_tool_result(
    tmp_path,
):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_event(
        ConversationEvent(
            chat_id="chat",
            turn_id="turn-1",
            event_id="assistant:1",
            role="assistant",
            kind=EventKind.ASSISTANT_TOOL_CALL,
            tool_calls=({"id": "call-1"},),
        )
    )

    await log.append_event(
        ConversationEvent(
            chat_id="chat",
            turn_id="turn-1",
            event_id="terminal:1",
            role="system",
            kind=EventKind.TURN_TERMINAL,
            terminal_status="incomplete",
        )
    )

    report = await log.validate_turn("turn-1")

    assert report.valid is False
    assert report.missing_tool_result_ids == ("call-1",)
    with pytest.raises(EventLogInvariantError, match="terminal turn"):
        await log.append_event(
            ConversationEvent(
                chat_id="chat",
                turn_id="turn-1",
                event_id="late:1",
                role="system",
                kind=EventKind.SYSTEM_EVENT,
                content="late",
            )
        )


@pytest.mark.asyncio
async def test_late_delivery_is_recorded_as_orphan_with_lineage(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))

    event = await log.append_late_delivery_event(
        chat_id="chat",
        original_turn_id="turn-1",
        delivery_id="delivery-1",
        content="late answer",
        message_id="platform-1",
    )

    assert event.kind is EventKind.SYSTEM_EVENT
    assert event.session_kind == "late_orphan"
    assert '"original_turn_id": "turn-1"' in event.content
    assert '"delivery_id": "delivery-1"' in event.content
    report = await log.validate_turn(event.turn_id)
    assert report.valid is True


@pytest.mark.asyncio
async def test_turn_source_date_is_fixed_by_first_event(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"), timezone_name="UTC")
    first_timestamp = datetime(2025, 1, 1, 23, 59, tzinfo=timezone.utc).timestamp()
    next_day_timestamp = datetime(2025, 1, 2, 0, 1, tzinfo=timezone.utc).timestamp()

    first = await log.append_user_message(
        chat_id="chat",
        turn_id="turn-1",
        message_id="message-1",
        content="跨午夜",
        timestamp=first_timestamp,
    )
    second = await log.append_accepted_delivery(
        chat_id="chat",
        turn_id="turn-1",
        delivery_id="delivery-1",
        content="完成",
        timestamp=next_day_timestamp,
    )

    assert first.source_date == "2025-01-01"
    assert second.source_date == first.source_date


@pytest.mark.asyncio
async def test_protocol_wire_orders_results_by_assistant_call_order(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="message-1", content="并行"
    )
    await log.append_event(
        ConversationEvent(
            chat_id="chat",
            turn_id="turn-1",
            event_id="assistant:1",
            role="assistant",
            kind=EventKind.ASSISTANT_TOOL_CALL,
            tool_calls=(
                {"id": "call-1", "function": {"name": "one"}},
                {"id": "call-2", "function": {"name": "two"}},
            ),
        )
    )
    await log.append_event(
        ConversationEvent(
            chat_id="chat",
            turn_id="turn-1",
            event_id="tool:call-2",
            role="tool",
            kind=EventKind.TOOL_RESULT,
            tool_call_id="call-2",
            content="二",
        )
    )
    await log.append_event(
        ConversationEvent(
            chat_id="chat",
            turn_id="turn-1",
            event_id="tool:call-1",
            role="tool",
            kind=EventKind.TOOL_RESULT,
            tool_call_id="call-1",
            content="一",
        )
    )

    wire = await log.protocol_wire("turn-1")

    assert [item.get("tool_call_id") for item in wire[1:]] == ["call-1", "call-2"]


@pytest.mark.asyncio
async def test_legacy_repair_groups_tool_wire_events_into_turns_idempotently(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    history = [
        {"role": "user", "content": "执行任务", "message_id": "m1", "timestamp": 1},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "function": {"name": "task"}}],
            "timestamp": 2,
        },
        {
            "role": "tool",
            "content": "完成",
            "tool_call_id": "call-1",
            "tool_name": "task",
            "timestamp": 3,
        },
        {"role": "assistant", "content": "已经完成", "timestamp": 4},
        {"role": "user", "content": "下一件事", "message_id": "m2", "timestamp": 5},
    ]

    first = await log.repair_from_legacy_history("chat", history)
    second = await log.repair_from_legacy_history("chat", history)
    turns = await log.snapshot_turns("chat", include_internal=True)
    events = await log.snapshot_events("chat", include_internal=True)

    assert first == 5
    assert second == 0
    assert len(turns.turns) == 2
    assert turns.turns[0].status == "completed"
    assert turns.turns[1].status == "open"
    assert [event.kind for event in events.events] == [
        EventKind.USER_MESSAGE,
        EventKind.ASSISTANT_TOOL_CALL,
        EventKind.TOOL_RESULT,
        EventKind.ACCEPTED_DELIVERY,
        EventKind.TURN_TERMINAL,
        EventKind.USER_MESSAGE,
    ]
