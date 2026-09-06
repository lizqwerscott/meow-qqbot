import asyncio
import sqlite3
from datetime import datetime, timezone

import pytest

from core.engine.conversation_event_log import (
    ConversationEvent,
    ConversationEventLog,
    EventKind,
    EventLogInvariantError,
    TurnKind,
    TurnStatus,
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
async def test_user_messages_by_id_reads_only_requested_chat_events(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat-a", turn_id="turn-a", message_id="message-a", content="A"
    )
    await log.append_user_message(
        chat_id="chat-a", turn_id="turn-b", message_id="message-b", content="B"
    )
    await log.append_user_message(
        chat_id="chat-b", turn_id="turn-a", message_id="message-a", content="other"
    )

    events = await log.user_messages_by_id("chat-a", ("message-b", "message-a"))

    assert [event.message_id for event in events] == ["message-a", "message-b"]
    assert [event.content for event in events] == ["A", "B"]
    assert await log.user_messages_by_id("chat-a", ("missing",)) == ()


@pytest.mark.asyncio
async def test_turn_ids_are_scoped_to_chat(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    for chat_id in ("chat-a", "chat-b"):
        await log.append_user_message(
            chat_id=chat_id,
            turn_id="same-turn",
            message_id=f"message-{chat_id}",
            content=chat_id,
        )

    turns = await log.snapshot_turns("chat-a", include_internal=True)
    assert turns.turns[0].turn_id == "same-turn"
    assert (await log.validate_turn("same-turn", chat_id="chat-b")).valid is True
    with pytest.raises(EventLogInvariantError, match="ambiguous turn"):
        await log.validate_turn("same-turn")
    with pytest.raises(EventLogInvariantError, match="ambiguous turn"):
        await log.protocol_snapshot("same-turn")


@pytest.mark.asyncio
async def test_history_can_read_only_latest_events(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    for index in range(3):
        turn_id = f"turn-{index}"
        await log.append_user_message(
            chat_id="chat",
            turn_id=turn_id,
            message_id=f"message-{index}",
            content=f"question-{index}",
        )
        await log.append_accepted_delivery(
            chat_id="chat",
            turn_id=turn_id,
            delivery_id=f"delivery-{index}",
            content=f"answer-{index}",
        )

    latest = await log.history("chat", max_events=2)

    assert [item["content"] for item in latest] == ["question-2", "answer-2"]


@pytest.mark.asyncio
async def test_legacy_migration_watermark_persists(tmp_path):
    path = tmp_path / "events.sqlite3"
    log = ConversationEventLog(str(path))

    assert await log.legacy_migration_is_complete() is False
    await log.mark_legacy_migration_complete()
    assert await log.legacy_migration_is_complete() is True
    await log.close()

    reopened = ConversationEventLog(str(path))
    assert await reopened.legacy_migration_is_complete() is True
    await reopened.close()


@pytest.mark.asyncio
async def test_event_log_upgrades_missing_per_chat_migration_checkpoint(tmp_path):
    path = tmp_path / "events.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE conversation_event_log_schema (version INTEGER NOT NULL)"
    )
    conn.execute("INSERT INTO conversation_event_log_schema(version) VALUES (2)")
    conn.execute(
        "CREATE TABLE conversation_migration_state "
        "(migration_key TEXT PRIMARY KEY, completed_at REAL NOT NULL)"
    )
    conn.commit()
    conn.close()

    log = ConversationEventLog(str(path))

    assert await log.legacy_chat_migration_is_complete("chat") is False
    await log.mark_legacy_chat_migration_complete("chat")
    assert await log.legacy_chat_migration_is_complete("chat") is True
    await log.close()


@pytest.mark.asyncio
async def test_legacy_conflict_event_ids_are_bounded_and_content_free(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_event(
        ConversationEvent(
            chat_id="chat",
            turn_id="turn",
            event_id="user:message:legacy-conflict:abc",
            role="user",
            kind=EventKind.USER_MESSAGE,
            content="敏感内容不应返回",
        )
    )

    assert await log.legacy_conflict_event_ids("chat") == (
        "user:message:legacy-conflict:abc",
    )
    await log.close()


@pytest.mark.asyncio
async def test_legacy_chat_migration_checkpoint_persists_per_chat(tmp_path):
    path = str(tmp_path / "events.sqlite3")
    log = ConversationEventLog(path)

    assert await log.legacy_chat_migration_is_complete("chat-1") is False
    await log.mark_legacy_chat_migration_complete("chat-1")
    assert await log.legacy_chat_migration_is_complete("chat-1") is True
    assert await log.legacy_chat_migration_is_complete("chat-2") is False
    await log.close()

    reopened = ConversationEventLog(path)
    assert await reopened.legacy_chat_migration_is_complete("chat-1") is True
    assert await reopened.legacy_chat_migration_is_complete("chat-2") is False
    await reopened.close()


@pytest.mark.asyncio
async def test_accepted_delivery_content_reads_by_anchor(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_accepted_delivery(
        chat_id="chat",
        turn_id="turn-1",
        delivery_id="delivery-1",
        message_id="anchor-1",
        content="恢复内容",
    )

    assert await log.accepted_delivery_content("chat", "anchor-1") == "恢复内容"
    assert await log.accepted_delivery_content("chat", "delivery-1") == "恢复内容"
    assert await log.accepted_delivery_content("chat", "missing") is None
    await log.close()


@pytest.mark.asyncio
async def test_event_log_distinguishes_ai_and_ambient_turns(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))

    await log.append_user_message(
        chat_id="group",
        turn_id="ambient-turn",
        message_id="ambient-message",
        content="群里闲聊",
        turn_kind=TurnKind.AMBIENT,
    )
    await log.append_turn_terminal(
        chat_id="group", turn_id="ambient-turn", status="completed"
    )
    await log.append_user_message(
        chat_id="group",
        turn_id="ai-turn",
        message_id="ai-message",
        content="请回答我",
        turn_kind=TurnKind.AI,
    )
    await log.append_turn_terminal(
        chat_id="group", turn_id="ai-turn", status="completed"
    )

    snapshot = await log.snapshot_turns("group", include_internal=True)

    assert {turn.turn_id: turn.turn_kind for turn in snapshot.turns} == {
        "ambient-turn": TurnKind.AMBIENT,
        "ai-turn": TurnKind.AI,
    }
    await log.close()


@pytest.mark.asyncio
async def test_event_log_migrates_existing_turns_with_unknown_kind(tmp_path):
    path = tmp_path / "events.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE conversation_turns (
            chat_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            turn_sequence INTEGER NOT NULL,
            status TEXT NOT NULL,
            started_seq INTEGER NOT NULL,
            ended_seq INTEGER NOT NULL DEFAULT 0,
            terminal_event_id TEXT NOT NULL DEFAULT '',
            source_date TEXT NOT NULL,
            event_count INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (chat_id, turn_id),
            UNIQUE (chat_id, turn_sequence),
            UNIQUE (turn_id)
        );
        CREATE TABLE conversation_event_log_schema (version INTEGER NOT NULL);
        INSERT INTO conversation_event_log_schema(version) VALUES (1);
        INSERT INTO conversation_turns
            (chat_id, turn_id, turn_sequence, status, started_seq,
             source_date, created_at, updated_at)
        VALUES ('chat', 'legacy-turn', 1, 'completed', 1,
                '2026-01-01', 1, 1);
        """)
    conn.commit()
    conn.close()

    log = ConversationEventLog(str(path))
    turns = await log.snapshot_turns("chat", include_internal=True)

    assert turns.turns[0].turn_kind is TurnKind.UNKNOWN
    await log.append_user_message(
        chat_id="other-chat",
        turn_id="legacy-turn",
        message_id="other-message",
        content="same turn id in another chat",
    )
    await log.close()


@pytest.mark.asyncio
async def test_turn_page_can_exclude_prompt_hidden_events(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="message-1", content="问题"
    )
    await log.append_accepted_delivery(
        chat_id="chat", turn_id="turn-1", delivery_id="delivery-1", content="回答"
    )

    page = await log.snapshot_turn_page(
        "chat",
        include_internal=False,
        exclude_event_ids=("user:message-1",),
    )

    assert page.total_turns == 1
    assert [event.event_id for event in page.events] == ["delivery:delivery-1"]


@pytest.mark.asyncio
async def test_turn_page_can_select_archived_turns_by_event_ids(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    for index in range(1, 3):
        await log.append_user_message(
            chat_id="chat",
            turn_id=f"turn-{index}",
            message_id=f"message-{index}",
            content=f"问题 {index}",
        )
        await log.append_accepted_delivery(
            chat_id="chat",
            turn_id=f"turn-{index}",
            delivery_id=f"delivery-{index}",
            content=f"回答 {index}",
        )

    page = await log.snapshot_turn_page(
        "chat",
        include_internal=True,
        include_event_ids=("user:message-1",),
    )

    assert page.total_turns == 1
    assert [turn.turn_id for turn in page.turns] == ["turn-1"]
    assert {event.event_id for event in page.events} == {
        "user:message-1",
        "delivery:delivery-1",
    }


@pytest.mark.asyncio
async def test_session_summary_uses_aggregate_metadata_without_event_bodies(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="message-1", content="问题"
    )
    await log.append_accepted_delivery(
        chat_id="chat", turn_id="turn-1", delivery_id="delivery-1", content="回答"
    )
    log.snapshot_events = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("session summary must not load event bodies")
    )

    summary = await log.session_summary("chat")

    assert summary["message_count"] == 2
    assert summary["event_count"] == 2
    assert summary["protocol_count"] == 0
    assert summary["wire_count"] == 0
    await log.close()


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
async def test_open_turn_with_missing_tool_result_is_not_valid_for_completion(tmp_path):
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

    report = await log.validate_turn("turn-1")

    assert report.valid is False
    assert report.reason == "missing_tool_result"
    with pytest.raises(EventLogInvariantError, match="invalid turn"):
        await log.append_turn_terminal(chat_id="chat", turn_id="turn-1")


@pytest.mark.asyncio
async def test_turn_integrity_rejects_corrupt_persisted_metadata(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="message-1", content="问题"
    )
    await log.append_turn_terminal(chat_id="chat", turn_id="turn-1")

    conn = await log._ensure_open()
    conn.execute(
        "UPDATE conversation_turns SET event_count = event_count + 1 "
        "WHERE turn_id = ?",
        ("turn-1",),
    )
    conn.commit()

    report = await log.validate_turn("turn-1")

    assert report.valid is False
    assert report.reason == "event_count_mismatch"
    await log.close()


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

    first = await log.repair_from_legacy_history(
        "chat", history, session_kind="private"
    )
    second = await log.repair_from_legacy_history(
        "chat", history, session_kind="private"
    )
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
    assert all(event.session_kind == "private" for event in events.events)


@pytest.mark.asyncio
async def test_legacy_repair_does_not_reclose_turn_from_second_source(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    active = [
        {
            "role": "user",
            "content": "任务",
            "message_id": "turn-1",
            "timestamp": 1,
        },
        {"role": "assistant", "content": "完成", "timestamp": 2},
    ]
    archive = [
        {
            "role": "user",
            "content": "任务",
            "message_id": "turn-1",
            "timestamp": 1,
        },
        {"role": "assistant", "content": "完成", "timestamp": 2},
    ]

    assert (
        await log.repair_from_legacy_history("chat", active, source_id="legacy-active")
        == 2
    )
    assert (
        await log.repair_from_legacy_history(
            "chat", archive, source_id="legacy-archive"
        )
        == 0
    )

    events = await log.snapshot_events("chat", include_internal=True)
    assert [event.kind for event in events.events] == [
        EventKind.USER_MESSAGE,
        EventKind.ACCEPTED_DELIVERY,
        EventKind.TURN_TERMINAL,
    ]


@pytest.mark.asyncio
async def test_legacy_repair_serializes_concurrent_retries(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    history = [
        {"role": "user", "content": "并发任务", "message_id": "m1", "timestamp": 1},
        {"role": "assistant", "content": "完成", "timestamp": 2},
    ]

    imported = await asyncio.gather(
        log.repair_from_legacy_history("chat", history, source_id="legacy-active"),
        log.repair_from_legacy_history("chat", history, source_id="legacy-active"),
    )

    assert sorted(imported) == [0, 2]
    events = (await log.snapshot_events("chat", include_internal=True)).events
    assert [event.kind for event in events] == [
        EventKind.USER_MESSAGE,
        EventKind.ACCEPTED_DELIVERY,
        EventKind.TURN_TERMINAL,
    ]


@pytest.mark.asyncio
async def test_legacy_repair_handles_multiple_assistant_records_without_terminal_append_error(
    tmp_path,
):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))

    imported = await log.repair_from_legacy_history(
        "chat",
        [
            {"role": "user", "content": "问题", "message_id": "m1", "timestamp": 1},
            {"role": "assistant", "content": "第一段", "timestamp": 2},
            {"role": "assistant", "content": "第二段", "timestamp": 3},
        ],
        source_id="legacy-active",
    )

    assert imported == 3
    turns = await log.snapshot_turns("chat", include_internal=True)
    assert len(turns.turns) == 2
    assert all(turn.is_terminal for turn in turns.turns)


@pytest.mark.asyncio
async def test_legacy_repair_does_not_rescan_turns_for_every_record(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    original_snapshot_turns = log.snapshot_turns
    snapshot_calls = 0

    async def counted_snapshot_turns(*args, **kwargs):
        nonlocal snapshot_calls
        snapshot_calls += 1
        return await original_snapshot_turns(*args, **kwargs)

    log.snapshot_turns = counted_snapshot_turns

    imported = await log.repair_from_legacy_history(
        "chat",
        [
            {"role": "user", "content": "问题", "message_id": "m1", "timestamp": 1},
            {"role": "assistant", "content": "第一段", "timestamp": 2},
            {"role": "assistant", "content": "第二段", "timestamp": 3},
        ],
        source_id="legacy-active",
    )

    assert imported == 3
    assert snapshot_calls == 1


@pytest.mark.asyncio
async def test_legacy_repair_marks_cutoff_tool_turn_incomplete(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))

    await log.repair_from_legacy_history(
        "chat",
        [
            {
                "role": "user",
                "content": "需要工具",
                "message_id": "m1",
                "timestamp": 1,
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "function": {"name": "lookup"}}],
                "timestamp": 2,
            },
            {"role": "user", "content": "下一条", "message_id": "m2", "timestamp": 3},
        ],
        source_id="legacy-active",
    )

    turns = await log.snapshot_turns("chat", include_internal=True)

    assert len(turns.turns) == 2
    assert turns.turns[0].status.value == "incomplete"
    assert turns.turns[1].status.value == "open"


@pytest.mark.asyncio
async def test_repair_revision_is_append_only_and_idempotent(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat",
        turn_id="broken-turn",
        message_id="m1",
        content="需要工具",
    )
    await log.append_event(
        ConversationEvent(
            chat_id="chat",
            turn_id="broken-turn",
            event_id="call-event",
            role="assistant",
            kind=EventKind.ASSISTANT_TOOL_CALL,
            tool_calls=({"id": "call-1", "function": {"name": "lookup"}},),
        )
    )
    await log.append_turn_terminal(
        chat_id="chat", turn_id="broken-turn", status=TurnStatus.INCOMPLETE
    )

    first = await log.append_repair_revision(
        chat_id="chat",
        original_turn_id="broken-turn",
        revision_id="rev-1",
        reason="外部工具已在旧系统完成，无法补录原始结果",
        operator="admin",
    )
    second = await log.append_repair_revision(
        chat_id="chat",
        original_turn_id="broken-turn",
        revision_id="rev-1",
        reason="外部工具已在旧系统完成，无法补录原始结果",
        operator="admin",
    )

    original = await log.validate_turn("broken-turn")
    revisions = await log.repair_revisions("chat", original_turn_id="broken-turn")
    assert not original.valid
    assert original.reason == "missing_tool_result"
    assert first == second
    assert len(revisions) == 1
    assert revisions[0].revision_turn_id == first.revision_turn_id
    assert revisions[0].original_reason == "missing_tool_result"
    assert revisions[0].reason == "外部工具已在旧系统完成，无法补录原始结果"
    revision_events = [
        event
        for event in (await log.snapshot_events("chat", include_internal=True)).events
        if event.turn_id == first.revision_turn_id
    ]
    assert all(event.session_kind == "repair" for event in revision_events)


@pytest.mark.asyncio
async def test_repair_revision_rejects_valid_or_cross_chat_turn(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="valid-turn", message_id="m1", content="ok"
    )
    await log.append_turn_terminal(chat_id="chat", turn_id="valid-turn")

    with pytest.raises(EventLogInvariantError, match="requires invalid turn"):
        await log.append_repair_revision(
            chat_id="chat",
            original_turn_id="valid-turn",
            revision_id="rev-1",
            reason="不应记录",
        )
    with pytest.raises(EventLogInvariantError, match="chat mismatch"):
        await log.append_repair_revision(
            chat_id="other",
            original_turn_id="valid-turn",
            revision_id="rev-2",
            reason="跨会话不应记录",
        )


@pytest.mark.asyncio
async def test_legacy_repair_strips_nested_display_prefixes(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))

    await log.repair_from_legacy_history(
        "chat",
        [
            {
                "role": "user",
                "content": (
                    "[用户 在 2026-07-13 17:14:22]: "
                    "[用户 在 2026-07-13 17:14:22]: 原始内容"
                ),
                "message_id": "m1",
                "timestamp": 1,
            }
        ],
        source_id="legacy-active",
    )

    event = (await log.snapshot_events("chat")).events[0]
    assert event.content == "原始内容"
    assert (await log.history("chat"))[0]["content"] == "原始内容"


@pytest.mark.asyncio
async def test_validate_turns_matches_single_turn_validation(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    for index in range(3):
        turn_id = f"turn-{index}"
        await log.append_user_message(
            chat_id="chat",
            turn_id=turn_id,
            message_id=f"message-{index}",
            content=f"question-{index}",
        )
        await log.append_turn_terminal(chat_id="chat", turn_id=turn_id)

    batch = await log.validate_turns(["turn-0", "turn-1", "turn-0", "missing"])

    assert set(batch) == {"turn-0", "turn-1", "missing"}
    assert batch["turn-0"] == await log.validate_turn("turn-0")
    assert batch["turn-1"] == await log.validate_turn("turn-1")
    assert batch["missing"].reason == "turn_not_found"


@pytest.mark.asyncio
async def test_integrity_summary_is_content_free_and_counts_invalid_turns(tmp_path):
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

    summary = await log.integrity_summary("chat")

    assert summary == {
        "turn_count": 1,
        "invalid_turn_count": 1,
        "incomplete_turn_count": 0,
        "open_turn_count": 0,
        "waiting_tool_turn_count": 1,
        "invalid_reasons": {"missing_tool_result": 1},
    }
    assert "content" not in str(summary)


@pytest.mark.asyncio
async def test_validate_turn_reports_corrupt_persisted_status(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="message-1", content="问题"
    )
    conn = await log._ensure_open()
    conn.execute(
        "UPDATE conversation_turns SET status = ? WHERE turn_id = ?",
        ("corrupt", "turn-1"),
    )
    conn.commit()

    report = await log.validate_turn("turn-1")

    assert report.valid is False
    assert report.reason == "invalid_turn_status"
    assert report.status == "corrupt"
    await log.close()


@pytest.mark.asyncio
async def test_validate_turn_reports_corrupt_persisted_metadata(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="message-1", content="问题"
    )
    conn = await log._ensure_open()
    conn.execute(
        "UPDATE conversation_turns SET event_count = ? WHERE turn_id = ?",
        ("not-a-number", "turn-1"),
    )
    conn.commit()

    report = await log.validate_turn("turn-1")

    assert report.valid is False
    assert report.reason == "persisted_turn_metadata_invalid"
    await log.close()


@pytest.mark.asyncio
async def test_validate_turn_rejects_empty_persisted_turn(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    conn = await log._ensure_open()
    conn.execute(
        "INSERT INTO conversation_turns "
        "(chat_id, turn_id, turn_sequence, status, started_seq, turn_kind, "
        "source_date, event_count, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("chat", "turn-empty", 1, "open", 0, "unknown", "2026-09-04", 0, 1, 1),
    )
    conn.commit()

    report = await log.validate_turn("turn-empty")

    assert report.valid is False
    assert report.reason == "empty_turn"
    await log.close()


@pytest.mark.asyncio
async def test_validate_turn_rejects_turn_sequence_index_mismatch(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="message-1", content="问题"
    )
    conn = await log._ensure_open()
    conn.execute(
        "UPDATE conversation_turns SET turn_sequence = turn_sequence + 1 "
        "WHERE turn_id = ?",
        ("turn-1",),
    )
    conn.commit()

    report = await log.validate_turn("turn-1")

    assert report.valid is False
    assert report.reason == "turn_sequence_index_mismatch"
    await log.close()


@pytest.mark.asyncio
async def test_validate_turn_rejects_source_date_mismatch(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="message-1", content="问题"
    )
    conn = await log._ensure_open()
    conn.execute(
        "UPDATE conversation_events SET source_date = ? WHERE turn_id = ?",
        ("2000-01-01", "turn-1"),
    )
    conn.commit()

    report = await log.validate_turn("turn-1")

    assert report.valid is False
    assert report.reason == "source_date_mismatch"
    await log.close()
