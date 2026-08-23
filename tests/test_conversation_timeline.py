import pytest

from core.engine.conversation_timeline import ConversationTimeline


@pytest.mark.asyncio
async def test_timeline_assigns_per_chat_sequence_and_is_idempotent(tmp_path):
    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))

    first = await timeline.append_user_message(
        chat_id="chat",
        message_id="m1",
        content="hello",
        sender_id="user-1",
        timestamp=10,
    )
    duplicate = await timeline.append_user_message(
        chat_id="chat",
        message_id="m1",
        content="changed",
        sender_id="user-1",
        timestamp=11,
    )
    second = await timeline.append_accepted_delivery(
        chat_id="chat",
        delivery_id="d1",
        content="answer",
        delivery_kind="response",
    )
    other = await timeline.append_user_message(
        chat_id="other",
        message_id="m2",
        content="separate",
        sender_id="user-2",
        timestamp=12,
    )

    assert first.seq == 1
    assert duplicate == first
    assert second.seq == 2
    assert other.seq == 1
    assert await timeline.latest_seq("chat") == 2
    assert await timeline.chat_ids() == ["chat", "other"]
    await timeline.close()


@pytest.mark.asyncio
async def test_timeline_snapshot_can_freeze_before_later_delivery(tmp_path):
    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    await timeline.append_user_message(
        chat_id="chat",
        message_id="m1",
        content="question",
        sender_id="user",
        timestamp=1,
    )
    frozen_seq = await timeline.latest_seq("chat")
    await timeline.append_accepted_delivery(
        chat_id="chat",
        delivery_id="d1",
        content="later answer",
        delivery_kind="response",
    )

    snapshot = await timeline.snapshot("chat", upto_seq=frozen_seq)

    assert [event.content for event in snapshot] == ["question"]
    assert snapshot[0].event_kind == "user_message"
    await timeline.close()


@pytest.mark.asyncio
async def test_timeline_migration_report_is_content_free_and_requires_protocol_cutover(
    tmp_path,
):
    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    await timeline.append_user_message(
        chat_id="chat",
        message_id="m1",
        content="hello",
        sender_id="user",
        timestamp=1,
    )
    await timeline.append_accepted_delivery(
        chat_id="chat",
        delivery_id="d1",
        content="answer",
        delivery_kind="response",
        timestamp=2,
    )

    report = await timeline.migration_report(
        "chat",
        [
            {"role": "user", "raw_content": "hello"},
            {"role": "assistant", "content": "answer"},
            {
                "role": "assistant",
                "content": "internal",
                "tool_calls": [{"id": "call-1"}],
            },
            {"role": "tool", "content": "result", "tool_call_id": "call-1"},
        ],
    )

    assert report.to_dict() == {
        "chat_id": "chat",
        "timeline_visible_count": 2,
        "legacy_visible_count": 2,
        "missing_legacy_visible_count": 0,
        "extra_timeline_visible_count": 0,
        "legacy_protocol_count": 2,
        "ready_for_legacy_read_removal": False,
    }
    assert "hello" not in str(report.to_dict())
    await timeline.close()


@pytest.mark.asyncio
async def test_timeline_migration_report_marks_complete_visible_history_ready(tmp_path):
    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    await timeline.append_user_message(
        chat_id="chat",
        message_id="m1",
        content="hello",
        sender_id="user",
        timestamp=1,
    )

    report = await timeline.migration_report(
        "chat", [{"role": "user", "raw_content": "hello"}]
    )

    assert report.ready_for_legacy_read_removal is True
    await timeline.close()


def test_timeline_migration_summary_counts_incomplete_sessions_without_content():
    from core.engine.conversation_timeline import TimelineMigrationReport

    reports = [
        TimelineMigrationReport(
            chat_id="ready",
            timeline_visible_count=1,
            legacy_visible_count=1,
            missing_legacy_visible_count=0,
            extra_timeline_visible_count=0,
            legacy_protocol_count=0,
        ),
        TimelineMigrationReport(
            chat_id="legacy",
            timeline_visible_count=1,
            legacy_visible_count=2,
            missing_legacy_visible_count=1,
            extra_timeline_visible_count=0,
            legacy_protocol_count=2,
        ),
    ]

    summary = ConversationTimeline.migration_summary(reports, scan_errors=1)

    assert summary.to_dict() == {
        "session_count": 3,
        "sessions_with_missing_legacy_visible": 1,
        "sessions_with_legacy_protocol": 1,
        "sessions_ready_for_legacy_read_removal": 1,
        "sessions_with_scan_errors": 1,
        "ready_for_legacy_read_removal": False,
    }
    assert "message body" not in str(summary.to_dict())


@pytest.mark.asyncio
async def test_timeline_does_not_share_event_keys_between_chats(tmp_path):
    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    first = await timeline.append(
        chat_id="chat-a",
        event_id="same",
        role="assistant",
        content="a",
        event_kind="delivery",
        delivery_kind="response",
    )
    second = await timeline.append(
        chat_id="chat-b",
        event_id="same",
        role="assistant",
        content="b",
        event_kind="delivery",
        delivery_kind="response",
    )

    assert first.content == "a"
    assert second.content == "b"
    await timeline.close()


@pytest.mark.asyncio
async def test_timeline_history_zero_limit_is_empty(tmp_path):
    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    await timeline.append_user_message(
        chat_id="chat",
        message_id="m1",
        content="hello",
        sender_id="user",
        timestamp=1,
    )

    assert await timeline.history("chat", max_events=0) == []
    await timeline.close()


@pytest.mark.asyncio
async def test_timeline_session_summary_aggregates_without_content(tmp_path):
    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    await timeline.append_user_message(
        chat_id="chat",
        message_id="m1",
        content="12345678",
        sender_id="user",
        timestamp=10,
    )
    await timeline.append_accepted_delivery(
        chat_id="chat",
        delivery_id="d1",
        content="abcd",
        delivery_kind="response",
        timestamp=20,
    )

    summary = await timeline.session_summary("chat")

    assert summary == {
        "message_count": 2,
        "last_activity": 20.0,
        "estimated_tokens": 3,
    }
    await timeline.close()


@pytest.mark.asyncio
async def test_timeline_repairs_visible_legacy_history_without_tool_protocol(tmp_path):
    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))

    migrated = await timeline.repair_from_legacy_history(
        "chat",
        [
            {
                "role": "user",
                "raw_content": "old question",
                "message_id": "m1",
                "sender_id": "user-1",
                "timestamp": 10,
            },
            {
                "role": "assistant",
                "content": "old answer",
                "message_id": "m1",
                "timestamp": 11,
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1"}],
                "timestamp": 12,
            },
            {
                "role": "tool",
                "content": "tool result",
                "tool_call_id": "call-1",
                "timestamp": 13,
            },
        ],
    )

    assert migrated == 2
    history = await timeline.history("chat")
    assert [item["content"] for item in history] == ["old question", "old answer"]
    assert [item["role"] for item in history] == ["user", "assistant"]
    assert (
        await timeline.repair_from_legacy_history(
            "chat", [{"role": "user", "message_id": "m1", "content": "changed"}]
        )
        == 0
    )
    assert (
        await timeline.repair_from_legacy_history(
            "chat",
            [{"role": "user", "message_id": "m2", "content": "new question"}],
        )
        == 1
    )
    await timeline.close()


@pytest.mark.asyncio
async def test_timeline_repair_does_not_duplicate_accepted_assistant_content(tmp_path):
    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    await timeline.append_accepted_delivery(
        chat_id="chat",
        delivery_id="delivery-1",
        content="same answer",
        delivery_kind="response",
    )

    migrated = await timeline.repair_from_legacy_history(
        "chat", [{"role": "assistant", "content": "same answer"}]
    )

    assert migrated == 0
    history = await timeline.history("chat")
    assert [item["event_id"] for item in history] == ["delivery:delivery-1"]
    await timeline.close()
