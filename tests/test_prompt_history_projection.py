import sqlite3

import pytest

from core.engine.conversation_event_log import (
    ConversationEvent,
    ConversationEventLog,
    EventKind,
)
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
async def test_prompt_projection_selects_turns_before_loading_event_bodies(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    for index in range(40):
        turn_id = f"turn-{index}"
        await log.append_user_message(
            chat_id="chat",
            turn_id=turn_id,
            message_id=f"message-{index}",
            content=f"历史消息 {index}",
            timestamp=index + 1,
        )
        await log.append_accepted_delivery(
            chat_id="chat",
            turn_id=turn_id,
            delivery_id=f"delivery-{index}",
            content=f"历史回答 {index}",
            timestamp=index + 1.5,
        )
        await log.append_turn_terminal(
            chat_id="chat", turn_id=turn_id, timestamp=index + 2
        )

    original_snapshot_events = log.snapshot_events
    snapshot_calls = []

    async def tracked_snapshot_events(*args, **kwargs):
        snapshot_calls.append(kwargs)
        return await original_snapshot_events(*args, **kwargs)

    log.snapshot_events = tracked_snapshot_events
    snapshot = await PromptHistoryProjection(
        log, max_tokens=12, max_turns=2
    ).snapshot_for_prompt("chat")

    assert {event.turn_id for event in snapshot.events} == {"turn-39", "turn-38"}
    assert snapshot_calls
    assert all(call.get("turn_ids") for call in snapshot_calls)
    await log.close()


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
    await log.append_turn_terminal(chat_id="chat", turn_id="turn-2")

    snapshot = await PromptHistoryProjection(
        log, max_tokens=100, max_turns=1
    ).snapshot_for_prompt("chat", current_turn_id="turn-1")

    assert {event.turn_id for event in snapshot.events} == {"turn-1", "turn-2"}


@pytest.mark.asyncio
async def test_prompt_projection_excludes_incomplete_historical_turn(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat",
        turn_id="completed-turn",
        message_id="completed-message",
        content="completed",
    )
    await log.append_turn_terminal(chat_id="chat", turn_id="completed-turn")
    await log.append_user_message(
        chat_id="chat",
        turn_id="incomplete-turn",
        message_id="incomplete-message",
        content="incomplete",
    )

    snapshot = await PromptHistoryProjection(
        log, max_tokens=100, max_turns=10
    ).snapshot_for_prompt("chat")

    assert [event.turn_id for event in snapshot.events] == ["completed-turn"]
    await log.close()


@pytest.mark.asyncio
async def test_prompt_projection_excludes_invalid_terminal_historical_turn(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat",
        turn_id="invalid-turn",
        message_id="invalid-message",
        content="invalid",
    )
    await log.append_turn_terminal(chat_id="chat", turn_id="invalid-turn")
    await log.append_user_message(
        chat_id="chat",
        turn_id="valid-turn",
        message_id="valid-message",
        content="valid",
    )
    await log.append_turn_terminal(chat_id="chat", turn_id="valid-turn")
    conn = await log._ensure_open()
    conn.execute(
        "UPDATE conversation_turns SET event_count = event_count + 1 "
        "WHERE turn_id = ?",
        ("invalid-turn",),
    )
    conn.commit()

    snapshot = await PromptHistoryProjection(
        log, max_tokens=100, max_turns=10
    ).snapshot_for_prompt("chat")

    assert [event.turn_id for event in snapshot.events] == ["valid-turn"]
    assert snapshot.degraded_reason == "invalid_historical_turn_excluded"
    await log.close()


@pytest.mark.asyncio
async def test_prompt_projection_refills_window_after_invalid_turn(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    for turn_id in ("old-turn", "invalid-turn", "new-turn"):
        await log.append_user_message(
            chat_id="chat",
            turn_id=turn_id,
            message_id=f"message-{turn_id}",
            content=turn_id,
        )
        await log.append_turn_terminal(chat_id="chat", turn_id=turn_id)
    conn = await log._ensure_open()
    conn.execute(
        "UPDATE conversation_turns SET event_count = event_count + 1 "
        "WHERE turn_id = ?",
        ("invalid-turn",),
    )
    conn.commit()

    snapshot = await PromptHistoryProjection(
        log, max_tokens=100, max_turns=2
    ).snapshot_for_prompt("chat")

    assert [event.turn_id for event in snapshot.events] == ["old-turn", "new-turn"]
    await log.close()


@pytest.mark.asyncio
async def test_prompt_projection_skips_invalid_turns_during_refill(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    for turn_id in ("old-turn", "invalid-old", "invalid-new", "new-turn"):
        await log.append_user_message(
            chat_id="chat",
            turn_id=turn_id,
            message_id=f"message-{turn_id}",
            content=turn_id,
        )
        await log.append_turn_terminal(chat_id="chat", turn_id=turn_id)
    conn = await log._ensure_open()
    conn.execute(
        "UPDATE conversation_turns SET event_count = event_count + 1 "
        "WHERE turn_id IN (?, ?)",
        ("invalid-old", "invalid-new"),
    )
    conn.commit()

    snapshot = await PromptHistoryProjection(
        log, max_tokens=100, max_turns=2
    ).snapshot_for_prompt("chat")

    assert [event.turn_id for event in snapshot.events] == ["old-turn", "new-turn"]
    assert snapshot.degraded_reason == "invalid_historical_turn_excluded"
    await log.close()


@pytest.mark.asyncio
async def test_prompt_projection_repair_voices_are_not_visible(tmp_path):
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
        chat_id="chat", turn_id="broken-turn", status="incomplete"
    )
    await log.append_repair_revision(
        chat_id="chat",
        original_turn_id="broken-turn",
        revision_id="rev-1",
        reason="保留异常",
    )
    projection = PromptHistoryProjection(
        log, metadata_path=str(tmp_path / "projection.sqlite3")
    )

    report = await projection.repair_chat("chat")
    visible_ids = await projection.visibility_event_ids("chat")

    assert report.source_event_count == 3
    assert visible_ids == {"user:m1", "call-event", "terminal:broken-turn"}


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
    await log.append_turn_terminal(chat_id="chat", turn_id="turn-1")
    await log.append_turn_terminal(chat_id="chat", turn_id="turn-2")
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
    with pytest.raises(
        RuntimeError, match="archive projection membership changed at watermark"
    ):
        await projection.apply_archive_retention(
            "chat", operation_id="stale-operation", retained_event_ids=("user:m1",)
        )

    assert await projection.hidden_event_ids("chat") == {"user:m1"}


@pytest.mark.asyncio
async def test_archive_retention_same_watermark_same_membership_is_idempotent_but_conflict_fails(
    tmp_path,
):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="m1", content="one"
    )
    await log.append_user_message(
        chat_id="chat", turn_id="turn-2", message_id="m2", content="two"
    )
    cutoff = await log.latest_event_seq("chat")
    projection = PromptHistoryProjection(
        log, metadata_path=str(tmp_path / "projection.sqlite3")
    )

    await projection.apply_archive_retention(
        "chat",
        operation_id="archive-1",
        captured_cutoff_seq=cutoff,
        hidden_event_ids=("user:m1",),
    )
    await projection.apply_archive_retention(
        "chat",
        operation_id="archive-1-replay",
        captured_cutoff_seq=cutoff,
        hidden_event_ids=("user:m1",),
    )

    with pytest.raises(
        RuntimeError, match="archive projection membership changed at watermark"
    ):
        await projection.apply_archive_retention(
            "chat",
            operation_id="archive-conflict",
            captured_cutoff_seq=cutoff,
            hidden_event_ids=("user:m2",),
        )

    assert await projection.hidden_event_ids("chat") == {"user:m1"}
    await projection.close()
    await log.close()


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
async def test_archive_retention_validates_captured_ledger_source_hash(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn", message_id="message", content="one"
    )
    projection = PromptHistoryProjection(
        log, metadata_path=str(tmp_path / "projection.sqlite3")
    )

    with pytest.raises(ValueError, match="archive ledger source hash mismatch"):
        await projection.apply_archive_retention(
            "chat",
            operation_id="archive-invalid-source",
            hidden_event_ids=("user:message",),
            captured_source_hash="not-the-ledger-snapshot",
        )

    await projection.close()
    await log.close()


@pytest.mark.asyncio
async def test_archive_retention_rejects_unknown_ledger_event_identity(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn", message_id="message", content="one"
    )
    projection = PromptHistoryProjection(
        log, metadata_path=str(tmp_path / "projection.sqlite3")
    )

    with pytest.raises(ValueError, match="unknown ledger events"):
        await projection.apply_archive_retention(
            "chat", operation_id="archive-unknown", hidden_event_ids=("missing",)
        )

    assert await projection.visibility_event_ids("chat") == set()
    await projection.close()
    await log.close()


@pytest.mark.asyncio
async def test_prompt_projection_repair_chat_adds_explicit_visibility_rows(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn", message_id="message", content="hello"
    )
    projection = PromptHistoryProjection(
        log, metadata_path=str(tmp_path / "projection.sqlite3")
    )

    first = await projection.repair_chat("chat")
    second = await projection.repair_chat("chat")

    assert first.inserted_event_count == 1
    assert second.inserted_event_count == 0
    assert await projection.visibility_event_ids("chat") == {"user:message"}


@pytest.mark.asyncio
async def test_prompt_projection_repair_chat_ignores_background_task_events(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="task:1",
        turn_id="turn",
        message_id="message",
        content="background",
        session_kind="chat",
    )
    projection = PromptHistoryProjection(
        log, metadata_path=str(tmp_path / "projection.sqlite3")
    )

    report = await projection.repair_chat("task:1")

    assert report.source_event_count == 0
    assert report.inserted_event_count == 0


@pytest.mark.asyncio
async def test_prompt_projection_repair_chat_includes_private_events(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="private-chat",
        turn_id="turn",
        message_id="message",
        content="private",
        session_kind="private",
    )
    projection = PromptHistoryProjection(
        log, metadata_path=str(tmp_path / "projection.sqlite3")
    )

    report = await projection.repair_chat("private-chat")

    assert report.source_event_count == 1
    assert report.inserted_event_count == 1
    assert await projection.visibility_event_ids("private-chat") == {"user:message"}


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


@pytest.mark.asyncio
async def test_prompt_projection_repair_chat_reads_identities_only(tmp_path, monkeypatch):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await log.append_user_message(
        chat_id="chat", turn_id="turn", message_id="message", content="hello"
    )
    projection = PromptHistoryProjection(
        log, metadata_path=str(tmp_path / "projection.sqlite3")
    )

    async def reject_body_snapshot(*args, **kwargs):
        raise AssertionError("prompt repair must not load event bodies")

    monkeypatch.setattr(log, "snapshot_events", reject_body_snapshot)

    report = await projection.repair_chat("chat")

    assert report.inserted_event_count == 1
    await projection.close()
    await log.close()


@pytest.mark.asyncio
async def test_prompt_projection_upgrades_legacy_metadata_schema(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    metadata_path = tmp_path / "projection.sqlite3"
    connection = sqlite3.connect(metadata_path)
    connection.executescript(
        """
        CREATE TABLE prompt_archive_operations (
            operation_id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL,
            cutoff_seq INTEGER NOT NULL,
            projection_version INTEGER NOT NULL,
            source_hash TEXT NOT NULL,
            hidden_ids TEXT NOT NULL,
            retained_ids TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE prompt_event_visibility (
            chat_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            prompt_visible INTEGER NOT NULL,
            storage_tier TEXT NOT NULL,
            archive_batch_id TEXT NOT NULL DEFAULT '',
            operation_id TEXT NOT NULL,
            PRIMARY KEY (chat_id, event_id)
        );
        CREATE TABLE prompt_projection_watermarks (
            chat_id TEXT PRIMARY KEY,
            cutoff_seq INTEGER NOT NULL,
            projection_version INTEGER NOT NULL,
            source_hash TEXT NOT NULL,
            operation_id TEXT NOT NULL
        );
        """
    )
    connection.commit()
    connection.close()

    projection = PromptHistoryProjection(log, metadata_path=str(metadata_path))
    await projection.hidden_event_ids("chat")
    await projection.close()


    await log.close()

    connection = sqlite3.connect(metadata_path)
    for table in ("prompt_archive_operations", "prompt_projection_watermarks"):
        columns = {
            row[1] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        assert "ledger_source_hash" in columns
    connection.close()


@pytest.mark.asyncio
async def test_prompt_projection_status_reports_visibility_lag(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    await log.append_user_message(
        chat_id="chat", turn_id="turn", message_id="message", content="hello"
    )
    await projection.repair_chat("chat")
    await log.append_user_message(
        chat_id="chat", turn_id="turn-2", message_id="message-2", content="later"
    )

    status = await projection.status("chat")

    assert status["visibility_count"] == 1
    assert status["visible_count"] == 1
    assert status["hidden_count"] == 0
    assert status["watermark_count"] == 1
    assert status["projection_lag"] > 0
    await log.close()
    await projection.close()
