"""测试 ArchiveManager 内部逻辑：_extract_replay_messages、_build_summary_group、
consume_summary、archive_if_stale 跨天触发与同日守卫。"""

import asyncio
import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.command_handlers.archive import ArchiveCommand
from core.engine.archive_index import ArchiveIndex
from core.engine.conversation_event_log import (
    ConversationEvent,
    ConversationEventLog,
    EventKind,
)
from core.engine.prompt_history_projection import PromptHistoryProjection
from core.engine.turn_summary import TurnSummaryStore
from core.managers.archive_manager import (
    ArchiveManager,
    ArchiveResult,
    ArchiveTurnRecord,
)
from core.managers.archive_manifest import ArchiveManifestStore
from core.message import InputMessage

# ── helpers ──


def make_msg(role="user", content="hello", msg_id="m1", sender_id="u1", **kw):
    timestamp = kw.pop("timestamp", 100.0)
    return ChatMessage(
        role=role,
        content=content,
        timestamp=timestamp,
        message_id=msg_id,
        sender_id=sender_id,
        name=None,
        **kw,
    )


async def _append_completed_turn(event_log, chat_id, turn_id, timestamp, content="x"):
    await event_log.append_user_message(
        chat_id=chat_id,
        turn_id=turn_id,
        message_id=f"message-{turn_id}",
        content=content,
        timestamp=timestamp,
    )
    await event_log.append_accepted_delivery(
        chat_id=chat_id,
        turn_id=turn_id,
        delivery_id=f"delivery-{turn_id}",
        content=f"answer-{turn_id}",
        timestamp=timestamp + 1,
    )
    await event_log.append_turn_terminal(
        chat_id=chat_id,
        turn_id=turn_id,
        timestamp=timestamp + 2,
    )


async def _make_event_archive_repair_manager(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    summary_store = TurnSummaryStore(
        event_log, path=str(tmp_path / "summaries.sqlite3")
    )
    archive_index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    manager = ArchiveManager(
        context_manager=MagicMock(),
        memory_dir=str(tmp_path / "archives"),
        archive_index=archive_index,
    )
    manager.set_event_log(event_log, projection, summary_store)
    return manager, event_log, projection, archive_index


@pytest.mark.asyncio
async def test_event_archive_rejects_empty_batch(tmp_path):
    manager, event_log, projection, archive_index = (
        await _make_event_archive_repair_manager(tmp_path)
    )
    batch = await archive_index.prepare_batch(
        batch_id="batch:empty",
        operation_id="empty-operation",
        chat_id="chat",
        captured_cutoff_seq=0,
        turn_records=[],
        event_ids=[],
    )

    with pytest.raises(RuntimeError, match="empty archive batch"):
        await manager._finish_event_archive_batch(batch)

    await event_log.close()
    await projection.close()
    await archive_index.close()


@pytest.fixture
def mgr(tmp_path):
    cm = MagicMock()
    return ArchiveManager(
        context_manager=cm,
        memory_dir=str(tmp_path / "mem"),
        summary_count=200,
        merge_window_seconds=300,
    )


@pytest.mark.asyncio
async def test_event_archive_repair_does_not_split_turn_at_cutoff(tmp_path):
    manager, event_log, projection, archive_index = (
        await _make_event_archive_repair_manager(tmp_path)
    )
    await event_log.append_user_message(
        chat_id="chat",
        turn_id="turn",
        message_id="message",
        content="old",
        timestamp=1,
    )
    cutoff = await event_log.latest_event_seq("chat")
    await event_log.append_accepted_delivery(
        chat_id="chat",
        turn_id="turn",
        delivery_id="delivery",
        content="answer",
        timestamp=2,
    )
    await event_log.append_turn_terminal(chat_id="chat", turn_id="turn", timestamp=3)

    result = await manager.repair_event_log_archives(
        "chat", before_date="1970-01-02", captured_cutoff_seq=cutoff
    )

    assert result.batches == []
    assert result.skipped_turns[0]["reason"] == "cutoff_splits_turn"
    assert await archive_index.list_for_webui("chat", state="committed") == []
    assert await projection.hidden_event_ids("chat") == frozenset()


@pytest.mark.asyncio
async def test_event_archive_repair_validates_date(tmp_path):
    manager, _, _, _ = await _make_event_archive_repair_manager(tmp_path)

    with pytest.raises(ValueError, match="before_date must be YYYY-MM-DD"):
        await manager.repair_event_log_archives("chat", before_date="2026/09/04")


@pytest.mark.asyncio
async def test_event_archive_repair_is_idempotent_and_reports_invalid_turn(tmp_path):
    manager, event_log, projection, archive_index = (
        await _make_event_archive_repair_manager(tmp_path)
    )
    await _append_completed_turn(event_log, "chat", "complete", 1)
    await event_log.append_user_message(
        chat_id="chat",
        turn_id="incomplete",
        message_id="incomplete-message",
        content="pending",
        timestamp=10,
    )

    first = await manager.repair_event_log_archives("chat", before_date="1970-01-02")
    second = await manager.repair_event_log_archives("chat", before_date="1970-01-02")

    assert first.reason == "repair"
    assert first.batches and first.batches[0].event_count == 3
    assert second.batches == []
    assert second.skipped_turns[0]["reason"] == "incomplete_turn"
    assert len(await archive_index.list_for_webui("chat", state="committed")) == 1
    assert len(await projection.hidden_event_ids("chat")) == 3


@pytest.mark.asyncio
async def test_event_archive_repair_archives_hot_visibility_when_membership_missing(
    tmp_path,
):
    manager, event_log, projection, archive_index = (
        await _make_event_archive_repair_manager(tmp_path)
    )
    await _append_completed_turn(event_log, "chat", "hot-old", 1)
    await projection.repair_chat("chat")

    result = await manager.repair_event_log_archives("chat", before_date="1970-01-02")

    assert result.batches and result.batches[0].event_count == 3
    assert len(await archive_index.list_for_webui("chat", state="committed")) == 1
    assert len(await projection.hidden_event_ids("chat")) == 3


@pytest.mark.asyncio
async def test_event_archive_static_repair_reads_only_selected_turn_bodies(
    tmp_path, monkeypatch
):
    manager, event_log, projection, archive_index = (
        await _make_event_archive_repair_manager(tmp_path)
    )
    await _append_completed_turn(event_log, "chat", "turn-1", 1)
    original_snapshot_events = event_log.snapshot_events

    async def bounded_snapshot(*args, **kwargs):
        if kwargs.get("turn_ids") is None and kwargs.get("event_ids") is None:
            raise AssertionError("static repair must not load unbounded event bodies")
        return await original_snapshot_events(*args, **kwargs)

    monkeypatch.setattr(event_log, "snapshot_events", bounded_snapshot)

    result = await manager.repair_event_log_archives("chat", before_date="1970-01-02")

    assert result.batches and result.batches[0].event_count == 3
    await event_log.close()
    await projection.close()
    await archive_index.close()


@pytest.mark.asyncio
async def test_event_archive_repair_never_archives_explicit_incomplete_turn(tmp_path):
    manager, event_log, projection, archive_index = (
        await _make_event_archive_repair_manager(tmp_path)
    )
    await event_log.append_user_message(
        chat_id="chat",
        turn_id="incomplete",
        message_id="message-incomplete",
        content="pending",
        timestamp=1,
    )
    await event_log.append_turn_terminal(
        chat_id="chat", turn_id="incomplete", status="incomplete", timestamp=2
    )
    await projection.repair_chat("chat")

    result = await manager.repair_event_log_archives("chat", before_date="1970-01-02")

    assert result.batches == []
    assert result.skipped_turns[0]["reason"] == "incomplete_turn"
    assert await archive_index.list_for_webui("chat", state="committed") == []


@pytest.mark.asyncio
async def test_event_archive_rejects_partial_turn_batch(tmp_path):
    manager, event_log, projection, archive_index = (
        await _make_event_archive_repair_manager(tmp_path)
    )
    await _append_completed_turn(event_log, "chat", "turn-1", 1)
    snapshot = await event_log.snapshot_events("chat", include_internal=True)
    turn = (await event_log.snapshot_turns("chat", include_internal=True)).turns[0]
    batch = await archive_index.prepare_batch(
        batch_id="batch:partial",
        operation_id="partial",
        chat_id="chat",
        captured_cutoff_seq=snapshot.cutoff_seq,
        turn_records=[
            ArchiveTurnRecord(
                turn_id=turn.turn_id,
                turn_sequence=turn.turn_sequence,
                source_date=turn.source_date,
                event_count=1,
                estimated_tokens=1,
            )
        ],
        event_ids=[(snapshot.events[0].event_id, turn.turn_id)],
    )

    with pytest.raises(RuntimeError, match="partial turn"):
        await manager._finish_event_archive_batch(batch)
    assert await archive_index.list_for_webui("chat", state="committed") == []
    assert await projection.hidden_event_ids("chat") == frozenset()


@pytest.mark.asyncio
async def test_event_archive_rejects_partial_turn_after_prior_hidden_event(tmp_path):
    manager, event_log, projection, archive_index = (
        await _make_event_archive_repair_manager(tmp_path)
    )
    await _append_completed_turn(event_log, "chat", "turn-1", 1)
    snapshot = await event_log.snapshot_events("chat", include_internal=True)
    await projection.apply_archive_retention(
        "chat",
        operation_id="prior-hidden",
        hidden_event_ids=(snapshot.events[0].event_id,),
        captured_cutoff_seq=snapshot.cutoff_seq,
    )
    turn = (await event_log.snapshot_turns("chat", include_internal=True)).turns[0]
    remaining = [
        event
        for event in snapshot.events
        if event.event_id != snapshot.events[0].event_id
    ]
    batch = await archive_index.prepare_batch(
        batch_id="batch:partial-after-hidden",
        operation_id="partial-after-hidden",
        chat_id="chat",
        captured_cutoff_seq=snapshot.cutoff_seq,
        turn_records=[
            ArchiveTurnRecord(
                turn_id=turn.turn_id,
                turn_sequence=turn.turn_sequence,
                source_date=turn.source_date,
                event_count=len(remaining),
                estimated_tokens=1,
            )
        ],
        event_ids=[(event.event_id, turn.turn_id) for event in remaining],
    )

    with pytest.raises(RuntimeError, match="partial turn"):
        await manager._finish_event_archive_batch(batch)


@pytest.mark.asyncio
async def test_event_archive_rejects_unknown_event_membership(tmp_path):
    manager, event_log, projection, archive_index = (
        await _make_event_archive_repair_manager(tmp_path)
    )
    await _append_completed_turn(event_log, "chat", "turn-1", 1)
    batch = await archive_index.prepare_batch(
        batch_id="batch:unknown",
        operation_id="unknown",
        chat_id="chat",
        captured_cutoff_seq=await event_log.latest_event_seq("chat"),
        turn_records=[],
        event_ids=[("missing-event", "missing-turn")],
    )

    with pytest.raises(RuntimeError, match="missing from ledger"):
        await manager._finish_event_archive_batch(batch)


@pytest.mark.asyncio
async def test_event_archive_rejects_events_appended_after_batch_cutoff(tmp_path):
    manager, event_log, projection, archive_index = (
        await _make_event_archive_repair_manager(tmp_path)
    )
    await event_log.append_user_message(
        chat_id="chat",
        turn_id="turn-1",
        message_id="message-1",
        content="问题",
        timestamp=1,
    )
    await event_log.append_accepted_delivery(
        chat_id="chat",
        turn_id="turn-1",
        delivery_id="delivery-1",
        content="回答",
        timestamp=2,
    )
    cutoff = await event_log.latest_event_seq("chat")
    turn = (await event_log.snapshot_turns("chat", include_internal=True)).turns[0]
    source = (await event_log.snapshot_events("chat", include_internal=True)).events
    batch = await archive_index.prepare_batch(
        batch_id="batch:late-event",
        operation_id="late-event",
        chat_id="chat",
        captured_cutoff_seq=cutoff,
        turn_records=[
            ArchiveTurnRecord(
                turn_id=turn.turn_id,
                turn_sequence=turn.turn_sequence,
                source_date=turn.source_date,
                event_count=len(source),
                estimated_tokens=1,
            )
        ],
        event_ids=[(event.event_id, turn.turn_id) for event in source],
    )
    await event_log.append_turn_terminal(chat_id="chat", turn_id="turn-1", timestamp=3)

    with pytest.raises(RuntimeError, match="partial turn"):
        await manager._finish_event_archive_batch(batch)


@pytest.mark.asyncio
async def test_event_archive_waits_for_dynamic_retention_threshold(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    manager = ArchiveManager(
        context_manager=MagicMock(),
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
        hot_max_tokens=1000,
        hot_max_turns=10,
        hot_max_bytes=100000,
        hot_max_age_seconds=86400,
    )
    manager.set_event_log(event_log, projection)
    await _append_completed_turn(event_log, "chat", "turn-1", time.time())

    assert await manager.archive_if_stale("chat", False) is None
    assert await index.list_for_webui("chat") == []
    await event_log.close()
    await projection.close()
    await index.close()


@pytest.mark.asyncio
async def test_event_archive_without_prompt_projection_still_commits(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    manager = ArchiveManager(
        context_manager=MagicMock(),
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
        hot_max_tokens=1,
        hot_max_turns=10,
        hot_max_bytes=100000,
        hot_max_age_seconds=86400,
    )
    manager.set_event_log(event_log)
    await _append_completed_turn(event_log, "chat", "turn-1", time.time())

    result = await manager.archive_if_stale("chat", False)

    assert result is not None
    assert await index.list_for_webui("chat", state="committed")
    await event_log.close()
    await index.close()


@pytest.mark.asyncio
async def test_event_archive_does_not_materialize_active_projection(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    projection.snapshot_for_active = AsyncMock(
        side_effect=AssertionError("archive must not load active event bodies")
    )
    index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    manager = ArchiveManager(
        context_manager=MagicMock(),
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
        hot_max_tokens=1,
        hot_max_turns=10,
        hot_max_bytes=100000,
        hot_max_age_seconds=86400,
    )
    manager.set_event_log(event_log, projection)
    await _append_completed_turn(event_log, "chat", "turn-1", time.time())

    result = await manager.archive_if_stale("chat", False)

    assert result is not None
    assert await index.list_for_webui("chat", state="committed")
    await event_log.close()
    await projection.close()
    await index.close()


@pytest.mark.asyncio
async def test_event_archive_reads_only_selected_turn_bodies(tmp_path, monkeypatch):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    manager = ArchiveManager(
        context_manager=MagicMock(),
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
        hot_max_tokens=1,
        hot_max_turns=10,
        hot_max_bytes=100000,
        hot_max_age_seconds=86400,
    )
    manager.set_event_log(event_log, projection)
    await _append_completed_turn(event_log, "chat", "turn-1", time.time())

    original_snapshot_events = event_log.snapshot_events

    async def bounded_snapshot(*args, **kwargs):
        if kwargs.get("turn_ids") is None and kwargs.get("event_ids") is None:
            raise AssertionError("archive must not load unbounded event bodies")
        return await original_snapshot_events(*args, **kwargs)

    monkeypatch.setattr(event_log, "snapshot_events", bounded_snapshot)

    result = await manager.archive_if_stale("chat", False)

    assert result is not None
    assert await index.list_for_webui("chat", state="committed")
    await event_log.close()
    await projection.close()
    await index.close()


@pytest.mark.asyncio
async def test_archive_recovery_sync_reads_event_identities_only(tmp_path, monkeypatch):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    manager = ArchiveManager(
        context_manager=MagicMock(),
        memory_dir=str(tmp_path / "memory"),
    )
    manager.set_event_log(event_log, projection)
    await _append_completed_turn(event_log, "chat", "turn-1", time.time())
    event_id = (await event_log.event_ids("chat", include_internal=True))[0]

    async def reject_body_snapshot(*args, **kwargs):
        raise AssertionError("archive recovery must not load event bodies")

    monkeypatch.setattr(event_log, "snapshot_events", reject_body_snapshot)

    await manager._sync_prompt_projection(
        {
            "chat_id": "chat",
            "operation_id": "recovery-op",
            "batches": [{"records": [{"event_id": event_id}]}],
        }
    )

    assert event_id in await projection.hidden_event_ids("chat")
    await event_log.close()
    await projection.close()


@pytest.mark.asyncio
async def test_archive_recovery_replays_at_captured_cutoff_after_late_event(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    manager = ArchiveManager(
        context_manager=MagicMock(),
        memory_dir=str(tmp_path / "memory"),
    )
    manager.set_event_log(event_log, projection)
    await _append_completed_turn(event_log, "chat", "turn-1", time.time())
    event_ids = await event_log.event_ids("chat", include_internal=True)
    cutoff_seq = await event_log.latest_event_seq("chat")
    manifest = {
        "chat_id": "chat",
        "operation_id": "recovery-op",
        "captured_cutoff_seq": cutoff_seq,
        "batches": [{"records": [{"event_id": event_ids[0]}]}],
    }

    await manager._sync_prompt_projection(manifest)
    await event_log.append_user_message(
        chat_id="chat",
        turn_id="late-turn",
        message_id="late-message",
        content="late",
        timestamp=time.time(),
    )
    await manager._sync_prompt_projection(manifest)

    assert event_ids[0] in await projection.hidden_event_ids("chat")
    await event_log.close()
    await projection.close()


@pytest.mark.asyncio
async def test_event_archive_repair_scope_excludes_later_events_from_operation(
    tmp_path,
):
    manager, event_log, projection, archive_index = (
        await _make_event_archive_repair_manager(tmp_path)
    )
    await event_log.append_user_message(
        chat_id="chat",
        turn_id="old-incomplete",
        message_id="old-message",
        content="pending",
        timestamp=1,
    )

    first = await manager.repair_event_log_archives("chat", before_date="1970-01-02")
    await event_log.append_user_message(
        chat_id="chat",
        turn_id="today-incomplete",
        message_id="today-message",
        content="new",
        timestamp=172800,
    )
    second = await manager.repair_event_log_archives("chat", before_date="1970-01-02")

    assert first.operation_id == second.operation_id
    assert await projection.visibility_event_ids("chat") == {"user:old-message"}
    assert await archive_index.list_for_webui("chat", state="committed") == []
    await event_log.close()
    await projection.close()
    await archive_index.close()


@pytest.mark.asyncio
async def test_event_log_status_does_not_fallback_to_legacy_history(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    context_manager = MagicMock()
    context_manager.get_chat_history_async = AsyncMock(
        side_effect=AssertionError("event-log status must not read legacy history")
    )
    manager = ArchiveManager(
        context_manager=context_manager,
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
    )
    manager.set_event_log(event_log)

    assert await manager.get_session_status_async("empty-ledger-chat") == {
        "message_count": 0,
        "last_activity": 0.0,
        "archive_count": 0,
    }
    context_manager.get_chat_history_async.assert_not_awaited()
    await event_log.close()
    await index.close()


@pytest.mark.asyncio
async def test_event_archive_rejects_unpaired_tool_turn(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    manager = ArchiveManager(
        context_manager=MagicMock(),
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
        hot_max_tokens=1,
        hot_max_turns=10,
        hot_max_bytes=100000,
        hot_max_age_seconds=86400,
    )
    manager.set_event_log(event_log, projection)
    await event_log.append_user_message(
        chat_id="chat",
        turn_id="turn-tool",
        message_id="message-tool",
        content="tool",
        timestamp=time.time(),
    )
    await event_log.append_event(
        ConversationEvent(
            chat_id="chat",
            turn_id="turn-tool",
            event_id="call:turn-tool",
            role="assistant",
            kind=EventKind.ASSISTANT_TOOL_CALL,
            tool_calls=({"id": "call-1", "function": {"name": "x"}},),
        )
    )
    await event_log.append_turn_terminal(
        chat_id="chat", turn_id="turn-tool", status="incomplete"
    )

    assert await manager.archive_if_stale("chat", False) is None
    assert await index.list_for_webui("chat") == []
    await event_log.close()
    await projection.close()
    await index.close()


@pytest.mark.asyncio
async def test_event_archive_recovery_resumes_after_summary_failure(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    summary_store = TurnSummaryStore(
        event_log, path=str(tmp_path / "summaries.sqlite3")
    )
    index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    manager = ArchiveManager(
        context_manager=MagicMock(),
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
        hot_max_tokens=1,
        hot_max_turns=10,
        hot_max_bytes=100000,
        hot_max_age_seconds=86400,
    )
    manager.set_event_log(event_log, projection, summary_store)
    await _append_completed_turn(event_log, "chat", "turn-1", time.time())

    original_ensure = summary_store.ensure_for_archived_events

    async def fail_summary(*args, **kwargs):
        raise OSError("summary unavailable")

    summary_store.ensure_for_archived_events = fail_summary
    with pytest.raises(OSError, match="summary unavailable"):
        await manager.archive_if_stale("chat", False)

    pending = await index.list_pending("chat")
    assert len(pending) == 1
    assert await projection.hidden_event_ids("chat") == frozenset()

    summary_store.ensure_for_archived_events = original_ensure
    recovered = ArchiveManager(
        context_manager=MagicMock(),
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
    )
    recovered.set_event_log(event_log, projection, summary_store)

    await recovered._recover_event_log_archives("chat")

    committed = await index.list_for_webui("chat", state="committed")
    assert len(committed) == 1
    assert await index.event_ids(committed[0]["batch_id"]) == {
        "user:message-turn-1",
        "delivery:delivery-turn-1",
        "terminal:turn-1",
    }
    assert await projection.hidden_event_ids("chat") == {
        "user:message-turn-1",
        "delivery:delivery-turn-1",
        "terminal:turn-1",
    }
    await event_log.close()
    await projection.close()
    await summary_store.close()
    await index.close()


@pytest.mark.asyncio
async def test_archive_export_failure_does_not_rollback_core_batch(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    manager = ArchiveManager(
        context_manager=MagicMock(),
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
        hot_max_tokens=1,
        hot_max_turns=10,
        hot_max_bytes=100000,
        hot_max_age_seconds=86400,
    )
    manager.set_event_log(event_log, projection)
    export_adapter = MagicMock()
    export_adapter.export_batch = AsyncMock(side_effect=RuntimeError("disk full"))
    manager.set_export_adapter(export_adapter)
    await _append_completed_turn(event_log, "chat", "turn-1", time.time())

    result = await manager.archive_if_stale("chat", False)

    assert result is not None
    batch = (await index.list_for_webui("chat"))[0]
    assert batch["state"] == "committed"
    assert batch["export_status"] == "failed"
    assert "disk full" in batch["export_error"]
    await event_log.close()
    await projection.close()
    await index.close()


@pytest.mark.asyncio
async def test_committed_archive_reconciles_missing_prompt_visibility(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    manager = ArchiveManager(
        context_manager=MagicMock(),
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
        hot_max_tokens=1,
        hot_max_turns=10,
        hot_max_bytes=100000,
        hot_max_age_seconds=86400,
    )
    manager.set_event_log(event_log, projection)
    await _append_completed_turn(event_log, "chat", "turn-1", time.time())

    assert await manager.archive_if_stale("chat", False) is not None
    archived_ids = set(await projection.hidden_event_ids("chat"))
    assert archived_ids

    await _append_completed_turn(event_log, "chat", "turn-2", time.time())
    await projection.repair_chat("chat")

    metadata = await projection._ensure_metadata_open()
    metadata.execute("DELETE FROM prompt_event_visibility WHERE chat_id = ?", ("chat",))
    metadata.commit()
    assert await projection.hidden_event_ids("chat") == frozenset()

    await manager._recover_event_log_archives("chat")

    assert await projection.hidden_event_ids("chat") == archived_ids
    await event_log.close()
    await projection.close()
    await index.close()


@pytest.mark.asyncio
async def test_static_repair_restores_visibility_for_unarchived_incomplete_turn(
    tmp_path, monkeypatch
):
    manager, event_log, projection, archive_index = (
        await _make_event_archive_repair_manager(tmp_path)
    )
    await _append_completed_turn(event_log, "chat", "complete", 1)
    await event_log.append_user_message(
        chat_id="chat",
        turn_id="incomplete",
        message_id="message-incomplete",
        content="pending",
        timestamp=10,
    )
    await projection.repair_chat("chat")
    metadata = await projection._ensure_metadata_open()
    metadata.execute("DELETE FROM prompt_event_visibility WHERE chat_id = ?", ("chat",))
    metadata.commit()
    monkeypatch.setattr(manager, "_recover_event_log_archives", AsyncMock())

    result = await manager.repair_event_log_archives("chat", before_date="1970-01-02")

    assert result.batches
    assert await projection.visibility_event_ids("chat") >= {"user:message-incomplete"}
    assert "user:message-incomplete" not in await projection.hidden_event_ids("chat")
    await event_log.close()
    await projection.close()
    await archive_index.close()


@pytest.mark.asyncio
async def test_static_repair_rehides_committed_event_when_recovery_was_degraded(
    tmp_path, monkeypatch
):
    manager, event_log, projection, archive_index = (
        await _make_event_archive_repair_manager(tmp_path)
    )
    await _append_completed_turn(event_log, "chat", "complete", 1)
    assert await manager.archive_if_stale("chat", False) is not None
    archived_ids = set(await archive_index.committed_event_ids("chat"))
    metadata = await projection._ensure_metadata_open()
    metadata.execute("DELETE FROM prompt_event_visibility WHERE chat_id = ?", ("chat",))
    metadata.commit()
    monkeypatch.setattr(manager, "_recover_event_log_archives", AsyncMock())

    result = await manager.repair_event_log_archives("chat", before_date="1970-01-02")

    assert result.batches == []
    assert await projection.hidden_event_ids("chat") == archived_ids
    await event_log.close()
    await projection.close()
    await archive_index.close()


@pytest.mark.asyncio
async def test_startup_archive_recovery_covers_all_ledger_chats(tmp_path, monkeypatch):
    manager, event_log, projection, archive_index = (
        await _make_event_archive_repair_manager(tmp_path)
    )
    await event_log.append_user_message(
        chat_id="chat-1", turn_id="turn-1", message_id="message-1", content="one"
    )
    await event_log.append_user_message(
        chat_id="chat-2", turn_id="turn-2", message_id="message-2", content="two"
    )
    calls = []

    async def recover(chat_id):
        calls.append(chat_id)

    monkeypatch.setattr(manager, "_recover_event_log_archives", recover)

    assert await manager.recover_event_log_archives_async() == 2
    assert calls == ["chat-1", "chat-2"]
    await event_log.close()
    await projection.close()
    await archive_index.close()


@pytest.mark.asyncio
async def test_event_archive_recovery_retries_transcript_rotation(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    manager = ArchiveManager(
        context_manager=MagicMock(),
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
        hot_max_tokens=1,
        hot_max_turns=10,
        hot_max_bytes=100000,
        hot_max_age_seconds=86400,
    )
    manager.set_event_log(event_log, projection)
    scope = SimpleNamespace(key="private:chat")
    calls = 0

    async def rotate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("transcript unavailable")

    transcript = SimpleNamespace(
        scopes_for_chat=AsyncMock(return_value=(scope,)),
        snapshot=AsyncMock(return_value=SimpleNamespace(source_event_ids=set())),
        rotate_for_hidden_sources=rotate,
    )
    manager.set_model_context_transcript(transcript)
    await _append_completed_turn(event_log, "chat", "turn-1", time.time())

    with pytest.raises(OSError, match="transcript unavailable"):
        await manager.archive_if_stale("chat", False)
    assert len(await index.list_pending("chat")) == 1
    assert await projection.hidden_event_ids("chat")

    await manager._recover_event_log_archives("chat")

    assert calls == 2
    assert await index.list_pending("chat") == []
    assert len(await index.list_for_webui("chat", state="committed")) == 1
    await event_log.close()
    await projection.close()
    await index.close()


@pytest.mark.asyncio
async def test_event_archive_recovery_retries_prompt_projection(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    manager = ArchiveManager(
        context_manager=MagicMock(),
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
        hot_max_tokens=1,
        hot_max_turns=10,
        hot_max_bytes=100000,
        hot_max_age_seconds=86400,
    )
    manager.set_event_log(event_log, projection)
    await _append_completed_turn(event_log, "chat", "turn-1", time.time())
    original_apply = projection.apply_archive_retention
    calls = 0

    async def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("projection unavailable")
        return await original_apply(*args, **kwargs)

    projection.apply_archive_retention = fail_once
    with pytest.raises(OSError, match="projection unavailable"):
        await manager.archive_if_stale("chat", False)
    assert len(await index.list_pending("chat")) == 1

    projection.apply_archive_retention = original_apply
    await manager._recover_event_log_archives("chat")

    assert await index.list_pending("chat") == []
    assert await projection.hidden_event_ids("chat")
    await event_log.close()
    await projection.close()
    await index.close()


@pytest.mark.asyncio
async def test_event_archive_recovery_retries_archive_index_commit(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    manager = ArchiveManager(
        context_manager=MagicMock(),
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
        hot_max_tokens=1,
        hot_max_turns=10,
        hot_max_bytes=100000,
        hot_max_age_seconds=86400,
    )
    manager.set_event_log(event_log, projection)
    await _append_completed_turn(event_log, "chat", "turn-1", time.time())
    original_mark_state = index.mark_state
    calls = 0

    async def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("archive index unavailable")
        return await original_mark_state(*args, **kwargs)

    index.mark_state = fail_once
    with pytest.raises(OSError, match="archive index unavailable"):
        await manager.archive_if_stale("chat", False)
    assert len(await index.list_pending("chat")) == 1

    index.mark_state = original_mark_state
    await manager._recover_event_log_archives("chat")

    assert await index.list_pending("chat") == []
    assert len(await index.list_for_webui("chat", state="committed")) == 1
    await event_log.close()
    await projection.close()
    await index.close()


@pytest.mark.asyncio
async def test_event_archive_recovery_finishes_when_index_committed_before_manifest(
    tmp_path,
):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    manager = ArchiveManager(
        context_manager=MagicMock(),
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
        hot_max_tokens=1,
        hot_max_turns=10,
        hot_max_bytes=100000,
        hot_max_age_seconds=86400,
    )
    manager.set_event_log(event_log, projection)
    await _append_completed_turn(event_log, "chat", "turn-1", time.time())

    original_write_phase = manager._write_event_manifest_phase
    failed = False

    def fail_after_index_commit(operation_id, phase, *, committed=False):
        nonlocal failed
        if committed and not failed:
            failed = True
            raise OSError("manifest commit marker unavailable")
        return original_write_phase(operation_id, phase, committed=committed)

    manager._write_event_manifest_phase = fail_after_index_commit
    with pytest.raises(OSError, match="manifest commit marker unavailable"):
        await manager.archive_if_stale("chat", False)

    committed = await index.list_for_webui("chat", state="committed")
    assert len(committed) == 1
    assert await index.list_pending("chat") == []
    assert await projection.hidden_event_ids("chat")

    manager._write_event_manifest_phase = original_write_phase
    await manager._recover_event_log_archives("chat")

    assert len(await index.list_for_webui("chat", state="committed")) == 1
    assert await projection.hidden_event_ids("chat")
    assert manager._manifest_store.load_pending(kind="event_log") == []
    await event_log.close()
    await projection.close()
    await index.close()


@pytest.mark.asyncio
async def test_event_archive_recovery_retries_manifest_after_projection_commit(
    tmp_path,
):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    manager = ArchiveManager(
        context_manager=MagicMock(),
        archive_index=index,
        memory_dir=str(tmp_path / "memory"),
        hot_max_tokens=1,
        hot_max_turns=10,
        hot_max_bytes=100000,
        hot_max_age_seconds=86400,
    )
    manager.set_event_log(event_log, projection)
    await _append_completed_turn(event_log, "chat", "turn-1", time.time())
    original_write_phase = manager._write_event_manifest_phase
    calls = 0

    def fail_once(operation_id, phase, *, committed=False):
        nonlocal calls
        if phase == "prompt_visibility_written" and calls == 0:
            calls += 1
            raise OSError("manifest unavailable")
        return original_write_phase(operation_id, phase, committed=committed)

    manager._write_event_manifest_phase = fail_once
    with pytest.raises(OSError, match="manifest unavailable"):
        await manager.archive_if_stale("chat", False)
    assert len(await index.list_pending("chat")) == 1
    assert await projection.hidden_event_ids("chat")

    manager._write_event_manifest_phase = original_write_phase
    await manager._recover_event_log_archives("chat")

    assert await index.list_pending("chat") == []
    assert len(await index.list_for_webui("chat", state="committed")) == 1
    await event_log.close()
    await projection.close()
    await index.close()


def test_archive_manifest_rejects_invalid_state_transition(tmp_path):
    store = ArchiveManifestStore(str(tmp_path))
    manifest = {
        "version": 1,
        "operation_id": "op-1",
        "chat_id": "chat",
        "state": "prepared",
        "batches": [],
    }

    store.write(manifest)
    for state in (
        "archive_written",
        "active_written",
        "summary_written",
        "committed",
    ):
        manifest["state"] = state
        store.write(manifest)
    manifest["state"] = "archive_written"

    with pytest.raises(RuntimeError, match="state transition"):
        store.write(manifest)


def test_archive_manifest_clear_pending_is_scoped_to_chat(tmp_path):
    store = ArchiveManifestStore(str(tmp_path))
    pending = {
        "version": 2,
        "kind": "event_log",
        "operation_id": "op-pending",
        "chat_id": "chat-pending",
        "state": "prepared",
        "phase": "prepared",
        "batches": [
            {
                "batch_id": "batch-pending",
                "partition_date": "2026-01-01",
                "state": "prepared",
            }
        ],
    }
    committed = {
        **pending,
        "operation_id": "op-committed",
        "chat_id": "chat-other",
        "state": "committed",
    }
    store.write(pending)
    store.write(committed)

    assert store.clear_pending("chat-pending") == 1
    assert store.load("op-pending") is None
    assert store.load("op-committed") is not None


@pytest.mark.asyncio
async def test_archive_command_resolves_other_target_type_in_manager():
    """命令归档其他会话时，不得携带发起会话的 is_group。"""

    class _CommandManager:
        replay_gap_seconds = 60
        summary_count = 15

        def __init__(self):
            self.calls = []

        async def archive_manual(self, *args):
            self.calls.append(args)
            return ArchiveResult("target_chat", "manual")

    manager = _CommandManager()
    command = ArchiveCommand(manager)
    message = InputMessage("m1", "admin", "admin_chat", "", False)

    await command._run_archive(message, "target_chat")

    assert manager.calls == [("target_chat", None)]


@pytest.mark.asyncio
async def test_archive_command_exposes_integrity_and_repair_actions():
    class _CommandManager:
        async def get_event_integrity_async(self, chat_id):
            assert chat_id == "target_chat"
            return {
                "turn_count": 3,
                "invalid_turn_count": 1,
                "incomplete_turn_count": 1,
                "open_turn_count": 0,
                "invalid_reasons": {"missing_tool_result": 1},
                "invalid_turns": [
                    {
                        "turn_id": "broken-turn",
                        "status": "incomplete",
                        "reason": "missing_tool_result",
                    }
                ],
            }

        async def repair_event_log_archives(self, chat_id, *, before_date):
            assert (chat_id, before_date) == ("target_chat", "2026-09-04")
            return ArchiveResult(
                chat_id,
                "repair",
                batches=[
                    SimpleNamespace(event_count=2, unit_count=1),
                ],
                skipped_turns=[{"reason": "incomplete_turn"}],
            )

    command = ArchiveCommand(_CommandManager())
    message = InputMessage("m1", "admin", "admin_chat", "", False)

    integrity = await command.execute(message, "完整性 target_chat")
    assert "missing_tool_result=1" in integrity[0]["content"]
    repair = await command.execute(message, "修复 2026-09-04 target_chat")
    assert "新增归档: 2 事件 / 1 turns" in repair[0]["content"]
    assert "incomplete_turn=1" in repair[0]["content"]


async def test_archive_command_records_append_only_repair_revision():
    class _CommandManager:
        async def record_turn_repair_revision_async(
            self, chat_id, turn_id, revision_id, reason, *, operator
        ):
            assert (chat_id, turn_id, revision_id, reason, operator) == (
                "admin_chat",
                "broken-turn",
                "rev-1",
                "保留原始异常",
                "admin",
            )
            return SimpleNamespace(
                original_turn_id=turn_id,
                revision_id=revision_id,
            )

    command = ArchiveCommand(_CommandManager())
    message = InputMessage("m1", "admin", "admin_chat", "", False)

    reply = await command.execute(message, "修订 broken-turn rev-1 保留原始异常")

    assert "原始事件与完整性状态未修改" in reply[0]["content"]
