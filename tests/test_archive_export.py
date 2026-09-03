import json
from pathlib import Path

import pytest

from core.engine.archive_export import ArchiveJSONLExportAdapter
from core.engine.archive_index import ArchiveIndex, ArchiveTurnRecord
from core.engine.conversation_event_log import ConversationEventLog, TurnKind


async def _completed_turn(event_log: ConversationEventLog) -> None:
    await event_log.append_user_message(
        chat_id="chat",
        turn_id="turn-1",
        message_id="message-1",
        content="hello",
        timestamp=100,
        turn_kind=TurnKind.AI,
    )
    await event_log.append_accepted_delivery(
        chat_id="chat",
        turn_id="turn-1",
        delivery_id="delivery-1",
        content="world",
        timestamp=101,
    )
    await event_log.append_turn_terminal(
        chat_id="chat", turn_id="turn-1", timestamp=102
    )


@pytest.mark.asyncio
async def test_jsonl_export_is_one_way_and_idempotent(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    index = ArchiveIndex(str(tmp_path / "index.sqlite3"))
    await _completed_turn(event_log)
    events = await event_log.snapshot_events("chat", include_internal=True)
    batch = await index.prepare_batch(
        batch_id="batch-1",
        operation_id="operation-1",
        chat_id="chat",
        captured_cutoff_seq=events.cutoff_seq,
        turn_records=[
            ArchiveTurnRecord("turn-1", 1, "1970-01-01", 3, 3, TurnKind.AI.value)
        ],
        event_ids=[(event.event_id, event.turn_id) for event in events.events],
    )
    await index.mark_state(batch.batch_id, "committed")

    adapter = ArchiveJSONLExportAdapter(
        event_log, index, str(tmp_path / "exports"), enabled=True
    )
    first = await adapter.export_batch(batch.batch_id)
    second = await adapter.export_batch(batch.batch_id)

    assert first.status == "exported"
    assert second.content_hash == first.content_hash
    lines = [
        json.loads(line)
        for line in Path(first.path).read_text(encoding="utf-8").splitlines()
    ]
    assert lines[0]["record_type"] == "archive_export_manifest"
    assert [line["event"]["event_id"] for line in lines[1:]] == [
        event.event_id for event in events.events
    ]
    archived_turns = await index.turns_for_batch(batch.batch_id)
    assert archived_turns[0].turn_kind == TurnKind.AI.value
    assert await index.count_turns_for_batch(batch.batch_id) == 1
    listed = await index.list_for_webui("chat")
    assert listed[0]["export_status"] == "disabled"
    await index.record_export(
        batch.batch_id,
        status=first.status,
        path=first.path,
        content_hash=first.content_hash,
        manifest_hash=first.manifest_hash,
    )
    listed = await index.list_for_webui("chat")
    assert listed[0]["export_status"] == "exported"
    await event_log.close()
    await index.close()


@pytest.mark.asyncio
async def test_archive_index_webui_queries_support_pagination_and_aggregation(tmp_path):
    index = ArchiveIndex(str(tmp_path / "index.sqlite3"))
    for index_number in range(3):
        await index.prepare_batch(
            batch_id=f"batch-{index_number}",
            operation_id=f"operation-{index_number}",
            chat_id=f"chat-{index_number}",
            captured_cutoff_seq=index_number,
            turn_records=[
                ArchiveTurnRecord(
                    f"turn-{index_number}", index_number, "2026-09-01", 2, 10
                )
            ],
            event_ids=[
                (f"user:{index_number}", f"turn-{index_number}"),
                (f"delivery:{index_number}", f"turn-{index_number}"),
            ],
        )
        await index.mark_state(f"batch-{index_number}", "committed")

    page = await index.list_for_webui("chat-1", limit=1, offset=0, state="committed")
    summaries, total = await index.chat_summaries_for_webui(
        query="chat-", limit=2, offset=1
    )

    assert len(page) == 1
    assert await index.count_for_webui("chat-1", state="committed") == 1
    assert total == 3
    assert [item["chat_id"] for item in summaries] == ["chat-1", "chat-0"]
    await index.close()
