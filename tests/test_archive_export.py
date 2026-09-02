import json
from pathlib import Path

import pytest

from core.engine.archive_export import ArchiveJSONLExportAdapter
from core.engine.archive_index import ArchiveIndex, ArchiveTurnRecord
from core.engine.conversation_event_log import ConversationEventLog


async def _completed_turn(event_log: ConversationEventLog) -> None:
    await event_log.append_user_message(
        chat_id="chat",
        turn_id="turn-1",
        message_id="message-1",
        content="hello",
        timestamp=100,
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
        turn_records=[ArchiveTurnRecord("turn-1", 1, "1970-01-01", 3, 3)],
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
