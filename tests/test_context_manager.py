import importlib

import pytest

from core.engine.conversation_event_log import ConversationEventLog
from core.engine.prompt_history_projection import PromptHistoryProjection
from core.managers.context_manager import ChatContextManager


def test_legacy_message_list_compaction_api_is_retired():
    assert not hasattr(ChatContextManager, "compact_history_if_needed")
    assert not hasattr(ChatContextManager, "compaction_threshold_tokens")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("core.managers.context_compactor")


def test_legacy_active_store_modules_are_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("core.managers.chat_context")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("core.managers.context_store")


@pytest.mark.asyncio
async def test_event_log_session_enumeration_is_ledger_only(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await event_log.append_user_message(
        chat_id="ledger-chat",
        turn_id="turn-1",
        message_id="message-1",
        content="账本会话",
    )
    manager = ChatContextManager()
    manager.set_event_log(event_log)

    assert await manager.get_all_disk_chat_ids_async() == ["ledger-chat"]
    assert await manager.get_all_chat_ids_async() == ["ledger-chat"]
    await event_log.close()


@pytest.mark.asyncio
async def test_event_log_user_dedup_uses_identity_lookup_without_snapshot(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    manager = ChatContextManager()
    manager.set_event_log(event_log)
    await event_log.append_user_message(
        chat_id="ledger-chat",
        turn_id="turn-1",
        message_id="message-1",
        content="已有消息",
    )

    assert (
        await manager.add_user_message_async(
            "ledger-chat", "已有消息", message_id="message-1"
        )
        is False
    )
    assert (
        await manager.add_user_message_async(
            "ledger-chat", "新消息", message_id="message-2"
        )
        is True
    )
    await event_log.close()


@pytest.mark.asyncio
async def test_recent_user_contents_uses_bounded_prompt_projection(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    manager = ChatContextManager()
    manager.set_event_log(event_log)
    manager.set_prompt_projection(projection)
    await event_log.append_user_message(
        chat_id="chat",
        turn_id="turn-old",
        message_id="old",
        content="old duplicate",
    )
    await event_log.append_turn_terminal(chat_id="chat", turn_id="turn-old")
    await event_log.append_user_message(
        chat_id="chat",
        turn_id="turn-new",
        message_id="new",
        content="new message",
    )
    await event_log.append_turn_terminal(chat_id="chat", turn_id="turn-new")
    await projection.apply_archive_retention(
        "chat",
        operation_id="archive-old",
        hidden_event_ids=("user:old", "terminal:turn-old"),
        captured_cutoff_seq=2,
    )

    assert await manager.get_recent_user_contents_async("chat", count=2) == [
        "new message"
    ]
    await event_log.close()
    await projection.close()


@pytest.mark.asyncio
async def test_session_clear_resets_ledger(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    manager = ChatContextManager()
    manager.set_event_log(event_log)
    await event_log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="m1", content="hello"
    )

    await manager.clear_chat_history_async("chat")

    assert await event_log.history("chat") == []
    await event_log.close()


@pytest.mark.asyncio
async def test_recent_user_contents_bounds_ledger_read(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await event_log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="message-1", content="hello"
    )
    manager = ChatContextManager()
    manager.set_event_log(event_log)

    assert await manager.get_recent_user_contents_async("chat", count=2) == ["hello"]
    await event_log.close()


@pytest.mark.asyncio
async def test_recent_user_contents_uses_bounded_event_window():
    calls = []

    class EventLog:
        async def history(self, chat_id, **kwargs):
            calls.append((chat_id, kwargs))
            return [{"role": "user", "content": "hello"}]

    manager = ChatContextManager()
    manager.set_event_log(EventLog())

    assert await manager.get_recent_user_contents_async("chat", count=3) == ["hello"]
    assert calls == [("chat", {"max_events": 20})]


@pytest.mark.asyncio
async def test_event_log_history_defaults_to_bounded_window():
    calls = []

    class EventLog:
        async def history(self, chat_id, **kwargs):
            calls.append((chat_id, kwargs))
            return [{"role": "user", "content": "hello"}]

    manager = ChatContextManager()
    manager.set_event_log(EventLog())

    assert await manager.get_chat_history_async("chat") == [
        {"role": "user", "content": "hello"}
    ]
    assert calls == [("chat", {"max_events": 100})]


@pytest.mark.asyncio
async def test_legacy_archive_readers_always_empty():
    manager = ChatContextManager()
    manager.set_event_log(object())

    assert await manager.get_archived_sessions_summary_async() == {}
    assert await manager.get_archived_files_async("chat") == []
    assert await manager.read_archived_messages_async("archive.jsonl") == []


@pytest.mark.asyncio
async def test_remove_message_if_is_noop_in_ledger_mode(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    manager = ChatContextManager()
    manager.set_event_log(event_log)
    await event_log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="m1", content="hello"
    )

    assert await manager.remove_last_user_message_if_async("chat", "m1") is False
    assert len(await manager.get_chat_history_async("chat")) == 1
    await event_log.close()


@pytest.mark.asyncio
async def test_clear_history_resets_ledger_without_touching_timeline(tmp_path):
    """Ledger clear must not drive the preserved Timeline/ProtocolHistory."""

    class Timeline:
        def __init__(self):
            self.cleared = []

        async def clear_chat(self, chat_id):
            self.cleared.append(chat_id)

    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await event_log.append_user_message(
        chat_id="chat", turn_id="turn-1", message_id="m1", content="hello"
    )
    timeline = Timeline()
    manager = ChatContextManager()
    manager.set_event_log(event_log)
    manager.set_timeline(timeline)

    await manager.clear_chat_history_async("chat")

    assert await event_log.history("chat") == []
    assert timeline.cleared == []
    await event_log.close()


@pytest.mark.asyncio
async def test_cleanup_inactive_contexts_is_noop_in_ledger_mode():
    manager = ChatContextManager()
    assert await manager.cleanup_inactive_contexts_async(max_inactivity=0) == []


# ── 聊天类型 ──


@pytest.mark.asyncio
async def test_get_chat_type_uses_index():
    from core.engine.chat_type_index import ChatTypeIndex
    from core.session_identity import DeliveryTarget, DeliveryTargetCatalog

    catalog = DeliveryTargetCatalog(":memory:")
    index = ChatTypeIndex(catalog)
    manager = ChatContextManager()
    manager.set_chat_type_index(index)
    index.observe(DeliveryTarget("qq", "default", "group", "chat_001"))
    assert manager.get_chat_type("chat_001") is True
    assert manager.get_chat_type("chat_unknown") is None
