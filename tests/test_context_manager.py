import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.engine.conversation_event_log import ConversationEventLog
from core.managers.context_compactor import CompactionResult
from core.managers.context_manager import ChatContextManager
from core.managers.context_store import MemoryContextStore


@pytest.fixture
def store():
    return MemoryContextStore()


class FakeCompactor:
    compact_threshold_tokens = 100
    keep_recent_tokens = 10

    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def compact(self, messages, *, force=False):
        self.calls.append((list(messages), force))
        if self.result is not None:
            return self.result
        return CompactionResult(False, list(messages), None)


@pytest.fixture
def compactor():
    return FakeCompactor()


@pytest.fixture
def mgr(store, compactor):
    return ChatContextManager(store=store, compactor=compactor)


@pytest.mark.asyncio
async def test_first_history_access_creates_new(mgr):
    assert await mgr.get_chat_history_async("chat_001") == []
    assert await mgr.get_all_chat_ids_async() == ["chat_001"]


@pytest.mark.asyncio
async def test_first_history_access_reuses_existing(mgr):
    await mgr.get_chat_history_async("chat_001")
    await mgr.get_chat_history_async("chat_001")
    assert await mgr.get_context_count_async() == 1


@pytest.mark.asyncio
async def test_event_log_session_enumeration_does_not_scan_legacy_store(tmp_path):
    class LegacyStore(MemoryContextStore):
        def get_all_disk_ids(self):
            raise AssertionError("event-log mode must not scan legacy sessions")

    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await event_log.append_user_message(
        chat_id="ledger-chat",
        turn_id="turn-1",
        message_id="message-1",
        content="账本会话",
    )
    manager = ChatContextManager(store=LegacyStore())
    manager.set_event_log(event_log)

    assert await manager.get_all_disk_chat_ids_async() == ["ledger-chat"]
    await event_log.close()


@pytest.mark.asyncio
async def test_history_access_loads_from_store(mgr, store):
    store.flush("chat_001", [{"role": "user", "content": "hello", "timestamp": 100.0}])
    history = await mgr.get_chat_history_async("chat_001")
    assert len(history) == 1
    assert history[0]["raw_content"] == "hello"


@pytest.mark.asyncio
async def test_concurrent_first_access_restores_once(compactor):
    class BlockingStore(MemoryContextStore):
        def __init__(self):
            super().__init__()
            self.load_calls = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def load_async(self, chat_id):
            self.load_calls += 1
            self.started.set()
            await self.release.wait()
            return None

    store = BlockingStore()
    mgr = ChatContextManager(store=store, compactor=compactor)
    first = asyncio.create_task(mgr.get_chat_history_async("chat_001"))
    await store.started.wait()
    second = asyncio.create_task(mgr.get_chat_history_async("chat_001"))
    await asyncio.sleep(0)
    assert store.load_calls == 1
    store.release.set()
    first_history, second_history = await asyncio.gather(first, second)
    assert first_history == second_history == []


@pytest.mark.asyncio
async def test_first_access_different_chats_restores_in_parallel(compactor):
    class ParallelStore(MemoryContextStore):
        def __init__(self):
            super().__init__()
            self.started = set()
            self.release = asyncio.Event()

        async def load_async(self, chat_id):
            self.started.add(chat_id)
            if len(self.started) == 2:
                self.release.set()
            await self.release.wait()
            return None

    store = ParallelStore()
    mgr = ChatContextManager(store=store, compactor=compactor)
    await asyncio.wait_for(
        asyncio.gather(
            mgr.get_chat_history_async("chat_001"),
            mgr.get_chat_history_async("chat_002"),
        ),
        timeout=1,
    )
    assert store.started == {"chat_001", "chat_002"}


@pytest.mark.asyncio
async def test_add_user_message_async(mgr):
    await mgr.add_user_message_async(
        "chat_001", "hello", message_id="msg_001", sender_id="user_001"
    )
    history = await mgr.get_chat_history_async("chat_001")
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["raw_content"] == "hello"


@pytest.mark.asyncio
async def test_add_assistant_message_async(mgr):
    await mgr.add_assistant_message_async("chat_001", "hi there", message_id="msg_002")
    history = await mgr.get_chat_history_async("chat_001")
    assert history[0]["role"] == "assistant"


@pytest.mark.asyncio
async def test_add_tool_result_async(mgr):
    await mgr.add_tool_result_async(
        "chat_001", "search", '{"result": "ok"}', "call_001"
    )
    history = await mgr.get_chat_history_async("chat_001")
    assert history[0]["role"] == "tool"


def test_set_messages_persists_without_event_loop():
    from core.managers.chat_context import ChatContext
    from core.managers.chat_message import ChatMessage

    store = MemoryContextStore()
    ctx = ChatContext("chat_001", store)
    ctx.set_messages([ChatMessage(role="assistant", content="summary", timestamp=1)])

    assert store.load("chat_001")[0]["content"] == "summary"


@pytest.mark.asyncio
async def test_set_messages_respects_max_history_and_restores():
    from core.managers.chat_context import ChatContext
    from core.managers.chat_message import ChatMessage

    store = MemoryContextStore()
    ctx = ChatContext("chat_001", store, max_history=2)
    messages = [
        ChatMessage(role="user", content="one", timestamp=1),
        ChatMessage(role="user", content="two", timestamp=2),
        ChatMessage(role="assistant", content="three", timestamp=3),
    ]

    ctx.set_messages(messages)
    await ctx._save_task
    restored = ChatContext("chat_001", store, max_history=2)
    assert restored.restore_from_store() is True
    assert [item.content for item in restored.get_history()] == ["two", "three"]


@pytest.mark.asyncio
async def test_set_messages_reschedules_after_pending_save():
    from core.managers.chat_context import ChatContext
    from core.managers.chat_message import ChatMessage

    class BlockingStore(MemoryContextStore):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()
            self.flush_count = 0

        def flush(self, chat_id, messages):
            self.flush_count += 1
            if self.flush_count == 1:
                self.started.set()
                self.release.wait(timeout=1)
            super().flush(chat_id, messages)

    store = BlockingStore()
    ctx = ChatContext("chat_001", store)
    ctx.add_message("user", "before")
    await asyncio.to_thread(store.started.wait, 1)
    ctx.set_messages([ChatMessage(role="assistant", content="after", timestamp=2)])
    store.release.set()
    await ctx._save_task
    await asyncio.sleep(0)
    if ctx._save_pending or store.flush_count == 1:
        await ctx._save_task

    assert store.flush_count == 2
    assert store.load("chat_001")[0]["content"] == "after"


@pytest.mark.asyncio
async def test_set_messages_persists_replacement():
    from core.managers.chat_context import ChatContext
    from core.managers.chat_message import ChatMessage

    store = MemoryContextStore()
    ctx = ChatContext("chat_001", store)
    replacement = ChatMessage(role="assistant", content="summary", timestamp=1)

    ctx.set_messages([replacement])
    await ctx._save_task

    assert store.load("chat_001")[0]["content"] == "summary"


@pytest.mark.asyncio
async def test_remove_last_user_message_match(mgr):
    await mgr.add_user_message_async("chat_001", "hello", message_id="msg_001")
    result = await mgr.remove_last_user_message_if_async("chat_001", "msg_001")
    assert result is True
    assert await mgr.get_chat_history_async("chat_001") == []


@pytest.mark.asyncio
async def test_remove_last_user_message_no_match(mgr):
    await mgr.add_user_message_async("chat_001", "hello", message_id="msg_001")
    result = await mgr.remove_last_user_message_if_async("chat_001", "wrong_id")
    assert result is False
    assert len(await mgr.get_chat_history_async("chat_001")) == 1


@pytest.mark.asyncio
async def test_remove_last_user_message_wrong_role(mgr):
    await mgr.add_assistant_message_async("chat_001", "hello", message_id="msg_001")
    result = await mgr.remove_last_user_message_if_async("chat_001", "msg_001")
    assert result is False


@pytest.mark.asyncio
async def test_remove_last_user_message_empty_chat(mgr):
    result = await mgr.remove_last_user_message_if_async("unknown", "msg_001")
    assert result is False


# ── compact_history_if_needed ──


@pytest.mark.asyncio
async def test_compact_history_if_needed_noop(mgr):
    compacted, usage, count = await mgr.compact_history_if_needed("chat_001")
    assert compacted is False
    assert usage is None
    assert count == 0


@pytest.mark.asyncio
async def test_compact_history_if_needed_forwards_force_and_applies_result(
    mgr, compactor
):
    from core.managers.chat_message import ChatMessage

    await mgr.add_user_message_async("chat_001", "old")
    original = ChatMessage(role="user", content="old", timestamp=1)
    replacement = ChatMessage(role="assistant", content="summary", timestamp=1)
    compactor.result = CompactionResult(True, [replacement], {"total_tokens": 3})

    compacted, usage, count = await mgr.compact_history_if_needed(
        "chat_001", force=True
    )

    assert compacted is True
    assert usage == {"total_tokens": 3}
    assert count == 1
    assert (await mgr.get_chat_history_async("chat_001"))[0]["content"] == "summary"
    assert compactor.calls[0][1] is True
    assert compactor.calls[0][0][0].content == original.content


@pytest.mark.asyncio
async def test_compaction_materializes_timeline_only_visible_events(mgr, compactor):
    mgr.set_timeline(
        SimpleNamespace(
            snapshot=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        event_id="delivery:d1",
                        role="assistant",
                        message_id="",
                        content="accepted answer",
                        timestamp=1,
                    ),
                )
            )
        )
    )

    await mgr.compact_history_if_needed("chat_001", force=True)

    assert [message.content for message in compactor.calls[0][0]] == ["accepted answer"]


@pytest.mark.asyncio
async def test_clear_history_also_clears_timeline(mgr):
    class Timeline:
        def __init__(self):
            self.cleared = []

        async def clear_chat(self, chat_id):
            self.cleared.append(chat_id)

    timeline = Timeline()
    mgr.set_timeline(timeline)
    await mgr.clear_chat_history_async("chat_001")

    assert timeline.cleared == ["chat_001"]


@pytest.mark.asyncio
async def test_compaction_same_chat_is_serialized(mgr, compactor):
    started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    maximum_active = 0

    async def compact(messages, *, force=False):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        started.set()
        await release.wait()
        active -= 1
        return CompactionResult(False, list(messages), None)

    compactor.compact = compact
    first = asyncio.create_task(mgr.compact_history_if_needed("chat_001"))
    await started.wait()
    second = asyncio.create_task(mgr.compact_history_if_needed("chat_001"))
    await asyncio.sleep(0)
    assert maximum_active == 1
    release.set()
    await asyncio.gather(first, second)
    assert maximum_active == 1


@pytest.mark.asyncio
async def test_compaction_different_chats_can_run_in_parallel(mgr, compactor):
    started = []
    release = asyncio.Event()

    async def compact(messages, *, force=False):
        started.append(len(started))
        if len(started) == 2:
            release.set()
        await release.wait()
        return CompactionResult(False, list(messages), None)

    compactor.compact = compact
    tasks = [
        asyncio.create_task(mgr.compact_history_if_needed("chat_001")),
        asyncio.create_task(mgr.compact_history_if_needed("chat_002")),
    ]
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
    assert len(started) == 2

    await mgr.add_user_message_async("chat_001", "keep")
    before = await mgr.get_chat_history_async("chat_001")

    compacted, usage, _ = await mgr.compact_history_if_needed("chat_001")

    assert compacted is False
    assert usage is None
    assert await mgr.get_chat_history_async("chat_001") == before


@pytest.mark.asyncio
async def test_cleanup_inactive_contexts(mgr):
    store = mgr.store
    store.flush("chat_001", [{"role": "user", "content": "old", "timestamp": 0}])
    await mgr.get_chat_history_async("chat_001")
    removed = await mgr.cleanup_inactive_contexts_async(max_inactivity=0)
    assert "chat_001" in removed
    assert "chat_001" not in await mgr.get_all_chat_ids_async()


@pytest.mark.asyncio
async def test_cleanup_does_not_release_file_lock_for_retained_context(compactor):
    class TrackingStore(MemoryContextStore):
        def __init__(self):
            super().__init__()
            self.released = []

        def release_file_lock(self, chat_id):
            self.released.append(chat_id)

    store = TrackingStore()
    mgr = ChatContextManager(store=store, compactor=compactor)
    await mgr.add_user_message_async("chat_001", "active")

    removed = await mgr.cleanup_inactive_contexts_async(max_inactivity=3600)

    assert removed == []
    assert store.released == []


@pytest.mark.asyncio
async def test_remove_context_waits_for_pending_save(compactor):
    class BlockingStore(MemoryContextStore):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def flush(self, chat_id, messages):
            self.started.set()
            self.release.wait(timeout=1)
            super().flush(chat_id, messages)

    store = BlockingStore()
    mgr = ChatContextManager(store=store, compactor=compactor)
    await mgr.add_user_message_async("chat_001", "pending")
    await asyncio.to_thread(store.started.wait, 1)

    removal = asyncio.create_task(mgr.remove_context_async("chat_001"))
    await asyncio.sleep(0)
    assert not removal.done()

    store.release.set()
    await asyncio.wait_for(removal, timeout=1)
    assert await mgr.get_all_chat_ids_async() == []
    assert store.load("chat_001")[0]["raw_content"] == "pending"


# ── 聊天类型 ──


@pytest.mark.asyncio
async def test_record_and_get_chat_type(mgr):
    await mgr.record_chat_type("chat_001", True)
    assert mgr.get_chat_type("chat_001") is True
