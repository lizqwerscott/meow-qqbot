from unittest.mock import MagicMock

import pytest

from core.managers.context_manager import ChatContextManager
from core.managers.context_store import MemoryContextStore


@pytest.fixture
def store():
    return MemoryContextStore()


@pytest.fixture
def mgr(store):
    return ChatContextManager(store=store)


@pytest.mark.asyncio
async def test_get_context_async_creates_new(mgr):
    ctx = await mgr.get_context_async("chat_001")
    assert ctx is not None
    assert ctx.chat_id == "chat_001"
    assert "chat_001" in mgr.contexts


@pytest.mark.asyncio
async def test_get_context_async_reuses_existing(mgr):
    ctx1 = await mgr.get_context_async("chat_001")
    ctx2 = await mgr.get_context_async("chat_001")
    assert ctx1 is ctx2


@pytest.mark.asyncio
async def test_get_context_async_loads_from_store(mgr, store):
    store.flush("chat_001", [{"role": "user", "content": "hello", "timestamp": 100.0}])
    ctx = await mgr.get_context_async("chat_001")
    assert len(ctx.history) == 1
    assert ctx.history[0].content == "hello"


@pytest.mark.asyncio
async def test_add_user_message_async(mgr):
    await mgr.add_user_message_async("chat_001", "hello", message_id="msg_001", sender_id="user_001")
    ctx = await mgr.get_context_async("chat_001")
    assert len(ctx.history) == 1
    assert ctx.history[0].role == "user"
    assert ctx.history[0].content == "hello"


@pytest.mark.asyncio
async def test_add_assistant_message_async(mgr):
    await mgr.add_assistant_message_async("chat_001", "hi there", message_id="msg_002")
    ctx = await mgr.get_context_async("chat_001")
    assert ctx.history[0].role == "assistant"


@pytest.mark.asyncio
async def test_add_tool_result_async(mgr):
    await mgr.add_tool_result_async("chat_001", "search", '{"result": "ok"}', "call_001")
    ctx = await mgr.get_context_async("chat_001")
    assert ctx.history[0].role == "tool"


# ── remove_last_user_message_if_async ──

@pytest.mark.asyncio
async def test_remove_last_user_message_match(mgr):
    await mgr.add_user_message_async("chat_001", "hello", message_id="msg_001")
    result = await mgr.remove_last_user_message_if_async("chat_001", "msg_001")
    assert result is True
    ctx = await mgr.get_context_async("chat_001")
    assert len(ctx.history) == 0


@pytest.mark.asyncio
async def test_remove_last_user_message_no_match(mgr):
    await mgr.add_user_message_async("chat_001", "hello", message_id="msg_001")
    result = await mgr.remove_last_user_message_if_async("chat_001", "wrong_id")
    assert result is False
    ctx = await mgr.get_context_async("chat_001")
    assert len(ctx.history) == 1


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
    ai_service = MagicMock()
    compacted, usage, ctx = await mgr.compact_history_if_needed("chat_001", ai_service)
    assert compacted is False
    assert usage is None


# ── cleanup_inactive_contexts ──

@pytest.mark.asyncio
async def test_cleanup_inactive_contexts(mgr):
    ctx = await mgr.get_context_async("chat_001")
    ctx.last_activity = 0  # 强制过期
    removed = mgr.cleanup_inactive_contexts(max_inactivity=0)
    assert "chat_001" in removed
    assert "chat_001" not in mgr.contexts


# ── 聊天类型 ──

@pytest.mark.asyncio
async def test_record_and_get_chat_type(mgr):
    await mgr.record_chat_type("chat_001", True)
    assert mgr.get_chat_type("chat_001") is True
