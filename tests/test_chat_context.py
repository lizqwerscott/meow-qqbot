import time

import pytest

from core.managers.chat_context import ChatContext
from core.managers.chat_message import ChatMessage
from core.managers.context_store import MemoryContextStore


# ── helpers ──


def make_ctx(chat_id="test_001", max_history=100):
    store = MemoryContextStore()
    ctx = ChatContext(chat_id=chat_id, store=store, max_history=max_history)
    return ctx, store


def add_assistant_with_toolcalls(ctx, msg_id: str, call_ids: list[str], content="思考中"):
    calls = [{"id": cid, "type": "function", "function": {"name": "tool", "arguments": "{}"}} for cid in call_ids]
    ctx.add_assistant_message(content, message_id=msg_id, tool_calls=calls)


def add_tool_result(ctx, call_id: str, content="ok"):
    ctx.add_tool_result("some_tool", content, tool_call_id=call_id)


# ── 消息写入/读取 ──


def test_add_and_read_user_message():
    ctx, _ = make_ctx()
    ctx.add_user_message("hello", message_id="m1", sender_id="u1", name="Alice")
    assert len(ctx.history) == 1
    assert ctx.history[0].role == "user"
    assert ctx.history[0].content == "hello"
    assert ctx.history[0].message_id == "m1"


def test_add_and_read_assistant():
    ctx, _ = make_ctx()
    ctx.add_assistant_message("hi", message_id="m2")
    assert ctx.history[0].role == "assistant"


def test_add_and_read_tool_result():
    ctx, _ = make_ctx()
    ctx.add_tool_result("search", '{"result": "ok"}', "call_001")
    assert ctx.history[0].role == "tool"


def test_max_history_limit():
    ctx, _ = make_ctx(max_history=3)
    for i in range(5):
        ctx.add_user_message(f"msg{i}", message_id=f"m{i}")
    assert len(ctx.history) == 3
    assert ctx.history[0].content == "msg2"
    assert ctx.history[-1].content == "msg4"


# ── remove_last_message_if ──


def test_remove_last_message_match():
    ctx, _ = make_ctx()
    ctx.add_user_message("hello", message_id="m1")
    assert ctx.remove_last_message_if("user", "m1") is True
    assert ctx.is_empty()


def test_remove_last_message_no_match():
    ctx, _ = make_ctx()
    ctx.add_user_message("hello", message_id="m1")
    assert ctx.remove_last_message_if("user", "wrong_id") is False
    assert ctx.get_history_count() == 1


def test_remove_last_message_wrong_role():
    ctx, _ = make_ctx()
    ctx.add_assistant_message("hello", message_id="m1")
    assert ctx.remove_last_message_if("user", "m1") is False


def test_remove_last_message_empty():
    ctx, _ = make_ctx()
    assert ctx.remove_last_message_if("user", "m1") is False


# ── remove_orphaned_tool_calls ──


def test_no_orphans_clean():
    ctx, _ = make_ctx()
    add_assistant_with_toolcalls(ctx, "a1", ["c1", "c2"])
    add_tool_result(ctx, "c1")
    add_tool_result(ctx, "c2")
    assert ctx.remove_orphaned_tool_calls() == 0
    assert ctx.get_history_count() == 3


def test_orphan_assistant_removed():
    ctx, _ = make_ctx()
    add_assistant_with_toolcalls(ctx, "a1", ["c1"])  # no tool response → orphan
    ctx.add_user_message("hello", message_id="m1")
    assert ctx.remove_orphaned_tool_calls() == 1
    assert ctx.get_history_count() == 1  # only user msg left


def test_orphan_tool_message_removed():
    ctx, _ = make_ctx()
    add_tool_result(ctx, "c1")  # no preceding assistant → orphan
    ctx.add_user_message("hi", message_id="m1")
    assert ctx.remove_orphaned_tool_calls() == 1
    assert ctx.get_history_count() == 1


def test_partial_orphan_tool_calls():
    """部分 tool_calls 无对应响应 → 整条 assistant 被移除。"""
    ctx, _ = make_ctx()
    add_assistant_with_toolcalls(ctx, "a1", ["c1", "c2"])
    add_tool_result(ctx, "c1")  # c2 is orphan
    assert ctx.remove_orphaned_tool_calls() == 2  # assistant + tool(c1)
    assert ctx.get_history_count() == 0


def test_orphan_removed_then_re_added():
    ctx, _ = make_ctx()
    add_assistant_with_toolcalls(ctx, "a1", ["c1"])
    assert ctx.remove_orphaned_tool_calls() == 1  # a1 removed
    ctx.add_user_message("retry", message_id="m2")
    add_assistant_with_toolcalls(ctx, "a2", ["c2"])
    add_tool_result(ctx, "c2")
    assert ctx.remove_orphaned_tool_calls() == 0  # clean


# ── restore_from_store ──


def test_restore_from_store():
    ctx, store = make_ctx()
    store.flush(ctx.chat_id, [
        {"role": "user", "content": "hello", "timestamp": 100.0, "message_id": "m1"},
        {"role": "assistant", "content": "hi", "timestamp": 101.0, "message_id": "m2"},
    ])
    assert ctx.restore_from_store() is True
    assert ctx.get_history_count() == 2
    assert ctx.history[0].content == "hello"


def test_restore_from_store_no_data():
    ctx, _ = make_ctx()
    assert ctx.restore_from_store() is False
    assert ctx.is_empty()


def test_restore_from_store_corrupted_entry():
    ctx, store = make_ctx()
    store.flush(ctx.chat_id, [
        {"role": "user", "content": "good", "timestamp": 100.0},
        {"role": "assistant", "content": "fine", "timestamp": 101.0},
    ])
    assert ctx.restore_from_store() is True
    assert ctx.get_history_count() == 2


# ── get_pruned_history ──


def test_pruned_empty():
    ctx, _ = make_ctx()
    assert ctx.get_pruned_history() == []


def test_pruned_simple():
    ctx, _ = make_ctx()
    ctx.add_user_message("hello", message_id="m1")
    ctx.add_assistant_message("world", message_id="m2")
    result = ctx.get_pruned_history(max_messages=10)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"


def test_pruned_tool_result_overflow():
    ctx, _ = make_ctx()
    ctx.add_user_message("tell me", message_id="m1")
    add_assistant_with_toolcalls(ctx, "a1", ["c1", "c2", "c3", "c4", "c5", "c6"])
    for i in range(6):
        add_tool_result(ctx, f"c{i+1}", "x" * 100)
    result = ctx.get_pruned_history(max_messages=20, max_tool_results=5)
    tool_roles = [d for d in result if d.get("role") == "tool"]
    assert len(tool_roles) == 6  # all within protected boundary


def test_pruned_tool_soft_trim():
    """大量 filler 消息把 tool result 推出保护区 → 触发裁剪。"""
    ctx, _ = make_ctx(max_history=200)
    # 前 10 条 user/assistant 推后保护区边界
    for i in range(10):
        ctx.add_user_message(f"q{i}", message_id=f"m_q{i}")
        ctx.add_assistant_message(f"a{i}", message_id=f"m_a{i}")
    add_assistant_with_toolcalls(ctx, "a_tool", ["c1"])
    add_tool_result(ctx, "c1", "x" * 30000)
    result = ctx.get_pruned_history(max_messages=20, max_tool_results=0, soft_trim=20000, hard_clear=180000)
    tool_entry = next(d for d in result if d.get("role") == "tool")
    assert "中间内容已裁剪" in tool_entry["content"]


def test_pruned_hard_clear():
    ctx, _ = make_ctx(max_history=200)
    for i in range(10):
        ctx.add_user_message(f"q{i}", message_id=f"m_q{i}")
        ctx.add_assistant_message(f"a{i}", message_id=f"m_a{i}")
    add_assistant_with_toolcalls(ctx, "a_tool", ["c1"])
    add_tool_result(ctx, "c1", "x" * 200000)
    result = ctx.get_pruned_history(max_messages=20, max_tool_results=0, soft_trim=20000, hard_clear=180000)
    tool_entry = next(d for d in result if d.get("role") == "tool")
    assert "已裁剪" in tool_entry["content"]
    assert "x" * 200000 not in tool_entry["content"]


def test_pruned_keeps_last_assistants():
    ctx, _ = make_ctx()
    ctx.add_user_message("q1", message_id="m1")
    ctx.add_assistant_message("a1", message_id="a1")
    for i in range(5):
        ctx.add_user_message(f"q{i+2}", message_id=f"m{i+2}")
        ctx.add_assistant_message(f"a{i+2}", message_id=f"a{i+2}")
    result = ctx.get_pruned_history(max_messages=20)
    assert len(result) == 12


# ── 状态查询 ──


def test_get_inactivity_time():
    ctx, _ = make_ctx()
    assert ctx.get_inactivity_time() >= 0


def test_is_empty():
    ctx, _ = make_ctx()
    assert ctx.is_empty() is True
    ctx.add_user_message("hi", message_id="m1")
    assert ctx.is_empty() is False


def test_get_last_message():
    ctx, _ = make_ctx()
    assert ctx.get_last_message() is None
    ctx.add_user_message("hi", message_id="m1")
    assert ctx.get_last_message().content == "hi"


def test_clear_history():
    ctx, store = make_ctx()
    ctx.add_user_message("hi", message_id="m1")
    assert ctx.is_empty() is False
    ctx.clear_history()
    assert ctx.is_empty() is True


def test_set_messages():
    ctx, _ = make_ctx()
    msgs = [ChatMessage(role="user", content="a", timestamp=1.0),
            ChatMessage(role="assistant", content="b", timestamp=2.0)]
    ctx.set_messages(msgs)
    assert ctx.get_history_count() == 2


# ── 过期检查 ──


def test_is_expired_with_history():
    ctx, _ = make_ctx()
    ctx.add_user_message("old", message_id="m1")
    ctx.history[-1].timestamp = time.time() - 1000  # 1000 < 86400
    assert ctx._is_expired(max_age=86400) is False


def test_is_expired_empty():
    ctx, _ = make_ctx()
    assert ctx._is_expired() is True


# ── estimate_tokens ──


def test_estimate_tokens():
    ctx, _ = make_ctx()
    ctx.add_user_message("hello world", message_id="m1")
    n = ctx.estimate_tokens_for_history()
    assert n > 0


def test_estimate_tokens_empty():
    ctx, _ = make_ctx()
    assert ctx.estimate_tokens_for_history() == 0
