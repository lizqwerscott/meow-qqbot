"""测试 ArchiveManager 内部纯逻辑方法：_extract_replay_messages、_build_summary_group、consume_summary。"""

import time
from unittest.mock import MagicMock

import pytest

from core.managers.archive_manager import (
    ArchiveManager, _build_summary_group, _get_memory_dir,
)
from core.managers.chat_message import ChatMessage


# ── helpers ──


def make_msg(role="user", content="hello", msg_id="m1", sender_id="u1", **kw):
    return ChatMessage(
        role=role, content=content, timestamp=100.0,
        message_id=msg_id, sender_id=sender_id, name=None,
        **kw,
    )


@pytest.fixture
def mgr():
    cm = MagicMock()
    return ArchiveManager(
        context_manager=cm,
        memory_dir="/tmp/mem",
        summary_count=200,
        merge_window_seconds=300,
    )


# ── _extract_replay_messages ──


def test_extract_skips_tool_messages(mgr):
    msgs = [
        make_msg("user", "你好", "m1"),
        make_msg("assistant", "思考中", "m2", tool_calls=[{"id": "c1"}]),
        make_msg("tool", '{"ok": true}', "m3", sender_id="tool"),
        make_msg("assistant", "回复", "m4"),
    ]
    result = mgr._extract_replay_messages(msgs, count=10)
    assert len(result) == 2
    assert all(m.role != "tool" for m in result)


def test_extract_skips_emoji_assistant(mgr):
    msgs = [
        make_msg("user", "发个表情", "m1"),
        make_msg("assistant", "[助手发送了一个表情]", "m2"),
    ]
    result = mgr._extract_replay_messages(msgs, count=10)
    assert len(result) == 1
    assert result[0].role == "user"


def test_extract_skips_system(mgr):
    msgs = [
        make_msg("user", "hi", "m1"),
        make_msg("assistant", "hello", "m2", sender_id="system"),
    ]
    result = mgr._extract_replay_messages(msgs, count=10)
    assert len(result) == 1


def test_extract_respects_count(mgr):
    msgs = [make_msg("user", f"msg{i}", f"m{i}") for i in range(10)]
    result = mgr._extract_replay_messages(msgs, count=3)
    assert len(result) == 3
    assert result[-1].content == "msg9"  # 最新的


def test_extract_preserves_order(mgr):
    msgs = [
        make_msg("user", "第一", "m1"),
        make_msg("assistant", "回复一", "m2"),
        make_msg("user", "第二", "m3"),
    ]
    result = mgr._extract_replay_messages(msgs, count=10)
    assert result[0].content == "第一"
    assert result[1].content == "回复一"
    assert result[2].content == "第二"


# ── _build_summary_group ──


def test_build_summary_group_user_only():
    lines = []
    group = [make_msg("user", "hello world", "m1")]
    _build_summary_group(lines, group, window_seconds=300)
    assert len(lines) == 1
    assert "hello world" in lines[0]


def test_build_summary_group_user_assistant():
    lines = []
    group = [
        make_msg("user", "你好", "m1"),
        make_msg("assistant", "嗨！", "m2"),
    ]
    _build_summary_group(lines, group, window_seconds=300)
    assert len(lines) == 1
    assert "你好" in lines[0]
    assert "嗨！" in lines[0]


def test_build_summary_group_assistant_only():
    lines = []
    group = [make_msg("assistant", "你好", "m1")]
    _build_summary_group(lines, group, window_seconds=300)
    assert len(lines) == 1


# ── consume_summary ──


def test_consume_summary_not_pending(mgr):
    mgr._pending_injection = set()
    result = mgr.consume_summary("chat_001")
    assert result is None


def test_consume_summary_consumes_and_removes(mgr, tmp_path):
    mgr._pending_injection = {"chat_001"}
    mgr._memory_dir = str(tmp_path)
    mgr._summary_days = 10  # 覆盖更多日期
    mem_dir = _get_memory_dir(mgr._memory_dir, "chat_001")
    mem_dir.mkdir(parents=True, exist_ok=True)
    from core.managers.archive_manager import _date_str
    recent = _date_str(time.time() - 86400)
    (mem_dir / f"{recent}.md").write_text("昨天的重要对话", encoding="utf-8")

    result = mgr.consume_summary("chat_001")
    assert result is not None
    assert "昨天的重要对话" in result
    assert "chat_001" not in mgr._pending_injection


def test_consume_summary_only_once(mgr, tmp_path):
    mgr._pending_injection = {"chat_001"}
    mgr._memory_dir = str(tmp_path)
    mgr._summary_days = 10
    mem_dir = _get_memory_dir(mgr._memory_dir, "chat_001")
    mem_dir.mkdir(parents=True, exist_ok=True)
    from core.managers.archive_manager import _date_str
    recent = _date_str(time.time())
    (mem_dir / f"{recent}.md").write_text("摘要内容", encoding="utf-8")

    first = mgr.consume_summary("chat_001")
    second = mgr.consume_summary("chat_001")
    assert first is not None
    assert second is None  # 一次性消费


# ── _get_memory_dir ──


def test_get_memory_dir():
    path = _get_memory_dir("/root/mem", "chat_001")
    assert str(path) == "/root/mem/chat_001"


def test_get_memory_dir_nested():
    path = _get_memory_dir("/root/mem", "group_001")
    assert str(path) == "/root/mem/group_001"


# ── _format_summary_text ──


def test_format_summary_text(mgr):
    msgs = [
        make_msg("user", "你好", "m1"),
        make_msg("assistant", "嗨！有什么可以帮你的？", "m2"),
    ]
    result = mgr._format_summary_text(msgs, count=200, is_group=True, chat_id="chat_001", date="2024-06-15")
    assert result is not None
    assert "Session: 2024-06-15" in result
    assert "你好" in result
    assert "嗨！有什么可以帮你的？" in result


def test_format_summary_text_empty(mgr):
    result = mgr._format_summary_text([], count=200, is_group=True, chat_id="chat_001", date="2024-06-15")
    assert result is None


def test_format_summary_text_filters_tool(mgr):
    msgs = [
        make_msg("user", "搜索一下", "m1"),
        make_msg("assistant", "思考中", "m2", tool_calls=[{"id": "c1"}]),
        make_msg("tool", '{"result": "找到了"}', "m3"),
        make_msg("assistant", "找到了，在这里", "m4"),
    ]
    result = mgr._format_summary_text(msgs, count=200, is_group=True, chat_id="chat_001", date="2024-06-15")
    # tool 和 tool_calls 的 assistant 被过滤
    assert result is not None
    assert "搜索一下" in result
    assert "找到了，在这里" in result
