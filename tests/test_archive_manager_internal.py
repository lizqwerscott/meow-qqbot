"""测试 ArchiveManager 内部逻辑：_extract_replay_messages、_build_summary_group、
consume_summary、archive_if_stale 跨天触发与同日守卫。"""

import asyncio
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.command_handlers.archive import ArchiveCommand
from core.managers.archive_manager import (
    ArchiveManager,
    ArchiveResult,
    _build_summary_group,
    _date_str,
    _get_memory_dir,
)
from core.managers.chat_context import ChatContext
from core.managers.chat_message import ChatMessage
from core.managers.context_store import JSONLContextStore
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


class _FakeCM:
    """最小 context_manager 替身：_with_context_locked 直接执行 func(ctx)。"""

    def __init__(self, ctx, store=None, chat_types=None):
        self._ctx = ctx
        self.store = store or MagicMock()
        self._chat_types = chat_types or {}

    def get_chat_type(self, chat_id):
        return self._chat_types.get(chat_id)

    async def _with_context_locked(self, chat_id, func):
        return await func(self._ctx)


class _MutableCtx:
    """供 JSONL 归档集成测试使用的最小可变 context。"""

    def __init__(self, messages):
        self.history = list(messages)
        self.last_activity = 0.0

    def is_empty(self):
        return not self.history

    def get_history(self):
        return list(self.history)

    def set_messages(self, messages):
        self.history = list(messages)


def _make_ctx(msgs):
    ctx = MagicMock()
    ctx.is_empty.return_value = False
    ctx.get_history.return_value = msgs
    return ctx


@pytest.fixture
def mgr(tmp_path):
    cm = MagicMock()
    return ArchiveManager(
        context_manager=cm,
        memory_dir=str(tmp_path / "mem"),
        summary_count=200,
        merge_window_seconds=300,
    )


def test_tool_message_serialization_preserves_archive_metadata():
    """tool 记录落 JSONL 时保留原始时间，供归档单元分区使用。"""
    message = ChatMessage(
        role="tool",
        content='{"content":"ok"}',
        timestamp=123.0,
        tool_call_id="call_001",
        tool_name="read_file",
    )

    wire = message.to_dict()
    stored = message.to_storage_dict()
    restored = ChatMessage.from_dict(stored)

    assert wire == {
        "role": "tool",
        "tool_call_id": "call_001",
        "content": '{"content":"ok"}',
    }
    assert stored["timestamp"] == 123.0
    assert stored["tool_name"] == "read_file"
    assert restored.timestamp == 123.0
    assert restored.tool_name == "read_file"


@pytest.mark.asyncio
async def test_archive_projection_prefers_timeline_visible_text_but_keeps_protocol():
    manager = ArchiveManager(context_manager=MagicMock())
    manager.set_timeline(
        SimpleNamespace(
            snapshot=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        role="user", message_id="user-1", content="canonical user"
                    ),
                    SimpleNamespace(
                        role="assistant", message_id="user-1", content="canonical reply"
                    ),
                )
            )
        )
    )
    tool_calls = [{"id": "call-1", "function": {"name": "read_file"}}]
    messages = [
        ChatMessage("user", "stale user", 1, message_id="user-1"),
        ChatMessage(
            "assistant", "internal", 2, message_id="user-1", tool_calls=tool_calls
        ),
    ]

    projected = await manager._apply_timeline_projection("chat", messages)

    assert projected[0].content == "canonical user"
    assert projected[1] is messages[1]
    assert projected[1].tool_calls == tool_calls


@pytest.mark.asyncio
async def test_archive_materializes_timeline_only_visible_events():
    manager = ArchiveManager(context_manager=MagicMock())
    manager.set_timeline(
        SimpleNamespace(
            snapshot=AsyncMock(
                return_value=(
                    SimpleNamespace(
                        event_id="user:u1",
                        role="user",
                        message_id="u1",
                        content="canonical user",
                        sender_id="u",
                        timestamp=1,
                    ),
                    SimpleNamespace(
                        event_id="delivery:d1",
                        role="assistant",
                        message_id="",
                        content="accepted answer",
                        sender_id="",
                        timestamp=2,
                    ),
                )
            )
        )
    )
    protocol = ChatMessage(
        "assistant",
        "internal",
        1.5,
        message_id="u1",
        tool_calls=[{"id": "call-1"}],
    )

    merged = await manager._merge_timeline_messages("chat", [protocol])

    assert [message.content for message in merged] == [
        "canonical user",
        "internal",
        "accepted answer",
    ]
    assert merged[1].tool_calls == [{"id": "call-1"}]


@pytest.mark.asyncio
async def test_archive_repairs_legacy_visible_gap_before_materializing_timeline():
    manager = ArchiveManager(context_manager=MagicMock())
    timeline = SimpleNamespace(
        repair_from_legacy_history=AsyncMock(),
        snapshot=AsyncMock(return_value=()),
    )
    manager.set_timeline(timeline)
    message = make_msg("user", "legacy", "u1", timestamp=1.0)

    merged = await manager._merge_timeline_messages("chat", [message])

    assert merged == [message]
    timeline.repair_from_legacy_history.assert_awaited_once_with(
        "chat", [message.to_storage_dict()]
    )


# ── _extract_replay_messages ──


def test_extract_skips_tool_messages(mgr):
    msgs = [
        make_msg("user", "你好", "m1"),
        make_msg("assistant", "思考中", "m2", tool_calls=[{"id": "c1"}]),
        make_msg("tool", '{"ok": true}', "m3", sender_id="tool"),
        make_msg("assistant", "回复", "m4"),
    ]
    result = mgr._extract_replay_messages(msgs)
    assert [message.role for message in result] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_extract_skips_emoji_assistant(mgr):
    msgs = [
        make_msg("user", "发个表情", "m1"),
        make_msg("assistant", "[助手发送了一个表情]", "m2"),
    ]
    result = mgr._extract_replay_messages(msgs)
    assert len(result) == 1
    assert result[0].role == "user"


def test_extract_skips_system(mgr):
    msgs = [
        make_msg("user", "hi", "m1"),
        make_msg("assistant", "hello", "m2", sender_id="system"),
    ]
    result = mgr._extract_replay_messages(msgs)
    assert len(result) == 1


def test_extract_replays_complete_recent_time_segment(mgr):
    """回放按消息间隔切分，不能用固定条数从最近对话中间截断。"""
    msgs = [
        make_msg("user", "早先的问题", "m1", timestamp=1_000),
        make_msg("assistant", "早先的回答", "m2", timestamp=1_010),
        make_msg("user", "最近的问题", "m3", timestamp=2_000),
        make_msg("assistant", "最近的回答", "m4", timestamp=2_010),
    ]
    result = mgr._extract_replay_messages(msgs)
    assert [message.content for message in result] == ["最近的问题", "最近的回答"]


def test_extract_replay_zero_gap_returns_nothing(mgr):
    msgs = [make_msg("user", "a", "m0")]
    assert mgr._extract_replay_messages(msgs, gap_seconds=0) == []


def test_extract_preserves_order(mgr):
    msgs = [
        make_msg("user", "第一", "m1"),
        make_msg("assistant", "回复一", "m2"),
        make_msg("user", "第二", "m3"),
    ]
    result = mgr._extract_replay_messages(msgs)
    assert result[0].content == "第一"
    assert result[1].content == "回复一"
    assert result[2].content == "第二"


def test_extract_replay_skips_empty_content(mgr):
    """回放与摘要共用过滤谓词：空内容消息不参与回放。"""
    msgs = [
        make_msg("user", "", "m0"),
        make_msg("user", "有内容", "m1"),
    ]
    result = mgr._extract_replay_messages(msgs)
    assert len(result) == 1
    assert result[0].content == "有内容"


# ── archive_if_stale（跨天触发 / 同日守卫）──


def test_archive_messages_same_timestamp_does_not_overwrite(tmp_path):
    """存储层必须在同一秒内分配不同 archive 路径。"""
    store = JSONLContextStore(str(tmp_path / "sessions"))

    first = store.archive_messages(
        "chat_001", "2025-01-02T10-00-00", [{"content": "first"}]
    )
    second = store.archive_messages(
        "chat_001", "2025-01-02T10-00-00", [{"content": "second"}]
    )

    assert first is not None
    assert second is not None
    assert first != second
    assert [item["content"] for item in store.read_archive(first)] == ["first"]
    assert [item["content"] for item in store.read_archive(second)] == ["second"]


@pytest.mark.asyncio
async def test_archive_waits_for_pending_context_save(tmp_path, monkeypatch):
    """旧快照保存未完成时，归档不得让它在最终写入后复活旧 history。"""
    day1 = datetime(2025, 1, 1, 20, 0, 0).timestamp()
    day2 = datetime(2025, 1, 2, 10, 0, 0).timestamp()

    class BlockingStore(JSONLContextStore):
        def __init__(self, base_dir):
            super().__init__(base_dir)
            self.started = threading.Event()
            self.release = threading.Event()
            self.flush_count = 0

        def flush(self, chat_id, messages):
            self.flush_count += 1
            if self.flush_count == 1:
                self.started.set()
                self.release.wait(timeout=2)
            super().flush(chat_id, messages)

    store = BlockingStore(str(tmp_path / "sessions"))
    ctx = ChatContext("chat_001", store)
    ctx.add_message("user", "old", timestamp=day1)
    await asyncio.to_thread(store.started.wait, 1)
    ctx.add_message("user", "today", timestamp=day2)

    manager = ArchiveManager(
        context_manager=_FakeCM(ctx, store), memory_dir=str(tmp_path / "mem")
    )
    monkeypatch.setattr("core.managers.archive_manager.time.time", lambda: day2)
    archive_task = asyncio.create_task(
        manager.archive_if_stale("chat_001", is_group=False)
    )
    await asyncio.sleep(0)
    assert not archive_task.done()

    store.release.set()
    result = await archive_task
    await ctx.wait_for_save_async()

    assert result is not None
    active = store.load("chat_001")
    assert active is not None
    assert [item["raw_content"] for item in active] == ["today"]
    assert [
        item["raw_content"] for item in store.read_archive(result.archive_path)
    ] == ["old"]


@pytest.mark.asyncio
async def test_archive_if_stale_crossed_day_archives(tmp_path):
    now = time.time()
    msgs = [
        ChatMessage(role="user", content="昨天", timestamp=now - 86400),
        ChatMessage(role="user", content="今天", timestamp=now),
    ]
    cm = _FakeCM(_make_ctx(msgs))
    mgr = ArchiveManager(context_manager=cm, memory_dir=str(tmp_path / "mem"))
    result = await mgr.archive_if_stale("chat_001", is_group=False)
    assert result is not None
    assert result.reason == "daily"
    assert "chat_001" in mgr._pending_injection  # 自动归档注入摘要
    assert mgr._last_daily_archive.get("chat_001") == _date_str(now)


@pytest.mark.asyncio
async def test_archive_if_stale_same_day_guard(tmp_path):
    now = time.time()
    chat_id = "chat_001"
    ctx = _MutableCtx(
        [
            ChatMessage(role="user", content="昨天", timestamp=now - 86400),
            ChatMessage(role="user", content="今天", timestamp=now),
        ]
    )
    store = JSONLContextStore(str(tmp_path / "sessions"))
    cm = _FakeCM(ctx, store)
    mgr = ArchiveManager(context_manager=cm, memory_dir=str(tmp_path / "mem"))
    first = await mgr.archive_if_stale(chat_id, is_group=False)
    second = await mgr.archive_if_stale(chat_id, is_group=False)
    assert first is not None
    assert second is None  # 同一天没有新的旧消息时不重复归档


@pytest.mark.asyncio
async def test_archive_if_stale_archives_late_old_message_after_daily_guard(
    tmp_path, monkeypatch
):
    """当天已归档后，迟到的旧消息仍必须在同一天补归档。"""
    day1 = datetime(2025, 1, 1, 20, 0, 0).timestamp()
    day2 = datetime(2025, 1, 2, 10, 0, 0).timestamp()
    chat_id = "chat_001"
    store = JSONLContextStore(str(tmp_path / "sessions"))
    ctx = _MutableCtx(
        [
            ChatMessage(role="user", content="on-time-old", timestamp=day1),
            ChatMessage(role="user", content="today", timestamp=day2),
        ]
    )
    manager = ArchiveManager(
        context_manager=_FakeCM(ctx, store), memory_dir=str(tmp_path / "mem")
    )

    monkeypatch.setattr("core.managers.archive_manager.time.time", lambda: day2)
    assert await manager.archive_if_stale(chat_id, is_group=False) is not None
    ctx.history.append(
        ChatMessage(role="user", content="late-old", timestamp=day1 + 60)
    )

    assert await manager.archive_if_stale(chat_id, is_group=False) is not None
    archive_contents = [
        [message["raw_content"] for message in store.read_archive(item["path"])]
        for item in store.list_archives(chat_id)
    ]
    assert sorted(archive_contents) == [["late-old"], ["on-time-old"]]


@pytest.mark.asyncio
async def test_archive_if_stale_today_only_no_trigger(tmp_path):
    now = time.time()
    msgs = [ChatMessage(role="user", content="今天", timestamp=now)]
    cm = _FakeCM(_make_ctx(msgs))
    mgr = ArchiveManager(context_manager=cm, memory_dir=str(tmp_path / "mem"))
    result = await mgr.archive_if_stale("chat_001", is_group=False)
    assert result is None


@pytest.mark.asyncio
async def test_archive_keeps_today_messages(tmp_path):
    """延迟触发时，今天早先的消息必须全部保留；已有今天单元则不回放昨天。"""
    now = time.time()
    yesterday = now - 86400
    msgs = [
        ChatMessage(
            role="user",
            content=f"昨天{i}",
            timestamp=yesterday - (4 - i) * 60,
            message_id=f"y{i}",
        )
        for i in range(5)
    ] + [
        ChatMessage(
            role="user",
            content=f"今天{i}",
            timestamp=now - (2 - i) * 60,
            message_id=f"t{i}",
        )
        for i in range(3)
    ]
    ctx = _make_ctx(msgs)
    cm = _FakeCM(ctx)
    mgr = ArchiveManager(context_manager=cm, memory_dir=str(tmp_path / "mem"))
    result = await mgr.archive_if_stale("chat_001", is_group=False)
    assert result is not None
    assert result.replay_count == 3  # 今天消息全部保留，昨天不回放

    kept = ctx.set_messages.call_args[0][0]
    contents = [m.content for m in kept]
    assert contents == ["今天0", "今天1", "今天2"]


@pytest.mark.asyncio
async def test_archive_replays_across_midnight_when_same_time_segment(tmp_path):
    """跨午夜间隔很短时，昨天尾段和今天消息属于同一个完整片段。"""
    today_start = datetime(2025, 1, 2, 0, 0, 0).timestamp()
    msgs = [
        ChatMessage(
            role="user",
            content="午夜前的问题",
            timestamp=today_start - 20,
        ),
        ChatMessage(
            role="assistant",
            content="午夜前的回答",
            timestamp=today_start - 10,
        ),
        ChatMessage(
            role="user",
            content="午夜后的补充",
            timestamp=today_start + 10,
        ),
    ]
    ctx = _make_ctx(msgs)
    cm = _FakeCM(ctx)
    mgr = ArchiveManager(
        context_manager=cm,
        memory_dir=str(tmp_path / "mem"),
        replay_gap_seconds=60,
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "core.managers.archive_manager.time.time", lambda: today_start + 30
        )
        result = await mgr.archive_if_stale("chat_001", is_group=False)

    assert result is not None
    kept = ctx.set_messages.call_args[0][0]
    assert [message.content for message in kept] == [
        "午夜前的问题",
        "午夜前的回答",
        "午夜后的补充",
    ]


@pytest.mark.asyncio
async def test_archive_does_not_replay_yesterday_when_today_has_content(tmp_path):
    """当天已有单元时，不再补昨天的原始消息。"""
    now = time.time()
    yesterday = now - 86400
    msgs = [
        ChatMessage(role="user", content="昨天的问题", timestamp=yesterday),
        ChatMessage(role="assistant", content="昨天的回答", timestamp=yesterday + 5),
        ChatMessage(role="user", content="今天已开始", timestamp=now),
    ]
    ctx = _make_ctx(msgs)
    cm = _FakeCM(ctx)
    mgr = ArchiveManager(context_manager=cm, memory_dir=str(tmp_path / "mem"))

    result = await mgr.archive_if_stale("chat_001", is_group=False)

    assert result is not None
    kept = ctx.set_messages.call_args[0][0]
    assert [message.content for message in kept] == ["今天已开始"]


@pytest.mark.asyncio
async def test_archive_replays_recent_yesterday_time_segment(tmp_path):
    """首条当天消息前，回放昨天最后一个连续时间段而非固定消息数。"""
    now = time.time()
    yesterday = now - 86400
    msgs = [
        ChatMessage(role="user", content="早先问题", timestamp=yesterday - 1_000),
        ChatMessage(role="assistant", content="早先回答", timestamp=yesterday - 995),
        ChatMessage(role="user", content="最近问题", timestamp=yesterday - 20),
        ChatMessage(role="assistant", content="最近回答", timestamp=yesterday - 15),
    ]
    ctx = _make_ctx(msgs)
    cm = _FakeCM(ctx)
    mgr = ArchiveManager(
        context_manager=cm,
        memory_dir=str(tmp_path / "mem"),
        replay_gap_seconds=60,
    )

    result = await mgr.archive_if_stale("chat_001", is_group=False)

    assert result is not None
    kept = ctx.set_messages.call_args[0][0]
    assert [message.content for message in kept] == ["最近问题", "最近回答"]


@pytest.mark.asyncio
async def test_archive_does_not_replay_yesterday_when_today_sparse(tmp_path):
    """今天已有单元时，不以固定条数补昨天。"""
    now = time.time()
    yesterday = now - 86400
    msgs = [
        ChatMessage(
            role="user",
            content=f"昨天{i}",
            timestamp=yesterday - (3 - i) * 60,
            message_id=f"y{i}",
        )
        for i in range(4)
    ] + [
        ChatMessage(
            role="user",
            content="今天0",
            timestamp=now,
            message_id="t0",
        )
    ]
    ctx = _make_ctx(msgs)
    cm = _FakeCM(ctx)
    mgr = ArchiveManager(context_manager=cm, memory_dir=str(tmp_path / "mem"))
    result = await mgr.archive_if_stale("chat_001", is_group=False)
    assert result is not None

    kept = ctx.set_messages.call_args[0][0]
    assert [message.content for message in kept] == ["今天0"]


@pytest.mark.asyncio
async def test_archive_does_not_archive_replayed_messages_twice(tmp_path, monkeypatch):
    """连续两天归档时，第一天留下的回放尾部不能写入第二天 archive。"""
    day1 = datetime(2025, 1, 1, 20, 0, 0).timestamp()
    day2 = datetime(2025, 1, 2, 10, 0, 0).timestamp()
    day3 = datetime(2025, 1, 3, 10, 0, 0).timestamp()
    chat_id = "chat_001"
    store = JSONLContextStore(str(tmp_path / "sessions"))
    ctx = _MutableCtx(
        [
            ChatMessage(
                role="user",
                content="day1-a",
                timestamp=day1 - 60,
                message_id="day1-a",
            ),
            ChatMessage(
                role="user",
                content="day1-b",
                timestamp=day1,
                message_id="day1-b",
            ),
        ]
    )
    cm = _FakeCM(ctx, store)
    mgr = ArchiveManager(
        context_manager=cm,
        memory_dir=str(tmp_path / "mem"),
    )

    monkeypatch.setattr("core.managers.archive_manager.time.time", lambda: day2)
    assert await mgr.archive_if_stale(chat_id, is_group=False) is not None
    ctx.history.append(
        ChatMessage(
            role="user",
            content="day2",
            timestamp=day2,
            message_id="day2",
        )
    )

    monkeypatch.setattr("core.managers.archive_manager.time.time", lambda: day3)
    assert await mgr.archive_if_stale(chat_id, is_group=False) is not None

    archive_contents = []
    for archive in store.list_archives(chat_id):
        archive_contents.append(
            [message["raw_content"] for message in store.read_archive(archive["path"])]
        )
    assert sorted(archive_contents) == [["day1-a", "day1-b"], ["day2"]]
    active_contents = [message["raw_content"] for message in store.load(chat_id) or []]
    assert active_contents == ["day2"]


@pytest.mark.asyncio
async def test_archive_clears_active_store_when_nothing_is_kept(tmp_path, monkeypatch):
    """旧历史没有可回放消息时，部分归档后不能留下旧 active JSONL。"""
    day1 = datetime(2025, 1, 1, 20, 0, 0).timestamp()
    day2 = datetime(2025, 1, 2, 10, 0, 0).timestamp()
    chat_id = "chat_001"
    store = JSONLContextStore(str(tmp_path / "sessions"))
    ctx = _MutableCtx(
        [
            ChatMessage(
                role="tool",
                content='{"content":"old result"}',
                timestamp=day1,
                tool_call_id="call_001",
                tool_name="read_file",
            )
        ]
    )
    cm = _FakeCM(ctx, store)
    mgr = ArchiveManager(context_manager=cm, memory_dir=str(tmp_path / "mem"))

    monkeypatch.setattr("core.managers.archive_manager.time.time", lambda: day2)
    result = await mgr.archive_if_stale(chat_id, is_group=False)
    assert result is not None
    assert result.archive_path is not None
    assert ctx.history == []
    assert store.load(chat_id) is None
    assert [message["role"] for message in store.read_archive(result.archive_path)] == [
        "tool"
    ]


@pytest.mark.asyncio
async def test_archive_groups_cross_day_tool_transaction_by_start_time(
    tmp_path, monkeypatch
):
    """工具事务按 assistant 开始日归档，但每条记录的 timestamp 不变。"""
    day1 = datetime(2025, 1, 1, 20, 0, 0).timestamp()
    day2 = datetime(2025, 1, 2, 10, 0, 0).timestamp()
    chat_id = "chat_001"
    tool_call = {
        "id": "call_001",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
    }
    store = JSONLContextStore(str(tmp_path / "sessions"))
    ctx = _MutableCtx(
        [
            ChatMessage(
                role="assistant",
                content="",
                timestamp=day1,
                message_id="assistant_001",
                tool_calls=[tool_call],
            ),
            ChatMessage(
                role="tool",
                content='{"content":"ok"}',
                timestamp=day2,
                tool_call_id="call_001",
                tool_name="read_file",
            ),
        ]
    )
    cm = _FakeCM(ctx, store)
    mgr = ArchiveManager(context_manager=cm, memory_dir=str(tmp_path / "mem"))

    monkeypatch.setattr("core.managers.archive_manager.time.time", lambda: day2)
    result = await mgr.archive_if_stale(chat_id, is_group=False)

    assert result is not None
    assert result.archive_path is not None
    archived = store.read_archive(result.archive_path)
    assert [message["role"] for message in archived] == ["assistant", "tool"]
    assert archived[0]["timestamp"] == day1
    assert archived[1]["timestamp"] == day2
    assert archived[1]["content"] == '{"content":"ok"}'
    assert ctx.history == []


@pytest.mark.asyncio
async def test_archive_groups_cross_day_tool_transaction_atomic(tmp_path):
    """工具调用及其结果以调用开始时间为整体归档。"""
    now = time.time()
    tool_call = {
        "id": "call_001",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
    }
    msgs = [
        ChatMessage(
            role="assistant",
            content="",
            timestamp=now - 86400,
            message_id="assistant_001",
            tool_calls=[tool_call],
        ),
        ChatMessage(
            role="tool",
            content='{"content":"ok"}',
            timestamp=now,
            tool_call_id="call_001",
            tool_name="read_file",
        ),
    ]
    ctx = _make_ctx(msgs)
    cm = _FakeCM(ctx)
    mgr = ArchiveManager(context_manager=cm, memory_dir=str(tmp_path / "mem"))

    result = await mgr.archive_if_stale("chat_001", is_group=False)
    assert result is not None

    kept = ctx.set_messages.call_args[0][0]
    assert kept == []


@pytest.mark.asyncio
async def test_archive_groups_all_results_of_cross_day_multi_tool_call(tmp_path):
    """多工具调用的全部结果按同一个调用开始时间整体归档。"""
    now = time.time()
    calls = [
        {
            "id": "call_001",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
        },
        {
            "id": "call_002",
            "type": "function",
            "function": {"name": "list_files", "arguments": "{}"},
        },
    ]
    msgs = [
        ChatMessage(
            role="assistant",
            content="",
            timestamp=now - 86400,
            message_id="assistant_001",
            tool_calls=calls,
        ),
        ChatMessage(
            role="tool",
            content='{"content":"a"}',
            timestamp=now - 86399,
            tool_call_id="call_001",
            tool_name="read_file",
        ),
        ChatMessage(
            role="tool",
            content='{"content":"b"}',
            timestamp=now,
            tool_call_id="call_002",
            tool_name="list_files",
        ),
    ]
    ctx = _make_ctx(msgs)
    cm = _FakeCM(ctx)
    mgr = ArchiveManager(context_manager=cm, memory_dir=str(tmp_path / "mem"))

    result = await mgr.archive_if_stale("chat_001", is_group=False)
    assert result is not None

    kept = ctx.set_messages.call_args[0][0]
    assert kept == []


@pytest.mark.asyncio
async def test_archive_manual_does_not_inject(tmp_path):
    now = time.time()
    msgs = [ChatMessage(role="user", content="随便", timestamp=now - 86400)]
    cm = _FakeCM(_make_ctx(msgs))
    mgr = ArchiveManager(context_manager=cm, memory_dir=str(tmp_path / "mem"))
    result = await mgr.archive_manual("chat_001", is_group=False)
    assert result is not None
    assert result.reason == "manual"
    assert "chat_001" not in mgr._pending_injection  # 手动归档不注入摘要


@pytest.mark.asyncio
async def test_manual_archive_persists_replay_prefix_across_restart(
    tmp_path, monkeypatch
):
    """手动归档保留的回放段重启后不能再次写入 archive。"""
    day1 = datetime(2025, 1, 1, 20, 0, 0).timestamp()
    day2 = datetime(2025, 1, 2, 10, 0, 0).timestamp()
    day3 = datetime(2025, 1, 3, 10, 0, 0).timestamp()
    chat_id = "chat_001"
    store = JSONLContextStore(str(tmp_path / "sessions"))
    ctx = _MutableCtx(
        [
            ChatMessage(role="user", content="day1-a", timestamp=day1),
            ChatMessage(role="assistant", content="day1-b", timestamp=day1 + 5),
        ]
    )
    cm = _FakeCM(ctx, store)
    mem_dir = str(tmp_path / "mem")
    mgr1 = ArchiveManager(context_manager=cm, memory_dir=mem_dir)

    monkeypatch.setattr("core.managers.archive_manager.time.time", lambda: day2)
    assert await mgr1.archive_manual(chat_id, is_group=False) is not None
    ctx.history.append(ChatMessage(role="user", content="day2", timestamp=day2))

    mgr2 = ArchiveManager(context_manager=cm, memory_dir=mem_dir)
    monkeypatch.setattr("core.managers.archive_manager.time.time", lambda: day3)
    assert await mgr2.archive_if_stale(chat_id, is_group=False) is not None

    archive_contents = [
        [message["raw_content"] for message in store.read_archive(item["path"])]
        for item in store.list_archives(chat_id)
    ]
    assert sorted(archive_contents) == [["day1-a", "day1-b"], ["day2"]]


@pytest.mark.asyncio
async def test_archive_skips_replay_suffix_after_history_truncation(
    tmp_path, monkeypatch
):
    """bounded history 挤出回放前缀开头后，剩余部分仍视为已归档。"""
    day1 = datetime(2025, 1, 1, 20, 0, 0).timestamp()
    day2 = datetime(2025, 1, 2, 10, 0, 0).timestamp()
    day3 = datetime(2025, 1, 3, 10, 0, 0).timestamp()
    chat_id = "chat_001"
    store = JSONLContextStore(str(tmp_path / "sessions"))
    ctx = _MutableCtx(
        [
            ChatMessage(role="user", content="day1-a", timestamp=day1),
            ChatMessage(role="assistant", content="day1-b", timestamp=day1 + 5),
        ]
    )
    cm = _FakeCM(ctx, store)
    mem_dir = str(tmp_path / "mem")
    mgr1 = ArchiveManager(context_manager=cm, memory_dir=mem_dir)

    monkeypatch.setattr("core.managers.archive_manager.time.time", lambda: day2)
    assert await mgr1.archive_if_stale(chat_id, is_group=False) is not None
    ctx.history = ctx.history[1:]
    ctx.history.append(ChatMessage(role="user", content="day2", timestamp=day2))

    mgr2 = ArchiveManager(context_manager=cm, memory_dir=mem_dir)
    monkeypatch.setattr("core.managers.archive_manager.time.time", lambda: day3)
    assert await mgr2.archive_if_stale(chat_id, is_group=False) is not None

    archive_contents = [
        [message["raw_content"] for message in store.read_archive(item["path"])]
        for item in store.list_archives(chat_id)
    ]
    assert sorted(archive_contents) == [["day1-a", "day1-b"], ["day2"]]


@pytest.mark.asyncio
async def test_archive_replays_only_previous_calendar_day(tmp_path, monkeypatch):
    """离线多天后，更早历史只能归档，不能作为原始回放保留。"""
    day1 = datetime(2025, 1, 1, 20, 0, 0).timestamp()
    day3 = datetime(2025, 1, 3, 10, 0, 0).timestamp()
    msgs = [
        ChatMessage(role="user", content="day1", timestamp=day1),
        ChatMessage(role="user", content="day3", timestamp=day3),
    ]
    ctx = _make_ctx(msgs)
    cm = _FakeCM(ctx)
    mgr = ArchiveManager(context_manager=cm, memory_dir=str(tmp_path / "mem"))

    monkeypatch.setattr("core.managers.archive_manager.time.time", lambda: day3)
    assert await mgr.archive_if_stale("chat_001", is_group=False) is not None

    kept = ctx.set_messages.call_args[0][0]
    assert [message.content for message in kept] == ["day3"]


@pytest.mark.asyncio
async def test_manual_archive_uses_target_chat_type_for_summary(tmp_path, monkeypatch):
    """跨会话归档时，摘要类型必须来自目标会话的元数据。"""
    day1 = datetime(2025, 1, 1, 20, 0, 0).timestamp()
    day2 = datetime(2025, 1, 2, 10, 0, 0).timestamp()
    ctx = _MutableCtx([ChatMessage(role="user", content="target", timestamp=day1)])
    store = JSONLContextStore(str(tmp_path / "sessions"))
    cm = _FakeCM(ctx, store, chat_types={"target_chat": True})
    mgr = ArchiveManager(context_manager=cm, memory_dir=str(tmp_path / "mem"))

    monkeypatch.setattr("core.managers.archive_manager.time.time", lambda: day2)
    result = await mgr.archive_manual("target_chat")

    assert result.summary_path is not None
    assert "- **Type**: 群聊" in Path(result.summary_path).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_session_status_prefers_timeline_summary(tmp_path):
    context_manager = MagicMock()
    context_manager.get_chat_history_async = MagicMock(
        side_effect=AssertionError("legacy history should not be read")
    )
    timeline = MagicMock()
    timeline.session_summary = AsyncMock(
        return_value={
            "message_count": 2,
            "last_activity": 120.0,
            "estimated_tokens": 5,
        }
    )
    manager = ArchiveManager(
        context_manager=context_manager,
        memory_dir=str(tmp_path / "mem"),
    )
    manager.set_timeline(timeline)

    status = await manager.get_session_status_async("chat_001")

    assert status["message_count"] == 2
    assert status["last_activity"] == 120.0


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
async def test_daily_guard_survives_restart(tmp_path):
    """同日守卫状态持久化：重启后同一天不重复归档。"""
    now = time.time()
    msgs = [ChatMessage(role="user", content="昨天", timestamp=now - 86400)]
    mem_dir = str(tmp_path / "mem")
    cm = _FakeCM(_make_ctx(msgs))
    mgr1 = ArchiveManager(context_manager=cm, memory_dir=mem_dir)
    first = await mgr1.archive_if_stale("chat_001", is_group=False)
    assert first is not None

    mgr2 = ArchiveManager(context_manager=cm, memory_dir=mem_dir)
    second = await mgr2.archive_if_stale("chat_001", is_group=False)
    assert second is None


def test_replayed_prefix_migrates_legacy_daily_state(tmp_path):
    """旧版 {chat_id: date} state 可推断回放前缀并升级为指纹。"""
    chat_id = "chat_001"
    (tmp_path / "daily_archive_state.json").write_text(
        '{"chat_001":"2025-01-02"}', encoding="utf-8"
    )
    mgr = ArchiveManager(context_manager=MagicMock(), memory_dir=str(tmp_path / "mem"))
    messages = [
        ChatMessage(
            role="user",
            content="旧回放",
            timestamp=datetime(2025, 1, 1, 20, 0, 0).timestamp(),
        ),
        ChatMessage(
            role="user",
            content="当天消息",
            timestamp=datetime(2025, 1, 2, 10, 0, 0).timestamp(),
        ),
    ]

    assert mgr._replayed_prefix_length(chat_id, messages) == 1
    assert chat_id in mgr._replayed_prefix_known
    assert len(mgr._replayed_prefix_keys[chat_id]) == 1


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
    result = mgr._format_summary_text(
        msgs, count=200, is_group=True, chat_id="chat_001", date="2024-06-15"
    )
    assert result is not None
    assert "Session: 2024-06-15" in result
    assert "你好" in result
    assert "嗨！有什么可以帮你的？" in result


def test_format_summary_text_empty(mgr):
    result = mgr._format_summary_text(
        [], count=200, is_group=True, chat_id="chat_001", date="2024-06-15"
    )
    assert result is None


def test_format_summary_text_filters_tool(mgr):
    msgs = [
        make_msg("user", "搜索一下", "m1"),
        make_msg("assistant", "思考中", "m2", tool_calls=[{"id": "c1"}]),
        make_msg("tool", '{"result": "找到了"}', "m3"),
        make_msg("assistant", "找到了，在这里", "m4"),
    ]
    result = mgr._format_summary_text(
        msgs, count=200, is_group=True, chat_id="chat_001", date="2024-06-15"
    )
    # tool 和 tool_calls 的 assistant 被过滤
    assert result is not None
    assert "搜索一下" in result
    assert "找到了，在这里" in result


def test_format_summary_text_skips_empty_content(mgr):
    msgs = [
        make_msg("user", "", "m0"),
        make_msg("assistant", "回答", "m1"),
    ]
    result = mgr._format_summary_text(
        msgs, count=200, is_group=True, chat_id="chat_001", date="2024-06-15"
    )
    assert result is not None
    assert "回答" in result
