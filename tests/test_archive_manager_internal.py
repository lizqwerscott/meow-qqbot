"""测试 ArchiveManager 内部逻辑：_extract_replay_messages、_build_summary_group、
consume_summary、archive_if_stale 跨天触发与同日守卫。"""

import time
from unittest.mock import MagicMock

import pytest

from core.managers.archive_manager import (
    ArchiveManager,
    _build_summary_group,
    _date_str,
    _get_memory_dir,
)
from core.managers.chat_message import ChatMessage

# ── helpers ──


def make_msg(role="user", content="hello", msg_id="m1", sender_id="u1", **kw):
    return ChatMessage(
        role=role,
        content=content,
        timestamp=100.0,
        message_id=msg_id,
        sender_id=sender_id,
        name=None,
        **kw,
    )


class _FakeCM:
    """最小 context_manager 替身：_with_context_locked 直接执行 func(ctx)。"""

    def __init__(self, ctx):
        self._ctx = ctx
        self.store = MagicMock()

    async def _with_context_locked(self, chat_id, func):
        return await func(self._ctx)


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


def test_extract_replay_zero_count(mgr):
    """count<=0 时不返回任何消息（旧实现会误返回 1 条）。"""
    msgs = [make_msg("user", "a", "m0")]
    assert mgr._extract_replay_messages(msgs, 0) == []


def test_extract_replay_skips_empty_content(mgr):
    """回放与摘要共用过滤谓词：空内容消息不参与回放。"""
    msgs = [
        make_msg("user", "", "m0"),
        make_msg("user", "有内容", "m1"),
    ]
    result = mgr._extract_replay_messages(msgs, count=10)
    assert len(result) == 1
    assert result[0].content == "有内容"


# ── archive_if_stale（跨天触发 / 同日守卫）──


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
    msgs = [
        ChatMessage(role="user", content="昨天", timestamp=now - 86400),
        ChatMessage(role="user", content="今天", timestamp=now),
    ]
    cm = _FakeCM(_make_ctx(msgs))
    mgr = ArchiveManager(context_manager=cm, memory_dir=str(tmp_path / "mem"))
    first = await mgr.archive_if_stale("chat_001", is_group=False)
    second = await mgr.archive_if_stale("chat_001", is_group=False)
    assert first is not None
    assert second is None  # 同一天只归档一次


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
    """延迟触发场景：跨天后第一条是非 TEXT（未触发归档），等到某条 TEXT
    才归档时，今天早先的消息必须全部保留；且今天消息已 >= replay_count，
    昨天的消息无需回放（摘要已注入，回放仅用于上下文衔接）。"""
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
    mgr = ArchiveManager(
        context_manager=cm, memory_dir=str(tmp_path / "mem"), replay_count=2
    )
    result = await mgr.archive_if_stale("chat_001", is_group=False)
    assert result is not None
    assert result.replay_count == 3  # 今天 3 条已 >= 2，昨天不回放

    kept = ctx.set_messages.call_args[0][0]
    contents = [m.content for m in kept]
    assert contents == ["今天0", "今天1", "今天2"]


@pytest.mark.asyncio
async def test_archive_replays_yesterday_tail_when_today_sparse(tmp_path):
    """今天消息不足 replay_count 时，用昨天尾部补足上下文衔接
    （此时摘要尚未注入，昨天回放是唯一上下文）。"""
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
    mgr = ArchiveManager(
        context_manager=cm, memory_dir=str(tmp_path / "mem"), replay_count=2
    )
    result = await mgr.archive_if_stale("chat_001", is_group=False)
    assert result is not None
    assert result.replay_count == 2  # 昨天尾部 1 + 今天 1

    kept = ctx.set_messages.call_args[0][0]
    contents = [m.content for m in kept]
    assert contents == ["昨天3", "今天0"]


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
