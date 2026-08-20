"""测试 ArchiveManager 辅助函数。"""

import time
from datetime import datetime

from core.managers.archive_manager import (
    ArchiveManager,
    _date_str,
    _format_archive_timestamp,
)
from core.managers.chat_message import ChatMessage

# ── _date_str ──


def test_date_str():
    dt = datetime(2024, 3, 15, 10, 30, 0)
    assert _date_str(dt.timestamp()) == "2024-03-15"


# ── _crossed_day（跨天判断，按消息时间戳）──


def test_crossed_day_with_yesterday_message():
    now = time.time()
    yesterday = now - 86400
    today = _date_str(now)
    msgs = [
        ChatMessage(role="user", content="昨天", timestamp=yesterday),
        ChatMessage(role="user", content="今天", timestamp=now),
    ]
    assert ArchiveManager._crossed_day(msgs, today) is True


def test_crossed_day_today_only():
    now = time.time()
    today = _date_str(now)
    msgs = [ChatMessage(role="user", content="hi", timestamp=now)]
    assert ArchiveManager._crossed_day(msgs, today) is False


def test_crossed_day_empty():
    assert ArchiveManager._crossed_day([], "2024-01-01") is False


def test_crossed_day_future_message_not_triggered():
    """时钟偏移导致的未来时间戳不应算作跨天。"""
    now = time.time()
    future = now + 2 * 86400  # 后天
    today = _date_str(now)
    msgs = [ChatMessage(role="user", content="hi", timestamp=future)]
    assert ArchiveManager._crossed_day(msgs, today) is False


# ── _format_archive_timestamp ──


def test_format_archive_timestamp():
    dt = datetime(2024, 3, 15, 10, 30, 45)
    result = _format_archive_timestamp(dt.timestamp())
    assert result.startswith("2024-03-15T10-30-45")
