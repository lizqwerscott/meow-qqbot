"""测试 ArchiveManager 辅助函数。"""

import time

from core.managers.archive_manager import _daily_reset_at, _date_str, _format_archive_timestamp


# ── _daily_reset_at ──


def test_daily_reset_at_today():
    """当天重置时间应在今天或昨天。"""
    ts = time.time()
    reset = _daily_reset_at(hour=4, t=ts)
    assert reset <= ts


def test_daily_reset_at_after_midnight():
    """凌晨 3 点，重置时间为 4 点 → 回退到昨天。"""
    from datetime import datetime
    jan = datetime(2024, 1, 15, 3, 0, 0).timestamp()
    reset = _daily_reset_at(hour=4, t=jan)
    assert reset <= jan
    assert reset > jan - 172800  # 48h in seconds


def test_daily_reset_at_before_midnight():
    """下午 2 点，重置时间为 4 点 → 今天凌晨 4 点。"""
    from datetime import datetime
    afternoon = datetime(2024, 6, 15, 14, 0, 0).timestamp()
    reset = _daily_reset_at(hour=4, t=afternoon)
    assert reset <= afternoon
    assert reset > afternoon - 43200  # 12h in seconds


# ── 回归：ValueError 分支（hour > 23 等无效值）──

def test_daily_reset_at_invalid_hour():
    """hour=25 触发 ValueError → fallback 路径。"""
    ts = time.time()
    reset = _daily_reset_at(hour=25, t=ts)
    assert isinstance(reset, float)
    assert reset <= ts


# ── _date_str ──


def test_date_str():
    from datetime import datetime
    dt = datetime(2024, 3, 15, 10, 30, 0)
    assert _date_str(dt.timestamp()) == "2024-03-15"


# ── _format_archive_timestamp ──


def test_format_archive_timestamp():
    from datetime import datetime
    dt = datetime(2024, 3, 15, 10, 30, 45)
    result = _format_archive_timestamp(dt.timestamp())
    assert result.startswith("2024-03-15T10-30-45")
