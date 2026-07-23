"""HeartbeatSchedule — SHA-256 相位偏移调度 + 活跃时段 seek。

匹配 OpenClaw heartbeat-schedule.ts 设计。
"""

import hashlib
import time
from typing import Callable, Optional


def resolve_phase_ms(seed: str, agent_id: str, interval_ms: int) -> int:
    """SHA-256 确定性相位偏移。返回 [0, interval_ms) 内的毫秒偏移。"""
    interval_ms = max(1, interval_ms)
    h = hashlib.sha256(f"{seed}:{agent_id}".encode()).digest()
    return int.from_bytes(h[:4], "big") % interval_ms


def compute_next_phase_due_ms(now_ms: float, interval_ms: int, phase_ms: int) -> float:
    """返回下一个相位对齐时刻（毫秒级 epoch）。"""
    interval_ms = max(1, interval_ms)
    phase_ms = phase_ms % interval_ms
    cycle_pos = now_ms % interval_ms
    delta = (phase_ms - cycle_pos) % interval_ms
    if delta == 0:
        delta = interval_ms
    return now_ms + delta


# 最多 seek 7 天
MAX_SEEK_HORIZON_MS = 7 * 24 * 3600 * 1000
# 批次最小步长 30s（匹配 openclaw MIN_SEEK_STEP_MS）
MIN_SEEK_STEP_MS = 30_000


def seek_next_active_phase(
    start_ms: float,
    interval_ms: int,
    phase_ms: int,
    is_active: Optional[Callable[[float], bool]] = None,
    horizon_ms: int = MAX_SEEK_HORIZON_MS,
) -> float:
    """从 start_ms 向前搜索，找到第一个在活跃时段内的相位对齐 slot。

    对 >= 30s 间隔使用 step 倍率加速；
    对 < 30s 间隔的转态过渡使用二分搜索（openclaw 的 binary-search 优化）。
    7 天 horizon 内找不到则 fallback 到 start_ms。
    """
    if not is_active:
        return start_ms

    interval_ms = max(1, interval_ms)
    phase_ms = phase_ms % interval_ms
    horizon = start_ms + horizon_ms

    # 倍率：至少 30s 步进
    multiplier = max(1, MIN_SEEK_STEP_MS // interval_ms)
    batch_step_ms = interval_ms * multiplier

    candidate = start_ms
    prev_inactive_ms: Optional[float] = None

    while candidate < horizon:
        if is_active(candidate):
            if prev_inactive_ms is not None and multiplier > 1:
                # 二分搜索精确的 inactive→active 过渡点
                lo, hi = prev_inactive_ms, candidate
                while hi - lo > interval_ms:
                    mid = lo + ((hi - lo) // interval_ms // 2) * interval_ms
                    if is_active(mid):
                        hi = mid
                    else:
                        lo = mid
                return hi
            return candidate
        prev_inactive_ms = candidate
        candidate += batch_step_ms

    return start_ms


# ── 活跃时段辅助函数 ──


def is_in_active_hours_ts(
    ts: float,
    start_str: Optional[str],
    end_str: Optional[str],
    tz_str: str = "Asia/Shanghai",
) -> bool:
    """Unix 时间戳是否在活跃时段内。"""
    if not start_str or not end_str:
        return True
    from datetime import datetime, timezone, timedelta
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tz_str)
    except (ImportError, KeyError, TypeError):
        import re
        m = re.match(r"^UTC([+-]\d{1,2})(?::(\d{2}))?$", tz_str)
        if m:
            tz = timezone(timedelta(hours=int(m.group(1))))
        else:
            tz = timezone.utc

    now = datetime.fromtimestamp(ts, tz)
    cur = now.hour * 60 + now.minute
    sp, ep = start_str.split(":"), end_str.split(":")
    start_min = int(sp[0]) * 60 + int(sp[1])
    end_min = int(ep[0]) * 60 + int(ep[1])

    if end_min <= start_min:
        return cur >= start_min or cur < end_min
    return start_min <= cur < end_min
