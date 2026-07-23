"""HeartbeatCooldown — 纯函数式 defer 决策。

匹配 OpenClaw heartbeat-cooldown.ts 设计。

核心：should_defer_wake() 是纯函数，不持有状态。
      HeartbeatCooldown 类是状态包装器，持有 recent_runs buffer + next_due。
"""

import time
from dataclasses import dataclass
from typing import Optional

from core.tasks.heartbeat_wake import WakeIntent


@dataclass
class DeferDecision:
    defer: bool = False
    reason: str = ""


# ── 纯函数（可独立测试） ──


def should_defer_wake(
    intent: WakeIntent,
    now: float,
    next_due_ms: float,
    last_run_started_at_ms: Optional[float] = None,
    recent_run_starts: Optional[list[float]] = None,
    min_spacing_ms: int = 30_000,
    flood_window_ms: int = 60_000,
    flood_threshold: int = 5,
) -> DeferDecision:
    """纯函数：决定是否应推迟本次 wake。

    规则矩阵（匹配 OpenClaw）：
      manual     → 永远不 defer
      immediate  → 仅 flood guard（不检查 nextDueMs / min-spacing）
      scheduled  → defer if now < nextDueMs
      event      → 首次运行(无 prior)绕过；否则检查 nextDueMs + min-spacing
    """

    # manual → 永远不 defer
    if intent == WakeIntent.MANUAL:
        return DeferDecision(defer=False)

    # immediate → 仅 flood guard
    if intent == WakeIntent.IMMEDIATE:
        flood = _check_flood(now, recent_run_starts, flood_window_ms, flood_threshold)
        return flood or DeferDecision(defer=False)

    # flood guard（所有非 manual/immediate）
    flood = _check_flood(now, recent_run_starts, flood_window_ms, flood_threshold)
    if flood.defer:
        return flood

    # scheduled → nextDueMs 检查
    if intent == WakeIntent.SCHEDULED:
        if now < next_due_ms:
            return DeferDecision(defer=True, reason="not-due")
        return DeferDecision(defer=False)

    # event → 首次运行绕过 cooldown；之后检查 nextDueMs + min-spacing
    if last_run_started_at_ms is None:
        return DeferDecision(defer=False)

    if now < next_due_ms:
        return DeferDecision(defer=True, reason="not-due")

    if min_spacing_ms > 0 and (now - last_run_started_at_ms) < min_spacing_ms:
        return DeferDecision(defer=True, reason="min-spacing")

    return DeferDecision(defer=False)


def _check_flood(
    now: float,
    runs: Optional[list[float]],
    window_ms: int,
    threshold: int,
) -> DeferDecision:
    if not runs or len(runs) < threshold or window_ms <= 0:
        return DeferDecision(defer=False)
    cutoff = now - window_ms
    in_window = 0
    for t in reversed(runs):
        if t < cutoff:
            break
        in_window += 1
    if in_window >= threshold:
        return DeferDecision(defer=True, reason="flood")
    return DeferDecision(defer=False)


def record_run_start(buffer: list[float], ts: float, threshold: int = 5) -> list[float]:
    """往 buffer 追加时间戳，保持长度 ≤ threshold + 1。返回 buffer 自身。"""
    buffer.append(ts)
    while len(buffer) > threshold + 1:
        buffer.pop(0)
    return buffer


# ── 状态包装类（供现有调用者使用） ──


class HeartbeatCooldown:
    """Cooldown 状态包装器。

    持有：
      - recent_runs: list[float]       — 最近运行启动时间（ms）
      - last_run_started_at_ms: float  — 上次运行启动时间
      - next_due_ms: float             — 下次调度到期时间（由 WakeRunner/HeartbeatManager 设置）
    """

    def __init__(
        self,
        min_spacing_seconds: float = 30.0,
        flood_window_seconds: float = 60.0,
        flood_threshold: int = 5,
    ):
        self._min_spacing_ms = int(min_spacing_seconds * 1000)
        self._flood_window_ms = int(flood_window_seconds * 1000)
        self._flood_threshold = flood_threshold
        self._recent_runs: list[float] = []
        self._last_run_started_at_ms: Optional[float] = None
        self._next_due_ms: Optional[float] = None

    def should_defer(self, intent: WakeIntent) -> DeferDecision:
        now = time.time() * 1000
        return should_defer_wake(
            intent=intent,
            now=now,
            next_due_ms=self._next_due_ms or now,
            last_run_started_at_ms=self._last_run_started_at_ms,
            recent_run_starts=self._recent_runs,
            min_spacing_ms=self._min_spacing_ms,
            flood_window_ms=self._flood_window_ms,
            flood_threshold=self._flood_threshold,
        )

    def record_run_start(self) -> None:
        now = time.time() * 1000
        self._last_run_started_at_ms = now
        record_run_start(self._recent_runs, now, self._flood_threshold)

    def set_next_due(self, due_ms: float) -> None:
        self._next_due_ms = due_ms
