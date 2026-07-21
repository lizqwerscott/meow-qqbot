"""HeartbeatCooldown — 冷却门控

所有 wake 来源统一 defer 决策。参照 OpenClaw heartbeat-cooldown.ts 设计。
"""

import time
from dataclasses import dataclass

from core.tasks.heartbeat_wake import WakeIntent


@dataclass
class DeferDecision:
    defer: bool = False
    reason: str = ""
    retry_after_ms: int = 0


class HeartbeatCooldown:
    """冷却门控 — 所有 wake 来源统一 defer 决策。

    策略：
    - manual → 总执行（仅受 flood guard 限制）
    - immediate → 仅 flood guard
    - scheduled → 额外检查 min-spacing
    - event → 额外加长 min-spacing 抗抖动
    """

    def __init__(self, min_spacing_seconds: float = 30.0):
        self._last_run_start: float = 0.0
        self._flood_limit_seconds: float = 10.0
        self._min_spacing_seconds = min_spacing_seconds

    def should_defer(
        self,
        intent: WakeIntent,
        is_busy: bool = False,
    ) -> DeferDecision:
        now = time.time()
        elapsed = now - self._last_run_start if self._last_run_start > 0 else float("inf")

        # flood guard：所有来源必须间隔至少 flood_limit
        if elapsed < self._flood_limit_seconds:
            retry = int((self._flood_limit_seconds - elapsed) * 1000) + 1
            return DeferDecision(defer=True, reason="flood", retry_after_ms=retry)

        # manual：总执行，不检查 busy 和 min-spacing
        if intent == WakeIntent.MANUAL:
            return DeferDecision(defer=False)

        # busy
        if is_busy:
            return DeferDecision(defer=True, reason="busy", retry_after_ms=1000)

        # scheduled：检查 min-spacing
        if intent == WakeIntent.SCHEDULED and elapsed < self._min_spacing_seconds:
            retry = int((self._min_spacing_seconds - elapsed) * 1000) + 1
            return DeferDecision(defer=True, reason="not-due", retry_after_ms=retry)

        # event：加长 min-spacing 抗抖动
        if intent == WakeIntent.EVENT and elapsed < self._min_spacing_seconds * 2:
            retry = int((self._min_spacing_seconds * 2 - elapsed) * 1000) + 1
            return DeferDecision(defer=True, reason="min-spacing", retry_after_ms=retry)

        return DeferDecision(defer=False)

    def record_run_start(self):
        self._last_run_start = time.time()
