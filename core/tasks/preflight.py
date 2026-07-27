"""Preflight — 唤醒前预检链。

返回 PreflightResult struct，含唤醒分类标志 + skip reason。
匹配 OpenClaw heartbeat-runner.ts 的 resolveHeartbeatPreflight + 10-step skip chain。
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from core.tasks.heartbeat_wake import WakeIntent as WI
from .wake_coalescer import INTENT_MANUAL, INTENT_IMMEDIATE
from .heartbeat_schedule import is_in_active_hours_ts

_log = logging.getLogger(__name__)


@dataclass
class PreflightContext:
    source: str
    intent: str
    session_key: str
    cooldown: Any
    active_hours: tuple[Optional[str], Optional[str], Optional[str]]
    has_system_events: bool
    has_extra_prompt: bool
    is_session_active: bool
    has_cron_jobs: bool
    # C1: wake 分类
    source_is_interval: bool = False
    source_is_manual: bool = False
    source_is_exec: bool = False
    source_is_cron: bool = False
    source_is_background: bool = False
    # C2: 新增跳过检查
    is_main_lane_busy: bool = False
    is_agent_busy: bool = False
    skip_when_busy: bool = False
    is_session_lane_busy: bool = False
    # 新增：pending delivery deferral
    is_delivery_pending: bool = False


@dataclass
class PreflightResult:
    skip_reason: Optional[str] = None
    is_exec_event: bool = False
    is_cron_wake: bool = False
    is_wake_payload: bool = False
    pending_event_count: int = 0
    should_bypass_file_gates: bool = False


def run_preflight(ctx: PreflightContext) -> PreflightResult:
    """预检链。返回 PreflightResult（skip_reason=None = 通过）。

    步骤顺序：retryable reasons 在前，non-retryable 在后。
    """
    result = PreflightResult()
    _start = time.time()

    _log.info(
        "[Preflight] entry: source=%s intent=%s session=%s "
        "cron=%s session_active=%s agent_busy=%s events=%s prompt=%s "
        "active_hours=%s delivery_pending=%s",
        ctx.source, ctx.intent, ctx.session_key[:20],
        ctx.has_cron_jobs, ctx.is_session_active, ctx.is_agent_busy,
        ctx.has_system_events, ctx.has_extra_prompt,
        ctx.active_hours, ctx.is_delivery_pending,
    )

    # ── 分类唤醒载荷 ──
    if ctx.source_is_exec:
        result.is_exec_event = True
        result.should_bypass_file_gates = True
    elif ctx.source_is_cron:
        result.is_cron_wake = True
        result.should_bypass_file_gates = True
    elif ctx.source_is_background:
        result.is_wake_payload = True

    result.pending_event_count = 1 if ctx.has_system_events else 0

    # Step 02: 活跃时段
    if not ctx.source_is_manual:
        start, end, tz = ctx.active_hours
        if start and end:
            if not is_in_active_hours_ts(time.time(), start, end, tz or "Asia/Shanghai"):
                _log.info("[Preflight] step=02 active_hours: SKIP → quiet-hours")
                result.skip_reason = "quiet-hours"
                return result
    _log.debug("[Preflight] step=02 active_hours: pass")

    # Step 03: Cron 运行中 (retryable)
    if ctx.has_cron_jobs:
        _log.info("[Preflight] step=03 cron_in_progress: SKIP → cron-in-progress")
        result.skip_reason = "cron-in-progress"
        return result
    _log.debug("[Preflight] step=03 cron_in_progress: pass")

    # Step 04: 主 command lane 忙 (retryable)
    if ctx.is_main_lane_busy:
        _log.info("[Preflight] step=04 main_lane_busy: SKIP → requests-in-flight")
        result.skip_reason = "requests-in-flight"
        return result
    _log.debug("[Preflight] step=04 main_lane_busy: pass")

    # Step 05: Session 忙态 (retryable)
    if ctx.is_session_active:
        _log.info("[Preflight] step=05 session_active: SKIP → requests-in-flight")
        result.skip_reason = "requests-in-flight"
        return result
    _log.debug("[Preflight] step=05 session_active: pass")

    # Step 06: skipWhenBusy (retryable)
    if ctx.skip_when_busy and ctx.is_agent_busy:
        _log.info("[Preflight] step=06 skip_when_busy: SKIP → lanes-busy")
        result.skip_reason = "lanes-busy"
        return result
    _log.debug("[Preflight] step=06 skip_when_busy: pass")

    # Step 07: agent busy for non-immediate/manual (retryable)
    if ctx.intent not in ("immediate", "manual") and ctx.is_agent_busy:
        _log.info("[Preflight] step=07 agent_busy: SKIP → requests-in-flight")
        result.skip_reason = "requests-in-flight"
        return result
    _log.debug("[Preflight] step=07 agent_busy: pass")

    # Step 08: interval + 无事件 + 无 prompt (non-retryable)
    if ctx.source_is_interval:
        if not ctx.has_system_events and not ctx.has_extra_prompt:
            _log.info("[Preflight] step=08 no_event_no_prompt: SKIP → no-events")
            result.skip_reason = "no-events"
            return result
    _log.debug("[Preflight] step=08 no_event_no_prompt: pass")

    # Step 09: pending final delivery (retryable)
    if ctx.is_delivery_pending:
        _log.info("[Preflight] step=09 delivery_pending: SKIP → requests-in-flight")
        result.skip_reason = "requests-in-flight"
        return result
    _log.debug("[Preflight] step=09 delivery_pending: pass")

    # Step 10: 目标 session lane 忙 (retryable)
    if ctx.is_session_lane_busy:
        _log.info("[Preflight] step=10 session_lane_busy: SKIP → requests-in-flight")
        result.skip_reason = "requests-in-flight"
        return result
    _log.debug("[Preflight] step=10 session_lane_busy: pass")

    # Step 11: Cooldown (not-due/min-spacing/flood)
    intent_map = {
        INTENT_MANUAL: WI.MANUAL,
        INTENT_IMMEDIATE: WI.IMMEDIATE,
        "event": WI.EVENT,
        "scheduled": WI.SCHEDULED,
    }
    dec = ctx.cooldown.should_defer(
        intent=intent_map.get(ctx.intent, WI.SCHEDULED),
    )
    if dec.defer:
        _log.info("[Preflight] step=11 cooldown: SKIP → %s", dec.reason)
        result.skip_reason = dec.reason
        return result
    _log.debug("[Preflight] step=11 cooldown: pass")

    _log.info(
        "[Preflight] result: PASS (duration=%.1fms)",
        (time.time() - _start) * 1000,
    )
    return result
