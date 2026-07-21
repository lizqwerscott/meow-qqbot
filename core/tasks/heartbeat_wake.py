"""HeartbeatWakeScheduler — 调度/合并/重试层

参照 OpenClaw heartbeat-wake.ts 设计，管理请求合并、延迟执行和忙态退避。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional

_log = logging.getLogger(__name__)


class WakeIntent(Enum):
    MANUAL = "manual"
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    EVENT = "event"


class WakeSource(Enum):
    INTERVAL = "interval"
    COMMAND = "command"
    SYSTEM_EVENT = "system_event"
    TASK = "task"


class _TimerKind(Enum):
    NONE = ""
    COALESCE = "coalesce"
    RETRY = "retry"


_PRIORITY = {
    WakeIntent.MANUAL: 0,
    WakeIntent.IMMEDIATE: 1,
    WakeIntent.EVENT: 2,
    WakeIntent.SCHEDULED: 3,
}


@dataclass
class HeartbeatWakeRequest:
    source: WakeSource = WakeSource.INTERVAL
    intent: WakeIntent = WakeIntent.SCHEDULED
    reason: str = ""
    extra_prompt: str = ""
    timestamp: float = field(default_factory=time.time)


class HeartbeatWakeScheduler:
    """请求调度层。

    职责：
    1. request() 将请求合并到 pending slot（高优先级覆盖低）
    2. 延迟 coalesce_ms 后调用 handler（防抖）
    3. retry() 不受后续 request() 打断（timer_kind="retry" 保护）

    使用方式：
        scheduler = HeartbeatWakeScheduler()
        scheduler.set_handler(my_async_handler)
        scheduler.request(HeartbeatWakeRequest(...))
    """

    def __init__(self, coalesce_ms: int = 100):
        self._pending: Optional[HeartbeatWakeRequest] = None
        self._handler: Optional[Callable[[HeartbeatWakeRequest], Awaitable[None]]] = None
        self._coalesce_ms = coalesce_ms
        self._timer_task: Optional[asyncio.Task] = None
        self._timer_kind: _TimerKind = _TimerKind.NONE

    def set_handler(self, handler: Callable[[HeartbeatWakeRequest], Awaitable[None]]):
        self._handler = handler

    def request(self, req: HeartbeatWakeRequest):
        """提交 wake 请求。同 key 合并，高优先级覆盖低优先级。"""
        if self._handler is None:
            return

        if self._pending is None:
            self._pending = req
        else:
            existing_prio = _PRIORITY.get(self._pending.intent, 99)
            new_prio = _PRIORITY.get(req.intent, 99)
            if new_prio < existing_prio:
                self._pending = req
            elif new_prio == existing_prio:
                self._pending = req

        if self._timer_kind != _TimerKind.RETRY:
            self._cancel_timer()
            self._schedule(self._coalesce_ms, _TimerKind.COALESCE)

    def retry(self, delay_ms: int = 1000):
        """忙态重试，不会被后续 request() 的计时器覆盖。"""
        self._cancel_timer()
        self._schedule(delay_ms, _TimerKind.RETRY)

    def cancel_pending(self):
        """取消待处理的 wake 请求。"""
        self._cancel_timer()
        self._pending = None

    def _schedule(self, delay_ms: int, kind: _TimerKind):
        self._timer_kind = kind
        self._timer_task = asyncio.create_task(self._timer(delay_ms / 1000))

    async def _timer(self, delay: float):
        await asyncio.sleep(delay)
        req = self._pending
        self._pending = None
        self._timer_kind = _TimerKind.NONE
        if req and self._handler:
            try:
                await self._handler(req)
            except Exception as e:
                _log.error(f"Wake handler 异常: {e}", exc_info=True)

    def _cancel_timer(self):
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        self._timer_task = None
        self._timer_kind = _TimerKind.NONE
