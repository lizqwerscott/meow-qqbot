"""WakeDispatcher — 兼容包装层。

所有方法委托给 core.tasks.wake_coalescer + wake_runner。
计划在 Phase 2 所有调用者迁移完成后删除此文件。
新代码应直接调用 wake_coalescer.request_wake()。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import core.tasks.wake_coalescer as _coalescer

_log = logging.getLogger(__name__)

# ── 暴露常量（供现有调用者使用） ──

SOURCE_INTERVAL = _coalescer.SOURCE_INTERVAL
SOURCE_MANUAL = _coalescer.SOURCE_MANUAL
SOURCE_EXEC = _coalescer.SOURCE_EXEC
SOURCE_CRON = _coalescer.SOURCE_CRON
SOURCE_TASK = _coalescer.SOURCE_TASK
SOURCE_SYSTEM = _coalescer.SOURCE_SYSTEM

INTENT_MANUAL = _coalescer.INTENT_MANUAL
INTENT_IMMEDIATE = _coalescer.INTENT_IMMEDIATE
INTENT_EVENT = _coalescer.INTENT_EVENT
INTENT_SCHEDULED = _coalescer.INTENT_SCHEDULED


@dataclass
class WakeResult:
    """保留给现有调用者。新代码应使用 WakeRunResult。"""
    should_notify: bool = False
    notification_text: str = ""
    captured_replies: list[str] = field(default_factory=list)
    error: Optional[str] = None


class WakeDispatcher:
    """兼容包装层。委托给 wake_coalescer + wake_runner。"""

    def __init__(
        self,
        system_events: Any = None,
    ):
        self._events = system_events

    async def request(
        self,
        source: str = SOURCE_SYSTEM,
        intent: str = INTENT_EVENT,
        session_key: str = "",
        *,
        reason: str = "",
        extra_prompt: str = "",
        event_text: str = "",
        event_context_key: str = "",
        event_replace: bool = False,
        coalesce_ms: int = 1000,
    ) -> Optional[WakeResult]:
        """统一 wake 入口（兼容包装）。

        内部委托给 wake_coalescer。
        当 source=manual + intent=manual 时同步返回 WakeResult。
        """
        if not session_key:
            return None

        # 入队系统事件
        if event_text and self._events:
            self._events.enqueue(
                session_key=session_key,
                text=event_text,
                context_key=event_context_key,
                replace=event_replace,
            )

        # manual/manual → 同步返回
        if source == SOURCE_MANUAL and intent == INTENT_MANUAL:
            wr = await _coalescer.execute_immediate(
                source=source, intent=intent,
                session_key=session_key,
                extra_prompt=extra_prompt,
            )
            return self._to_wake_result(wr)

        # 其他来源 → fire-and-forget
        _coalescer.request_wake(
            source=source, intent=intent,
            session_key=session_key,
            extra_prompt=extra_prompt,
            coalesce_ms=coalesce_ms,
        )
        return None

    async def start_interval(self, every_minutes: float, prompt: str = "") -> None:
        """NOOP — interval loop 由 HeartbeatManager 接管。"""
        _log.debug("WakeDispatcher.start_interval() 不再处理，由 HeartbeatManager 接管")

    async def stop_interval(self) -> None:
        """NOOP"""
        _log.debug("WakeDispatcher.stop_interval() 不再需要")

    def set_active_hours(self, start: str, end: str, tz: str = "Asia/Shanghai") -> None:
        """NOOP — 由 WakeRunner 持有。"""
        pass

    def get_status(self) -> dict:
        return _coalescer.get_status()

    @staticmethod
    def _to_wake_result(wr) -> Optional[WakeResult]:
        if wr is None or not wr.result:
            return None
        r = wr.result
        return WakeResult(
            should_notify=getattr(r, 'should_notify', False),
            notification_text=getattr(r, 'notification_text', ''),
            captured_replies=getattr(r, 'captured_replies', []),
            error=getattr(r, 'error', None),
        )
