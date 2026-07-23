"""WakeDispatcher — 统一 wake/interval 调度器。

所有 wake 来源（interval/exec-event/cron/manual/background-task）统一入口。
替代原 WakeManager + HeartbeatWakeScheduler。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_log = logging.getLogger(__name__)

# ── 常量 ──

# Wake source
SOURCE_INTERVAL = "interval"
SOURCE_MANUAL = "manual"
SOURCE_EXEC = "exec-event"
SOURCE_CRON = "cron"
SOURCE_TASK = "background-task"
SOURCE_SYSTEM = "system"

# Wake intent（优先级从高到低）
INTENT_MANUAL = "manual"
INTENT_IMMEDIATE = "immediate"
INTENT_EVENT = "event"
INTENT_SCHEDULED = "scheduled"

_INTENT_PRIORITY = {
    INTENT_MANUAL: 0,
    INTENT_IMMEDIATE: 1,
    INTENT_EVENT: 2,
    INTENT_SCHEDULED: 3,
}


# ── 数据类型 ──


@dataclass
class PendingWake:
    source: str = SOURCE_INTERVAL
    intent: str = INTENT_SCHEDULED
    reason: str = ""
    session_key: str = ""
    extra_prompt: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class WakeResult:
    should_notify: bool = False
    notification_text: str = ""
    captured_replies: list[str] = field(default_factory=list)
    error: Optional[str] = None


# ── 调度器 ──


class WakeDispatcher:
    """统一 wake 调度器。

    职责：
    1. 接收所有来源的 wake 请求，按 session_key 合并（优先级感知）
    2. 延迟 coalesce_ms 后触发，忙态跳过并重试
    3. 执行预检（冷却门控、活跃时段、session 忙态）
    4. 调用 AgentEngine.run_wake_turn() 执行 AI turn
    5. 根据 source 调用对应的 delivery callback 投递结果

    用法：
        dispatcher = WakeDispatcher(system_events, agent_engine, cooldown)
        dispatcher.set_delivery_callback("interval", my_callback)
        dispatcher.set_delivery_callback("exec-event", my_callback)
        await dispatcher.request(
            source=SOURCE_INTERVAL, intent=INTENT_SCHEDULED,
            session_key="heartbeat:events",
        )
    """

    COALESCE_MS_DEFAULT = 1000

    def __init__(
        self,
        system_events: Any = None,
        agent_engine: Any = None,
        cooldown: Any = None,
    ):
        self._events = system_events
        self._agent = agent_engine
        self._cooldown = cooldown  # HeartbeatCooldown 实例

        # 调度状态
        self._pending: dict[str, PendingWake] = {}
        self._timers: dict[str, asyncio.TimerHandle] = {}
        self._retry_timers: dict[str, asyncio.TimerHandle] = {}
        self._exec_lock = asyncio.Lock()
        self._lock = asyncio.Lock()

        # delivery callbacks: source → async (WakeResult) → None
        self._callbacks: dict[str, Callable] = {}

        # 活跃时段（非 manual 触发检查）
        self._active_hours: tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None)

        # interval 定时循环
        self._interval_task: Optional[asyncio.Task] = None
        self._interval_every: float = 0
        self._interval_running = False

    # ── 配置 ──

    def set_active_hours(self, start: str, end: str, tz: str = "Asia/Shanghai") -> None:
        self._active_hours = (start, end, tz)

    def set_delivery_callback(self, source: str, callback: Callable) -> None:
        self._callbacks[source] = callback

    # ── 公开 API ──

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
        coalesce_ms: int = COALESCE_MS_DEFAULT,
    ) -> Optional[WakeResult]:
        """统一 wake 入口。

        所有 source 都通过此方法请求 AI turn。
        返回 WakeResult 仅当 source=manual + intent=manual 时。
        常规 merge 路径返回 None。
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

        # 直接执行（manual/manual 跳过合并，但要经过 exec_lock）
        if source == SOURCE_MANUAL and intent == INTENT_MANUAL:
            async with self._exec_lock:
                return await self._execute_wake(source, intent, reason, session_key, extra_prompt)

        # 合并调度
        async with self._lock:
            existing = self._pending.get(session_key)
            if existing:
                existing_prio = _INTENT_PRIORITY.get(existing.intent, 99)
                new_prio = _INTENT_PRIORITY.get(intent, 99)
                if new_prio >= existing_prio:
                    return None  # 不覆盖
                if new_prio < existing_prio:
                    self._pending[session_key].intent = intent
                    self._pending[session_key].source = source
                    self._pending[session_key].reason = reason
                    self._pending[session_key].extra_prompt = extra_prompt
            else:
                self._pending[session_key] = PendingWake(
                    source=source,
                    intent=intent,
                    reason=reason,
                    session_key=session_key,
                    extra_prompt=extra_prompt,
                )

            # 取消旧计时器，重新调度
            old = self._timers.pop(session_key, None)
            if old:
                old.cancel()
            loop = asyncio.get_event_loop()
            self._timers[session_key] = loop.call_later(
                coalesce_ms / 1000,
                lambda sk=session_key: asyncio.create_task(self._fire(sk)),
            )
        return None

    async def start_interval(self, every_minutes: float, prompt: str = "") -> None:
        self._interval_every = every_minutes * 60
        self._interval_running = True
        self._interval_task = asyncio.create_task(self._interval_loop(prompt))

    async def stop_interval(self) -> None:
        self._interval_running = False
        if self._interval_task and not self._interval_task.done():
            self._interval_task.cancel()
            try:
                await self._interval_task
            except asyncio.CancelledError:
                pass

        # 清理待处理的计时器和 pending
        async with self._lock:
            for handle in self._timers.values():
                handle.cancel()
            self._timers.clear()
            self._pending.clear()
        for handle in self._retry_timers.values():
            handle.cancel()
        self._retry_timers.clear()

    # ── 内部：interval 循环 ──

    async def _interval_loop(self, prompt: str = "") -> None:
        while self._interval_running:
            await asyncio.sleep(self._interval_every)
            if not self._interval_running:
                break
            await self.request(
                source=SOURCE_INTERVAL,
                intent=INTENT_SCHEDULED,
                session_key="heartbeat:events",
                reason="定时心跳",
                extra_prompt=prompt,
                coalesce_ms=100,
            )

    # ── 内部：合并触发 ──

    async def _fire(self, session_key: str) -> None:
        async with self._lock:
            pending = self._pending.get(session_key)  # peek，不 pop
            if not pending:
                return

        # ── 预检：冷却门控（在 pop 之前，defer 不丢 pending） ──
        if self._cooldown:
            from core.tasks.heartbeat_wake import WakeIntent as WI
            intent_map = {
                INTENT_MANUAL: WI.MANUAL,
                INTENT_IMMEDIATE: WI.IMMEDIATE,
                INTENT_SCHEDULED: WI.SCHEDULED,
                INTENT_EVENT: WI.EVENT,
            }
            hw_intent = intent_map.get(pending.intent, WI.SCHEDULED)
            dec = self._cooldown.should_defer(
                intent=hw_intent,
                is_busy=self._is_session_busy(session_key),
            )
            if dec.defer:
                _log.debug("Wake defer [%s..]: %s", session_key[:12], dec.reason)
                if dec.retry_after_ms > 0:
                    self._schedule_retry(session_key, dec.retry_after_ms)
                return  # pending 保留，retry 能拿到

        # ── 现在 pop ──
        async with self._lock:
            pending = self._pending.pop(session_key, None)
            self._timers.pop(session_key, None)
            if not pending:
                return

        async with self._exec_lock:
            await self._execute_wake(
                pending.source, pending.intent, pending.reason,
                session_key, pending.extra_prompt,
            )

    def _schedule_retry(self, session_key: str, delay_ms: int) -> None:
        loop = asyncio.get_event_loop()
        handle = loop.call_later(
            delay_ms / 1000,
            lambda sk=session_key: asyncio.create_task(self._retry_fire(sk)),
        )
        old = self._retry_timers.pop(session_key, None)
        if old:
            old.cancel()
        self._retry_timers[session_key] = handle

    async def _retry_fire(self, session_key: str) -> None:
        self._retry_timers.pop(session_key, None)
        await self._fire(session_key)

    # ── 内部：执行 + 预检 ──

    async def _execute_wake(
        self,
        source: str,
        intent: str,
        reason: str,
        session_key: str,
        extra_prompt: str = "",
    ) -> WakeResult:
        """执行 wake AI turn。预检（cooldown）已由 _fire 完成。"""

        # ── 预检 1: 活跃时段（非 manual） ──
        if source != SOURCE_MANUAL and self._active_hours[0]:
            if not self._is_in_active_hours():
                _log.debug("Wake 跳过 [%s..]: 不在活跃时段", session_key[:12])
                return WakeResult()

        # ── 预检 3: interval 且无事件无任务时跳过 ──
        if source == SOURCE_INTERVAL:
            has_events = self._events and self._events.has_events(session_key)
            has_content = bool(extra_prompt and extra_prompt.strip())
            if not has_events and not has_content:
                _log.debug("Wake 跳过 [%s..]: 无事件且无任务", session_key[:12])
                return WakeResult()

        # ── 执行 AI ──
        if self._cooldown:
            self._cooldown.record_run_start()
        if not self._agent:
            return WakeResult(error="AgentEngine 未就绪")

        result = await self._agent.run_wake_turn(
            source=source,
            intent=intent,
            reason=reason,
            session_key=session_key,
            extra_prompt=extra_prompt,
        )

        # ── 投递 ──
        callback = self._callbacks.get(source)
        if callback:
            try:
                await callback(result)
            except Exception as e:
                _log.error("delivery callback 异常 [%s]: %s", source, e)

        return result

    # ── 辅助 ──

    def _is_session_busy(self, session_key: str) -> bool:
        if not self._agent:
            return False
        last_active = getattr(self._agent, "last_active_time", 0.0)
        last_chat = getattr(self._agent, "last_active_chat", "") or ""
        if last_active <= 0:
            return False
        # 仅当同一 session 最近活跃时才认为忙，不影响其他 session
        if last_chat and session_key not in (last_chat, "heartbeat:events"):
            return False
        return (time.time() - last_active) < 120

    def _is_in_active_hours(self) -> bool:
        start_str, end_str, tz_str = self._active_hours
        if not start_str or not end_str:
            return True
        try:
            from datetime import datetime, timezone, timedelta
            import re as _re
            tz = timezone.utc
            if tz_str:
                try:
                    import zoneinfo
                    tz = zoneinfo.ZoneInfo(tz_str)
                except (ImportError, KeyError, TypeError):
                    m = _re.match(r"^UTC([+-]\d{1,2})(?::(\d{2}))?$", tz_str)
                    if m:
                        hours = int(m.group(1))
                        tz = timezone(timedelta(hours=hours))
            now = datetime.now(tz)
            cur = now.hour * 60 + now.minute
            sp = start_str.split(":")
            ep = end_str.split(":")
            start_min = int(sp[0]) * 60 + int(sp[1])
            end_min = int(ep[0]) * 60 + int(ep[1])
            if end_min <= start_min:
                return cur >= start_min or cur < end_min
            return start_min <= cur < end_min
        except Exception:
            return True

    def get_status(self) -> dict:
        return {
            "pending": list(self._pending.keys()),
            "running": self._exec_lock.locked(),
            "interval_running": self._interval_running,
        }
