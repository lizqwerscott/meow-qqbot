"""WakeManager — 通用事件唤醒调度器。

类似 openclaw 的 heartbeat-wake.ts：
- 任何模块通过 notify() 入队事件 + 可选触发 AI turn
- 同 session_key 在 COALESCE_MS 内的多次 notify 合并为一次
- 忙态跳过（running 标志），v2 可加重试
"""

import asyncio
import logging
import time
from typing import Callable, Optional

_log = logging.getLogger(__name__)


class WakeManager:
    """Central wake/dispatch for background events.

    用法:
        wake_manager = WakeManager(system_events, agent_engine)
        await wake_manager.notify(
            session_key="group_123456",
            text="Exec completed: npm run build (code 0)",
            source="exec:exit",
            trigger_ai=True,
        )
    """

    COALESCE_MS = 1.0

    def __init__(self, system_events, agent_engine=None):
        self._events = system_events
        self._agent = agent_engine
        self._pending: dict[str, dict] = {}
        self._timers: dict[str, asyncio.TimerHandle] = {}
        self._running = False
        self._lock = asyncio.Lock()

    async def notify(
        self,
        session_key: str,
        text: str,
        context_key: Optional[str] = None,
        *,
        source: str = "system",
        trigger_ai: bool = False,
        replace: bool = False,
    ) -> None:
        if not session_key or not text:
            return

        self._events.enqueue(
            session_key=session_key,
            text=text,
            context_key=context_key,
            replace=replace,
        )

        if trigger_ai and self._agent:
            await self._schedule_wake(session_key, source)

    async def _schedule_wake(self, session_key: str, source: str) -> None:
        async with self._lock:
            old = self._timers.pop(session_key, None)
            if old:
                old.cancel()

            self._pending[session_key] = {
                "session_key": session_key,
                "source": source,
                "timestamp": time.time(),
            }

            loop = asyncio.get_event_loop()
            self._timers[session_key] = loop.call_later(
                self.COALESCE_MS,
                lambda: asyncio.create_task(self._fire(session_key)),
            )

    async def _fire(self, session_key: str) -> None:
        async with self._lock:
            self._pending.pop(session_key, None)
            self._timers.pop(session_key, None)

            if self._running:
                _log.debug("Wake 跳过 [%s..]: 当前 busy", session_key[:12])
                return

            self._running = True

        try:
            _log.debug("Wake 触发 [%s..]", session_key[:12])
            await self._agent.trigger_event_response(session_key)
        except Exception:
            _log.exception("Wake 异常 [%s..]", session_key[:12])
        finally:
            self._running = False
