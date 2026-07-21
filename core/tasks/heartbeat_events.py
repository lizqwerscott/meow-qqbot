"""HeartbeatEvents — 心跳事件广播

参照 OpenClaw heartbeat-events.ts 设计。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

_log = logging.getLogger(__name__)


@dataclass
class HeartbeatEvent:
    timestamp: float = field(default_factory=time.time)
    status: str = "running"
    duration_ms: float = 0.0
    result_text: Optional[str] = None
    source: str = ""
    reason: str = ""


class HeartbeatEvents:
    """全局单例，广播心跳状态。

    职责：
    1. emit() — 发布事件
    2. on_event() — 注册 listener
    3. get_last() — 获取最新事件（供调试/命令使用）
    """

    def __init__(self):
        self._last_event: Optional[HeartbeatEvent] = None
        self._listeners: list[Callable[[HeartbeatEvent], None]] = []

    def on_event(self, listener: Callable[[HeartbeatEvent], None]):
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[HeartbeatEvent], None]):
        self._listeners[:] = [l for l in self._listeners if l is not listener]

    def emit(self, event: HeartbeatEvent):
        self._last_event = event
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                _log.warning(f"HeartbeatEvents listener 异常: {e}")

    def get_last(self) -> Optional[HeartbeatEvent]:
        return self._last_event
