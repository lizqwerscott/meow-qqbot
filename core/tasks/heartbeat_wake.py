"""Wake enums — 供 HeartbeatCooldown 引用。HeartbeatWakeScheduler 已合并到 WakeDispatcher。"""

from enum import Enum


class WakeIntent(Enum):
    MANUAL = "manual"
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    EVENT = "event"
