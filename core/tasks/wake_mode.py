"""WakeMode — 事件完成后是否立即唤醒 AI。

NOW:             立即唤醒 AI（当前行为，向后兼容）
NEXT_HEARTBEAT:  只入队系统事件，等下次定时心跳消费

对应 OpenClaw 的 cron_wake_mode: "now" | "next-heartbeat"。
"""

from enum import Enum


class WakeMode(str, Enum):
    NOW = "now"
    NEXT_HEARTBEAT = "next-heartbeat"
