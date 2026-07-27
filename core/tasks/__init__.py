"""后台任务系统（Tasks + Cron Jobs）+ 心跳系统

仿 OpenClaw 设计哲学：后台任务是在主对话 Session 之外运行的工作记录。
每个任务拥有独立隔离的 Session（复用 SessionTaskManager 的 per-chat 队列锁机制）。

系统架构：
  models.py     → TaskRecord, CronJob, TaskStatus 数据模型
  store.py      → JSON 文件持久化
  manager.py    → TaskManager / CronJobManager CRUD 管理层
  scheduler.py  → CronJobScheduler 定时调度器
  runner.py     → BackgroundTaskRunner 任务执行器（复用 ToolLoop）

心跳子系统：
  heartbeat.py           → HeartbeatManager
  heartbeat_cooldown.py  → HeartbeatCooldown
  wake_coalescer.py      → WakeCoalescer (module-level singleton)
  wake_runner.py         → WakeRunner
  preflight.py           → run_preflight, PreflightContext, PreflightResult
  delivery_strategy.py   → HeartbeatDeliveryStrategy, ChatReplyDeliveryStrategy
  delivery_normalization.py → normalize_heartbeat_reply

外部依赖：croniter（cron 表达式解析）
"""

from .delivery_normalization import normalize_heartbeat_reply, strip_heartbeat_token
from .delivery_strategy import (
    HeartbeatDeliveryStrategy, ChatReplyDeliveryStrategy,
    SilentDeliveryStrategy, DeliveryStrategy,
)
from .heartbeat import HeartbeatManager
from .heartbeat_cooldown import HeartbeatCooldown
from .heartbeat_schedule import (
    resolve_phase_ms, compute_next_phase_due_ms, seek_next_active_phase,
    is_in_active_hours_ts,
)
from .heartbeat_wake import WakeIntent
from .manager import TaskManager, CronJobManager
from .models import TaskRecord, TaskStatus, CronJob, SessionMode
from .preflight import run_preflight, PreflightContext, PreflightResult
from .runner import BackgroundTaskRunner
from .scheduler import CronJobScheduler
from .store import TaskStore
from .wake_coalescer import (
    request_wake, execute_immediate, set_wake_handler, clear_pending,
    get_status, PendingWake, WakeRunResult, WakeTurnResult,
    SOURCE_INTERVAL, SOURCE_MANUAL, SOURCE_EXEC, SOURCE_CRON, SOURCE_TASK,
    INTENT_MANUAL, INTENT_IMMEDIATE, INTENT_EVENT, INTENT_SCHEDULED,
)
from .wake_runner import WakeRunner

__all__ = [
    # 后台任务
    "TaskRecord",
    "TaskStatus",
    "CronJob",
    "TaskStore",
    "TaskManager",
    "CronJobManager",
    "CronJobScheduler",
    "BackgroundTaskRunner",
    # 心跳系统
    "HeartbeatManager",
    "HeartbeatCooldown",
    "WakeIntent",
    "resolve_phase_ms",
    "compute_next_phase_due_ms",
    "seek_next_active_phase",
    "is_in_active_hours_ts",
    "request_wake",
    "execute_immediate",
    "set_wake_handler",
    "clear_pending",
    "get_status",
    "PendingWake",
    "WakeRunResult",
    "WakeTurnResult",
    "SOURCE_INTERVAL",
    "SOURCE_MANUAL",
    "SOURCE_EXEC",
    "SOURCE_CRON",
    "SOURCE_TASK",
    "INTENT_MANUAL",
    "INTENT_IMMEDIATE",
    "INTENT_EVENT",
    "INTENT_SCHEDULED",
    "WakeRunner",
    "run_preflight",
    "PreflightContext",
    "PreflightResult",
    "DeliveryStrategy",
    "HeartbeatDeliveryStrategy",
    "ChatReplyDeliveryStrategy",
    "SilentDeliveryStrategy",
    "normalize_heartbeat_reply",
    "strip_heartbeat_token",
]
