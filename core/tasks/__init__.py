"""后台任务系统（Tasks + Cron Jobs）

仿 OpenClaw 设计哲学：后台任务是在主对话 Session 之外运行的工作记录。
每个任务拥有独立隔离的 Session（复用 SessionTaskManager 的 per-chat 队列锁机制）。

系统架构：
  models.py     → TaskRecord, CronJob, TaskStatus 数据模型
  store.py      → JSON 文件持久化
  manager.py    → TaskManager / CronJobManager CRUD 管理层
  scheduler.py  → CronJobScheduler 定时调度器
  runner.py     → BackgroundTaskRunner 任务执行器（复用 ToolLoop）

外部依赖：croniter（cron 表达式解析）
"""

from .models import TaskRecord, TaskStatus, CronJob
from .store import TaskStore
from .manager import TaskManager, CronJobManager
from .scheduler import CronJobScheduler
from .runner import BackgroundTaskRunner

__all__ = [
    "TaskRecord",
    "TaskStatus",
    "CronJob",
    "TaskStore",
    "TaskManager",
    "CronJobManager",
    "CronJobScheduler",
    "BackgroundTaskRunner",
]
