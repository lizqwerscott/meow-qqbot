"""后台任务数据模型。"""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    """任务状态枚举。"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

    @classmethod
    def terminal(cls) -> set:
        """返回所有终态。"""
        return {cls.SUCCESS, cls.FAILED, cls.CANCELLED, cls.TIMEOUT}

    @classmethod
    def active(cls) -> set:
        """返回所有活跃态。"""
        return {cls.PENDING, cls.RUNNING}


def _now() -> float:
    return datetime.now(timezone.utc).timestamp()


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class TaskRecord:
    """后台任务记录。

    每次 cron 执行或手动后台任务都会创建一条 TaskRecord。
    每个 TaskRecord 拥有独立 session（session_id = 'task:<id>'）。
    """
    id: str = field(default_factory=_new_id)
    type: str = "manual"            # "cron" | "manual"
    status: TaskStatus = TaskStatus.PENDING
    prompt: str = ""                # AI 执行指令
    job_id: Optional[str] = None    # 关联的 CronJob ID
    session_id: str = ""            # 运行时填充：task:<id>
    created_at: float = field(default_factory=_now)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[str] = None    # AI 回复文本
    error: Optional[str] = None
    delivery_channel: Optional[str] = None  # 结果投递到的 chat_id

    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"task:{self.id}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TaskRecord":
        d = dict(d)
        d["status"] = TaskStatus(d.get("status", "pending"))
        return cls(**d)


@dataclass
class CronJob:
    """定时任务定义。

    持久化在 JSON 文件中，机器人重启后由 Scheduler 重新加载
    并重新计算 next_run_at。
    """
    id: str = field(default_factory=_new_id)
    name: str = ""
    cron_expression: str = "0 * * * *"  # 默认每小时
    prompt: str = ""                    # AI 执行指令
    enabled: bool = True
    catch_up: bool = True               # 重启时是否补跑错过的任务
    created_at: float = field(default_factory=_now)
    next_run_at: Optional[float] = None
    delivery_channel: Optional[str] = None  # 结果投递到的 chat_id

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CronJob":
        return cls(**d)
