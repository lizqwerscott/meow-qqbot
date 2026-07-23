"""后台任务数据模型。"""

import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import List, Optional

from croniter import croniter

_log = logging.getLogger(__name__)


class SessionMode(str, Enum):
    """任务执行 Session 模式。

    对应 OpenClaw 的 --session 机制：
    - isolated:  每次执行使用全新 session（task:<uuid>）
    - custom:    持久化命名 session（跨执行保留上下文）
    - main:      专用 cron:main 通道（系统提醒）
    """
    ISOLATED = "isolated"
    CUSTOM = "custom"
    MAIN = "main"


class TaskStatus(str, Enum):
    """任务状态枚举。"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    LOST = "lost"

    @classmethod
    def terminal(cls) -> set:
        """返回所有终态（不含丢失）。"""
        return {cls.SUCCESS, cls.FAILED, cls.CANCELLED, cls.TIMEOUT}

    @classmethod
    def active(cls) -> set:
        """返回所有活跃态。"""
        return {cls.PENDING, cls.RUNNING}

    @classmethod
    def lost(cls) -> set:
        """返回丢失态。"""
        return {cls.LOST}


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
    reply_to_message_id: str = ""           # 创建任务时的原始消息 ID（用于发消息时构造 msg_id）

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
    """定时 / 一次性任务定义。

    两种调度方式（二选一）：
    - cron_expression: 标准 cron 表达式，周期性执行
    - at: Unix 时间戳，一次性执行（执行后自动删除）

    持久化在 JSON 文件中，机器人重启后重新计算 next_run_at。
    """
    id: str = field(default_factory=_new_id)
    name: str = ""
    cron_expression: str = ""           # 周期性 cron，如 "0 8 * * *"
    at: Optional[float] = None          # 一次性执行时间戳（UTC），有值则优先
    prompt: str = ""                    # AI 执行指令
    enabled: bool = True
    catch_up: bool = True               # 重启时是否补跑错过的任务（仅 cron 有效）
    delete_after_run: bool = True       # 一次性任务执行后自动删除
    created_at: float = field(default_factory=_now)
    next_run_at: Optional[float] = None
    delivery_channel: Optional[str] = None  # 结果投递到的 chat_id
    is_group: bool = True  # 来源聊天是否为群聊（影响 send_emoji 等工具使用群聊还是私聊接口）
    enable_notify: bool = True  # 是否投递执行结果到频道
    session_mode: str = SessionMode.ISOLATED.value  # isolated/custom/main
    custom_session_id: Optional[str] = None  # custom 模式下的命名 session ID

    # 唤醒策略
    wake_mode: str = "now"                  # now / next-heartbeat

    # 载荷类型
    payload_type: str = "message"           # message / command / system_event
    command: str = ""                       # shell 命令（command 载荷时使用）

    # 工具权限
    tools_allow: Optional[List[str]] = None
    # None = 默认工具集（announce + 记忆 + 文件）
    # ["*"] = 所有 cron 允许的工具
    # [工具名1, 工具名2, ...] = 指定工具列表
    # [] = 仅 announce

    # AI 选项
    model: Optional[str] = None             # 模型覆盖
    thinking: Optional[str] = None          # 思考级别（off/low/medium/high）

    @property
    def is_one_shot(self) -> bool:
        """是否是一次性任务。"""
        return self.at is not None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def __post_init__(self):
        if self.wake_mode not in ("now", "next-heartbeat"):
            self.wake_mode = "now"

    @classmethod
    def from_dict(cls, d: dict) -> "CronJob":
        return cls(**d)


def recalculate_next_run(job: CronJob) -> None:
    """计算 CronJob 的 next_run_at。

    - 一次性任务（at 有值）：直接用 at 作为 next_run_at
    - 周期性任务（cron）：用 croniter 从当前北京时间计算
    """
    if job.at is not None:
        job.next_run_at = job.at
        return
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    try:
        cron = croniter(job.cron_expression, now)
        job.next_run_at = cron.get_next(float)
    except (ValueError, KeyError) as e:
        _log.error(f"定时任务 {job.name} cron 表达式解析失败: {e}")
        job.next_run_at = None
