"""Cron 结果投递决策。"""

from dataclasses import dataclass

from .delivery_normalization import normalize_heartbeat_reply
from .models import CronJob, TaskRecord, TaskStatus


@dataclass(frozen=True)
class DeliveryDecision:
    should_deliver: bool
    content: str = ""
    reason: str = ""


def _payload_prefix(payload_type: str) -> str:
    return {"command": "🖥️", "message": "📋"}.get(payload_type, "📋")


def decide_cron_delivery(
    job: CronJob,
    task: TaskRecord,
    *,
    tool_delivered: bool = False,
) -> DeliveryDecision:
    """根据任务最终结果决定是否发送自动回执。"""
    if job.payload_type == "system_event":
        return DeliveryDecision(False, reason="system_event")
    if not job.enable_notify or not job.delivery_channel:
        return DeliveryDecision(False, reason="delivery_disabled")
    if task.status != TaskStatus.SUCCESS:
        return _decide_failure_delivery(job, task)
    if tool_delivered:
        return DeliveryDecision(False, reason="already_delivered")
    if task.silent:
        return DeliveryDecision(False, reason="silent_final_reply")

    raw = task.result or ""
    content, silent = normalize_heartbeat_reply(raw)
    if silent or not content.strip():
        return DeliveryDecision(False, reason="silent_final_reply")

    prefix = _payload_prefix(job.payload_type)
    return DeliveryDecision(
        True,
        content=f"{prefix} 定时任务 [{job.name}] 执行成功：\n{content}",
        reason="final_reply",
    )


def _decide_failure_delivery(job: CronJob, task: TaskRecord) -> DeliveryDecision:
    """决定失败、超时任务是否发送独立的错误通知。"""
    error = (task.error or "任务执行失败").strip()
    prefix = _payload_prefix(job.payload_type)
    return DeliveryDecision(
        True,
        content=f"{prefix} 定时任务 [{job.name}] 执行失败：\n{error}",
        reason="execution_failure",
    )
