"""TaskManager / CronJobManager — 任务和定时任务的 CRUD 管理层。

职责：
- TaskManager: 创建/更新/查询/取消任务，执行任务触发
- CronJobManager: 创建/更新/删除/启用/禁用定时任务
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Callable, List, Optional

from croniter import croniter

from .models import CronJob, TaskRecord, TaskStatus
from .store import TaskStore

_log = logging.getLogger(__name__)


class TaskManager:
    """任务记录管理器。"""

    def __init__(self, store: TaskStore):
        self._store = store
        # 运行中任务的 asyncio.Task 集合（用于取消）
        self._running_tasks: dict[str, asyncio.Task] = {}

    # ── 创建 ──

    def create_task(
        self,
        prompt: str,
        task_type: str = "manual",
        job_id: Optional[str] = None,
        delivery_channel: Optional[str] = None,
    ) -> TaskRecord:
        """创建一个新的后台任务（状态 = pending）。"""
        task = TaskRecord(
            type=task_type,
            prompt=prompt,
            job_id=job_id,
            delivery_channel=delivery_channel,
        )
        self._store.add_task(task)
        _log.info(
            f"任务已创建: id={task.id[:12]}.. type={task_type} "
            f"prompt={prompt[:60]}"
        )
        return task

    # ── 更新 ──

    def start_task(self, task_id: str) -> Optional[TaskRecord]:
        """标记任务为 running。"""
        task = self._store.get_task(task_id)
        if task is None:
            return None
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self._store.update_task(task)
        return task

    def finish_task(
        self,
        task_id: str,
        status: TaskStatus = TaskStatus.SUCCESS,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[TaskRecord]:
        """完成任务。"""
        task = self._store.get_task(task_id)
        if task is None:
            return None
        task.status = status
        task.finished_at = time.time()
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        self._store.update_task(task)
        self._running_tasks.pop(task_id, None)
        _log.info(
            f"任务完成: id={task_id[:12]}.. status={status.value} "
            f"result_len={len(result or '')}"
        )
        return task

    # ── 查询 ──

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        return self._store.get_task(task_id)

    def list_tasks(
        self,
        limit: int = 50,
        status: Optional[TaskStatus] = None,
        job_id: Optional[str] = None,
    ) -> List[TaskRecord]:
        return self._store.list_tasks(limit=limit, status=status, job_id=job_id)

    def list_active_tasks(self) -> List[TaskRecord]:
        return self._store.list_tasks(limit=100)

    # ── 取消 ──

    async def cancel_task(self, task_id: str) -> bool:
        """取消一个 pending 或 running 的任务。"""
        task = self._store.get_task(task_id)
        if task is None:
            return False
        if task.status not in TaskStatus.active():
            _log.warning(f"任务 {task_id[:12]}.. 当前状态 {task.status.value} 不可取消")
            return False

        # 如果有 asyncio.Task 在运行，取消它
        runner = self._running_tasks.pop(task_id, None)
        if runner is not None and not runner.done():
            runner.cancel()
            try:
                await asyncio.wait_for(runner, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        task.status = TaskStatus.CANCELLED
        task.finished_at = time.time()
        task.error = "用户取消"
        self._store.update_task(task)
        _log.info(f"任务已取消: id={task_id[:12]}..")
        return True

    # ── 清理 ──

    def cleanup_old_tasks(self) -> int:
        return self._store.cleanup_old_tasks()


class CronJobManager:
    """定时任务定义管理器。"""

    def __init__(self, store: TaskStore):
        self._store = store

    # ── CRUD ──

    @staticmethod
    def _recalculate_next_run(job: CronJob) -> None:
        """计算 next_run_at。
        - 一次性任务（at 有值）：直接用 at 作为 next_run_at
        - 周期性任务（cron）：用 croniter 计算
        """
        if job.at is not None:
            job.next_run_at = job.at
            return
        try:
            # 使用 CST (UTC+8) 以匹配 AI prompt 注入的时间
            _tz = timezone(timedelta(hours=8))
            now = datetime.now(_tz)
            cron = croniter(job.cron_expression, now)
            job.next_run_at = cron.get_next(float)
        except (ValueError, KeyError) as e:
            _log.error(f"定时任务 {job.name} cron 表达式解析失败: {e}")
            job.next_run_at = None

    def create_job(
        self,
        name: str,
        cron_expression: str = "",
        prompt: str = "",
        *,
        at: Optional[float] = None,
        delivery_channel: Optional[str] = None,
        catch_up: bool = True,
        enabled: bool = True,
        delete_after_run: bool = True,
        session_mode: str = "isolated",
        custom_session_id: Optional[str] = None,
        payload_type: str = "message",
        command: str = "",
        model: Optional[str] = None,
        thinking: Optional[str] = None,
    ) -> CronJob:
        """创建定时或一次性任务。

        Args:
            name: 任务名称
            cron_expression: 周期性 cron 表达式（与 at 二选一）
            prompt: AI 执行指令
            at: 一次性执行 UTC 时间戳（与 cron_expression 二选一）
            delivery_channel: 结果投递 chat_id
            catch_up: 重启时补跑（仅周期任务）
            enabled: 是否启用
            delete_after_run: 一次性任务执行后自动删除
            session_mode: isolated/current/custom/main
            custom_session_id: custom 模式的命名 session ID
            payload_type: message/command/system_event
            command: shell 命令（payload_type=command 时使用）
            model: 模型覆盖
            thinking: 思考级别
        """
        if not cron_expression and at is None:
            raise ValueError("cron_expression 和 at 必须至少提供一个")
        if cron_expression and at is not None:
            raise ValueError("cron_expression 和 at 不能同时设置")

        job = CronJob(
            name=name,
            cron_expression=cron_expression,
            at=at,
            prompt=prompt,
            enabled=enabled,
            catch_up=catch_up,
            delete_after_run=delete_after_run,
            delivery_channel=delivery_channel,
            session_mode=session_mode,
            custom_session_id=custom_session_id,
            payload_type=payload_type,
            command=command,
            model=model,
            thinking=thinking,
        )
        self._recalculate_next_run(job)
        self._store.add_job(job)
        schedule_desc = f"at={at}" if at is not None else f"cron={cron_expression}"
        _log.info(
            f"定时任务已创建: id={job.id[:12]}.. name={name} "
            f"{schedule_desc} session={session_mode} "
            f"payload={payload_type} "
            f"next_run={job.next_run_at}"
        )
        return job

    def update_job(self, job: CronJob) -> None:
        self._store.update_job(job)

    def get_job(self, job_id: str) -> Optional[CronJob]:
        return self._store.get_job(job_id)

    def list_jobs(self) -> List[CronJob]:
        return self._store.list_jobs()

    def delete_job(self, job_id: str) -> bool:
        job = self._store.get_job(job_id)
        if job is None:
            return False
        self._store.delete_job(job_id)
        _log.info(f"定时任务已删除: id={job_id[:12]}.. name={job.name}")
        return True

    def find_jobs_by_name(self, name: str) -> List[CronJob]:
        """按名称模糊查找。"""
        name_lower = name.lower()
        return [j for j in self._store.list_jobs() if name_lower in j.name.lower()]

    # ── 启用/禁用 ──

    def enable_job(self, job_id: str) -> bool:
        job = self._store.get_job(job_id)
        if job is None:
            return False
        job.enabled = True
        self._recalculate_next_run(job)
        self._store.update_job(job)
        _log.info(f"定时任务已启用: {job.name}")
        return True

    def disable_job(self, job_id: str) -> bool:
        job = self._store.get_job(job_id)
        if job is None:
            return False
        job.enabled = False
        self._store.update_job(job)
        _log.info(f"定时任务已禁用: {job.name}")
        return True
