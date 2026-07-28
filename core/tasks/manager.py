"""TaskManager / CronJobManager — 任务和定时任务的 CRUD 管理层。

职责：
- TaskManager: 创建/更新/查询/取消任务，执行任务触发
- CronJobManager: 创建/更新/删除/启用/禁用定时任务
"""

import asyncio
import logging
import time
from typing import Callable, List, Optional

from .models import CronJob, TaskRecord, TaskStatus, recalculate_next_run
from .store import TaskStore

_log = logging.getLogger(__name__)


class TaskManager:
    """任务记录管理器。"""

    def __init__(self, store: TaskStore):
        self._store = store
        # 运行中任务的 asyncio.Task 集合（用于取消）
        self._running_tasks: dict[str, asyncio.Task] = {}
        # 状态转换锁（防止 cancel_task / finish_task 竞态）
        self._status_lock = asyncio.Lock()

    # ── 创建 ──

    async def create_task(
        self,
        prompt: str,
        task_type: str = "manual",
        job_id: Optional[str] = None,
        delivery_channel: Optional[str] = None,
        reply_to_message_id: str = "",
    ) -> TaskRecord:
        """创建一个新的后台任务（状态 = pending）。"""
        task = TaskRecord(
            type=task_type,
            prompt=prompt,
            job_id=job_id,
            delivery_channel=delivery_channel,
            reply_to_message_id=reply_to_message_id,
        )
        await self._store.add_task(task)
        _log.info(
            f"任务已创建: id={task.id[:12]}.. type={task_type} "
            f"prompt={prompt[:60]}"
        )
        return task

    # ── 更新 ──

    async def start_task(self, task_id: str) -> Optional[TaskRecord]:
        """标记任务为 running，并注册当前 asyncio.Task 用于取消和 LOST 检测。"""
        task = self._store.get_task(task_id)
        if task is None:
            return None
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        await self._store.update_task(task)
        current = asyncio.current_task()
        if current is not None:
            self._running_tasks[task_id] = current
        return task

    async def update_task_record(self, task: TaskRecord) -> None:
        """更新任务记录到持久化存储（不改变状态）。"""
        await self._store.update_task(task)

    async def finish_task(
        self,
        task_id: str,
        status: TaskStatus = TaskStatus.SUCCESS,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[TaskRecord]:
        """完成任务。"""
        async with self._status_lock:
            task = self._store.get_task(task_id)
            if task is None:
                return None
            # 已被 cancel_task 取消，不覆写
            if task.status == TaskStatus.CANCELLED:
                self._running_tasks.pop(task_id, None)
                return task
            task.status = status
            task.finished_at = time.time()
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            await self._store.update_task(task)
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
        runner = None
        async with self._status_lock:
            task = self._store.get_task(task_id)
            if task is None:
                return False
            if task.status not in TaskStatus.active():
                _log.warning(f"任务 {task_id[:12]}.. 当前状态 {task.status.value} 不可取消")
                return False

            runner = self._running_tasks.pop(task_id, None)
            task.status = TaskStatus.CANCELLED
            task.finished_at = time.time()
            task.error = "用户取消"
            await self._store.update_task(task)

        if runner is not None and not runner.done():
            runner.cancel()
            try:
                await asyncio.wait_for(runner, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        _log.info(f"任务已取消: id={task_id[:12]}..")
        return True

    # ── LOST 检测 ──

    async def detect_lost_tasks(self, lost_detection_minutes: int = 30) -> int:
        """检测并标记丢失的任务。

        将满足以下条件的活跃任务标记为 LOST：
        - RUNNING：started_at + lost_detection_minutes 无更新
        - PENDING：created_at + lost_detection_minutes 未被调度

        跳过当前仍在 _running_tasks 中运行的 asyncio.Task。
        """
        now = time.time()
        cutoff = now - lost_detection_minutes * 60
        count = 0

        async with self._status_lock:
            for tid, t in self._store.all_tasks().items():
                if t.status not in TaskStatus.active():
                    continue

                # 跳过仍在本进程运行的任务
                runner = self._running_tasks.get(tid)
                if runner is not None and not runner.done():
                    continue

                # RUNNING：用 started_at，PENDING：用 created_at
                ts = t.started_at if t.status == TaskStatus.RUNNING else t.created_at
                if ts is None or ts > cutoff:
                    continue

                old_status = t.status.value
                t.status = TaskStatus.LOST
                t.finished_at = now
                t.error = "任务丢失（进程崩溃或重启导致）"
                await self._store.update_task(t)
                count += 1
                _log.warning(
                    f"任务标记为丢失: id={tid[:12]}.. "
                    f"之前状态={old_status} 检测阈值={lost_detection_minutes}分钟"
                )

        if count:
            _log.info(f"本次检测到 {count} 个丢失任务")
        return count

    # ── 清理 ──

    async def cleanup_old_tasks(self) -> int:
        return await self._store.cleanup_old_tasks()

    async def enforce_per_job_terminal_limit(self, max_per_job: int = 2000) -> int:
        return await self._store.enforce_per_job_terminal_limit(max_per_job)


class CronJobManager:
    """定时任务定义管理器。"""

    def __init__(self, store: TaskStore):
        self._store = store

    # ── CRUD ──

    async def create_job(
        self,
        name: str,
        cron_expression: str = "",
        prompt: str = "",
        *,
        at: Optional[float] = None,
        delivery_channel: Optional[str] = None,
        is_group: bool = True,
        enable_notify: bool = True,
        catch_up: bool = True,
        enabled: bool = True,
        delete_after_run: bool = True,
        session_mode: str = "isolated",
        custom_session_id: Optional[str] = None,
        payload_type: str = "message",
        command: str = "",
        model: Optional[str] = None,
        thinking: Optional[str] = None,
        tools_allow: Optional[List[str]] = None,
    ) -> CronJob:
        if not cron_expression and at is None:
            raise ValueError("cron_expression 和 at 必须至少提供一个")
        if cron_expression and at is not None:
            raise ValueError("cron_expression 和 at 不能同时设置")

        if payload_type == "command":
            prompt = ""
        elif payload_type == "system_event":
            command = ""
        else:
            command = ""

        job = CronJob(
            name=name,
            cron_expression=cron_expression,
            at=at,
            prompt=prompt,
            enabled=enabled,
            catch_up=catch_up,
            delete_after_run=delete_after_run,
            delivery_channel=delivery_channel,
            is_group=is_group,
            enable_notify=enable_notify,
            session_mode=session_mode,
            custom_session_id=custom_session_id,
            payload_type=payload_type,
            command=command,
            model=model,
            thinking=thinking,
            tools_allow=tools_allow,
        )
        recalculate_next_run(job)
        if job.next_run_at is None:
            raise ValueError(
                f"定时任务 {name} 调度表达式无效: cron={cron_expression!r} at={at}"
            )
        await self._store.add_job(job)
        schedule_desc = f"at={at}" if at is not None else f"cron={cron_expression}"
        _log.info(
            f"定时任务已创建: id={job.id[:12]}.. name={name} "
            f"{schedule_desc} session={session_mode} "
            f"payload={payload_type} "
            f"next_run={job.next_run_at}"
        )
        return job

    async def update_job(self, job: CronJob) -> None:
        await self._store.update_job(job)

    def get_job(self, job_id: str) -> Optional[CronJob]:
        return self._store.get_job(job_id)

    def list_jobs(self) -> List[CronJob]:
        return self._store.list_jobs()

    async def delete_job(self, job_id: str) -> bool:
        job = self._store.get_job(job_id)
        if job is None:
            return False
        await self._store.delete_job(job_id)
        _log.info(f"定时任务已删除: id={job_id[:12]}.. name={job.name}")
        return True

    def find_jobs_by_name(self, name: str) -> List[CronJob]:
        name_lower = name.lower()
        return [j for j in self._store.list_jobs() if name_lower in j.name.lower()]

    async def enable_job(self, job_id: str) -> bool:
        job = self._store.get_job(job_id)
        if job is None:
            return False
        job.enabled = True
        recalculate_next_run(job)
        await self._store.update_job(job)
        _log.info(f"定时任务已启用: {job.name}")
        return True

    async def disable_job(self, job_id: str) -> bool:
        job = self._store.get_job(job_id)
        if job is None:
            return False
        job.enabled = False
        await self._store.update_job(job)
        _log.info(f"定时任务已禁用: {job.name}")
        return True
