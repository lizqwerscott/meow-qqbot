"""CronJobScheduler — Cron 定时调度器。

轮询机制：
1. 启动时重新计算所有启用 job 的 next_run_at
2. 每 30 秒轮询一次（可配置）
3. 当 next_run_at <= now 时触发执行
4. 重启恢复：启动时检查 catch_up 标记，补跑离线期间错过的任务

使用 croniter 库解析 cron 表达式（标准 5 字段格式）。
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from croniter import croniter

from .models import CronJob, TaskRecord, TaskStatus

_log = logging.getLogger(__name__)


class CronJobScheduler:
    """Cron 定时调度器。

    Args:
        poll_interval: 轮询间隔（秒），默认 30
        catch_up_window: 重启时补跑窗口（秒），默认 3600（1 小时）
        max_concurrent: 最大并发执行任务数，默认 3
    """

    def __init__(
        self,
        poll_interval: float = 30.0,
        catch_up_window: float = 3600.0,
        max_concurrent: int = 3,
    ):
        self._poll_interval = poll_interval
        self._catch_up_window = catch_up_window
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # 需要外部注入的回调
        self._on_trigger: Optional[Callable] = None  # async (job: CronJob) -> None
        self._get_jobs: Optional[Callable] = None     # () -> list[CronJob]
        self._update_job: Optional[Callable] = None   # async (job: CronJob) -> None

        self._running = False
        self._task: Optional[asyncio.Task] = None

    def set_callbacks(
        self,
        on_trigger: Callable,
        get_jobs: Callable,
        update_job: Callable,
    ) -> None:
        """注入回调。

        on_trigger: async (job: CronJob) -> None — 执行任务
        get_jobs: () -> list[CronJob] — 获取所有 job
        update_job: (job: CronJob) -> None — 更新 job（同步）
        """
        self._on_trigger = on_trigger
        self._get_jobs = get_jobs
        self._update_job = update_job

    # ── 启动/停止 ──

    def start(self) -> None:
        """启动调度轮询。"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        _log.info(
            f"CronJobScheduler 已启动 (poll={self._poll_interval}s, "
            f"catch_up_window={self._catch_up_window}s, "
            f"max_concurrent={self._max_concurrent})"
        )

    async def stop(self) -> None:
        """停止调度器。"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        _log.info("CronJobScheduler 已停止")

    # ── 内部 ──

    def _recalculate_next_run(self, job: CronJob) -> None:
        """从当前时间重新计算 next_run_at。"""
        now = datetime.now(timezone.utc)
        try:
            cron = croniter(job.cron_expression, now)
            job.next_run_at = cron.get_next(float)
        except (ValueError, KeyError) as e:
            _log.error(f"定时任务 {job.name} cron 表达式解析失败: {e}")
            job.next_run_at = None

    def _recover_missed_jobs(self) -> list[CronJob]:
        """检查是否有需要补跑的 job。返回需要立即触发的 job 列表。"""
        now = time.time()
        to_trigger = []
        if not self._get_jobs:
            return to_trigger

        for job in self._get_jobs():
            if not job.enabled or not job.catch_up:
                continue
            if job.next_run_at is None:
                continue

            # 如果 next_run_at 在 1 秒前就应该触发（给予轮询精度缓冲）
            if job.next_run_at <= now - 1:
                delta = now - job.next_run_at
                if delta <= self._catch_up_window:
                    _log.info(
                        f"定时任务 {job.name} 需补跑 (missed={delta:.0f}s)"
                    )
                    to_trigger.append(job)
                else:
                    _log.info(
                        f"定时任务 {job.name} 错过超过补跑窗口 ({delta:.0f}s)，跳过"
                    )
                    # 跳过，重新计算下一次
                    self._recalculate_next_run(job)
                    if self._update_job:
                        self._update_job(job)

        return to_trigger

    async def _poll_loop(self) -> None:
        """主轮询循环。"""
        # 启动时：重新计算所有 job 的 next_run_at
        if self._get_jobs:
            for job in self._get_jobs():
                self._recalculate_next_run(job)
                if self._update_job:
                    self._update_job(job)

        # 启动时补跑（fire-and-forget）
        missed = self._recover_missed_jobs()
        for job in missed:
            if self._on_trigger:
                asyncio.create_task(self._on_trigger(job))

        # 轮询循环
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                _log.error(f"CronJobScheduler 轮询异常: {e}", exc_info=True)

    async def _tick(self) -> None:
        """单个轮询周期。"""
        if not self._get_jobs or not self._on_trigger:
            return

        now = time.time()
        for job in self._get_jobs():
            if not job.enabled:
                continue
            if job.next_run_at is None:
                continue

            if job.next_run_at <= now:
                _log.info(
                    f"定时任务到期: {job.name} (next_run_at={job.next_run_at})"
                )
                # 计算下一次，先更新再执行，防止重复触发
                self._recalculate_next_run(job)
                if self._update_job:
                    self._update_job(job)

                # 非阻塞触发（fire-and-forget，不阻塞轮询）
                asyncio.create_task(self._on_trigger(job))
