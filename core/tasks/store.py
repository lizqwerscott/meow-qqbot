"""TaskStore — JSON 文件持久化层。

使用文件锁确保多协程安全。存储两个集合：
  - tasks: 任务记录（append-only + 定期清理）
  - cron_jobs: 定时任务定义（CRUD）
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional

from .models import CronJob, TaskRecord, TaskStatus

_log = logging.getLogger(__name__)


class TaskStore:
    """JSON 文件持久化存储。

    文件结构：
      data/tasks/
        tasks.json       → TaskRecord 列表（最近 N 条）
        cron_jobs.json   → CronJob 列表
    """

    def __init__(
        self,
        data_dir: str = "data/tasks/",
        max_tasks: int = 1000,
        task_ttl_days: int = 30,
    ):
        self._data_dir = data_dir
        self._max_tasks = max_tasks
        self._task_ttl_days = task_ttl_days
        self._lock = asyncio.Lock()

        os.makedirs(self._data_dir, exist_ok=True)

        self._tasks_path = os.path.join(self._data_dir, "tasks.json")
        self._jobs_path = os.path.join(self._data_dir, "cron_jobs.json")

        # 内存缓存
        self._tasks: Dict[str, TaskRecord] = {}
        self._jobs: Dict[str, CronJob] = {}

        self._load_all()

    # ── 内部 I/O ──

    def _load_json(self, path: str) -> list:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as e:
            _log.warning(f"读取 {path} 失败: {e}")
            return []

    def _save_json(self, path: str, data: list) -> None:
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except OSError as e:
            _log.error(f"写入 {path} 失败: {e}")

    def _load_all(self) -> None:
        """从 JSON 文件加载所有数据到内存。"""
        # 加载 tasks
        for item in self._load_json(self._tasks_path):
            try:
                task = TaskRecord.from_dict(item)
                self._tasks[task.id] = task
            except Exception as e:
                _log.warning(f"跳过损坏的 task 记录: {e}")

        # 加载 cron_jobs
        for item in self._load_json(self._jobs_path):
            try:
                job = CronJob.from_dict(item)
                self._jobs[job.id] = job
            except Exception as e:
                _log.warning(f"跳过损坏的 cron_job 记录: {e}")

        _log.info(
            f"TaskStore 已加载: {len(self._tasks)} 条任务, {len(self._jobs)} 个定时任务"
        )

    async def save_tasks(self) -> None:
        """将内存中的 tasks 写回 JSON 文件。"""
        async with self._lock:
            tasks = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
            if len(tasks) > self._max_tasks:
                tasks = tasks[: self._max_tasks]
            data = [t.to_dict() for t in tasks]
            await asyncio.to_thread(self._save_json, self._tasks_path, data)

    async def save_jobs(self) -> None:
        """将内存中的 cron_jobs 写回 JSON 文件。"""
        async with self._lock:
            data = [j.to_dict() for j in self._jobs.values()]
            await asyncio.to_thread(self._save_json, self._jobs_path, data)

    # ── Task CRUD ──

    async def add_task(self, task: TaskRecord) -> None:
        async with self._lock:
            self._tasks[task.id] = task
        await self.save_tasks()

    async def update_task(self, task: TaskRecord) -> None:
        async with self._lock:
            self._tasks[task.id] = task
        await self.save_tasks()

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    def list_tasks(
        self,
        limit: int = 50,
        status: Optional[TaskStatus] = None,
        job_id: Optional[str] = None,
    ) -> List[TaskRecord]:
        results = list(self._tasks.values())
        if status:
            results = [t for t in results if t.status == status]
        if job_id:
            results = [t for t in results if t.job_id == job_id]
        results.sort(key=lambda t: t.created_at, reverse=True)
        return results[:limit]

    async def delete_task(self, task_id: str) -> bool:
        found = False
        async with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                found = True
        if found:
            await self.save_tasks()
        return found

    async def cleanup_old_tasks(self) -> int:
        """清理超过 TTL 的终态任务。返回清理数量。"""
        now = time.time()
        cutoff = now - self._task_ttl_days * 86400
        to_delete = [
            tid
            for tid, t in self._tasks.items()
            if t.status in TaskStatus.terminal()
            and (t.finished_at or t.created_at) < cutoff
        ]
        async with self._lock:
            for tid in to_delete:
                self._tasks.pop(tid, None)
        if to_delete:
            await self.save_tasks()
            _log.info(f"清理了 {len(to_delete)} 条过期任务记录")
        return len(to_delete)

    # ── CronJob CRUD ──

    async def add_job(self, job: CronJob) -> None:
        async with self._lock:
            self._jobs[job.id] = job
        await self.save_jobs()

    async def update_job(self, job: CronJob) -> None:
        async with self._lock:
            self._jobs[job.id] = job
        await self.save_jobs()

    def get_job(self, job_id: str) -> Optional[CronJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[CronJob]:
        return list(self._jobs.values())

    async def delete_job(self, job_id: str) -> bool:
        found = False
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                found = True
        if found:
            await self.save_jobs()
        return found

    # ── 统计 ──

    def get_stats(self) -> dict:
        active = sum(1 for t in self._tasks.values() if t.status in TaskStatus.active())
        failed = sum(
            1
            for t in self._tasks.values()
            if t.status in (TaskStatus.FAILED, TaskStatus.TIMEOUT)
        )
        jobs_enabled = sum(1 for j in self._jobs.values() if j.enabled)
        return {
            "total_tasks": len(self._tasks),
            "active_tasks": active,
            "failed_tasks": failed,
            "total_jobs": len(self._jobs),
            "enabled_jobs": jobs_enabled,
        }

    # ── 生命周期 ──

    async def close(self) -> None:
        await self.save_tasks()
        await self.save_jobs()
        _log.info("TaskStore 已关闭")
