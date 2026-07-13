"""BackgroundTaskRunner — 后台任务执行器。

每个后台任务在一个独立的隔离 Session 中执行：
  1. 创建合成 InputMessage（用任务 prompt 作为 content）
  2. 使用 task:<uuid> 作为 chat_id → 天然复用 SessionTaskManager 的队列锁隔离
  3. 通过 AgentEngine._process_message 路径执行（复用 ToolLoop）
  4. 执行结果回写到 TaskRecord
  5. 可选：结果投递到指定 chat_id

任务执行是 best-effort 的，异常不会影响主机器人。
"""

import asyncio
import logging
import time
from typing import Any, Callable, Optional

from .models import CronJob, TaskRecord, TaskStatus

_log = logging.getLogger(__name__)


class BackgroundTaskRunner:
    """后台任务执行器。

    通过回调与 AgentEngine 解耦。
    """

    def __init__(self, task_manager: Any = None):
        self._task_manager = task_manager

        # 外部注入的回调
        self._execute_prompt_cb: Optional[Callable] = None
        # async (task_chat_id, prompt, sender_id) -> (result_text, error_text)
        self._delivery_cb: Optional[Callable] = None
        # async (chat_id, content, message_id, is_group) -> None

    def set_execute_callback(self, cb: Callable) -> None:
        """注入任务执行回调。

        cb: async (chat_id: str, prompt: str, sender_id: str) -> (result: str | None, error: str | None)
        """
        self._execute_prompt_cb = cb

    def set_delivery_callback(self, cb: Callable) -> None:
        """注入结果投递回调。

        cb: async (chat_id: str, content: str, message_id: str, is_group: bool) -> None
        """
        self._delivery_cb = cb

    # ── 主入口 ──

    async def run_task(
        self,
        task: TaskRecord,
        timeout: float = 300.0,
    ) -> TaskRecord:
        """在独立 Session 中执行一个后台任务。

        Args:
            task: 待执行的任务记录（状态应为 PENDING）
            timeout: 单个工具循环超时（秒）

        Returns:
            更新后的 TaskRecord
        """
        if self._task_manager is None:
            _log.error("TaskManager 未注入，无法执行任务")
            task.status = TaskStatus.FAILED
            task.error = "TaskManager 未就绪"
            return task

        if self._execute_prompt_cb is None:
            _log.error("execute_callback 未注入，无法执行任务")
            task.status = TaskStatus.FAILED
            task.error = "执行器未就绪"
            return task

        # 标记为 running
        task = self._task_manager.start_task(task.id)
        if task is None:
            return task

        chat_id = task.session_id  # task:<id>
        _log.info(
            f"开始执行后台任务: id={task.id[:12]}.. "
            f"session={chat_id} prompt={task.prompt[:60]}"
        )

        try:
            # 在独立 session 中执行 prompt
            result, error = await asyncio.wait_for(
                self._execute_prompt_cb(
                    chat_id=chat_id,
                    prompt=task.prompt,
                    sender_id="system",
                ),
                timeout=timeout,
            )

            # 更新任务记录
            if error:
                task = self._task_manager.finish_task(
                    task.id,
                    status=TaskStatus.FAILED,
                    result=result,
                    error=error,
                )
            else:
                task = self._task_manager.finish_task(
                    task.id,
                    status=TaskStatus.SUCCESS,
                    result=result,
                )

        except asyncio.TimeoutError:
            _log.warning(f"后台任务超时: id={task.id[:12]}.. timeout={timeout}s")
            task = self._task_manager.finish_task(
                task.id,
                status=TaskStatus.TIMEOUT,
                error=f"执行超时 ({timeout}s)",
            )
        except asyncio.CancelledError:
            _log.warning(f"后台任务被取消: id={task.id[:12]}..")
            task = self._task_manager.finish_task(
                task.id,
                status=TaskStatus.CANCELLED,
                error="任务被取消",
            )
        except Exception as e:
            _log.error(
                f"后台任务异常: id={task.id[:12]}.. error={e}", exc_info=True
            )
            task = self._task_manager.finish_task(
                task.id,
                status=TaskStatus.FAILED,
                error=str(e),
            )

        return task

    async def run_cron_job(
        self,
        job: CronJob,
        timeout: float = 300.0,
    ) -> Optional[TaskRecord]:
        """执行一个 CronJob（创建 → 执行 → 可选投递）。"""
        if self._task_manager is None:
            _log.error("TaskManager 未注入")
            return None
        task = self._task_manager.create_task(
            prompt=job.prompt,
            task_type="cron",
            job_id=job.id,
            delivery_channel=job.delivery_channel,
        )

        task = await self.run_task(task, timeout=timeout)

        # 如果指定了投递频道且任务有结果，投递结果
        if job.delivery_channel and task and task.result and self._delivery_cb:
            try:
                await self._delivery_cb(
                    chat_id=job.delivery_channel,
                    content=f"📋 定时任务 [{job.name}] 执行结果：\n{task.result[:500]}",
                    message_id="",
                    is_group=True,
                )
            except Exception as e:
                _log.error(f"投递任务结果失败: {e}")

        return task

    async def run_manual_task(
        self,
        prompt: str,
        timeout: float = 300.0,
        delivery_channel: Optional[str] = None,
    ) -> TaskRecord:
        """执行一个手动后台任务。"""
        if self._task_manager is None:
            raise RuntimeError("TaskManager 未注入")
        task = self._task_manager.create_task(
            prompt=prompt,
            task_type="manual",
            delivery_channel=delivery_channel,
        )
        return await self.run_task(task, timeout=timeout)
