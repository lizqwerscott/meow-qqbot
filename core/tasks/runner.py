"""BackgroundTaskRunner — 后台任务执行器。

每个后台任务在一个独立的隔离 Session 中执行：
  1. 创建合成 InputMessage（用任务 prompt 作为 content）
  2. 根据 session_mode 选择 chat_id → 复用 SessionTaskManager 的队列锁隔离
     - isolated: task:<uuid>（全新隔离上下文）
     - custom: cron:<custom_id>（持久化命名 session）
     - main: cron:main（专用 cron 通道）
  3. 通过 AgentEngine._process_message 路径执行（复用 ToolLoop）
  4. 执行结果回写到 TaskRecord
  5. 可选：结果投递到指定 chat_id

任务执行是 best-effort 的，异常不会影响主机器人。
"""

import asyncio
import logging
import os
import shlex
import subprocess
import time
from typing import Any, Callable, List, Optional

from .models import CronJob, SessionMode, TaskRecord, TaskStatus

# 安全黑名单（复用 SkillManagers 的配置）
_DENIED_COMMANDS: frozenset = frozenset({
    "rm", "chmod", "chown", "sudo", "su", "doas",
    "dd", "mkfs", "fdisk", "parted", "mkswap",
    "shutdown", "reboot", "poweroff", "halt", "init", "systemctl",
    "useradd", "usermod", "groupadd", "userdel", "groupdel",
    "setuid", "setgid", "chattr", "lsattr",
    "tcpdump", "nmap", "tshark",
    "pkill", "killall", "kill", "passwd",
    "service", "grub-install", "grub-mkconfig",
    "modprobe", "insmod", "rmmod",
    "iptables", "ufw",
})

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
        self._system_events: Optional[Any] = None
        self._wake_manager: Optional[Any] = None

    def set_system_events(self, system_events: Any) -> None:
        """注入系统事件队列。"""
        self._system_events = system_events

    def set_wake_manager(self, wake_manager: Any) -> None:
        """注入 WakeManager（事件完成时唤醒 AI）。"""
        self._wake_manager = wake_manager

    def set_execute_callback(self, cb: Callable) -> None:
        """注入任务执行回调。

        cb: async (chat_id: str, prompt: str, sender_id: str, is_group: bool = True, delivery_channel: str = "", reply_to_message_id: str = "", tools_allow: list[str] | None = None) -> (result: str | None, error: str | None)
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
        is_group: bool = True,
        tools_allow: Optional[List[str]] = None,
    ) -> TaskRecord:
        """在独立 Session 中执行一个后台任务。

        Args:
            task: 待执行的任务记录（状态应为 PENDING）
            timeout: 单个工具循环超时（秒）
            is_group: 来源聊天是否为群聊（影响工具如 send_emoji 的接口选择）
            tools_allow: 可用的工具列表（None=默认，["*"]=全部，[]=仅announce）

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
        task = await self._task_manager.start_task(task.id)
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
                    is_group=is_group,
                    delivery_channel=task.delivery_channel or "",
                    reply_to_message_id=task.reply_to_message_id,
                    tools_allow=tools_allow,
                ),
                timeout=timeout,
            )

            # 更新任务记录
            if error:
                task = await self._task_manager.finish_task(
                    task.id,
                    status=TaskStatus.FAILED,
                    result=result,
                    error=error,
                )
            else:
                task = await self._task_manager.finish_task(
                    task.id,
                    status=TaskStatus.SUCCESS,
                    result=result,
                )

        except asyncio.TimeoutError:
            _log.warning(f"后台任务超时: id={task.id[:12]}.. timeout={timeout}s")
            task = await self._task_manager.finish_task(
                task.id,
                status=TaskStatus.TIMEOUT,
                error=f"执行超时 ({timeout}s)",
            )
        except asyncio.CancelledError:
            _log.warning(f"后台任务被取消: id={task.id[:12]}..")
            task = await self._task_manager.finish_task(
                task.id,
                status=TaskStatus.CANCELLED,
                error="任务被取消",
            )
        except Exception as e:
            _log.error(
                f"后台任务异常: id={task.id[:12]}.. error={e}", exc_info=True
            )
            task = await self._task_manager.finish_task(
                task.id,
                status=TaskStatus.FAILED,
                error=str(e),
            )

        # ── 手动后台任务完成后：唤醒目标会话 ──
        if task and task.status in (TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.TIMEOUT) \
           and task.type == "manual":
            status_text = {
                TaskStatus.SUCCESS: "已完成",
                TaskStatus.FAILED: "执行失败",
                TaskStatus.TIMEOUT: "执行超时",
            }.get(task.status, "已完成")
            target = task.delivery_channel or f"task:{task.id}"
            if self._wake_manager:
                await self._wake_manager.notify(
                    session_key=target,
                    text=f"后台任务{status_text}",
                    context_key=f"task:{task.id}",
                    source="task",
                    trigger_ai=bool(task.delivery_channel),
                )
            elif self._system_events:
                self._system_events.enqueue(
                    session_key=target,
                    text=f"后台任务{status_text}",
                    context_key=f"task:{task.id}",
                )

        return task

    @staticmethod
    def _resolve_session_id(job: CronJob, task_id: str) -> str:
        """根据 job 的 session_mode 确定执行 chat_id。"""
        mode = job.session_mode
        if mode == "current":
            mode = SessionMode.ISOLATED.value  # backward compat
            _log.info(f"兼容旧 job: session_mode=current -> isolated (job={job.name})")
        if mode == SessionMode.CUSTOM.value:
            cid = job.custom_session_id
            return f"cron:{cid}" if cid else f"task:{task_id}"
        elif mode == SessionMode.MAIN.value:
            return "cron:main"
        else:  # isolated（默认）
            return f"task:{task_id}"

    @staticmethod
    def _check_command_safe(command: str) -> Optional[str]:
        """检查 shell 命令是否安全。返回 None 表示通过，否则返回拒绝原因。"""
        try:
            parts = shlex.split(command)
        except ValueError:
            return "命令格式无效（引号不匹配等）"
        if not parts:
            return "命令为空"
        cmd_name = os.path.basename(parts[0])
        if cmd_name in _DENIED_COMMANDS:
            return f"命令 '{cmd_name}' 被禁止执行"
        return None

    async def _execute_command_payload(
        self,
        job: CronJob,
        task: TaskRecord,
        timeout: float,
    ) -> TaskRecord:
        """执行 command 载荷（shell 命令），捕获 stdout/stderr。"""
        if not job.command:
            return await self._task_manager.finish_task(
                task.id, TaskStatus.FAILED, error="command 为空",
            )

        # 安全检查
        reason = self._check_command_safe(job.command)
        if reason:
            _log.warning(f"命令被拒绝 [{job.name}]: {reason}")
            return await self._task_manager.finish_task(
                task.id, TaskStatus.FAILED, error=f"命令被拒绝: {reason}",
            )

        await self._task_manager.start_task(task.id)
        effective_timeout = min(timeout, 120.0)
        _log.info(f"执行命令 [{job.name}]: {job.command[:100]}")

        try:
            proc = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    shlex.split(job.command),
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout,
                ),
                timeout=effective_timeout + 5,
            )
            stdout = (proc.stdout or "")[-10000:]
            stderr = (proc.stderr or "")[-5000:]

            if proc.returncode == 0:
                result = stdout if stdout else "命令执行成功（无输出）"
                return await self._task_manager.finish_task(
                    task.id, TaskStatus.SUCCESS, result=result,
                )
            else:
                error_msg = f"退出码 {proc.returncode}"
                if stderr:
                    error_msg += f"\nstderr: {stderr}"
                if stdout:
                    error_msg += f"\nstdout: {stdout}"
                return await self._task_manager.finish_task(
                    task.id, TaskStatus.FAILED, error=error_msg,
                )
        except asyncio.TimeoutError:
            return await self._task_manager.finish_task(
                task.id, TaskStatus.TIMEOUT, error=f"命令超时 ({effective_timeout}s)",
            )
        except Exception as e:
            return await self._task_manager.finish_task(
                task.id, TaskStatus.FAILED, error=f"命令执行异常: {e}",
            )

    @staticmethod
    def _resolve_event_target(job: CronJob, task_id: str) -> str:
        """确定系统事件投递的目标 session。

        delivery_channel 优先（由 /cron create 保证设置），
        降级到该 cron 的 AI 执行 session。
        """
        return job.delivery_channel or BackgroundTaskRunner._resolve_session_id(job, task_id)

    async def _enqueue_events(
        self,
        job: CronJob,
        task_id: str,
        text: str,
        context_key: str,
        *,
        replace: bool = False,
    ) -> None:
        """向目标 session 入队系统事件。如果 session_mode=main 同时投递心跳。

        Args:
            job: cron job 定义
            task_id: 当前任务 ID，用于降级路由
            text: 事件文本
            context_key: 去重/replace 键
            replace: replace=True 时同 context_key 覆盖旧事件
        """
        if not self._system_events:
            return
        target = self._resolve_event_target(job, task_id)
        self._system_events.enqueue(
            session_key=target,
            text=text,
            context_key=context_key,
            replace=replace,
        )
        _log.info("系统事件入队 [%s] → %s: %s", job.name, target[:24], text[:60])
        # main 模式额外投递心跳 session，让 HeartbeatManager 也能消费
        if job.session_mode == "main":
            self._system_events.enqueue(
                session_key="heartbeat:events",
                text=text,
                context_key=context_key,
                replace=replace,
            )
            _log.debug("系统事件额外入队 → heartbeat:events [%s]: %s", job.name, text[:60])

    async def _execute_system_event_payload(
        self,
        job: CronJob,
        task: TaskRecord,
    ) -> TaskRecord:
        """执行 system_event 载荷：推入对应 session 的系统事件队列。"""
        await self._task_manager.start_task(task.id)
        text = job.prompt or f"定时任务 [{job.name}] 触发"
        if self._wake_manager and job.delivery_channel:
            await self._wake_manager.notify(
                session_key=job.delivery_channel,
                text=text,
                context_key=f"cron:{job.id}",
                source="cron",
                trigger_ai=True,
                replace=True,
            )
        elif self._system_events:
            await self._enqueue_events(
                job, task.id, text,
                context_key=f"cron:{job.id}",
                replace=True,
            )
        return await self._task_manager.finish_task(
            task.id, TaskStatus.SUCCESS,
            result=f"[系统事件] {text}",
        )

    async def run_cron_job(
        self,
        job: CronJob,
        timeout: float = 300.0,
    ) -> Optional[TaskRecord]:
        """执行一个 CronJob（创建 → 执行 → 可选投递）。

        根据 job.payload_type 决定执行方式：
        - message: AI 智能体轮次
        - command: shell 命令
        - system_event: 系统事件通知
        """
        if self._task_manager is None:
            _log.error("TaskManager 未注入")
            return None
        task = await self._task_manager.create_task(
            prompt=job.prompt or job.command or job.name,
            task_type="cron",
            job_id=job.id,
            delivery_channel=job.delivery_channel,
        )

        # 根据 session_mode 覆盖 session_id
        task.session_id = self._resolve_session_id(job, task.id)
        await self._task_manager.update_task_record(task)
        _log.info(
            f"CronJob [{job.name}] payload={job.payload_type} "
            f"session={job.session_mode} session_id={task.session_id}"
        )

        # 按载荷类型分支执行
        if job.payload_type == "command":
            task = await self._execute_command_payload(job, task, timeout)
        elif job.payload_type == "system_event":
            task = await self._execute_system_event_payload(job, task)
        else:  # message（默认）
            task = await self.run_task(
                task, timeout=timeout,
                is_group=job.is_group,
                tools_allow=job.tools_allow,
            )

        # 投递结果
        if job.enable_notify and job.delivery_channel and task and self._delivery_cb:
            content = task.result or task.error or ""
            if content:
                prefix = {
                    "command": "🖥️",
                    "system_event": "🔔",
                    "message": "📋",
                }.get(job.payload_type, "📋")
                try:
                    await self._delivery_cb(
                        chat_id=job.delivery_channel,
                        content=f"{prefix} 定时任务 [{job.name}] 执行{'成功' if task.status == TaskStatus.SUCCESS else '失败'}：\n{content}",
                        message_id="",
                        is_group=job.is_group,
                    )
                except Exception as e:
                    _log.error(f"投递任务结果失败: {e}")

        # 系统事件：message/command 任务完成后通知目标 session
        # system_event 类型由 _execute_system_event_payload 自行入队，不重复
        if task and task.status in (TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.TIMEOUT) \
           and job.payload_type != "system_event":
            status_text = {
                TaskStatus.SUCCESS: "已完成",
                TaskStatus.FAILED: "执行失败",
                TaskStatus.TIMEOUT: "执行超时",
            }.get(task.status, "已完成")
            if self._wake_manager and job.delivery_channel:
                await self._wake_manager.notify(
                    session_key=job.delivery_channel,
                    text=f"任务 '{job.name}'{status_text}",
                    context_key=f"task:{task.id}",
                    source="cron",
                    trigger_ai=True,
                )
            elif self._system_events:
                await self._enqueue_events(
                    job, task.id,
                    text=f"任务 '{job.name}'{status_text}",
                    context_key=f"task:{task.id}",
                )

        return task


