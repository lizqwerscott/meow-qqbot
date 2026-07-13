"""Cron 定时任务管理命令 — /cron"""

import logging
import time
from typing import Any, Dict, List, Optional

from core.command_handlers.base import command, make_reply
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(name="cron", aliases=["cronjob", "定时"], permission="admin", description="管理定时任务")
class CronCommand:
    def __init__(self, cron_job_manager=None, background_task_runner=None, task_manager=None, agent_engine=None):
        self._cron_mgr = cron_job_manager
        self._runner = background_task_runner
        self._task_mgr = task_manager
        self._agent_engine = agent_engine

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        if self._cron_mgr is None:
            return make_reply(input_message, "定时任务系统未就绪。")

        parts = args.strip().split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "list"

        if subcmd == "list":
            jobs = self._cron_mgr.list_jobs()
            if not jobs:
                return make_reply(input_message, "暂无定时任务。\n用法: /cron create <name> <cron> <prompt>")
            lines = ["**定时任务列表**\n"]
            for j in jobs:
                enabled_icon = "✅" if j.enabled else "⛔"
                next_str = ""
                if j.next_run_at and j.enabled:
                    next_str = f" 下次: {time.strftime('%m-%d %H:%M', time.localtime(j.next_run_at))}"
                lines.append(
                    f"{enabled_icon} `{j.id[:10]}..` **{j.name}** "
                    f"`{j.cron_expression}`{next_str}"
                )
                lines.append(f"   └─ {j.prompt[:60]}...")
            return make_reply(input_message, "\n".join(lines))

        elif subcmd == "create":
            rest = parts[1] if len(parts) > 1 else ""
            if not rest:
                return make_reply(input_message, "用法: /cron create <name> <cron_expr> <prompt>")
            # 解析：name, cron_expr, prompt
            # 约定格式：name 是第一个词，cron_expr 是第二个词，剩余是 prompt
            tokens = rest.split(maxsplit=2)
            if len(tokens) < 3:
                return make_reply(input_message, "用法: /cron create <name> <cron_expr> <prompt>\n例: /cron create 早安 \"0 8 * * *\" 对大家说早上好")
            name, cron_expr, prompt = tokens
            job = self._cron_mgr.create_job(
                name=name,
                cron_expression=cron_expr,
                prompt=prompt,
                delivery_channel=input_message.chat_id,
            )
            return make_reply(input_message, f"✅ 定时任务已创建: `{job.id[:12]}..` **{name}** (`{cron_expr}`)")

        elif subcmd == "delete":
            target = parts[1] if len(parts) > 1 else ""
            if not target:
                return make_reply(input_message, "用法: /cron delete <id>")
            job = self._cron_mgr.get_job(target)
            if job is None:
                # 尝试模糊搜索
                matched = self._cron_mgr.find_jobs_by_name(target)
                if matched:
                    job = matched[0]
            if job is None:
                return make_reply(input_message, f"未找到定时任务: {target}")
            name = job.name
            mid = job.id[:12]
            self._cron_mgr.delete_job(job.id)
            return make_reply(input_message, f"🗑️ 定时任务已删除: `{mid}..` **{name}**")

        elif subcmd == "pause":
            target = parts[1] if len(parts) > 1 else ""
            if not target:
                return make_reply(input_message, "用法: /cron pause <id>")
            success = self._cron_mgr.disable_job(target)
            if success:
                return make_reply(input_message, f"⏸️ 定时任务 {target[:12]}.. 已暂停。")
            return make_reply(input_message, f"未找到定时任务: {target}")

        elif subcmd == "resume":
            target = parts[1] if len(parts) > 1 else ""
            if not target:
                return make_reply(input_message, "用法: /cron resume <id>")
            success = self._cron_mgr.enable_job(target)
            if success:
                return make_reply(input_message, f"▶️ 定时任务 {target[:12]}.. 已恢复。")
            return make_reply(input_message, f"未找到定时任务: {target}")

        elif subcmd == "run":
            target = parts[1] if len(parts) > 1 else ""
            if not target:
                return make_reply(input_message, "用法: /cron run <id>")
            job = self._cron_mgr.get_job(target)
            if job is None:
                matched = self._cron_mgr.find_jobs_by_name(target)
                if matched:
                    job = matched[0]
            if job is None:
                return make_reply(input_message, f"未找到定时任务: {target}")
            if self._runner is None or self._task_mgr is None:
                return make_reply(input_message, "任务执行器未就绪。")

            task = await self._runner.run_cron_job(
                job=job,
                timeout=300,
            )
            status_line = f"✅ 完成" if task.status.value == "success" else f"❌ {task.status.value}"
            lines = [
                f"**手动执行定时任务: {job.name}**",
                f"- 状态: {status_line}",
            ]
            if task.result:
                lines.append(f"- 结果: {task.result[:200]}")
            if task.error:
                lines.append(f"- 错误: `{task.error}`")
            return make_reply(input_message, "\n".join(lines))

        elif subcmd == "show":
            target = parts[1] if len(parts) > 1 else ""
            if not target:
                return make_reply(input_message, "用法: /cron show <id>")
            job = self._cron_mgr.get_job(target)
            if job is None:
                matched = self._cron_mgr.find_jobs_by_name(target)
                if matched:
                    job = matched[0]
            if job is None:
                return make_reply(input_message, f"未找到定时任务: {target}")
            lines = [
                f"**定时任务详情** `{job.id}`",
                f"- 名称: **{job.name}**",
                f"- 表达式: `{job.cron_expression}`",
                f"- 状态: {'✅ 已启用' if job.enabled else '⛔ 已暂停'}",
                f"- 补跑: {'是' if job.catch_up else '否'}",
                f"- 指令: {job.prompt}",
            ]
            if job.next_run_at:
                lines.append(f"- 下次执行: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.next_run_at))}")
            if job.delivery_channel:
                lines.append(f"- 投递频道: `{job.delivery_channel[:12]}..`")
            return make_reply(input_message, "\n".join(lines))

        else:
            return make_reply(input_message, "未知子命令。可用: list, create, delete, pause, resume, run, show")
