"""Cron 定时任务管理命令 — /cron"""

import logging
import shlex
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
                if j.is_one_shot:
                    schedule_str = f"🕐 {time.strftime('%m-%d %H:%M', time.localtime(j.at))}"
                else:
                    schedule_str = f"`{j.cron_expression}`"
                # 载荷标识
                payload_icons = {"command": "🖥️", "system_event": "🔔", "message": ""}
                payload_tag = f" {payload_icons.get(j.session_mode, '')}"
                session_tag = ""
                if j.session_mode != "isolated":
                    if j.session_mode == "custom":
                        session_tag = f" [session:cron:{j.custom_session_id}]"
                    else:
                        session_tag = f" [session:{j.session_mode}]"
                if j.payload_type != "message":
                    session_tag += f" [{j.payload_type}]"
                lines.append(
                    f"{enabled_icon} `{j.id[:10]}..` **{j.name}** "
                    f"{schedule_str}{next_str}{session_tag}"
                )
                lines.append(f"   └─ {j.prompt[:60]}...")
            return make_reply(input_message, "\n".join(lines))

        elif subcmd == "create":
            rest = parts[1] if len(parts) > 1 else ""
            if not rest:
                return make_reply(input_message,
                    "用法:\n"
                    "  /cron create <name> <cron_expr> <prompt>                   # 周期性 AI 消息\n"
                    "  /cron create <name> at:<ISO8601> <prompt>                  # 一次性 AI 消息\n"
                    "  /cron create <name> command:<shell> <prompt>               # 定时 shell 命令\n"
                    "  /cron create <name> event:<text>                           # 定时系统事件\n"
                    "  /cron create <name> <arg> session:<mode> <prompt>          # 指定 session\n"
                    "例: /cron create 早安 \"0 8 * * *\" 说早安\n"
                    "例: /cron create 提醒 at:2027-01-01T08:00:00Z 新年快乐\n"
                    "例: /cron create 备份 \"0 3 * * *\" command:scripts/backup.sh 执行备份\n"
                    "例: /cron create 健康检查 \"*/5 * * * *\" event:检查服务状态\n"
                    "session 模式: isolated(默认) / current / custom:<id> / main\n"
                    "载荷类型: message(默认) / command (command:) / system_event (event:)"
                )
            try:
                tokens = shlex.split(rest)
            except ValueError:
                return make_reply(input_message, "参数解析失败：请检查引号是否配对")
            if len(tokens) < 3:
                return make_reply(input_message, "用法见 /cron create 的提示")
            name, second, *rest_tokens = tokens
            prompt = " ".join(rest_tokens)

            # 解析 session 模式（如果 prompt 以 session: 开头）
            session_mode = "isolated"
            custom_session_id = None
            if prompt.startswith("session:"):
                session_part = prompt[len("session:"):].split(maxsplit=1)
                if len(session_part) == 2:
                    raw_mode, prompt = session_part
                else:
                    raw_mode = session_part[0]
                    prompt = ""
                if raw_mode in ("isolated", "current", "main"):
                    session_mode = raw_mode
                elif raw_mode.startswith("custom:"):
                    session_mode = "custom"
                    custom_session_id = raw_mode[len("custom:"):]
                else:
                    session_mode = raw_mode  # 原样传，由 manager 校验

            if second.startswith("at:"):
                at_str = second[3:]
                import time as _time
                from datetime import datetime as _dt
                try:
                    at_ts = _dt.fromisoformat(at_str.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    return make_reply(input_message, f"时间格式无效: {at_str}")
                job = self._cron_mgr.create_job(
                    name=name, cron_expression="", prompt=prompt,
                    at=at_ts, delivery_channel=input_message.chat_id,
                    session_mode=session_mode, custom_session_id=custom_session_id,
                )
                base = f"🕐 一次性提醒已创建: `{job.id[:12]}..` **{name}** (将在 {_time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(at_ts))} 执行)"
                return make_reply(input_message, base + (f" session={session_mode}" if session_mode != "isolated" else ""))

            elif second.startswith("command:"):
                payload_cmd = second[len("command:"):]
                job = self._cron_mgr.create_job(
                    name=name, cron_expression="", prompt=prompt,
                    delivery_channel=input_message.chat_id,
                    session_mode=session_mode, custom_session_id=custom_session_id,
                    payload_type="command", command=payload_cmd,
                )
                base = f"🖥️ 定时命令已创建: `{job.id[:12]}..` **{name}**\n命令: `{payload_cmd[:80]}`"
                return make_reply(input_message, base + (f"\nsession={session_mode}" if session_mode != "isolated" else ""))

            elif second.startswith("event:"):
                event_text = prompt if prompt else second[len("event:"):]
                job = self._cron_mgr.create_job(
                    name=name, cron_expression="", prompt=event_text,
                    delivery_channel=input_message.chat_id,
                    session_mode=session_mode, custom_session_id=custom_session_id,
                    payload_type="system_event",
                )
                base = f"🔔 定时系统事件已创建: `{job.id[:12]}..` **{name}**\n事件: {event_text[:80]}"
                return make_reply(input_message, base + (f"\nsession={session_mode}" if session_mode != "isolated" else ""))

            else:
                # 周期性 AI 消息模式（默认）
                job = self._cron_mgr.create_job(
                    name=name, cron_expression=second, prompt=prompt,
                    delivery_channel=input_message.chat_id,
                    session_mode=session_mode, custom_session_id=custom_session_id,
                )
                base = f"✅ 定时任务已创建: `{job.id[:12]}..` **{name}** (`{second}`)"
                return make_reply(input_message, base + (f" session={session_mode}" if session_mode != "isolated" else ""))

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
                f"**任务详情** `{job.id}`",
                f"- 名称: **{job.name}**",
            ]
            if job.is_one_shot:
                lines.append(f"- 类型: 🕐 一次性提醒")
                if job.at:
                    lines.append(f"- 执行时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.at))}")
            else:
                lines.append(f"- 类型: 🔄 周期性任务")
                lines.append(f"- 表达式: `{job.cron_expression}`")
                lines.append(f"- 补跑: {'是' if job.catch_up else '否'}")
            # 载荷信息
            payload_labels = {"message": "🤖 AI 消息", "command": "🖥️ Shell 命令", "system_event": "🔔 系统事件"}
            lines.append(f"- 载荷: {payload_labels.get(job.payload_type, job.payload_type)}")
            if job.payload_type == "command" and job.command:
                lines.append(f"- 命令: `{job.command[:100]}`")
            if job.model:
                lines.append(f"- 模型: `{job.model}`")
            if job.thinking:
                lines.append(f"- 思考: `{job.thinking}`")

            # session 信息
            if job.session_mode == "current":
                lines.append(f"- Session: 当前会话 (current)")
            elif job.session_mode == "custom":
                lines.append(f"- Session: 命名会话 cron:{job.custom_session_id} (custom)")
            elif job.session_mode == "main":
                lines.append(f"- Session: 系统通道 cron:main (main)")
            else:
                lines.append(f"- Session: 隔离执行 (isolated)")

            lines.extend([
                f"- 状态: {'✅ 已启用' if job.enabled else '⛔ 已暂停'}",
                f"- 指令: {job.prompt}",
            ])
            if job.next_run_at:
                lines.append(f"- 下次执行: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.next_run_at))}")
            if job.delivery_channel:
                lines.append(f"- 投递频道: `{job.delivery_channel[:12]}..`")
            return make_reply(input_message, "\n".join(lines))

        else:
            return make_reply(input_message, "未知子命令。可用: list, create, delete, pause, resume, run, show")
