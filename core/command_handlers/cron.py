"""Cron 定时任务管理命令 — /cron"""

import logging
import shlex
import time
from typing import Any, Dict, List, Optional

from core.command_handlers.base import command, make_reply
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(
    name="cron",
    aliases=["cronjob", "定时"],
    permission="admin",
    description="管理定时任务",
)
class CronCommand:
    def __init__(
        self,
        cron_job_manager=None,
        background_task_runner=None,
        task_manager=None,
        agent_engine=None,
    ):
        self._cron_mgr = cron_job_manager
        self._runner = background_task_runner
        self._task_mgr = task_manager

    @staticmethod
    def _parse_flags(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
        """从 token 列表中提取 --flag value 对。

        Returns:
            (flags_dict, remaining_tokens)
        """
        flags = {}
        remaining = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.startswith("--") and t[2:] in (
                "command",
                "event",
                "session",
                "target",
                "model",
                "thinking",
                "notify",
            ):
                flag_name = t[2:]
                if i + 1 < len(tokens):
                    flags[flag_name] = tokens[i + 1]
                    i += 2
                    continue
            remaining.append(t)
            i += 1
        return flags, remaining

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        if self._cron_mgr is None:
            return make_reply(input_message, "定时任务系统未就绪。")

        parts = args.strip().split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "list"

        if subcmd == "list":
            jobs = self._cron_mgr.list_jobs()
            if not jobs:
                return make_reply(
                    input_message,
                    "暂无定时任务。\n用法: /cron create <name> <cron> <prompt>",
                )
            lines = ["**定时任务列表**\n"]
            for j in jobs:
                enabled_icon = "✅" if j.enabled else "⛔"
                next_str = ""
                if j.next_run_at and j.enabled:
                    next_str = f" 下次: {time.strftime('%m-%d %H:%M', time.localtime(j.next_run_at))}"
                if j.is_one_shot:
                    schedule_str = (
                        f"🕐 {time.strftime('%m-%d %H:%M', time.localtime(j.at))}"
                    )
                else:
                    schedule_str = f"`{j.cron_expression}`"
                payload_icons = {"command": "🖥️ ", "system_event": "🔔 ", "message": ""}
                payload_tag = payload_icons.get(j.payload_type, "")
                session_tag = ""
                if j.session_mode != "isolated":
                    if j.session_mode == "custom":
                        session_tag = f" [session:cron:{j.custom_session_id}]"
                    else:
                        session_tag = f" [session:{j.session_mode}]"
                type_tag = ""
                if j.payload_type != "message":
                    type_tag = f" [{j.payload_type}]"
                lines.append(
                    f"{enabled_icon} `{j.id[:10]}..` **{j.name}** "
                    f"{payload_tag}{schedule_str}{next_str}{type_tag}{session_tag}"
                )
                if j.payload_type == "command" and j.command:
                    lines.append(f"   └─ {j.command[:60]}...")
                else:
                    lines.append(f"   └─ {j.prompt[:60]}...")
            return make_reply(input_message, "\n".join(lines))

        elif subcmd == "create":
            rest = parts[1] if len(parts) > 1 else ""
            if not rest:
                return make_reply(
                    input_message,
                    "用法:\n"
                    "  /cron create <name> <cron_expr> <prompt>                         # AI 消息（默认）\n"
                    "  /cron create <name> at:<ISO8601> <prompt>                        # 一次性 AI 消息\n"
                    "  /cron create <name> <cron_expr> --command <shell>                # 定时 shell 命令\n"
                    "  /cron create <name> <cron_expr> --event <text>                   # 定时系统事件\n"
                    "  /cron create <name> at:<ISO8601> --command <shell>               # 一次性命令\n"
                    "  /cron create <name> at:<ISO8601> --event <text>                  # 一次性事件\n"
                    "  /cron create <name> <cron_expr> <prompt> --session <mode>        # 指定 session\n"
                    "  /cron create <name> <cron_expr> <prompt> --target <mode>         # 指定 wake 目标\n"
                    "  /cron create <name> <cron_expr> <prompt> --model <m> --thinking <t>  # AI 参数\n"
                    "  /cron create <name> <cron_expr> <prompt> --notify off              # 静默执行不投递\n\n"
                    "兼容旧语法:\n"
                    "  /cron create <name> command:<shell> <prompt>                     # (无 cron 表达式)\n"
                    "  /cron create <name> event:<text>                                 # (无 cron 表达式)\n"
                    "  /cron create <name> <arg> session:<mode> <prompt>                # (旧 session 语法)\n\n"
                    '例: /cron create 早安 "0 8 * * *" 说早安\n'
                    '例: /cron create 备份 "0 3 * * *" --command scripts/backup.sh\n'
                    "例: /cron create 提醒 at:2027-01-01T08:00:00Z 新年快乐\n"
                    '例: /cron create 健康检查 "*/5 * * * *" --event 检查服务状态\n'
                    "session 模式: isolated(默认) / custom:<id> / main\n"
                    "target 模式: isolated(默认) / main（完成后 wake 心跳系统，不打扰用户）\n"
                    "载荷类型: message(默认) / command(--command) / system_event(--event)",
                )
            try:
                tokens = shlex.split(rest)
            except ValueError:
                return make_reply(input_message, "参数解析失败：请检查引号是否配对")
            if len(tokens) < 2:
                return make_reply(input_message, "用法见 /cron create 的提示")

            name, second, *rest_tokens = tokens
            flags, prompt_parts = self._parse_flags(rest_tokens)
            prompt = " ".join(prompt_parts)

            # 解析 session 模式（--session 标志优先，向后兼容 prompt 中 session: 前缀）
            session_mode = "isolated"
            custom_session_id = None
            if "session" in flags:
                raw = flags["session"]
                if raw in ("isolated", "main"):
                    session_mode = raw
                elif raw.startswith("custom:"):
                    session_mode = "custom"
                    custom_session_id = raw[len("custom:") :]
                else:
                    session_mode = raw
            elif prompt.startswith("session:"):
                session_part = prompt[len("session:") :].split(maxsplit=1)
                if len(session_part) == 2:
                    raw_mode, prompt = session_part
                else:
                    raw_mode = session_part[0]
                    prompt = ""
                if raw_mode in ("isolated", "main"):
                    session_mode = raw_mode
                elif raw_mode.startswith("custom:"):
                    session_mode = "custom"
                    custom_session_id = raw_mode[len("custom:") :]
                else:
                    session_mode = raw_mode

            # 解析 session_target（--target 标志）
            session_target = "isolated"
            if "target" in flags:
                raw = flags["target"]
                if raw in ("isolated", "main"):
                    session_target = raw
                else:
                    return make_reply(
                        input_message,
                        f"无效的 target: '{raw}'。允许的值: isolated, main",
                    )

            # 解析调度方式和载荷类型
            cron_expr = ""
            at_ts = None
            if second.startswith("at:"):
                at_str = second[3:]
                import time as _time
                from datetime import datetime as _dt

                try:
                    at_ts = _dt.fromisoformat(at_str.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    return make_reply(input_message, f"时间格式无效: {at_str}")
            elif second.startswith("command:"):
                if "command" not in flags:
                    flags["command"] = second[len("command:") :]
            elif second.startswith("event:"):
                if "event" not in flags:
                    flags["event"] = second[len("event:") :]
            else:
                cron_expr = second

            # 确定载荷类型
            if "command" in flags:
                payload_type = "command"
            elif "event" in flags:
                payload_type = "system_event"
            else:
                payload_type = "message"

            # 构建创建参数
            kwargs = {
                "name": name,
                "cron_expression": cron_expr,
                "prompt": prompt,
                "delivery_channel": input_message.chat_id,
                "is_group": input_message.is_group,
                "session_mode": session_mode,
                "session_target": session_target,
                "custom_session_id": custom_session_id,
            }
            if at_ts is not None:
                kwargs["at"] = at_ts
            if payload_type != "message":
                kwargs["payload_type"] = payload_type
            if payload_type == "command":
                kwargs["command"] = flags.get("command", "")
            if "model" in flags:
                kwargs["model"] = flags["model"]
            if "thinking" in flags:
                kwargs["thinking"] = flags["thinking"]
            if "notify" in flags:
                notify_val = flags["notify"].lower()
                if notify_val in ("off", "false", "no", "0"):
                    kwargs["enable_notify"] = False
                elif notify_val in ("on", "true", "yes", "1"):
                    kwargs["enable_notify"] = True

            job = await self._cron_mgr.create_job(**kwargs)

            # 构建回复消息
            if at_ts is not None:
                time_str = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(at_ts))
                if payload_type == "command":
                    base = f"🖥️ 一次性命令已创建: `{job.id[:12]}..` **{name}** (将在 {time_str} 执行)\n命令: `{kwargs['command'][:80]}`"
                elif payload_type == "system_event":
                    base = f"🔔 一次性事件已创建: `{job.id[:12]}..` **{name}** (将在 {time_str} 执行)\n事件: {prompt[:80]}"
                else:
                    base = f"🕐 一次性提醒已创建: `{job.id[:12]}..` **{name}** (将在 {time_str} 执行)"
            elif payload_type == "command":
                base = f"🖥️ 定时命令已创建: `{job.id[:12]}..` **{name}**\n命令: `{kwargs['command'][:80]}`"
            elif payload_type == "system_event":
                event_text = flags.get("event", prompt)
                base = f"🔔 定时系统事件已创建: `{job.id[:12]}..` **{name}**\n事件: {event_text[:80]}"
            else:
                base = (
                    f"✅ 定时任务已创建: `{job.id[:12]}..` **{name}** (`{cron_expr}`)"
                )

            if session_mode != "isolated":
                base += f"\nsession={session_mode}"
            if custom_session_id:
                base += f" ({custom_session_id})"
            if session_target != "isolated":
                base += f"\ntarget={session_target}"
            if "model" in flags:
                base += f" model={flags['model']}"
            if "thinking" in flags:
                base += f" thinking={flags['thinking']}"
            return make_reply(input_message, base)

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
            await self._cron_mgr.delete_job(job.id)
            return make_reply(input_message, f"🗑️ 定时任务已删除: `{mid}..` **{name}**")

        elif subcmd == "pause":
            target = parts[1] if len(parts) > 1 else ""
            if not target:
                return make_reply(input_message, "用法: /cron pause <id>")
            success = await self._cron_mgr.disable_job(target)
            if success:
                return make_reply(
                    input_message, f"⏸️ 定时任务 {target[:12]}.. 已暂停。"
                )
            return make_reply(input_message, f"未找到定时任务: {target}")

        elif subcmd == "resume":
            target = parts[1] if len(parts) > 1 else ""
            if not target:
                return make_reply(input_message, "用法: /cron resume <id>")
            success = await self._cron_mgr.enable_job(target)
            if success:
                return make_reply(
                    input_message, f"▶️ 定时任务 {target[:12]}.. 已恢复。"
                )
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
            status_line = (
                f"✅ 完成"
                if task.status.value == "success"
                else f"❌ {task.status.value}"
            )
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
                    lines.append(
                        f"- 执行时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.at))}"
                    )
            else:
                lines.append(f"- 类型: 🔄 周期性任务")
                lines.append(f"- 表达式: `{job.cron_expression}`")
                lines.append(f"- 补跑: {'是' if job.catch_up else '否'}")
            # 载荷信息
            payload_labels = {
                "message": "🤖 AI 消息",
                "command": "🖥️ Shell 命令",
                "system_event": "🔔 系统事件",
            }
            lines.append(
                f"- 载荷: {payload_labels.get(job.payload_type, job.payload_type)}"
            )
            if job.payload_type == "command" and job.command:
                lines.append(f"- 命令: `{job.command[:100]}`")
            if job.model:
                lines.append(f"- 模型: `{job.model}`")
            if job.thinking:
                lines.append(f"- 思考: `{job.thinking}`")

            # session 信息
            if job.session_mode == "custom":
                lines.append(
                    f"- Session: 命名会话 cron:{job.custom_session_id} (custom)"
                )
            elif job.session_mode == "main":
                lines.append(f"- Session: 系统通道 cron:main (main)")
            else:
                lines.append(f"- Session: 隔离执行 (isolated)")
            session_target = job.session_target or job.session_mode
            if session_target != job.session_mode:
                lines.append(f"- Target: {session_target}（完成后 wake 目标）")

            lines.extend(
                [
                    f"- 状态: {'✅ 已启用' if job.enabled else '⛔ 已暂停'}",
                    f"- 指令: {job.prompt}",
                ]
            )
            if job.next_run_at:
                lines.append(
                    f"- 下次执行: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.next_run_at))}"
                )
            if job.delivery_channel:
                lines.append(f"- 投递频道: `{job.delivery_channel[:12]}..`")
            return make_reply(input_message, "\n".join(lines))

        else:
            return make_reply(
                input_message,
                "未知子命令。可用: list, create, delete, pause, resume, run, show",
            )
