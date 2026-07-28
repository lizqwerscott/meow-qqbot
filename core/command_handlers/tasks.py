"""后台任务管理命令 — /tasks"""

import logging
import time
from typing import Any, Dict, List, Optional

from core.command_handlers.base import command, make_reply
from core.message import InputMessage

_log = logging.getLogger(__name__)


def _tasks_summary(tasks: list) -> str:
    """格式化任务列表为文本"""
    if not tasks:
        return "暂无任务记录。"
    lines = ["**后台任务列表**\n"]
    for t in tasks[:20]:
        status_icon = {
            "pending": "⏳",
            "running": "🔄",
            "success": "✅",
            "failed": "❌",
            "cancelled": "🚫",
            "timeout": "⏰",
            "lost": "💤",
        }.get(t.status.value, "❓")
        time_str = time.strftime(
            "%m-%d %H:%M", time.localtime(t.created_at)
        )
        lines.append(
            f"{status_icon} `{t.id[:10]}..` {time_str} "
            f"[{t.type}] {t.prompt[:40]}..."
        )
        if t.status.value == "failed" and t.error:
            lines.append(f"   └─ 错误: {t.error[:60]}")
        if t.status.value == "success" and t.result:
            lines.append(f"   └─ {t.result[:60]}...")
    return "\n".join(lines)


@command(name="tasks", aliases=["task", "tasklist"], permission="admin", description="管理后台任务")
class TasksCommand:
    def __init__(self, task_manager=None, background_task_runner=None, agent_engine=None):
        self._task_manager = task_manager
        self._runner = background_task_runner

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        if self._task_manager is None:
            return make_reply(input_message, "任务系统未就绪。")

        parts = args.strip().split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "list"

        if subcmd == "list":
            status_filter = None
            if len(parts) > 1:
                status_filter = parts[1].lower()
            tasks = self._task_manager.list_tasks(limit=30, status=status_filter)
            text = _tasks_summary(tasks)
            return make_reply(input_message, text)

        elif subcmd == "show":
            task_id = parts[1] if len(parts) > 1 else ""
            if not task_id:
                return make_reply(input_message, "请指定任务 ID。用法: /tasks show <id>")
            # 支持短 ID 搜索
            tasks = self._task_manager.list_tasks(limit=50)
            matched = [t for t in tasks if t.id.startswith(task_id)]
            if not matched:
                return make_reply(input_message, f"未找到任务: {task_id}")
            t = matched[0]
            lines = [
                f"**任务详情** `{t.id}`",
                f"- 类型: `{t.type}`",
                f"- 状态: `{t.status.value}`",
                f"- 创建: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t.created_at))}",
                f"- 指令: {t.prompt}",
            ]
            if t.started_at:
                lines.append(f"- 开始: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t.started_at))}")
            if t.finished_at:
                lines.append(f"- 结束: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t.finished_at))}")
            if t.job_id:
                lines.append(f"- 关联定时任务: `{t.job_id[:12]}..`")
            if t.result:
                lines.append(f"- 结果:\n```\n{t.result[:500]}\n```")
            if t.error:
                lines.append(f"- 错误: `{t.error}`")
            return make_reply(input_message, "\n".join(lines))

        elif subcmd == "cancel":
            task_id = parts[1] if len(parts) > 1 else ""
            if not task_id:
                return make_reply(input_message, "请指定任务 ID。用法: /tasks cancel <id>")
            success = await self._task_manager.cancel_task(task_id)
            if success:
                return make_reply(input_message, f"✅ 任务 {task_id[:12]}.. 已取消。")
            return make_reply(input_message, f"❌ 无法取消任务 {task_id[:12]}..，可能已完成或不存在。")

        else:
            return make_reply(input_message, "未知子命令。可用: list, show <id>, cancel <id>")
