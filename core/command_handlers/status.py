import logging
import time
from typing import Any, Dict, List

import psutil

from core.agent_engine import AgentEngine
from core.command_handlers.base import command, make_reply
from core.message import InputMessage

_log = logging.getLogger(__name__)


def _everos_status_line(health: dict) -> str:
    status = health.get("status", "unknown")
    if status == "disabled":
        return "未启用 🚫"
    if status == "ok":
        latency = health.get("latency_ms")
        if latency is not None:
            return f"已连接 ✅ ({latency}ms)"
        return "已连接 ✅"
    if status == "unknown":
        return "待检查 ⏳"
    error = health.get("error", "未知错误")
    return f"不可达 ❌ ({error})"


@command(name="状态", aliases=["status"], permission="admin", description="查看系统状态（管理员专用）")
class StatusCommand:
    def __init__(self, agent_engine: AgentEngine):
        self.agent_engine = agent_engine

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage("/")
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 ** 2)
            process_cpu = process.cpu_percent(interval=0.1)

            stats = self.agent_engine.get_stats()
            queue_sizes = stats.get("queue_sizes", {})
            total_queue = sum(queue_sizes.values())
            active_chats = stats.get("active_chats", 0)
            everos_health = stats.get("everos_health", {})

            status_text = [
                "=== 系统状态 ===",
                f"系统时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "=== 系统资源 ===",
                f"CPU使用率: {cpu_percent:.1f}%",
                f"内存使用: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)",
                f"磁盘使用: {disk.percent:.1f}% ({disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB)",
                "",
                "=== 进程状态 ===",
                f"进程内存: {process_memory:.1f}MB",
                f"进程CPU: {process_cpu:.1f}%",
                "",
                "=== 机器人状态 ===",
                f"消息队列: {total_queue} 条（{len(queue_sizes)} 个活跃会话）",
                f"活跃聊天: {active_chats} 个",
                "",
                "=== 记忆系统 ===",
                f"EverOS: {_everos_status_line(everos_health)}",
            ]
            return make_reply(input_message, "\n".join(status_text))
        except ImportError:
            return make_reply(input_message, "无法获取系统状态信息，请安装psutil库。")
        except Exception as e:
            _log.error(f"状态命令处理失败: {e}")
            return []
