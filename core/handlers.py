import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List

from botpy import logging

from core.client import MyClient
from core.message_queue import InputMessage

if TYPE_CHECKING:
    from core.client import MyClient

_log = logging.get_logger()


def handle_history_command(
    client: MyClient, input_message: InputMessage, args: str
) -> List[Dict[str, Any]]:
    """处理历史命令，显示最近的对话历史"""
    try:
        chat_id = input_message.chat_id

        # 获取历史记录
        history = client.context_manager.get_chat_history(chat_id)

        if not history:
            reply_content = "当前没有对话历史。"
        else:
            # 格式化历史记录
            history_text = []
            for i, msg in enumerate(history, 1):
                role = "用户" if msg.role == "user" else "助手"
                history_text.append(f"{i}. [{role}] {msg.content}")

            reply_content = "最近的对话历史：\n" + "\n".join(history_text)

            # 限制回复长度
            if len(reply_content) > 1000:
                reply_content = reply_content[:1000] + "...\n(历史记录过长，已截断)"

        # 返回消息列表
        return [
            {
                "chat_id": chat_id,
                "content": reply_content,
                "message_id": input_message.id,
                "is_group": input_message.is_group,
            }
        ]

    except Exception as e:
        _log.error(f"处理历史命令时出错: {e}")
        return []


def handle_clear_command(
    client: MyClient, input_message: InputMessage, args: str
) -> List[Dict[str, Any]]:
    """处理清空命令，清除当前对话历史"""
    try:
        chat_id = input_message.chat_id

        # 清空历史记录
        client.context_manager.clear_chat_history(chat_id)

        reply_content = "对话历史已清空。"

        # 返回消息列表
        return [
            {
                "chat_id": chat_id,
                "content": reply_content,
                "message_id": input_message.id,
                "is_group": input_message.is_group,
            }
        ]

    except Exception as e:
        _log.error(f"处理清空命令时出错: {e}")
        return []


def handle_help_command(
    client: MyClient, input_message: InputMessage, args: str
) -> List[Dict[str, Any]]:
    """处理帮助命令，显示可用命令列表"""
    try:
        chat_id = input_message.chat_id
        user_id = input_message.sender_id

        # 获取所有命令
        all_commands = client.command_manager.get_all_commands()

        # 过滤用户有权限的命令
        available_commands = []
        for cmd in all_commands:
            if client.command_manager.has_permission(cmd, user_id):
                available_commands.append(cmd)

        if not available_commands:
            reply_content = "没有可用的命令。"
        else:
            # 格式化帮助信息
            help_text = ["可用命令："]
            for cmd in available_commands:
                aliases_str = (
                    f"（别名: {', '.join(cmd.aliases)}）" if cmd.aliases else ""
                )
                help_text.append(f"• {cmd.name}{aliases_str}: {cmd.description}")

            reply_content = "\n".join(help_text)

        # 返回消息列表
        return [
            {
                "chat_id": chat_id,
                "content": reply_content,
                "message_id": input_message.id,
                "is_group": input_message.is_group,
            }
        ]

    except Exception as e:
        _log.error(f"处理帮助命令时出错: {e}")
        return []


def handle_status_command(
    client: MyClient, input_message: InputMessage, args: str
) -> List[Dict[str, Any]]:
    """处理状态命令，显示系统状态（管理员专用）"""
    try:
        chat_id = input_message.chat_id

        # 获取系统状态信息
        import time

        import psutil

        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / (1024**3)  # GB
        memory_total = memory.total / (1024**3)  # GB

        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # 磁盘使用情况
        disk = psutil.disk_usage("/")
        disk_percent = disk.percent
        disk_used = disk.used / (1024**3)  # GB
        disk_total = disk.total / (1024**3)  # GB

        # 进程信息
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024**2)  # MB
        process_cpu = process.cpu_percent(interval=0.1)

        # 消息队列状态
        input_queue_size = client.message_queue.input_queue.qsize()
        processed_queue_size = client.message_queue.processed_queue.qsize()

        # 上下文管理器状态
        active_chats = client.context_manager.get_context_count()

        # 格式化状态信息
        status_text = [
            "=== 系统状态 ===",
            f"系统时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "=== 系统资源 ===",
            f"CPU使用率: {cpu_percent:.1f}%",
            f"内存使用: {memory_percent:.1f}% ({memory_used:.1f}GB / {memory_total:.1f}GB)",
            f"磁盘使用: {disk_percent:.1f}% ({disk_used:.1f}GB / {disk_total:.1f}GB)",
            "",
            "=== 进程状态 ===",
            f"进程内存: {process_memory:.1f}MB",
            f"进程CPU: {process_cpu:.1f}%",
            "",
            "=== 机器人状态 ===",
            f"消息队列: 输入队列 {input_queue_size} 条，处理队列 {processed_queue_size} 条",
            f"活跃聊天: {active_chats} 个",
            f"管理员ID: {', '.join(client.admin_id) if client.admin_id else '未设置'}",
        ]

        reply_content = "\n".join(status_text)

        # 返回消息列表
        return [
            {
                "chat_id": chat_id,
                "content": reply_content,
                "message_id": input_message.id,
                "is_group": input_message.is_group,
            }
        ]

    except ImportError:
        reply_content = "无法获取系统状态信息，请安装psutil库。"
        return [
            {
                "chat_id": chat_id,
                "content": reply_content,
                "message_id": input_message.id,
                "is_group": input_message.is_group,
            }
        ]
    except Exception as e:
        _log.error(f"处理状态命令时出错: {e}")
        return []


def handle_list_chats_command(
    client: MyClient, input_message: InputMessage, args: str
) -> List[Dict[str, Any]]:
    """处理聊天列表命令，显示所有活跃聊天（管理员专用）"""
    try:
        chat_id = input_message.chat_id

        # 获取所有活跃聊天
        active_chats = client.context_manager.get_all_chats()

        if not active_chats:
            reply_content = "当前没有活跃的聊天。"
        else:
            # 格式化聊天列表
            chats_text = ["活跃聊天列表："]
            for i, (chat_id_item, chat_context) in enumerate(active_chats.items(), 1):
                history_count = len(chat_context.history)
                last_activity = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(chat_context.last_activity)
                )
                chats_text.append(f"{i}. 聊天ID: {chat_id_item}")
                chats_text.append(f"   历史记录: {history_count} 条")
                chats_text.append(f"   最后活动: {last_activity}")
                chats_text.append("")

            reply_content = "\n".join(chats_text)

            # 限制回复长度
            if len(reply_content) > 1500:
                reply_content = reply_content[:1500] + "...\n(列表过长，已截断)"

        # 返回消息列表
        return [
            {
                "chat_id": chat_id,
                "content": reply_content,
                "message_id": input_message.id,
                "is_group": input_message.is_group,
            }
        ]

    except Exception as e:
        _log.error(f"处理聊天列表命令时出错: {e}")
        return []
