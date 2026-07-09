import logging
import time
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.context_manager import ChatContextManager
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(name="聊天列表", aliases=["list", "chats"], permission="admin", description="查看所有聊天ID列表（管理员专用）")
class ListChatsCommand:
    def __init__(self, context_manager: ChatContextManager):
        self.context_manager = context_manager

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        try:
            active_chats = self.context_manager.get_all_chats()
            if not active_chats:
                return make_reply(input_message, "当前没有活跃的聊天。")

            lines = ["活跃聊天列表："]
            for i, (chat_id_item, chat_context) in enumerate(active_chats.items(), 1):
                history_count = len(chat_context.history)
                last_activity = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(chat_context.last_activity)
                )
                lines.append(f"{i}. 聊天ID: {chat_id_item}")
                lines.append(f"   历史记录: {history_count} 条")
                lines.append(f"   最后活动: {last_activity}")
                lines.append("")

            reply = "\n".join(lines)
            if len(reply) > 1500:
                reply = reply[:1500] + "...\n(列表过长，已截断)"

            return make_reply(input_message, reply)
        except Exception as e:
            _log.error(f"聊天列表命令处理失败: {e}")
            return []
