import logging
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.context_manager import ChatContextManager
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(name="历史", aliases=["history"], description="查看最近对话历史")
class HistoryCommand:
    def __init__(self, context_manager: ChatContextManager):
        self.context_manager = context_manager

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        try:
            history = self.context_manager.get_chat_history(input_message.chat_id)
            if not history:
                return make_reply(input_message, "当前没有对话历史。")

            lines = []
            for i, msg in enumerate(history, 1):
                role = "用户" if msg.role == "user" else "助手"
                lines.append(f"{i}. [{role}] {msg.content}")

            reply = "最近的对话历史：\n" + "\n".join(lines)
            if len(reply) > 1000:
                reply = reply[:1000] + "...\n(历史记录过长，已截断)"

            return make_reply(input_message, reply)
        except Exception as e:
            _log.error(f"历史命令处理失败: {e}")
            return []
