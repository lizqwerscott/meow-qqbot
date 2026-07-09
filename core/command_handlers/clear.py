import logging
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.context_manager import ChatContextManager
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(name="清空", aliases=["clear"], description="清空当前对话历史")
class ClearCommand:
    def __init__(self, context_manager: ChatContextManager):
        self.context_manager = context_manager

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        try:
            self.context_manager.clear_chat_history(input_message.chat_id)
            return make_reply(input_message, "对话历史已清空。")
        except Exception as e:
            _log.error(f"清空命令处理失败: {e}")
            return []
