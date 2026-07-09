import logging
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.command_manager import CommandManager
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(name="帮助", aliases=["help"], description="显示命令帮助")
class HelpCommand:
    def __init__(self, command_manager: CommandManager):
        self.command_manager = command_manager

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        try:
            user_id = input_message.sender_id
            all_commands = self.command_manager.get_all_commands()
            available = []
            for cmd in all_commands:
                if self.command_manager.has_permission(cmd, user_id):
                    available.append(cmd)

            if not available:
                return make_reply(input_message, "没有可用的命令。")

            lines = ["可用命令："]
            for cmd in available:
                aliases_str = f"（别名: {', '.join(cmd.aliases)}）" if cmd.aliases else ""
                lines.append(f"• {cmd.name}{aliases_str}: {cmd.description}")

            return make_reply(input_message, "\n".join(lines))
        except Exception as e:
            _log.error(f"帮助命令处理失败: {e}")
            return []
