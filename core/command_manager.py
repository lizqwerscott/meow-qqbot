"""命令管理器模块"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from core.commands import Command, CommandRegistry, PermissionLevel
from core.message_queue import InputMessage

_log = logging.getLogger(__name__)


class CommandManager:
    """命令管理器"""

    def __init__(self, client: "MyClient"):
        self.client = client
        self.registry = CommandRegistry()

    def register_command(self, command: Command) -> None:
        self.registry.register(command)

    def register_default_commands(self) -> None:
        """注册默认命令"""
        # 帮助命令
        self.registry.register(
            Command(
                name="帮助",
                handler=self.handle_help_command,
                aliases=["help"],
                permission=PermissionLevel.DEFAULT,
                description="显示命令帮助",
            )
        )

    def find_command(self, command_name: str) -> Command | None:
        """查找命令"""
        return self.registry.find(command_name)

    def get_all_commands(self) -> list[Command]:
        """获取所有命令"""
        return self.registry.get_all_commands()

    def has_permission(self, command: Command, user_id: str) -> bool:
        """检查用户是否有权限执行命令"""
        return command.has_permission(user_id, self.client.admin_id)

    def process_message(self, input_message) -> list[dict]:
        """
        处理消息，查找是否有符合的命令，如果有就执行并返回消息列表

        Args:
            input_message: 输入消息对象

        Returns:
            消息列表，每个消息是一个字典，包含chat_id, content, message_id, is_group等字段
            如果没有匹配的命令，返回空列表
        """
        try:
            content = input_message.content.strip()

            # 检查是否是命令（以猫猫开头或以/开头）
            if content.startswith("猫猫"):
                # 猫猫 命令名 参数 - 使用len("猫猫")而不是固定2
                parts = content[len("猫猫") :].strip().split(maxsplit=1)
                if not parts or not parts[0]:
                    return []
                command_name = parts[0]
                args = parts[1] if len(parts) > 1 else ""
            elif content.startswith("/"):
                # /命令名 参数
                parts = content[1:].split(maxsplit=1)
                command_name = parts[0]
                args = parts[1] if len(parts) > 1 else ""
            else:
                return []

            # 查找命令
            command = self.find_command(command_name)
            if not command:
                return []

            # 检查权限
            if not self.has_permission(command, input_message.sender_id):
                return [
                    {
                        "chat_id": input_message.chat_id,
                        "content": "您没有权限执行此命令。",
                        "message_id": input_message.id,
                        "is_group": input_message.is_group,
                    }
                ]

            # 执行命令并返回消息列表
            messages = command.execute(input_message, args)
            return messages if messages else []

        except Exception as e:
            _log.error(f"处理命令消息时出错: {e}")
            return []

    def handle_help_command(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        """处理帮助命令，显示可用命令列表"""
        try:
            chat_id = input_message.chat_id
            user_id = input_message.sender_id

            # 获取所有命令
            all_commands = self.get_all_commands()

            # 过滤用户有权限的命令
            available_commands = []
            for cmd in all_commands:
                if self.has_permission(cmd, user_id):
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
