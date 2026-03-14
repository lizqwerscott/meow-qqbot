"""命令管理器模块"""

from typing import TYPE_CHECKING

from botpy import logging

from core.commands import Command, CommandRegistry, PermissionLevel
from core.handlers import (
    handle_clear_command,
    handle_help_command,
    handle_history_command,
    handle_list_chats_command,
    handle_status_command,
)

if TYPE_CHECKING:
    from core.client import MyClient

_log = logging.get_logger()


class CommandManager:
    """命令管理器"""

    def __init__(self, client: MyClient):
        self.client = client
        self.registry = CommandRegistry()

    def register_default_commands(self) -> None:
        """注册默认命令"""
        # 历史命令
        self.registry.register(
            Command(
                name="历史",
                handler=lambda im, args: handle_history_command(self.client, im, args),
                aliases=["history"],
                permission=PermissionLevel.DEFAULT,
                description="查看最近对话历史",
            )
        )

        # 清空命令
        self.registry.register(
            Command(
                name="清空",
                handler=lambda im, args: handle_clear_command(self.client, im, args),
                aliases=["clear"],
                permission=PermissionLevel.DEFAULT,
                description="清空当前对话历史",
            )
        )

        # 帮助命令
        self.registry.register(
            Command(
                name="帮助",
                handler=lambda im, args: handle_help_command(self.client, im, args),
                aliases=["help"],
                permission=PermissionLevel.DEFAULT,
                description="显示命令帮助",
            )
        )

        # 状态命令（管理员专用）
        self.registry.register(
            Command(
                name="状态",
                handler=lambda im, args: handle_status_command(self.client, im, args),
                aliases=["status"],
                permission=PermissionLevel.ADMIN,
                description="查看系统状态（管理员专用）",
            )
        )

        # 列出所有聊天命令（管理员专用）
        self.registry.register(
            Command(
                name="聊天列表",
                handler=lambda im, args: handle_list_chats_command(
                    self.client, im, args
                ),
                aliases=["list", "chats"],
                permission=PermissionLevel.ADMIN,
                description="查看所有聊天ID列表（管理员专用）",
            )
        )

        _log.info(f"已注册 {self.registry.count()} 个命令")

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
            messages = command.handler(input_message, args)
            return messages if messages else []

        except Exception as e:
            _log.error(f"处理命令消息时出错: {e}")
            return []
