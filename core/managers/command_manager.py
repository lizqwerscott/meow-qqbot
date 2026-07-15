"""命令系统模块"""

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from core.message import InputMessage

_log = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """权限级别枚举"""

    DEFAULT = "default"
    ADMIN = "admin"


class Command:
    """命令类"""

    def __init__(
        self,
        name: str,
        handler: Callable,
        aliases: list[str] | None = None,
        permission: PermissionLevel = PermissionLevel.DEFAULT,
        description: str = "",
    ):
        self.name = name
        self.handler = handler
        self.aliases = aliases or []
        self.permission = permission
        self.description = description

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        result = self.handler(input_message, args)
        if asyncio.iscoroutine(result):
            result = await result
        return result


class CommandRegistry:
    """命令注册表"""

    def __init__(self):
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        self._commands[command.name.lower()] = command
        for alias in command.aliases:
            self._commands[alias.lower()] = command

    def find(self, command_name: str) -> Optional[Command]:
        return self._commands.get(command_name.lower())

    def get_all_commands(self) -> list[Command]:
        seen = set()
        commands = []
        for cmd in self._commands.values():
            if cmd.name not in seen:
                commands.append(cmd)
                seen.add(cmd.name)
        return commands

    def count(self) -> int:
        return len(set(cmd.name for cmd in self._commands.values()))

    def unregister(self, command_name: str) -> Optional[Command]:
        key = command_name.lower()
        cmd = self._commands.get(key)
        if cmd is None:
            return None
        keys_to_del = [cmd.name.lower()] + [a.lower() for a in cmd.aliases]
        for k in keys_to_del:
            self._commands.pop(k, None)
        return cmd

    def clear(self) -> None:
        self._commands.clear()


class CommandManager:
    """命令管理器"""

    def __init__(self, admin_id: list[str], permission_manager=None):
        self.admin_id = admin_id
        self._perm = permission_manager
        self.registry = CommandRegistry()

    def register_command(self, command: Command) -> None:
        self.registry.register(command)

    def unregister_command(self, command_name: str) -> Optional[Command]:
        return self.registry.unregister(command_name)

    def find_command(self, command_name: str) -> Command | None:
        """查找命令"""
        return self.registry.find(command_name)

    def get_all_commands(self) -> list[Command]:
        """获取所有命令"""
        return self.registry.get_all_commands()

    def has_permission(self, command: Command, user_id: str) -> bool:
        """检查用户是否有权限执行命令"""
        if command.permission == PermissionLevel.DEFAULT:
            return True
        if command.permission == PermissionLevel.ADMIN:
            return self._perm.get_user_role(user_id) == "admin" if self._perm else (user_id in self.admin_id)

    async def process_message(self, input_message: InputMessage) -> list[dict]:
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

            command_name = ""
            args = ""

            if input_message.is_group:
                # 群聊：只认 猫猫 /<命令>
                if content.startswith("猫猫 /"):
                    raw = content[len("猫猫 /"):].strip()
                    if raw:
                        parts = raw.split(maxsplit=1)
                        command_name = parts[0]
                        args = parts[1] if len(parts) > 1 else ""
            else:
                # 私聊：只认 /<命令>
                if content.startswith("/"):
                    parts = content[1:].split(maxsplit=1)
                    command_name = parts[0]
                    args = parts[1] if len(parts) > 1 else ""

            if not command_name:
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
            messages = await command.execute(input_message, args)
            return messages if messages else []

        except Exception as e:
            _log.error(f"处理命令消息时出错: {e}")
            return []
