"""命令系统模块"""

import asyncio
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from core.message import InputMessage


class PermissionLevel(Enum):
    """权限级别枚举"""

    DEFAULT = "default"  # 默认权限，所有用户可用
    ADMIN = "admin"  # 管理员权限，仅管理员可用


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
        """执行命令（支持 async handler）"""
        result = self.handler(input_message, args)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    def has_permission(self, user_id: str, admin_ids: list[str]) -> bool:
        """检查用户是否有权限执行此命令"""
        if self.permission == PermissionLevel.DEFAULT:
            return True
        elif self.permission == PermissionLevel.ADMIN:
            return user_id in admin_ids
        return False


class CommandRegistry:
    """命令注册表"""

    def __init__(self):
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        """注册单个命令"""
        # 注册主命令名
        self._commands[command.name.lower()] = command

        # 注册别名
        for alias in command.aliases:
            self._commands[alias.lower()] = command

    def find(self, command_name: str) -> Optional[Command]:
        """查找命令"""
        return self._commands.get(command_name.lower())

    def get_all_commands(self) -> list[Command]:
        """获取所有命令（去重）"""
        seen = set()
        commands = []
        for cmd in self._commands.values():
            if cmd.name not in seen:
                commands.append(cmd)
                seen.add(cmd.name)
        return commands

    def count(self) -> int:
        """获取命令数量（去重）"""
        return len(set(cmd.name for cmd in self._commands.values()))

    def clear(self) -> None:
        """清空所有命令"""
        self._commands.clear()
