"""PermissionManager — 权限与命令白名单管理器

从 allowlist.toml 加载配置，提供：
- 用户角色解析（admin / trusted / default）
- 工具权限校验
- 命令白名单 + 安全策略校验（管道、串联、重定向、长度）
"""

import logging
import os
import re
import shlex
import tomllib
from pathlib import Path
from typing import List, Optional

_log = logging.getLogger(__name__)

ROLE_LEVEL = {"admin": 3, "trusted": 2, "default": 1}


class PermissionManager:
    """权限与命令白名单管理器。"""

    def __init__(self, path: str = "allowlist.toml"):
        path_resolved = Path(path)
        if not path_resolved.exists():
            _log.warning(f"权限配置文件不存在: {path}，使用全部默认权限")
            self._data = {
                "roles": {},
                "tools": {},
                "commands": {"allowed": []},
                "security": {},
            }
        else:
            with open(path_resolved, "rb") as f:
                self._data = tomllib.load(f)
        _log.info(
            f"PermissionManager 已加载: roles={list(self._data.get('roles', {}))}"
        )

    # ── 角色解析 ──

    def get_user_role(self, sender_id: str) -> str:
        """返回用户角色: 'admin' | 'trusted' | 'default'"""
        roles = self._data.get("roles", {})
        for role_name, ids in roles.items():
            if sender_id in ids:
                return role_name
        return "default"

    def get_role_ids(self, role_name: str) -> list:
        """返回指定角色的用户 ID 列表。"""
        return list(self._data.get("roles", {}).get(role_name, []))

    def _role_level(self, role: str) -> int:
        return ROLE_LEVEL.get(role, 1)

    # ── 工具权限 ──

    def can_use_tool(self, tool_name: str, user_role: str) -> bool:
        """检查用户角色是否可以使用指定工具。未知工具默认 admin。"""
        required = self._require_level(tool_name)
        return self._role_level(user_role) >= self._role_level(required)

    def _require_level(self, tool_name: str) -> str:
        """返回工具所需的最低角色。"""
        tools = self._data.get("tools", {})
        return tools.get(tool_name, "admin")

    # ── 命令白名单 + 安全策略 ──

    def check_command_allowed(
        self, command: str, parts: List[str], user_role: str
    ) -> Optional[str]:
        """检查命令是否在安全限制内。

        Args:
            command: 原始命令字符串
            parts: shlex.split 后的命令参数列表
            user_role: 用户角色

        Returns:
            None 表示通过，str 表示拒绝原因
        """
        if user_role == "admin":
            return None

        if not parts:
            return "命令为空"

        raw = command.strip()

        # ── 命令串联检查 (; && || &) ──
        if self._get_config("security.deny_chaining", True):
            # 用管道符 | 分割后，再看各分段内是否有串联符
            pipe_segments = raw.split("|")
            for seg in pipe_segments:
                seg = seg.strip()
                if not seg:
                    continue
                if re.search(r'(?<!\|)[;&]|&&|\|\|(?<!\|)', seg):
                    return "禁止使用命令串联符 (; && || &)"

        # ── 管道检查 ──
        if "|" in raw:
            if self._get_config("security.deny_pipe", False):
                return "禁止使用管道 |"
            for seg in pipe_segments:
                seg = seg.strip()
                if not seg:
                    continue
                try:
                    seg_parts = shlex.split(seg)
                except ValueError:
                    return f"管道段格式无效: {seg[:60]}"
                if not self._is_command_allowed(seg_parts):
                    cmd_name = os.path.basename(seg_parts[0])
                    return f"管道中命令「{cmd_name}」不在白名单中"

        # ── 重定向检查 ──
        if self._get_config("security.deny_redirect", True):
            if re.search(r'(?:^|[^<>])>+[^<>]', raw):
                return "禁止使用输出重定向 (> / >>)"

        # ── 命令长度 ──
        max_len = self._get_config("security.max_command_length", 2000)
        if len(command) > max_len:
            return f"命令过长（{len(command)} > {max_len}）"

        # ── 命令名白名单 ──
        if not self._is_command_allowed(parts):
            cmd_name = os.path.basename(parts[0])
            return f"命令「{cmd_name}」不在白名单中"

        return None

    def _is_command_allowed(self, parts: List[str]) -> bool:
        """检查解析后的命令名是否在白名单内。"""
        allowed = self._data.get("commands", {}).get("allowed", [])
        if not allowed:
            return False
        cmd_name = os.path.basename(parts[0])
        return cmd_name in allowed

    def _get_config(self, key_path: str, default=None):
        """从嵌套 dict 中按点号路径取值。"""
        keys = key_path.split(".")
        val = self._data
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default
