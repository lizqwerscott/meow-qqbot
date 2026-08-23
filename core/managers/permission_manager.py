"""PermissionManager — 权限与命令白名单管理器

从 allowlist.toml 加载配置，提供：
- 用户角色解析（admin / trusted / default）
- 工具权限校验（支持 group:* 组级别权限）
- 命令白名单 + 安全策略校验（管道、串联、重定向、长度）
"""

import logging
import os
import re
import shlex
import tomllib
from pathlib import Path
from typing import List, Optional

from core.tools.catalog import SECTIONS

_log = logging.getLogger(__name__)

ROLE_LEVEL = {"system": 4, "admin": 3, "trusted": 2, "default": 1}


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

    def role_at_least(self, candidate_role: str, required_role: str) -> bool:
        """Compare roles without exposing the internal numeric ordering."""
        return self._role_level(candidate_role) >= self._role_level(required_role)

    def is_admin_role(self, role: str) -> bool:
        return self._role_level(role) >= 3

    # ── 工具权限 ──

    def can_use_tool(self, tool_name: str, user_role: str) -> bool:
        """检查用户角色是否可以使用指定工具。未知工具默认 admin。"""
        required = self._require_level(tool_name)
        return self._role_level(user_role) >= self._role_level(required)

    def _require_level(self, tool_name: str) -> str:
        """返回工具所需的最低角色。

        查找顺序：
        1. 精确工具名匹配（如 memory = "all"）
        2. 所属 group 匹配（如 group:memory = "all"），取最宽松的权限
        3. 默认 "admin"
        """
        tools = self._data.get("tools", {})
        if tool_name in tools:
            return tools[tool_name]
        best = "admin"
        best_level = ROLE_LEVEL.get(best, 3)
        for key, level in tools.items():
            if key.startswith("group:"):
                section = key[6:]
                if tool_name in SECTIONS.get(section, set()):
                    cur = ROLE_LEVEL.get(level, 1)
                    if cur < best_level:
                        best_level = cur
                        best = level
        return best

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
        if user_role in ("admin", "system"):
            return None

        if not parts:
            return "命令为空"

        raw = command.strip()

        # ── 命令替换检查 ($(...) / ``) ──
        if self._get_config("security.deny_substitution", True):
            if re.search(r"\$[({]|`[^`]*`", raw):
                return "禁止使用命令替换 ($(…) / ``)"

        # ── 命令串联检查 (; && || &) ──
        if self._get_config("security.deny_chaining", True):
            # 用管道符 | 分割后，再看各分段内是否有串联符
            pipe_segments = raw.split("|")
            for seg in pipe_segments:
                seg = seg.strip()
                if not seg:
                    continue
                if re.search(r"(?<!\|)[;&]|&&|\|\|(?<!\|)", seg):
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
            if re.search(r"(?:^|[^<>])>+[^<>]", raw):
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

    def get_allowed_commands(self) -> list:
        """返回 [commands].allowed 命令名列表（作为静态 allowlist 输入）。"""
        return list(self._data.get("commands", {}).get("allowed", []))

    # ── 执行超时 ──

    def get_default_timeout(self) -> int:
        """返回命令/脚本执行的默认超时秒数。"""
        return self._get_config("security.default_timeout", 60)

    def get_max_timeout(self) -> int:
        """返回命令/脚本执行的最大超时秒数。"""
        return self._get_config("security.max_timeout", 300)

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

    def get_security_config(self, name: str, default=None):
        """读取 [security] 下的单项安全策略配置。"""
        return self._get_config(f"security.{name}", default)

    # ── exec 审批策略（对齐 openclaw tools.exec.*）──

    def get_exec_policy(self):
        """读取 [exec] 段的 requested 策略（对齐 openclaw tools.exec.*）。

        Returns:
            dict: {"mode", "security", "ask", "ask_fallback", "strict_inline_eval", "auto_reviewer"}
        """
        return {
            "mode": self._get_config("exec.mode", "ask"),
            "security": self._get_config("exec.security", "allowlist"),
            "ask": self._get_config("exec.ask", "on-miss"),
            "ask_fallback": self._get_config("exec.ask_fallback", "deny"),
            "strict_inline_eval": self._get_config("exec.strict_inline_eval", True),
            "auto_reviewer": self._get_config("exec.auto_reviewer", None),
            "safe_bins": self._get_config("exec.safe_bins", None),
            "safe_bin_profiles": self._get_config("exec.safe_bin_profiles", None),
            "approval_timeout": self._get_config("exec.approval_timeout", 300),
        }
