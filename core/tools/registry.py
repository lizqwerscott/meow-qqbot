"""ToolRegistry — 类型安全的工具注册中心"""

import json
import logging
from typing import Optional

from core.tools._types import ToolContext, ToolEntry, ToolResult

_log = logging.getLogger(__name__)


class ToolRegistry:
    """类型安全的工具注册中心。

    职责：
    1. 注册/查找 ToolEntry
    2. 执行工具调用（含权限检查）
    3. 生成 OpenAI 格式的 tool specs
    """

    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}

    # ── 注册 ──

    def register(self, entry: ToolEntry):
        if entry.name in self._tools:
            _log.warning(f"工具 {entry.name} 重复注册，将被覆盖")
        self._tools[entry.name] = entry

    def get(self, name: str) -> Optional[ToolEntry]:
        return self._tools.get(name)

    @property
    def names(self) -> set[str]:
        return set(self._tools.keys())

    # ── 执行 ──

    async def execute(
        self,
        name: str,
        args: dict,
        ctx: ToolContext,
        permission_manager=None,
    ) -> ToolResult:
        entry = self._tools.get(name)
        if entry is None:
            _log.warning(f"未知工具调用: {name}")
            return ToolResult(content=json.dumps({"error": f"未知工具: {name}"}))

        if permission_manager:
            role = permission_manager.get_user_role(ctx.sender_id)
            if not permission_manager.can_use_tool(name, role):
                _log.warning(
                    f"工具权限拒绝: {name} role={role} sender={ctx.sender_id[:16]}.."
                )
                return ToolResult(
                    content=json.dumps(
                        {"error": "你没有权限使用该工具"},
                        ensure_ascii=False,
                    )
                )

        _log.info(f"[工具调用] {name}: {json.dumps(args, ensure_ascii=False)[:200]}")
        result = await entry.handler(args, ctx)
        _log.info(f"[工具调用] {name} 输出: {result.content[:200]}")
        return result

    # ── 生成 OpenAI tool specs ──

    def specs(self, names: set[str]) -> list[dict]:
        entries = [self._tools[n] for n in names if n in self._tools]
        return [
            {
                "type": "function",
                "function": {
                    "name": e.name,
                    "description": e.description,
                    "parameters": e.parameters,
                },
            }
            for e in entries
        ]

    def all_specs(self) -> list[dict]:
        return self.specs(self.names)
