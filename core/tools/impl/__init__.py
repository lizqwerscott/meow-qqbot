"""工具注册中心 — ToolRegistry 实例与 execute 封装。

工具注册改为显式调用 create_all_tool_entries(deps)。
"""

import logging

from core.tools._types import ToolContext, ToolResult
from core.tools.registry import ToolRegistry

_log = logging.getLogger(__name__)

registry = ToolRegistry()


async def execute(
    name: str,
    args: dict,
    ctx: ToolContext,
    permission_manager=None,
) -> ToolResult:
    return await registry.execute(name, args, ctx, permission_manager)
