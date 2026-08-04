"""工具模块 — 工具定义、执行器、子智能体管理器、以及调用上下文。"""

from core.tools._types import ToolContext, ToolResult
from core.tools.patch_parser import DiffError, apply_update_hunks, parse_patch_text
from core.tools.sub_agent_manager import SubAgentManager, SubAgentRecord

__all__ = [
    "ToolContext",
    "ToolResult",
    "SubAgentManager",
    "SubAgentRecord",
]
