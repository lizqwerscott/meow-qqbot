"""工具注册中心 — ToolRegistry 实例 + 引导"""

import importlib
import logging
from typing import Any

from core.tools._types import ToolEntry, ToolContext, ToolResult
from core.tools.registry import ToolRegistry

_log = logging.getLogger(__name__)

# 全局注册中心（替换 _TOOL_MAP + _DEPS）
registry = ToolRegistry()

_IMPL_MODULES = [
    "emoji", "user", "memory", "learner", "skill",
    "task", "file", "heartbeat", "sub_agent", "tts",
    "exec_process",
]


def _register(entry: ToolEntry):
    registry.register(entry)


def _bootstrap():
    for mod_name in _IMPL_MODULES:
        try:
            mod = importlib.import_module(f"core.tools.impl.{mod_name}")
            if hasattr(mod, "_register_all"):
                mod._register_all(_register)
        except Exception as e:
            _log.error(f"加载工具模块 {mod_name} 失败: {e}", exc_info=True)


# ── 兼容层：旧代码通过 from core.tools.impl import _DEPS, _TOOL_MAP 引用 ──
_DEPS = registry._deps
_TOOL_MAP = registry._tools
_HEARTBEAT_RESPONSE: dict = {}


def inject_deps(**deps):
    registry.inject(**deps)


def get_dep(name: str) -> Any:
    return registry.get_dep(name)


async def execute(
    name: str,
    args: dict,
    ctx: ToolContext,
    permission_manager=None,
) -> ToolResult:
    return await registry.execute(name, args, ctx, permission_manager)


def specs_by_names(names: set[str]) -> list[dict]:
    return registry.specs(names)


def get_entry(name: str) -> ToolEntry | None:
    return registry.get(name)


def consume_heartbeat_response() -> dict:
    resp = dict(_HEARTBEAT_RESPONSE)
    _HEARTBEAT_RESPONSE.clear()
    return resp


_bootstrap()
