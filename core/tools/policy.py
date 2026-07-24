"""声明式工具选择管线 — 替代 _factory.py"""

from dataclasses import dataclass
from typing import Optional

from core.tools.catalog import SECTIONS, PROFILES, CRON_ALLOWED, FEATURE_SECTION_MAP
from core.tools.impl import registry
from core.tools.deps import ToolDeps


@dataclass
class ChatContext:
    has_emojis: bool = False
    has_hindsight: bool = False
    has_users: bool = False
    is_group: bool = False
    has_skills: bool = False
    has_workspace: bool = False
    has_tasks: bool = False
    has_tts: bool = False
    has_sub_agents: bool = False
    has_learners: bool = False


def build_tools(
    profile: str,
    ctx: ChatContext,
    deps: ToolDeps | None = None,
    role: Optional[str] = None,
    tools_allow: Optional[list[str]] = None,
) -> list[dict]:
    """声明式工具选择管线。

    Args:
        profile: 工具集模板名 ("normal" | "heartbeat" | "task")
        ctx: 聊天的功能存在性上下文
        role: 用户角色（从 PermissionManager 获取），用于权限过滤
        tools_allow: task 模式专用的工具白名单

    Returns:
        OpenAI 格式的工具定义列表
    """
    # Step 1: 从 profile 取基础工具集
    names = set(PROFILES.get(profile, set()))

    # Step 2: 按功能存在性过滤
    # - FEATURE_SECTION_MAP 中定义的标志位 → 移除对应 section 的工具
    for flag, section in FEATURE_SECTION_MAP.items():
        if not getattr(ctx, flag, False) and section:
            names -= SECTIONS.get(section, set())

    # is_group + has_users 特殊处理（用户搜索仅在群聊且有用户数据时可用）
    if not ctx.is_group or not ctx.has_users:
        names -= SECTIONS.get("user", set())

    # cron + task 独立处理（has_tasks 同时控制两个 section）
    if not ctx.has_tasks:
        names -= SECTIONS.get("cron", set())
        names -= SECTIONS.get("task", set())

    # announce 仅在 task profile 中有效
    if profile != "task":
        names.discard("announce")

    # Step 3: 按角色过滤（从 allowlist.toml）
    if role:
        perm = deps.permission_manager if deps else None
        if perm:
            names = {n for n in names if perm.can_use_tool(n, role)}

    # Step 4 (task 专用): tools_allow 白名单
    if tools_allow is not None and profile == "task":
        names = _filter_task_allow(names, tools_allow)

    return registry.specs(names)


def _filter_task_allow(
    names: set[str],
    tools_allow: list[str],
) -> set[str]:
    if not tools_allow:
        return {"announce"}
    if tools_allow == ["*"]:
        return names
    allowed = {n for n in tools_allow if n in CRON_ALLOWED}
    return names & allowed


def format_task_tool_descriptions(names: set[str]) -> str:
    """为 task profile 格式化工具描述文本"""
    lines = ["可用工具："]
    DEFAULT_ORDER = [
        "announce", "search_user",
        "memory", "mark_important",
        "read_file", "write_file", "edit_file", "apply_patch",
        "exec", "view_skill", "execute_skill", "rescan_skills",
    ]
    for name in DEFAULT_ORDER:
        if name in names:
            entry = registry.get(name)
            desc = entry.description[:60] if entry else name
            lines.append(f"- {name}：{desc}")
    return "\n".join(lines)
