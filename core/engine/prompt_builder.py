"""PromptBuilder — AI 请求消息组装器

组装 AI 请求的 messages 列表：
- 静态 system prompt（模板渲染）
- 对话历史（上下文管理器）
- 动态 system 消息（记忆、时间、用户列表、技能条目）
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, List, Optional, Tuple

from core.message import InputMessage
from core.tools.definitions import (
    EMOJI_TOOLS,
    SEARCH_USER_TOOL,
    SEARCH_MEMORY_TOOL,
    SEARCH_RELATION_TOOL,
    MARK_IMPORTANT_TOOL,
    SKILL_TOOLS,
    EXECUTE_COMMAND_TOOL,
    LEARNER_TOOLS,
    TASK_TOOLS,
    FILE_TOOLS,
)

_log = logging.getLogger(__name__)

_MEMORY_SYSTEM_DESC = (
    "【记忆系统】\n"
    "你可以使用以下工具查询和保存长期记忆。\n"
    "\n"
    "**重要原则：不确定的先查记忆，不要猜测！**\n"
    "- 当用户询问关于某人的背景、偏好、说过的话、过往经历时→ 先 search_memory，不要凭印象回答\n"
    "- 当用户提到以前的事、上次的约定、之前讨论过的内容→ 先 search_memory 确认事实\n"
    "- 当需要确认某个具体事实（如生日、爱好、说过的话）→ 先 search_memory 再回答\n"
    "- 如果 search_memory 没有找到相关信息，如实告诉用户你不知道，不要编造\n"
    "\n"
    "可用工具：\n"
    "- search_memory：搜索记忆（指定 person_name 可查群友，不指定则查当前用户），可查画像、经历、事实等\n"
    "- search_relation：搜索两个人之间的关系记忆，系统会同时搜索双方记忆和当前用户的记载\n"
    "- mark_important：记录重要信息。用户解释背景/喜好/事实时主动调用，立即存入长期记忆\n"
)

_DIRTY_PATTERNS = (
    "<available_skills", "<skill>", "<description>",
    "<name>", "【工具配合原则】", "【记忆系统】",
    "--- 技能系统 ---", "工具配合原则",
)


class PromptBuilder:
    """AI 请求消息组装器。

    职责：
    1. 触发历史压缩（compaction）
    2. 根据可用能力确定工具列表
    3. 渲染静态 system prompt（模板）
    4. 拼接动态上下文（记忆、时间、表情标签、群友列表、技能条目）
    """

    def __init__(
        self,
        template_manager: Any,
        context_manager: Any,
        ai_service: Any,
        *,
        bot_id: str = "",
        nickname_manager: Any = None,
        emoji_manager: Any = None,
        skill_managers: Any = None,
        hindsight_memory: Any = None,
        search_top_k: int = 3,
        admin_ids: Optional[List[str]] = None,
        learning_orchestrator: Any = None,
        has_tasks: bool = False,
        permission_manager=None,
        workspace_manager=None,
    ):
        self.template_manager = template_manager
        self.context_manager = context_manager
        self.ai_service = ai_service
        self._bot_id = bot_id
        self._nm = nickname_manager
        self.emoji_manager = emoji_manager
        self._skill_managers = skill_managers
        self.hindsight = hindsight_memory
        self.learners = learning_orchestrator
        self._search_top_k = search_top_k
        self._admin_ids = admin_ids or []
        self._has_tasks = has_tasks
        self._perm = permission_manager
        self._workspace_manager = workspace_manager

    async def build(
        self,
        chat_id: str,
        is_group: bool,
        user_nickname: str,
        sender_id: str,
        input_message: InputMessage,
        cost_tracker: Any = None,
    ) -> Tuple[List[dict], Optional[List[dict]]]:
        """组装 AI 请求的 messages 列表。

        Returns:
            (messages, tools_to_use):
            - messages: OpenAI 格式的消息列表
            - tools_to_use: 本次可用的工具定义列表，或 None
        """
        # ── 1. Token 阈值触发 compaction ──
        _, compact_usage, ctx = await self.context_manager.compact_history_if_needed(
            chat_id, self.ai_service
        )
        if compact_usage and cost_tracker:
            cost_tracker.record_turn(chat_id, self.ai_service.model, compact_usage)

        # ── 1b. 防御：清理 context 历史中孤立的 tool_calls ──
        cleaned = ctx.remove_orphaned_tool_calls()
        if cleaned:
            _log.info(f"清理了 {cleaned} 条孤立 tool_calls 消息 [{chat_id[:12]}..]")

        # ── 2. 确定可用工具 / 能力状态 ──
        has_emojis = (
            self.emoji_manager is not None
            and self.emoji_manager.count_emojis() > 0
        )
        if is_group and self._nm:
            has_users = any(
                k != self._bot_id for k in self._nm.nicknames
            ) or any(
                k != self._bot_id for k in self._nm.auto_nicknames
            )
        else:
            has_users = False

        tools_to_use: List[dict] = []
        if has_emojis:
            tools_to_use.extend(EMOJI_TOOLS)
        if has_users:
            tools_to_use.extend(SEARCH_USER_TOOL)
        if self.hindsight:
            tools_to_use.extend(SEARCH_MEMORY_TOOL)
            tools_to_use.extend(SEARCH_RELATION_TOOL)
            tools_to_use.extend(MARK_IMPORTANT_TOOL)
        if self._skill_managers and self._skill_managers.has_skills:
            tools_to_use.extend(SKILL_TOOLS)
        if self.learners:
            tools_to_use.extend(LEARNER_TOOLS)
        if self._has_tasks:
            tools_to_use.extend(TASK_TOOLS)
        if self._workspace_manager:
            tools_to_use.extend(FILE_TOOLS)
        tools_to_use = tools_to_use or None

        # ── 3. 静态 system prompt ──
        skill_intro = (
            self._skill_managers.get_skill_system_intro()
            if (self._skill_managers and self._skill_managers.has_skills)
            else ""
        )
        memory_desc = _MEMORY_SYSTEM_DESC if self.hindsight else ""

        if is_group:
            static_prompt = self.template_manager.get_group_chat_prompt(
                has_emojis=has_emojis,
                has_users=has_users,
                memory_system_desc=memory_desc,
                skill_system_intro=skill_intro,
            )
        else:
            static_prompt = self.template_manager.get_private_chat_prompt(
                user_nickname,
                has_emojis=has_emojis,
                has_users=has_users,
                memory_system_desc=memory_desc,
                skill_system_intro=skill_intro,
            )

        # ── 4. 完整历史 ──
        history = ctx.get_history_as_dicts()

        messages: List[dict] = [{"role": "system", "content": static_prompt}]
        messages.extend(history)

        # ── 4b. 按角色过滤工具列表 ──
        if tools_to_use and self._perm:
            role = self._perm.get_user_role(sender_id)
            before_count = len(tools_to_use)
            tools_to_use = [
                t for t in tools_to_use
                if self._perm.can_use_tool(t["function"]["name"], role)
            ]
            if len(tools_to_use) < before_count:
                _log.info(
                    f"角色过滤: sender={sender_id[:16]}.. role={role} "
                    f"tools={before_count}→{len(tools_to_use)}"
                )

        # ── 5. 动态上下文（末尾单独一个 system 消息） ──
        dynamic_parts: List[str] = []

        # 技能条目列表
        if self._skill_managers and self._skill_managers.has_skills:
            entries = self._skill_managers.get_skill_entries_block()
            if entries:
                dynamic_parts.append(entries)

        # 记忆上下文
        memory_text = await self._build_memory_context(
            sender_id=sender_id,
            input_message=input_message,
        )
        if memory_text:
            dynamic_parts.append(memory_text)

        # 学习上下文（社群俚语词典）
        if self.learners:
            learning_ctx = await self.learners.enrich_prompt_context(
                chat_id=chat_id,
                sender_id=sender_id,
                message_text=input_message.content,
            )
            if learning_ctx:
                dynamic_parts.append(learning_ctx)

        # 当前时间
        _tz = timezone(timedelta(hours=8))
        now = datetime.now(_tz)
        weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        time_info = now.strftime(f"%Y-%m-%d %H:%M:%S ({weekday_names[now.weekday()]})")
        dynamic_parts.append(f"当前时间: {time_info} (CST/UTC+8)")
        dynamic_parts.append("注意：创建定时任务时请使用北京时间 (CST/UTC+8)，不要使用 UTC。")

        # 工作区上下文
        if self._workspace_manager:
            ws_type = "群聊" if is_group else "私聊"
            dynamic_parts.append(f"当前{ws_type}工作区: {chat_id[:12]}")
            dynamic_parts.append(
                "read_file / write_file / edit_file / list_files / search_files 五个文件工具仅限当前工作区内使用。"
            )

        # HEARTBEAT.md（管理员的私聊专属）
        if (
            self._workspace_manager
            and not is_group
            and chat_id in self._admin_ids
        ):
            hb_path = self._workspace_manager.heartbeat_path()
            if hb_path.exists():
                try:
                    hb_content = hb_path.read_text(encoding="utf-8").strip()
                    if hb_content:
                        dynamic_parts.append("【心跳配置 (HEARTBEAT.md)】\n你可以在本工作区查看和管理心跳配置。\n当前心跳配置内容如下：\n\n" + hb_content)
                except Exception:
                    pass

        # 表情标签列表
        if has_emojis and self.emoji_manager:
            tags = self.emoji_manager.get_all_tags()
            if tags:
                dynamic_parts.append("可用表情标签：" + "、".join(tags))

        # 自身 ID 映射
        if is_group and self._bot_id:
            dynamic_parts.append(f"你的 ID: {self._bot_id}（群友 @ 你时显示为 @{self._bot_id}）")

        # 群友列表
        if has_users and self._nm:
            lines = ["【群友列表】"]
            for uid, aliases in sorted(self._nm.iter_users(), key=lambda x: "，".join(x[1])):
                alias_str = "，".join(aliases)
                lines.append(f"- {uid}（{alias_str}）")
            if lines:
                dynamic_parts.append("\n".join(lines))

        if dynamic_parts:
            messages.append({
                "role": "system",
                "content": "\n\n".join(dynamic_parts),
            })

        # ── 6. 防御：清理孤立的 tool_calls（防止 compaction 拆散配对） ──
        from core.tools.tool_loop import ensure_messages_consistent
        ensure_messages_consistent(messages)

        _log.debug(
            f"请求 AI messages:\n{json.dumps(messages, ensure_ascii=False, indent=2)}"
        )
        if tools_to_use:
            _log.info(
                f"本次请求注入 {len(tools_to_use)} 个工具: "
                f"{[t['function']['name'] for t in tools_to_use]}"
            )

        return messages, tools_to_use

    async def build_task_messages(
        self,
        chat_id: str,
        prompt: str,
    ) -> Tuple[List[dict], Optional[List[dict]]]:
        """组装后台任务的 messages 列表（使用 task_chat.j2 模板）。

        工具集与主对话一致（排除表情工具和递归任务工具）。
        Returns:
            (messages, tools_to_use)
        """
        tools_to_use: List[dict] = []
        from core.tools.definitions import (
            EMOJI_TOOLS,
            SEARCH_MEMORY_TOOL,
            SEARCH_RELATION_TOOL,
            MARK_IMPORTANT_TOOL,
            EXECUTE_COMMAND_TOOL,
            EXECUTE_SKILL_TOOL,
            VIEW_SKILL_TOOL,
        )

        if self.emoji_manager and self.emoji_manager.count_emojis() > 0:
            tools_to_use.extend(EMOJI_TOOLS)
        if self.hindsight:
            tools_to_use.extend(SEARCH_MEMORY_TOOL)
            tools_to_use.extend(SEARCH_RELATION_TOOL)
            tools_to_use.extend(MARK_IMPORTANT_TOOL)
        if self._skill_managers and self._skill_managers.has_skills:
            tools_to_use.extend(EXECUTE_COMMAND_TOOL)
            tools_to_use.extend(EXECUTE_SKILL_TOOL)
            tools_to_use.extend(VIEW_SKILL_TOOL)
        if self._workspace_manager:
            tools_to_use.extend(FILE_TOOLS)
        tools_to_use = tools_to_use or None

        # 渲染 task prompt
        from datetime import datetime, timezone, timedelta
        _tz = timezone(timedelta(hours=8))
        now = datetime.now(_tz)
        system_prompt = self.template_manager.get_task_chat_prompt(
            current_time=now.strftime("%Y-%m-%d %H:%M:%S (CST/UTC+8)"),
        )

        messages: List[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return messages, tools_to_use

    async def build_memory_context(self, sender_id: str, input_message: InputMessage) -> str:
        """构建记忆上下文字符串（公开，供 ToolLoop Queue Steering 使用）。"""
        return await self._build_memory_context(sender_id, input_message)

    async def _build_memory_context(
        self,
        sender_id: str,
        input_message: InputMessage,
    ) -> str:
        """查询 Hindsight 记忆，返回格式化的上下文字符串，或空字符串。"""
        if not self.hindsight:
            return ""

        query = input_message.content.strip()
        if not query:
            return ""

        try:
            result = await self.hindsight.search(
                user_id=sender_id,
                query=query,
                top_k=5,
                include_profile=True,
            )

            episodes = result.get("episodes", [])
            profiles = result.get("profiles", [])

            if not episodes and not profiles:
                return ""

            parts = ["--- 相关记忆 ---"]

            if profiles:
                for p in profiles[:1]:
                    pd = p.get("profile_data", {})
                    if isinstance(pd, dict):
                        for k, v in pd.items():
                            if isinstance(v, str) and any(p in v for p in _DIRTY_PATTERNS):
                                continue
                            parts.append(f"- [{k}]: {str(v)[:150]}")

            if episodes:
                count = 0
                for e in episodes:
                    if count >= 3:
                        break
                    summary = (e.get("summary", "") or e.get("episode", "")).strip()
                    if not summary:
                        continue
                    if any(p in summary for p in _DIRTY_PATTERNS):
                        continue
                    parts.append(f"- {summary[:150]}")
                    count += 1

            if len(parts) == 1:
                return ""

            parts.append("--- 相关记忆结束 ---")

            _log.info(
                f"自动记忆注入: sender={sender_id[:16]}.. "
                f"注入{len(episodes)}条经历, {len(profiles)}条画像"
            )
            return "\n".join(parts)
        except Exception as e:
            _log.warning(f"自动记忆注入失败: {e!r}")
            return ""
