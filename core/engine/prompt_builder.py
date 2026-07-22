"""PromptBuilder — AI 请求消息组装器

组装 AI 请求的 messages 列表：
- 静态 system prompt（模板渲染）
- 对话历史（上下文管理器）
- 动态 system 消息（记忆、时间、用户列表、技能条目）
"""

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, List, Optional, Tuple

from core.message import InputMessage
from core.tools.policy import build_tools, ChatContext, format_task_tool_descriptions

_log = logging.getLogger(__name__)

_MEMORY_SYSTEM_DESC = (
    "【记忆系统】\n"
    "你可以使用以下工具查询和保存长期记忆。\n"
    "\n"
    "**重要原则：不确定的先查记忆，不要猜测！**\n"
    "- 当用户询问关于某人的背景、偏好、说过的话、过往经历时→ 先 memory(action=search)，不要凭印象回答\n"
    "- 当用户提到以前的事、上次的约定、之前讨论过的内容→ 先 memory(action=search) 确认事实\n"
    "- 当需要确认某个具体事实（如生日、爱好、说过的话）→ 先 memory(action=search) 再回答\n"
    "- 如果 memory(action=search) 没有找到相关信息，如实告诉用户你不知道，不要编造\n"
    "\n"
    "可用工具：\n"
    "- memory：记忆搜索和关系查询。action=search 搜索记忆（指定 person_name 可查群友）；action=relation 查两人关系（指定 person_a + person_b）；可查画像、经历、事实等\n"
    "- mark_important：记录重要信息。用户解释背景/喜好/事实时主动调用，立即存入长期记忆\n"
)

HEARTBEAT_MINIMAL_SYSTEM_PROMPT = (
    "你是一个群聊助手的心跳检查器，运行在独立的 Heartbeat Session 中。\n\n"
    "可用工具：文件工具、记忆工具、命令工具、heartbeat_respond。\n\n"
    "回应方式：\n"
    "- 如果没有任何需要关注的事项，调用 heartbeat_respond(notify=false)\n"
    "- 如果有需要提醒的事项，调用 heartbeat_respond(notify=true, notification_text=\"...\")\n"
    "  提醒文本简洁明了，不要超过 200 字\n"
    "- 不要闲聊或回复额外内容\n\n"
    "规则：\n"
    "1. 不要凭印象回答，不确定的先查记忆\n"
    "2. 不要编造事实\n"
    "3. 只汇报真正需要关注的事项，不要过度打扰管理员"
)

HEARTBEAT_BEHAVIOR_BLOCK = (
    "\n\n【心跳检查模式】\n"
    "你当前处于定期心跳检查，这不是正常对话，请严格遵守以下规则：\n"
    "- 使用 heartbeat_respond(notify=true/false, notification_text=\"...\") 工具回应，不要直接输出文本聊天\n"
    "- 不要闲聊、不要使用猫娘语气卖萌、不要加表情符号\n"
    "- 只汇报需要提醒管理员的事项\n"
    "- 如果没有需要关注的事项，调用 heartbeat_respond(notify=false) 静默结束"
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

    def __init__(self, ctx):
        self.template_manager = ctx.prompt.template_manager
        self.context_manager = ctx.mgmt.context_manager
        self.ai_service = ctx.ai.ai_service
        self._bot_id = ctx.sys.bot_id
        self._nm = ctx.prompt.nickname_manager
        self.emoji_manager = ctx.prompt.emoji_manager
        self._skill_managers = ctx.prompt.skill_managers
        self.hindsight = ctx.memory.hindsight_memory
        self.learners = ctx.prompt.learning_orchestrator
        self._search_top_k = ctx.memory.search_top_k
        self._admin_ids = list(ctx.sys.admin_ids)
        self._has_tasks = ctx.bg.task_manager is not None
        self._has_sub_agents = ctx.sub.sub_agent_manager is not None
        self._perm = ctx.mgmt.permission_manager
        self._workspace_manager = ctx.mgmt.workspace_manager
        self._archive_manager = ctx.mgmt.archive_manager
        self._system_events = ctx.mgmt.system_events
        self._tts_service = None

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
        has_tts = bool(self._tts_service)
        if is_group and self._nm:
            has_users = any(
                k != self._bot_id for k in self._nm.nicknames
            ) or any(
                k != self._bot_id for k in self._nm.auto_nicknames
            )
        else:
            has_users = False

        role = self._perm.get_user_role(sender_id) if self._perm else None
        tools_to_use: Optional[List[dict]] = build_tools(
            "normal",
            ChatContext(
                has_emojis=has_emojis,
                has_hindsight=bool(self.hindsight),
                has_users=has_users,
                is_group=is_group,
                has_skills=bool(self._skill_managers and self._skill_managers.has_skills),
                has_workspace=bool(self._workspace_manager),
                has_tasks=self._has_tasks,
                has_tts=has_tts,
                has_sub_agents=self._has_sub_agents,
                has_learners=bool(self.learners),
            ),
            role=role,
        ) or None

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
                has_tts=has_tts,
                has_users=has_users,
                memory_system_desc=memory_desc,
                skill_system_intro=skill_intro,
            )
        else:
            static_prompt = self.template_manager.get_private_chat_prompt(
                user_nickname,
                has_emojis=has_emojis,
                has_tts=has_tts,
                has_users=has_users,
                memory_system_desc=memory_desc,
                skill_system_intro=skill_intro,
            )

        # ── 4. 完整历史 ──
        history = ctx.get_history_as_dicts()

        messages: List[dict] = [{"role": "system", "content": static_prompt}]
        messages.extend(history)

        # ── 5. 动态上下文（末尾单独一个 system 消息） ──
        dynamic_parts: List[str] = []

        # 系统事件（会话外部感知上下文，最优先显示）
        if self._system_events:
            events = self._system_events.peek_and_snapshot(chat_id)
            if events:
                lines = []
                for e in events:
                    ts = time.strftime("%H:%M:%S", time.localtime(e.ts))
                    lines.append(f"System: [{ts}] {e.text}")
                lines.append("")
                lines.append("处理完成后，如果没有需要关注的事项，回复 NO_REPLY 静默结束，无需向用户发送消息。")
                dynamic_parts.append("【系统事件】\n" + "\n".join(lines))

        # send_message 工具投递提示
        dynamic_parts.append(
            "【消息投递】你的工具调用之间的文本正常展示给用户。"
            "如果需要在最终回复中使用 send_message 工具，send_message 投递后你的后续文本将不再自动发送。"
        )

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

        # 归档摘要注入（归档后首次 build 时仅注入一次）
        if self._archive_manager:
            summary_text = self._archive_manager.consume_summary(chat_id)
            if summary_text:
                _log.info(
                    "注入归档摘要 [%s..] (%d 字符)",
                    chat_id[:12], len(summary_text),
                )
                dynamic_parts.append(
                    "以下内容来自过去几天的对话记录，"
                    "帮助你了解之前聊过什么：\n" + summary_text
                )

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
            is_admin_private = not is_group and chat_id in self._admin_ids
            dynamic_parts.append(f"当前{ws_type}工作区: {chat_id[:12]}")
            if is_admin_private:
                dynamic_parts.append(
                    "read_file / write_file / edit_file / apply_patch 四个文件工具可访问整个 workspaces/ 根目录（管理员权限）。搜索文件内容请使用 exec + rg。"
                )
            else:
                dynamic_parts.append(
                    "read_file / write_file / edit_file / apply_patch 四个文件工具仅限当前工作区内使用。搜索文件内容请使用 exec + rg。"
                )

        # HEARTBEAT.md（管理员的私聊专属）
        if self._workspace_manager and not is_group and chat_id in self._admin_ids:
            hb_path = self._workspace_manager.heartbeat_path()
            if hb_path.exists():
                dynamic_parts.append(
                    "【心跳配置 (HEARTBEAT.md)】\n"
                    "心跳配置文件存在于 workspaces/HEARTBEAT.md，"
                    "你可以使用 read_file 工具查看和 write_file 工具修改。"
                    "心跳执行时 AI 会自主读取此文件。"
                )
            else:
                dynamic_parts.append(
                    "【心跳配置 (HEARTBEAT.md)】\n"
                    "你可以在 workspaces/ 根目录创建 HEARTBEAT.md 来定义心跳检查清单，"
                    "文件不存在时心跳自动跳过。使用 write_file 工具写入 HEARTBEAT.md 即可。"
                )

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

    def _resolve_task_tools(
        self,
        tools_allow: Optional[List[str]],
    ) -> Tuple[Optional[List[dict]], str]:
        ctx = ChatContext(
            has_hindsight=bool(self.hindsight),
            has_workspace=bool(self._workspace_manager),
            has_skills=bool(self._skill_managers and self._skill_managers.has_skills),
        )
        tools_defs = build_tools("task", ctx, tools_allow=tools_allow) or []
        names = {t["function"]["name"] for t in tools_defs}
        desc_text = format_task_tool_descriptions(names)
        return tools_defs or None, desc_text

    async def build_task_messages(
        self,
        chat_id: str,
        prompt: str,
        tools_allow: Optional[List[str]] = None,
    ) -> Tuple[List[dict], Optional[List[dict]]]:
        """组装后台任务的 messages 列表（使用 task_chat.j2 模板）。

        工具集根据 tools_allow 动态决定。
        Returns:
            (messages, tools_to_use)
        """
        tools_to_use, tool_descriptions = self._resolve_task_tools(tools_allow)

        # 渲染 task prompt
        from datetime import datetime, timezone, timedelta
        _tz = timezone(timedelta(hours=8))
        now = datetime.now(_tz)
        system_prompt = self.template_manager.get_task_chat_prompt(
            current_time=now.strftime("%Y-%m-%d %H:%M:%S (CST/UTC+8)"),
            tool_descriptions=tool_descriptions,
        )

        messages: List[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return messages, tools_to_use

    async def build_heartbeat_messages(
        self,
        prompt: str,
        *,
        system_prompt_mode: str = "minimal",
        session_mode: str = "isolated",
        admin_chat_id: str = "",
        chat_id: str = "heartbeat:events",
        system_event_key: str = "heartbeat:events",
    ) -> Tuple[List[dict], Optional[List[dict]]]:
        """组装心跳检查的 messages 列表。

        工具集：文件工具 + 记忆工具 + heartbeat_respond + exec。
        system_prompt_mode:
          - "normal": 复用正常聊天的角色卡 SP（角色卡 + 技能介绍 + 记忆系统描述 + 动态上下文）
          - "minimal": 极简 heartbeat 专用 SP
        session_mode="main": 从管理员真实 chat_id 读取历史，作为独立消息对注入。
        chat_id: 心跳上下文的存储 key（用于构建 system prompt 中的工作区信息）。
        system_event_key: 系统事件队列的 drain key，与上下文存储解耦。

        Returns:
            (messages, tools_to_use)
        """
        tools_to_use = build_tools(
            "heartbeat",
            ChatContext(
                has_hindsight=bool(self.hindsight),
                has_workspace=bool(self._workspace_manager),
                has_tasks=self._has_tasks,
            ),
        ) or None

        # ── system message ──
        if system_prompt_mode == "normal":
            system_prompt = self._build_normal_heartbeat_system_prompt(chat_id)
            system_prompt += HEARTBEAT_BEHAVIOR_BLOCK
        else:
            system_prompt = HEARTBEAT_MINIMAL_SYSTEM_PROMPT

        messages: List[dict] = [{"role": "system", "content": system_prompt}]

        # ── 系统事件（心跳触发时注入） ──
        if self._system_events:
            events = self._system_events.peek_and_snapshot(system_event_key)
            if events:
                lines = []
                for e in events:
                    ts = time.strftime("%H:%M:%S", time.localtime(e.ts))
                    lines.append(f"System: [{ts}] {e.text}")
                messages.insert(1, {
                    "role": "system",
                    "content": "【系统事件（本次心跳触发）】\n" + "\n".join(lines),
                })

        # ── 聊天历史（作为独立消息对，仅 session=main） ──
        if session_mode == "main" and admin_chat_id:
            try:
                recent = await self.context_manager.get_chat_history_async(
                    admin_chat_id, max_messages=20,
                )
                inserted = 0
                for msg in recent[-15:]:
                    role = msg.get("role")
                    if role in ("user", "assistant"):
                        content = (msg.get("content") or "")
                        if content:
                            messages.append({"role": role, "content": content[:300]})
                            inserted += 1
                if inserted:
                    _log.info(f"心跳历史注入: {inserted} 条消息（来自 {admin_chat_id[:12]}..）")
            except Exception as e:
                _log.warning(f"心跳读取历史失败，回退隔离模式: {e}")

        # ── heartbeat user message ──
        messages.append({"role": "user", "content": prompt})

        return messages, tools_to_use

    def _build_normal_heartbeat_system_prompt(self, chat_id: str) -> str:
        """构建"normal"模式的 system prompt（类似正常聊天的 SP，不含历史）。"""
        memory_desc = _MEMORY_SYSTEM_DESC if self.hindsight else ""

        skill_intro = (
            self._skill_managers.get_skill_system_intro()
            if (self._skill_managers and self._skill_managers.has_skills)
            else ""
        )

        static_prompt = self.template_manager.get_private_chat_prompt(
            user_name="管理员",
            has_emojis=False,
            has_users=False,
            memory_system_desc=memory_desc,
            skill_system_intro=skill_intro,
        )

        # 动态上下文
        dynamic_parts: List[str] = []

        _tz = timezone(timedelta(hours=8))
        now = datetime.now(_tz)
        weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        dynamic_parts.append(
            f"当前时间: {now.strftime(f'%Y-%m-%d %H:%M:%S ({weekday_names[now.weekday()]})')} (CST/UTC+8)"
        )

        if self._workspace_manager:
            dynamic_parts.append(f"当前工作区: {chat_id[:12]}")
            dynamic_parts.append(
                "read_file / write_file / edit_file / apply_patch "
                "四个文件工具可访问整个 workspaces/ 根目录（管理员权限）。搜索文件内容请使用 exec + rg。"
            )

        return static_prompt + "\n\n" + "\n\n".join(dynamic_parts)

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
