"""PromptBuilder — AI 请求消息组装器

组装 AI 请求的 messages 列表：
- 静态 system prompt（模板渲染）
- 对话历史（上下文管理器）
- 动态 system 消息（记忆、时间、用户列表、技能条目）
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from core.engine.conversation_timeline import TimelineEvent
    from core.engine.turn_protocol_history import ProtocolEvent

from core.engine.batch_media_context import BatchMediaContext
from core.engine.delivery_prompt_contract import DeliveryPromptContract
from core.engine.model_context_transcript import (
    ModelContextScope,
    ModelContextSnapshot,
)
from core.engine.turn_protocol_history import TurnProtocolHistory
from core.message import InputMessage
from core.tools.policy import ChatContext, build_tools, format_task_tool_descriptions

from .dynamic_context import DynamicContextBuilder

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptBuildResult:
    messages: List[dict]
    tools: Optional[List[dict]]
    model_context_scope: Optional[ModelContextScope] = None
    model_context_fingerprint: Optional[str] = None

    def __iter__(self):
        yield self.messages
        yield self.tools


_MEMORY_SYSTEM_DESC = (
    "【记忆系统】\n"
    "你可以使用 memory 和 mark_important 工具管理长期记忆。\n"
    "\n"
    "**核心原则：不确定的先查记忆，不要猜测或编造！**\n"
    "- 涉及用户的背景、偏好、过往经历、约定 → 先 memory(action=search) 确认\n"
    "- 需要核实某个具体事实（生日、爱好、说过的话）→ 先 memory(action=search) 再回答\n"
    "- 搜不到相关信息时，如实告诉用户你不知道\n"
    "\n"
    "工具说明：\n"
    "- memory：action=search 查画像/经历/事实（指定 person_name 可查群友），action=relation 查两人关系\n"
    "- mark_important：记录重要信息到长期记忆。用户解释背景/喜好/事实时主动调用\n"
)

HEARTBEAT_MINIMAL_SYSTEM_PROMPT = (
    "你是一个群聊助手的心跳检查器，运行在独立的 Heartbeat Session 中。\n\n"
    "可用工具：read_file / write_file / edit_file / apply_patch / list_dir / search_content / find_files、记忆工具、exec、heartbeat_respond。\n"
    "文件路径均相对于 workspaces/ 根目录，例如 read_file(file_path='HEARTBEAT.md')。\n\n"
    "回应方式：\n"
    "- 如果没有任何需要关注的事项，调用 heartbeat_respond(notify=false)\n"
    '- 如果有需要提醒的事项，调用 heartbeat_respond(notify=true, notification_text="...")\n'
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
    '- 使用 heartbeat_respond(notify=true/false, notification_text="...") 工具回应，不要直接输出文本聊天\n'
    "- 不要闲聊、不要使用猫娘语气卖萌、不要加表情符号\n"
    "- 只汇报需要提醒管理员的事项\n"
    "- 如果没有需要关注的事项，调用 heartbeat_respond(notify=false) 静默结束"
)

SYSTEM_EVENT_SYSTEM_PROMPT = (
    "你是一个群聊助手的系统事件处理器，运行在独立的 System Event Session 中。\n\n"
    "你收到一个系统事件通知。请判断是否需要管理员或用户关注，并相应回应。\n\n"
    "可用工具：read_file / write_file / edit_file / apply_patch / list_dir / search_content / find_files、记忆工具、exec、heartbeat_respond。\n"
    "文件路径均相对于 workspaces/ 根目录。\n\n"
    "回应方式：\n"
    "- 如果无需关注，调用 heartbeat_respond(notify=false)\n"
    '- 如果需通知管理员，调用 heartbeat_respond(notify=true, notification_text="...")\n'
    '- 如果该事件的结果需要告知用户，调用 heartbeat_respond(notify=true, notification_text="...", deliver_to_user="用户的chat_id")\n'
    "- 不要闲聊或回复额外内容\n\n"
    "规则：\n"
    "1. 不要凭印象回答，不确定的先查记忆或文件\n"
    "2. 不要编造事实\n"
    "3. 只汇报真正需要关注的事项，不要过度打扰管理员"
)


class PromptBuilder:
    """AI 请求消息组装器。

    职责：
    1. 触发历史压缩（compaction）
    2. 根据可用能力确定工具列表
    3. 渲染静态 system prompt（模板）
    4. 拼接动态上下文（记忆、时间、表情标签、群友列表、技能条目）
    """

    def __init__(self, ctx, deps=None):
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
        self._deps = deps
        self.timeline = None
        self.model_context_transcript = None
        self.media_service = None
        self._dynamic_ctx_builder = DynamicContextBuilder(
            hindsight=self.hindsight,
            search_top_k=self._search_top_k,
            learners=self.learners,
            archive_manager=self._archive_manager,
            system_events=self._system_events,
            skill_managers=self._skill_managers,
            workspace_manager=self._workspace_manager,
            perm=self._perm,
            admin_ids=self._admin_ids,
            nm=self._nm,
            bot_id=self._bot_id,
            emoji_manager=self.emoji_manager,
        )

    async def build(
        self,
        chat_id: str,
        is_group: bool,
        user_nickname: str,
        sender_id: str,
        input_message: InputMessage,
        cost_tracker: Any = None,
        timeline_snapshot: Optional[Sequence["TimelineEvent"]] = None,
        protocol_snapshot: Optional[Sequence["ProtocolEvent"]] = None,
        model_context_snapshot: Optional[ModelContextSnapshot] = None,
        model_context_scope: Optional[ModelContextScope] = None,
        model_context_identity: Optional[Sequence[str]] = None,
        model_context_provider_identity: Optional[str] = None,
        delivery_contract: Optional[DeliveryPromptContract] = None,
        media_context: Optional[BatchMediaContext] = None,
    ) -> PromptBuildResult:
        """组装 AI 请求的 messages 列表。

        Returns:
            (messages, tools_to_use):
            - messages: OpenAI 格式的消息列表
            - tools_to_use: 本次可用的工具定义列表，或 None
        """
        # ── 1. Token 阈值触发 compaction ──
        try:
            _, compact_usage, _ = await self.context_manager.compact_history_if_needed(
                chat_id
            )
            if compact_usage and cost_tracker:
                cost_tracker.record_turn(
                    chat_id,
                    self.ai_service.model,
                    compact_usage,
                    metadata={"usage_kind": "compaction"},
                )
        except Exception as e:
            _log.warning("历史压缩失败 [%s..]: %s", chat_id[:12], e)

        # ── 1b. 防御：清理 context 历史中孤立的 tool_calls ──
        cleaned = await self.context_manager.remove_orphaned_tool_calls_async(chat_id)
        if cleaned:
            _log.info(f"清理了 {cleaned} 条孤立 tool_calls 消息 [{chat_id[:12]}..]")

        # ── 2. 确定可用工具 / 能力状态 ──
        has_emojis = (
            self.emoji_manager is not None and self.emoji_manager.count_emojis() > 0
        )
        has_tts = bool(self._tts_service)
        if is_group and self._nm:
            has_users = any(k != self._bot_id for k in self._nm.nicknames) or any(
                k != self._bot_id for k in self._nm.auto_nicknames
            )
        else:
            has_users = False

        role = self._perm.get_user_role(sender_id) if self._perm else None
        tools_to_use: Optional[List[dict]] = (
            build_tools(
                "normal",
                ChatContext(
                    has_emojis=has_emojis,
                    has_hindsight=bool(self.hindsight),
                    has_users=has_users,
                    is_group=is_group,
                    has_skills=bool(
                        self._skill_managers and self._skill_managers.has_skills
                    ),
                    has_workspace=bool(self._workspace_manager),
                    has_tasks=self._has_tasks,
                    has_tts=has_tts,
                    has_sub_agents=self._has_sub_agents,
                    has_learners=bool(self.learners),
                    has_web=bool(getattr(self._deps, "web", None)),
                    has_media=bool(
                        self.media_service and self.media_service.tools_enabled
                    ),
                ),
                deps=self._deps,
                role=role,
            )
            or None
        )

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

        if (
            model_context_scope is not None
            and self.model_context_transcript is not None
        ):
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "provider": type(self.ai_service).__name__,
                        "model": getattr(self.ai_service, "model", ""),
                        "model_identity": list(model_context_identity or ()),
                        "system": static_prompt,
                        "tools": tools_to_use or [],
                        "delivery_contract": (
                            delivery_contract.fingerprint(tools_to_use)
                            if delivery_contract is not None
                            else ""
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            try:
                model_context_scope = (
                    await self.model_context_transcript.ensure_generation(
                        model_context_scope,
                        fingerprint,
                        provider_identity=model_context_provider_identity,
                    )
                )
                model_context_snapshot = await self.model_context_transcript.snapshot(
                    model_context_scope
                )
            except Exception as exc:
                _log.warning("模型上下文 generation 检查失败: %s", exc)
                model_context_scope = None
                model_context_snapshot = None

        # ── 4. 完整历史 ──
        if model_context_snapshot is not None and timeline_snapshot is not None:
            history = self._model_context_history(
                model_context_snapshot, timeline_snapshot
            )
        elif timeline_snapshot is None:
            history = await self.context_manager.get_history_as_dicts_merged_async(
                chat_id
            )
            history = self._apply_timeline_snapshot(history, timeline_snapshot)
        else:
            history = self._timeline_history(timeline_snapshot)
        if protocol_snapshot:
            history.extend(TurnProtocolHistory.to_wire_messages(protocol_snapshot))

        messages: List[dict] = [{"role": "system", "content": static_prompt}]
        if delivery_contract is not None:
            messages.append(
                {
                    "role": "system",
                    "content": delivery_contract.render(tools_to_use),
                }
            )
        messages.extend(history)

        # ── 5. 动态上下文（委托给 DynamicContextBuilder） ──
        dynamic_text = await self._dynamic_ctx_builder.build(
            chat_id=chat_id,
            is_group=is_group,
            sender_id=sender_id,
            input_message=input_message,
            has_emojis=has_emojis,
            has_users=has_users,
        )
        if dynamic_text:
            messages.append(
                {
                    "role": "system",
                    "content": dynamic_text,
                }
            )

        if self.media_service and self.media_service.tools_enabled:
            media_rules = []
            if self.media_service.image_tools_enabled:
                media_rules.append(
                    "图片摘要足以回答时直接回答，不要重复调用工具；"
                    "需要确认细节、文字或人物关系时调用 image。"
                    "image 只能使用当前消息、引用消息或近期媒体目录中的 media:// 引用。"
                )
            if self.media_service.file_tools_enabled:
                media_rules.append(
                    "需要查看 TXT、Markdown、JSON 或 CSV 文件的完整内容时调用 read_file；"
                    "read_file 只能使用当前消息、引用消息或近期媒体目录中的 media:// 引用。"
                )
            if self.media_service.pdf_tools_enabled:
                media_rules.append(
                    "需要分析 PDF 文档时调用 pdf；"
                    "pdf 只能使用当前消息、引用消息或近期媒体目录中的 media:// 引用。"
                )
            media_rules.append("语音已自动转写时直接使用转写内容，不要重复调用工具。")
            media_rules.append("多张媒体指代不清时先询问用户，不要猜测。")
            messages.append(
                {
                    "role": "system",
                    "content": "媒体协作规则：" + "".join(media_rules),
                }
            )

        if media_context is not None:
            media_text = media_context.as_text()
            if media_text:
                messages.append({"role": "system", "content": media_text})

        if delivery_contract is not None:
            messages.append(
                {
                    "role": "system",
                    "content": delivery_contract.render_target(),
                }
            )

        # ── 6. 防御：清理孤立的 tool_calls（防止 compaction 拆散配对） ──
        from core.ai.protocol import ensure_messages_consistent

        ensure_messages_consistent(messages)

        _log.debug(
            f"请求 AI messages:\n{json.dumps(messages, ensure_ascii=False, indent=2)}"
        )
        if tools_to_use:
            _log.info(
                f"本次请求注入 {len(tools_to_use)} 个工具: "
                f"{[t['function']['name'] for t in tools_to_use]}"
            )

        return PromptBuildResult(
            messages=messages,
            tools=tools_to_use,
            model_context_scope=model_context_scope,
            model_context_fingerprint=(
                fingerprint
                if "fingerprint" in locals() and model_context_scope is not None
                else None
            ),
        )

    @classmethod
    def _model_context_history(
        cls,
        model_context_snapshot: ModelContextSnapshot,
        timeline_snapshot: Sequence["TimelineEvent"],
    ) -> List[dict]:
        history = model_context_snapshot.to_wire()
        if model_context_snapshot.events:
            inherited_event_ids = model_context_snapshot.source_event_ids
            current_events = tuple(
                event
                for event in timeline_snapshot
                if event.role == "user" and event.event_id not in inherited_event_ids
            )
        else:
            current_events = tuple(timeline_snapshot)
        history.extend(cls._timeline_history(current_events))
        return history

    @staticmethod
    def _timeline_history(
        timeline_snapshot: Sequence["TimelineEvent"],
    ) -> List[dict]:
        """Build visible prompt history from a frozen timeline snapshot.

        A provided empty snapshot is authoritative: it must not fall back to
        the shared legacy context, which may contain another turn's protocol.
        Protocol events are appended separately by the caller for the active
        turn only.
        """
        history: List[dict] = []
        for event in timeline_snapshot:
            if event.role not in {"user", "assistant"} or not event.content:
                continue
            if event.role == "user":
                name = event.sender_id or "未知"
                timestamp = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp)
                )
                content = f"[{name} 在 {timestamp}]: {event.content}"
            else:
                content = event.content
            history.append({"role": event.role, "content": content})
        return history

    @staticmethod
    def _apply_timeline_snapshot(
        history: List[dict], timeline_snapshot: Optional[Sequence["TimelineEvent"]]
    ) -> List[dict]:
        """Use the frozen timeline as authority for matching user message text.

        The legacy context still defines protocol ordering and display metadata.
        Replacing only message-id matches avoids duplicating user content or
        separating assistant tool calls from their result messages during the
        dual-write migration.
        """
        if not timeline_snapshot:
            return history
        user_events = {
            event.message_id: event
            for event in timeline_snapshot
            if event.role == "user" and event.message_id
        }
        if not user_events:
            return history
        projected: List[dict] = []
        for message in history:
            event = user_events.get(message.get("message_id"))
            if message.get("role") != "user" or event is None:
                projected.append(message)
                continue
            copied = dict(message)
            name = copied.get("name") or copied.get("sender_id") or event.sender_id
            timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp)
            )
            copied["raw_content"] = event.content
            copied["content"] = f"[{name} 在 {timestamp}]: {event.content}"
            projected.append(copied)
        return projected

    def _resolve_task_tools(
        self,
        tools_allow: Optional[List[str]],
    ) -> Tuple[Optional[List[dict]], str]:
        ctx = ChatContext(
            has_hindsight=bool(self.hindsight),
            has_workspace=bool(self._workspace_manager),
            has_skills=bool(self._skill_managers and self._skill_managers.has_skills),
            has_web=bool(getattr(self._deps, "web", None)),
        )
        tools_defs = (
            build_tools("task", ctx, deps=self._deps, tools_allow=tools_allow) or []
        )
        names = {t["function"]["name"] for t in tools_defs}
        desc_text = format_task_tool_descriptions(names)
        return tools_defs or None, desc_text

    async def build_task_messages(
        self,
        chat_id: str,
        prompt: str,
        tools_allow: Optional[List[str]] = None,
        media_context: Optional[BatchMediaContext] = None,
    ) -> Tuple[List[dict], Optional[List[dict]]]:
        """组装后台任务的 messages 列表（使用 task_chat.j2 模板）。

        工具集根据 tools_allow 动态决定。
        Returns:
            (messages, tools_to_use)
        """
        tools_to_use, tool_descriptions = self._resolve_task_tools(tools_allow)

        # 渲染 task prompt
        from datetime import datetime, timedelta, timezone

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
        if media_context is not None and media_context.as_text():
            messages.insert(
                1,
                {
                    "role": "system",
                    "content": "【已授权的媒体上下文】\n" + media_context.as_text(),
                },
            )
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
        tools_to_use = (
            build_tools(
                "heartbeat",
                ChatContext(
                    has_hindsight=bool(self.hindsight),
                    has_workspace=bool(self._workspace_manager),
                    has_tasks=self._has_tasks,
                    has_web=bool(getattr(self._deps, "web", None)),
                ),
                deps=self._deps,
            )
            or None
        )

        # ── system message ──
        if system_prompt_mode == "normal":
            system_prompt = self._build_normal_heartbeat_system_prompt(chat_id)
            system_prompt += HEARTBEAT_BEHAVIOR_BLOCK
        else:
            system_prompt = HEARTBEAT_MINIMAL_SYSTEM_PROMPT

        messages: List[dict] = [{"role": "system", "content": system_prompt}]

        # ── 系统事件（心跳触发时注入） ──
        if self._system_events:
            lease = self._system_events.claim_snapshot(system_event_key)
            if lease:
                lines = []
                for e in lease.events:
                    ts = time.strftime("%H:%M:%S", time.localtime(e.ts))
                    lines.append(f"System: [{ts}] {e.text}")
                lines.append("")
                lines.append(
                    '处理完成后，如果没有需要关注的事项，使用 heartbeat_respond(notify=false) 静默结束。如果有需要通知用户的事项，使用 heartbeat_respond(notify=true, notification_text="...")。'
                )
                messages.insert(
                    1,
                    {
                        "role": "system",
                        "content": "【系统事件（本次心跳触发）】\n" + "\n".join(lines),
                    },
                )

        # ── 聊天历史（作为独立消息对，仅 session=main） ──
        if session_mode == "main" and admin_chat_id:
            try:
                recent = (
                    await self.timeline.history(admin_chat_id, max_events=20)
                    if self.timeline is not None
                    else []
                )
                legacy_recent = await self.context_manager.get_chat_history_async(
                    admin_chat_id,
                    max_messages=20,
                )
                if self.timeline is not None:
                    await self.timeline.repair_from_legacy_history(
                        admin_chat_id, legacy_recent
                    )
                    recent = await self.timeline.history(admin_chat_id, max_events=20)
                if not recent:
                    recent = legacy_recent
                inserted = 0
                for msg in recent[-15:]:
                    role = msg.get("role")
                    if role in ("user", "assistant"):
                        content = msg.get("content") or ""
                        if content:
                            messages.append({"role": role, "content": content[:300]})
                            inserted += 1
                if inserted:
                    _log.info(
                        f"心跳历史注入: {inserted} 条消息（来自 {admin_chat_id[:12]}..）"
                    )
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
        weekday_names = [
            "星期一",
            "星期二",
            "星期三",
            "星期四",
            "星期五",
            "星期六",
            "星期日",
        ]
        dynamic_parts.append(
            f"当前时间: {now.strftime(f'%Y-%m-%d %H:%M:%S ({weekday_names[now.weekday()]})')} (CST/UTC+8)"
        )

        if self._workspace_manager:
            ws_root = str(self._workspace_manager.root_dir())
            dynamic_parts.append(f"当前工作区: {ws_root}/")
            dynamic_parts.append(
                "文件工具 (read_file / write_file / edit_file / apply_patch / list_dir) 和搜索工具 (search_content / find_files) 可访问整个 workspaces/ 目录（管理员权限）。"
                "文件路径请使用相对于工作区根目录的相对路径。"
                "如需访问外部文件，使用 .. 路径越界，系统会发送审批请求。"
            )

        return static_prompt + "\n\n" + "\n\n".join(dynamic_parts)

    async def build_system_event_messages(
        self,
        prompt: str,
        *,
        system_event_key: str = "system:events",
    ) -> Tuple[List[dict], Optional[List[dict]]]:
        """组装系统事件通知的 messages 列表。

        通用方法：cron/exec/background 等所有非人类事件都走此路径。
        工具集：heartbeat_respond + 记忆 + 文件 + exec，不加载聊天历史。
        """
        tools_to_use = (
            build_tools(
                "cron",
                ChatContext(
                    has_hindsight=bool(self.hindsight),
                    has_workspace=bool(self._workspace_manager),
                    has_tasks=self._has_tasks,
                    has_web=bool(getattr(self._deps, "web", None)),
                ),
                deps=self._deps,
            )
            or None
        )

        messages: List[dict] = [
            {"role": "system", "content": SYSTEM_EVENT_SYSTEM_PROMPT},
        ]

        # 系统事件注入
        if self._system_events:
            lease = self._system_events.claim_snapshot(system_event_key)
            if lease:
                lines = []
                for e in lease.events:
                    ts = time.strftime("%H:%M:%S", time.localtime(e.ts))
                    lines.append(f"System: [{ts}] {e.text}")
                lines.append("")
                lines.append(
                    "处理完成后，如果没有需要关注的事项，使用 heartbeat_respond(notify=false) 静默结束。"
                )
                messages.insert(
                    1,
                    {
                        "role": "system",
                        "content": "【系统事件（本次触发）】\n" + "\n".join(lines),
                    },
                )

        messages.append({"role": "user", "content": prompt or "[系统事件] 事件通知。"})

        return messages, tools_to_use

    async def build_memory_context(
        self, sender_id: str, input_message: InputMessage
    ) -> str:
        """构建记忆上下文字符串（公开，供 ToolLoop Queue Steering 使用）。"""
        return await self._dynamic_ctx_builder.memory_builder.build_memory_context(
            sender_id,
            input_message,
        )
