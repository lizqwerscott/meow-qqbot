"""Agent Engine — 会话管理、AI 编排、工具执行、自动复读检测

全局单例，独立于 BotEngine/WebSocket 生命周期。
"""

import asyncio
import itertools
import json
import logging
from datetime import datetime, timezone, timedelta
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set

from core.ai_service import AIService
from core.context_manager import ChatContextManager
from core.cost_tracker import CostTracker
from core.emoji import EmojiManager
from core.message import InputMessage
from core.nickname_manager import NicknameManager
from core.template_manager import TemplateManager
from core.tools import ToolExecutor, ToolContext
from core.tools.definitions import (
    EMOJI_TOOLS,
    SEARCH_USER_TOOL,
    SEARCH_MEMORY_TOOL,
    SEARCH_RELATION_TOOL,
    MARK_IMPORTANT_TOOL,
    SKILL_TOOLS,
)

_log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# SessionTaskManager — 每会话队列 + 锁
# ════════════════════════════════════════════════════════════

class SessionTaskManager:
    """管理每个 chat_id 的异步队列和锁，实现会话级隔离。"""

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._running: Set[str] = set()
        self._lock = asyncio.Lock()

    async def get_queue(self, chat_id: str) -> asyncio.Queue:
        async with self._lock:
            if chat_id not in self._queues:
                self._queues[chat_id] = asyncio.Queue()
            return self._queues[chat_id]

    async def get_lock(self, chat_id: str) -> asyncio.Lock:
        async with self._lock:
            if chat_id not in self._locks:
                self._locks[chat_id] = asyncio.Lock()
            return self._locks[chat_id]

    async def try_start_consumer(self, chat_id: str) -> bool:
        async with self._lock:
            if chat_id in self._running:
                return False
            self._running.add(chat_id)
            return True

    async def mark_consumer_done(self, chat_id: str):
        async with self._lock:
            self._running.discard(chat_id)

    def get_queue_sizes(self) -> Dict[str, int]:
        return {cid: q.qsize() for cid, q in self._queues.items() if q.qsize() > 0}

    async def cleanup_session(self, chat_id: str):
        async with self._lock:
            self._queues.pop(chat_id, None)
            self._locks.pop(chat_id, None)
            self._running.discard(chat_id)

    async def cleanup_all(self):
        async with self._lock:
            self._queues.clear()
            self._locks.clear()
            self._running.clear()


# ════════════════════════════════════════════════════════════
# AgentEngine — 核心业务引擎
# ════════════════════════════════════════════════════════════

class AgentEngine:
    """
    核心业务引擎。

    管理所有会话的短期记忆（通过 ChatContextManager）、消息队列、
    AI 调用和工具执行。内部使用 chat_id 级的锁和队列实现会话隔离。
    """

    def __init__(
        self,
        ai_service: AIService,
        template_manager: TemplateManager,
        context_manager: ChatContextManager,
        bot_id: str,
        admin_id: List[str],
        openai_config: dict,
        nickname_manager: Optional[NicknameManager] = None,
        emoji_manager: Optional[EmojiManager] = None,
        everos_memory: Optional[Any] = None,
        search_top_k: int = 3,
        skill_managers: Optional[Any] = None,
        max_tool_rounds: int = -1,
        cost_tracker: Optional[CostTracker] = None,
        context_window: int = 1000000,
    ):
        self.ai_service = ai_service
        self._max_tool_rounds = max_tool_rounds
        self.template_manager = template_manager
        self.context_manager = context_manager
        self._context_window = context_window
        self._bot_id = bot_id
        self._admin_id = admin_id
        self._openai_config = openai_config

        self._nm = nickname_manager
        self.emoji_manager = emoji_manager
        self.media_uploader = None
        self.multimodal_service = None
        self._api_client = None

        self.everos = everos_memory
        self._search_top_k = search_top_k
        self._skill_managers = skill_managers
        self.cost_tracker = cost_tracker or CostTracker()

        self.tool_executor = ToolExecutor(
            emoji_manager=emoji_manager,
            everos=everos_memory,
            bot_id=bot_id,
            nickname_manager=nickname_manager,
            skill_managers=skill_managers,
            admin_ids=admin_id,
        )

        self._auto_replied: Dict[str, str] = {}

        self._processed_ids: OrderedDict[str, bool] = OrderedDict()
        self._max_processed_ids = 1000

        self._consumer_tasks: Set[asyncio.Task] = set()

        self.session_manager = SessionTaskManager()

        _log.info("AgentEngine 已初始化")

    # ── 懒注入 ──

    def set_media_uploader(self, media_uploader: Any):
        self.media_uploader = media_uploader
        self.tool_executor.set_media_uploader(media_uploader)
        _log.info("AgentEngine: MediaUploader 已注入")

    def set_api_client(self, api_client: Any):
        self._api_client = api_client
        self.tool_executor.set_api_client(api_client)
        _log.info("AgentEngine: QQApiClient 已注入")

    def set_multimodal_service(self, multimodal_service: Any):
        self.multimodal_service = multimodal_service

    def set_emoji_manager(self, emoji_manager: EmojiManager):
        self.emoji_manager = emoji_manager

    def set_nickname_manager(self, nm: NicknameManager):
        self._nm = nm
        self.tool_executor.set_nickname_manager(nm)

    # ── 消息分发 ──

    async def dispatch(
        self,
        input_message: InputMessage,
        reply_callback: Callable,
        get_user_nickname: Callable[[str], str],
    ) -> None:
        if input_message.id in self._processed_ids:
            _log.debug(f"跳过重复消息: {input_message.id}")
            return
        self._processed_ids[input_message.id] = True
        self._processed_ids.move_to_end(input_message.id)
        if len(self._processed_ids) > self._max_processed_ids:
            self._processed_ids.popitem(last=False)

        chat_id = input_message.chat_id

        user_nickname = get_user_nickname(input_message.sender_id)

        content_with_context = input_message.content
        if input_message.replied_content:
            if input_message.replied_author:
                context_prefix = f"[正在回复 {input_message.replied_author}: {input_message.replied_content}]"
            else:
                context_prefix = f"[正在回复: {input_message.replied_content}]"
            if content_with_context:
                content_with_context = context_prefix + "\n" + content_with_context
            else:
                content_with_context = context_prefix

        await self.context_manager.add_user_message_async(
            chat_id,
            content_with_context,
            input_message.id,
            sender_id=input_message.sender_id,
            name=user_nickname,
        )

        if self.everos:
            await self.everos.add_message(
                session_id=chat_id,
                sender_id=input_message.sender_id,
                sender_name=user_nickname,
                content=content_with_context,
            )
            keywords = ["我喜欢", "我讨厌", "我叫", "我是", "我的", "记住", "我不喜欢", "我有", "别忘了"]
            if any(k in input_message.content for k in keywords):
                await self.everos.flush(session_id=chat_id)

        if not (input_message.content.startswith("[表情:") or input_message.content == "[自定义表情]"):
            reply_content = await self._check_auto_reply_duplicate(input_message)
            if reply_content is not None:
                _log.info(
                    f"检测到重复消息 [{chat_id}]，自动复读: {reply_content[:30]}"
                )
                await reply_callback(
                    chat_id=chat_id,
                    content=reply_content,
                    message_id=input_message.id,
                    is_group=True,
                )
                await self.context_manager.add_assistant_message_async(
                    chat_id, reply_content, input_message.id,
                )
                self._auto_replied[chat_id] = reply_content
                return

        if input_message.is_group and not input_message.is_at_mention:
            if not input_message.content.startswith("猫猫"):
                _log.debug(f"跳过 AI 回复（非@且非猫猫开头）: {input_message.content[:30]}")
                return

        queue = await self.session_manager.get_queue(chat_id)
        await queue.put(input_message)

        should_start = await self.session_manager.try_start_consumer(chat_id)
        if should_start:
            task = asyncio.create_task(
                self._consumer(chat_id, reply_callback, get_user_nickname)
            )
            self._consumer_tasks.add(task)
            task.add_done_callback(self._consumer_tasks.discard)
            _log.debug(f"已启动会话 {chat_id[:12]}.. 的消费者")

    # ── 会话消费者循环 ──

    async def _consumer(
        self,
        chat_id: str,
        reply_callback: Callable,
        get_user_nickname: Callable[[str], str],
    ) -> None:
        session_lock = await self.session_manager.get_lock(chat_id)

        async with session_lock:
            while True:
                try:
                    queue = await self.session_manager.get_queue(chat_id)
                    input_message = await asyncio.wait_for(
                        queue.get(), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    break

                try:
                    await self._process_message(
                        input_message, reply_callback, get_user_nickname
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    _log.error(f"消费者处理消息 {input_message.id} 时出错: {e}")
                    try:
                        await reply_callback(
                            chat_id=chat_id,
                            content="抱歉，处理您的消息时出现了问题，请稍后再试。",
                            message_id=input_message.id,
                            is_group=input_message.is_group,
                        )
                    except Exception:
                        pass

        await self.session_manager.mark_consumer_done(chat_id)

    # ── 常量文本块（供模板使用） ──

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

    async def _process_message(
        self,
        input_message: InputMessage,
        reply_callback: Callable,
        get_user_nickname: Callable[[str], str],
    ) -> None:
        chat_id = input_message.chat_id
        is_group = input_message.is_group
        user_nickname = get_user_nickname(input_message.sender_id)

        # ── 1. Token 阈值触发 compaction ──
        compacted = await self.context_manager.get_context_async(chat_id)
        compacted_result, compact_usage = await compacted.compact_history_if_needed(self.ai_service)
        if compact_usage:
            self.cost_tracker.record_turn(chat_id, self.ai_service.model, compact_usage)

        # ── 2. 确定可用工具 / 能力状态 ──
        has_emojis = self.emoji_manager is not None and self.emoji_manager.count_emojis() > 0
        if is_group and self._nm:
            has_users = any(
                k != self._bot_id for k in self._nm.nicknames
            ) or any(
                k != self._bot_id for k in self._nm.auto_nicknames
            )
        else:
            has_users = False

        tools_to_use = []
        if has_emojis:
            tools_to_use.extend(EMOJI_TOOLS)
        if has_users:
            tools_to_use.extend(SEARCH_USER_TOOL)
        if self.everos:
            tools_to_use.extend(SEARCH_MEMORY_TOOL)
            tools_to_use.extend(SEARCH_RELATION_TOOL)
            tools_to_use.extend(MARK_IMPORTANT_TOOL)
        if self._skill_managers and self._skill_managers.has_skills:
            tools_to_use.extend(SKILL_TOOLS)
        tools_to_use = tools_to_use or None

        # ── 3. 静态 system prompt（包含所有不变的引导说明） ──
        skill_intro = self._skill_managers.get_skill_system_intro() if (self._skill_managers and self._skill_managers.has_skills) else ""
        memory_desc = self._MEMORY_SYSTEM_DESC if self.everos else ""

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
        history = compacted.get_history_as_dicts()

        messages = [{"role": "system", "content": static_prompt}]
        messages.extend(history)

        # ── 5. 动态上下文 → 末尾单独一个 system 消息（仅放变化的内容） ──
        dynamic_parts = []

        # 技能条目列表（高优先级——仅 XML，不包含原则说明）
        if self._skill_managers and self._skill_managers.has_skills:
            entries = self._skill_managers.get_skill_entries_block()
            if entries:
                dynamic_parts.append(entries)

        # 记忆上下文（查询 EverOS）
        memory_text = await self._build_memory_context(
            sender_id=input_message.sender_id,
            input_message=input_message,
        )
        if memory_text:
            dynamic_parts.append(memory_text)

        # 当前时间
        _tz = timezone(timedelta(hours=8))
        now = datetime.now(_tz)
        weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        time_info = now.strftime(f"%Y-%m-%d %H:%M:%S ({weekday_names[now.weekday()]})")
        dynamic_parts.append(f"当前时间: {time_info}")

        # 表情标签列表（动态）
        if has_emojis and self.emoji_manager:
            tags = self.emoji_manager.get_all_tags()
            if tags:
                dynamic_parts.append("可用表情标签：" + "、".join(tags))

        # 群友列表（动态）
        if has_users and self._nm:
            merged = self._nm.all_merged()
            others = {uid: name for uid, name in merged.items() if uid != self._bot_id}
            if others:
                lines = ["【群友列表】"]
                for uid, name in sorted(others.items(), key=lambda x: x[1]):
                    lines.append(f"- {name} (id: {uid[:12]}..)")
                dynamic_parts.append("\n".join(lines))

        if dynamic_parts:
            messages.append({
                "role": "system",
                "content": "\n\n".join(dynamic_parts),
            })

        _log.info(
            f"请求 AI messages:\n{json.dumps(messages, ensure_ascii=False, indent=2)}"
        )
        if tools_to_use:
            _log.info(f"本次请求注入 {len(tools_to_use)} 个工具: {[t['function']['name'] for t in tools_to_use]}")

        # ── 6. 工具调用循环 ──
        await self._execute_tool_calls(
            messages=messages,
            tools=tools_to_use,
            chat_id=chat_id,
            is_group=is_group,
            reply_to=input_message.id,
            reply_callback=reply_callback,
            sender_id=input_message.sender_id,
            get_user_nickname=get_user_nickname,
        )

        _log.info(f"消息处理完成: {input_message.id}")

    # ── 构建记忆上下文（返回字符串，不修改 messages） ──

    _DIRTY_PATTERNS = (
        "<available_skills", "<skill>", "<description>",
        "<name>", "【工具配合原则】", "【记忆系统】",
        "--- 技能系统 ---", "工具配合原则",
    )

    async def _build_memory_context(
        self,
        sender_id: str,
        input_message: InputMessage,
    ) -> str:
        """查询 EverOS 记忆，返回记忆上下文字符串，或空字符串。"""
        if not self.everos:
            return ""

        query = input_message.content.strip()
        if not query:
            return ""

        try:
            result = await self.everos.search(
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
                            if isinstance(v, str) and any(p in v for p in self._DIRTY_PATTERNS):
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
                    if any(p in summary for p in self._DIRTY_PATTERNS):
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

    # ════════════════════════════════════════════════════════════
    # 工具调用循环
    # ════════════════════════════════════════════════════════════

    async def _execute_tool_calls(
        self,
        messages: list,
        tools: Optional[list],
        chat_id: str,
        is_group: bool,
        reply_to: str,
        reply_callback: Callable,
        sender_id: str = "",
        get_user_nickname: Optional[Callable[[str], str]] = None,
    ) -> bool:
        """
        执行 AI 工具调用循环。

        AI 返回 → 若有 tool_calls → 执行 → 结果追加到 messages → 继续下一轮
        直到 AI 不再返回 tool_calls 或达到最大轮数。

        每轮 AI 返回的文本会立即发送并记录到上下文。
        """
        sent_emoji = False

        if self._max_tool_rounds == -1:
            _rounds: Any = itertools.count()
        else:
            _rounds = range(self._max_tool_rounds)

        for round_idx in _rounds:
            message, usage = await self.ai_service.chat_completion_with_tools(
                messages=messages,
                tools=tools,
            )
            if usage:
                self.cost_tracker.record_turn(chat_id, self.ai_service.model, usage)

            if message is None:
                await reply_callback(chat_id, "AI 服务异常", reply_to, is_group)
                break

            response_text = message.content or ""
            tool_calls = message.tool_calls or []

            reasoning = getattr(message, "reasoning_content", None) or None
            if reasoning:
                _log.info(
                    f"[工具循环 第{round_idx + 1}轮 思考过程]\n{reasoning}"
                )

            _log.info(
                f"[工具循环 第{round_idx + 1}轮] "
                f"text={response_text[:50]!r}... "
                f"tool_calls={[tc.function.name for tc in tool_calls]}"
            )

            tool_calls_data = None
            if tool_calls:
                tool_calls_data = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]

            if response_text:
                await reply_callback(
                    chat_id=chat_id,
                    content=response_text,
                    message_id=reply_to,
                    is_group=is_group,
                )

            if response_text or tool_calls:
                await self.context_manager.add_assistant_message_async(
                    chat_id,
                    response_text or "",
                    reply_to,
                    tool_calls=tool_calls_data,
                    reasoning_content=reasoning,
                )

            if not tool_calls:
                break

            assistant_msg = {"role": "assistant", "content": response_text or None}
            reasoning_content = getattr(message, "reasoning_content", None)
            if reasoning_content:
                assistant_msg["reasoning_content"] = reasoning_content
            assistant_msg["tool_calls"] = tool_calls_data
            messages.append(assistant_msg)

            ctx = ToolContext(
                chat_id=chat_id,
                is_group=is_group,
                reply_to=reply_to,
                sender_id=sender_id,
                reply_callback=reply_callback,
            )

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    _log.warning(f"工具参数解析失败: {tc.function.arguments}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": "参数解析失败"}),
                    })
                    continue

                result = await self.tool_executor.execute(tc.function.name, args, ctx)
                if result.sent_emoji:
                    sent_emoji = True
                    await self.context_manager.add_assistant_message_async(
                        chat_id, "[助手发送了一个表情]", reply_to,
                    )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result.content,
                })
                await self.context_manager.add_tool_result_async(
                    chat_id, tc.function.name, result.content, tc.id,
                )

            if get_user_nickname:
                steer_msgs = await self._drain_steering_messages(
                    chat_id=chat_id,
                    current_sender_id=sender_id,
                    messages=messages,
                    get_user_nickname=get_user_nickname,
                )
                messages.extend(steer_msgs)

        return sent_emoji

    # ── Queue Steering ──

    async def _drain_steering_messages(
        self,
        chat_id: str,
        current_sender_id: str,
        messages: list,
        get_user_nickname: Callable[[str], str],
    ) -> List[dict]:
        """从会话队列中 drain 新消息，注入到当前工具循环。

        同一用户 → 不注入记忆（减少冗余调用）
        不同用户 → 走一次 _build_memory_context
        """
        queue = await self.session_manager.get_queue(chat_id)
        drained: List[InputMessage] = []
        while not queue.empty():
            try:
                drained.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        steered: List[dict] = []
        for msg in drained:
            nick = get_user_nickname(msg.sender_id) or msg.sender_id
            content = f"[来自 {nick} 的新消息]: {msg.content}"
            user_msg: dict = {"role": "user", "content": content}
            steered.append(user_msg)

            await self.context_manager.add_user_message_async(
                chat_id, content, msg.id,
                sender_id=msg.sender_id, name=nick,
            )

            if self.everos:
                await self.everos.add_message(
                    session_id=chat_id,
                    sender_id=msg.sender_id,
                    sender_name=nick,
                    content=content,
                )
                keywords = ["我喜欢", "我讨厌", "我叫", "我是", "我的", "记住", "我不喜欢", "我有", "别忘了"]
                if any(k in msg.content for k in keywords):
                    await self.everos.flush(session_id=chat_id)

            if msg.sender_id != current_sender_id and self.everos:
                memory_text = await self._build_memory_context(
                    sender_id=msg.sender_id,
                    input_message=msg,
                )
                if memory_text:
                    steered.append({
                        "role": "system",
                        "content": memory_text,
                    })

            queue.task_done()

        return steered

    # ── 辅助：用户目录文本（仅供 prompt 拼接）──

    def _get_user_catalog_text(self, max_users: int = 30) -> str:
        if not self._nm:
            return ""
        merged = self._nm.all_merged()

        if not merged:
            return ""

        lines = []
        for uid, nickname in list(merged.items())[:max_users]:
            lines.append(f"- {nickname} (id: {uid})")

        catalog = "当前群聊中已知的用户：\n" + "\n".join(lines)
        catalog += (
            "\n\n你可以使用 search_user 工具搜索用户获取其ID，"
            "然后在回复中使用 <qqbot-at-user id=\"用户ID\" /> 来@该用户。"
        )
        return catalog

    # ── 自动复读检查 ──

    async def _check_auto_reply_duplicate(self, input_message: InputMessage) -> Optional[str]:
        if not input_message.is_group:
            return None

        context = await self.context_manager.get_context_async(input_message.chat_id)
        user_msgs = [m for m in context.history if m.role == "user"]
        if len(user_msgs) < 2:
            return None

        last_content = user_msgs[-1].content
        prev_content = user_msgs[-2].content
        if last_content != prev_content:
            return None

        if self._auto_replied.get(input_message.chat_id) == last_content:
            return None

        return last_content

    # ── 统计 ──

    def get_stats(self) -> dict:
        stats: dict = {
            "queue_sizes": self.session_manager.get_queue_sizes(),
            "active_chats": self.context_manager.get_context_count(),
            "total_messages": self.context_manager.get_total_messages_count(),
        }

        if self.everos:
            health = self.everos.last_health_status
            if health:
                stats["everos_health"] = health
            else:
                stats["everos_health"] = {"status": "unknown", "error": "待检查"}
        else:
            stats["everos_health"] = {"status": "disabled"}

        g = self.cost_tracker.get_global_stats()
        stats["cost"] = {
            "turn_count": g.turn_count,
            "prompt_tokens": g.prompt_tokens,
            "completion_tokens": g.completion_tokens,
            "cache_hit_tokens": g.cache_hit_tokens,
            "cache_miss_tokens": g.cache_miss_tokens,
            "cache_hit_rate": round(g.cache_hit_rate * 100, 1),
            "total_cost": round(g.cost, 4),
        }
        return stats

    # ── 生命周期 ──

    async def stop(self):
        if self._consumer_tasks:
            for task in list(self._consumer_tasks):
                task.cancel()
            await asyncio.wait(self._consumer_tasks, timeout=5.0)
            self._consumer_tasks.clear()
        await self.session_manager.cleanup_all()

        if self.everos:
            await self.everos.close()

        _log.info("AgentEngine 已停止")
