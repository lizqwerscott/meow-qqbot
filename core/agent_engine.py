"""Agent Engine — 会话管理、AI 编排、工具执行、自动复读检测

全局单例，独立于 BotEngine/WebSocket 生命周期。
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set

from qqbot_agent_sdk.constants import MEDIA_TYPE_IMAGE
from qqbot_agent_sdk.dto import MediaInfo, MessageToCreate, QQMessageType

from core.ai_service import AIService
from core.context_manager import ChatContextManager
from core.emoji import EmojiManager
from core.message import InputMessage
from core.template_manager import TemplateManager

_log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 工具（Function Calling）定义
# ════════════════════════════════════════════════════════════

EMOJI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_emoji",
            "description": "搜索表情图片。输入一个或多个标签，用空格分开。系统会匹配其中任意标签，按匹配数量排序返回。输入多个标签可以得到更精准的搜索结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于搜索的标签，多个标签用空格分隔，例如：开心 撒娇 猫娘。标签越具体搜索越精准。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_emoji",
            "description": "发送一个指定的表情图片到聊天中。需要提供通过 search_emoji 获取到的表情 hash。一条回复最多发送 1 个表情。",
            "parameters": {
                "type": "object",
                "properties": {
                    "emoji_hash": {
                        "type": "string",
                        "description": "表情的唯一标识 hash（完整 hash 或前 12 位短 hash），通过 search_emoji 获取",
                    },
                    "reason": {
                        "type": "string",
                        "description": "发送这个表情的原因或想表达的情绪，仅用于记录",
                    },
                },
                "required": ["emoji_hash", "reason"],
            },
        },
    },
]

SEARCH_USER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_user",
            "description": "根据昵称或昵称的一部分模糊搜索群里的用户。输入昵称的一部分（如'小'）即可找到所有匹配的人。返回用户的ID和昵称，获取到用户ID后你可以在回复中使用 <qqbot-at-user id=\"xxx\" /> 来@该用户。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，如用户名、昵称或ID的一部分",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ════════════════════════════════════════════════════════════
# EverOS 记忆工具
# ════════════════════════════════════════════════════════════

SEARCH_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "搜索记忆系统，可查询人物画像、过往经历、具体事实、"
                "用户偏好等任何信息。如果不指定 person_name，则搜索"
                "当前对话用户的记忆；如果指定 person_name，则搜索对应群友的记忆。"
                "person_name 支持模糊搜索，输入昵称的一部分也能匹配到。"
                "当需要了解某人的背景、确认某件事、查找说过的话时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题，例如 '他喜欢什么'、'上次提到的新显卡'、'生日是什么时候'",
                    },
                    "person_name": {
                        "type": "string",
                        "description": "要搜索的人名或昵称（可选）。支持模糊搜索，输入昵称的一部分（如'小'）也能匹配。不填则搜索当前对话用户。私聊中不可用。",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["hybrid", "agentic"],
                        "description": "检索方法。hybrid（默认）适合大多数情况；agentic 适合需要深度挖掘的复杂查询，会进行多轮检索。",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# ════════════════════════════════════════════════════════════
# 关系搜索工具
# ════════════════════════════════════════════════════════════

SEARCH_RELATION_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_relation",
            "description": (
                "搜索两个人之间的关系记忆。当你需要了解两个用户之间的关联、"
                "共同经历、相互评价、关系背景时使用。"
                "系统会同时搜索两个人的相关记忆以及当前用户的记载。"
                "人名支持模糊搜索，输入昵称的一部分即可匹配。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "person_a": {
                        "type": "string",
                        "description": "第一个人的人名或昵称，支持模糊搜索（部分匹配即可）。如果搜'我'或'自己'则代表当前说话者。",
                    },
                    "person_b": {
                        "type": "string",
                        "description": "第二个人的人名或昵称，支持模糊搜索（部分匹配即可）。如果搜'我'或'自己'则代表当前说话者。",
                    },
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题（可选）。例如'他们什么关系'、'一起做过什么'",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["hybrid", "agentic"],
                        "description": "检索方法。hybrid（默认）适合大多数情况；agentic 适合需要深度挖掘的复杂查询。",
                    },
                },
                "required": ["person_a", "person_b"],
            },
        },
    },
]

MARK_IMPORTANT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "mark_important",
            "description": (
                "当用户明确要求'记住这个'、'记好了'，或者当前讨论的内容"
                "非常重要时，调用此工具标记当前上下文，系统将立刻保存为长期记忆。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# ════════════════════════════════════════════════════════════
# SessionTaskManager — 每会话队列 + 锁
# ════════════════════════════════════════════════════════════

class SessionTaskManager:
    """管理每个 chat_id 的异步队列和锁，实现会话级隔离。"""

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._running: Set[str] = set()   # 正在运行消费者的会话
        self._lock = asyncio.Lock()        # 保护上述三个字典的操作

    async def get_queue(self, chat_id: str) -> asyncio.Queue:
        """获取或创建特定 chat_id 的队列。"""
        async with self._lock:
            if chat_id not in self._queues:
                self._queues[chat_id] = asyncio.Queue()
            return self._queues[chat_id]

    async def get_lock(self, chat_id: str) -> asyncio.Lock:
        """获取或创建特定 chat_id 的锁。"""
        async with self._lock:
            if chat_id not in self._locks:
                self._locks[chat_id] = asyncio.Lock()
            return self._locks[chat_id]

    async def try_start_consumer(self, chat_id: str) -> bool:
        """尝试启动消费者。如果该会话尚无消费者运行，标记为运行中并返回 True。"""
        async with self._lock:
            if chat_id in self._running:
                return False
            self._running.add(chat_id)
            return True

    async def mark_consumer_done(self, chat_id: str):
        """标记消费者已完成。"""
        async with self._lock:
            self._running.discard(chat_id)

    def get_queue_sizes(self) -> Dict[str, int]:
        """返回每个聊天队列的当前大小（用于状态命令）。"""
        return {cid: q.qsize() for cid, q in self._queues.items() if q.qsize() > 0}

    async def cleanup_session(self, chat_id: str):
        """清理指定会话的所有资源。"""
        async with self._lock:
            self._queues.pop(chat_id, None)
            self._locks.pop(chat_id, None)
            self._running.discard(chat_id)

    async def cleanup_all(self):
        """清理所有会话。"""
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
        emoji_manager: Optional[EmojiManager] = None,
        http_client: Optional[Any] = None,
        # EverOS 长期记忆
        everos_memory: Optional[Any] = None,
        search_top_k: int = 3,
    ):
        self.ai_service = ai_service
        self.template_manager = template_manager
        self.context_manager = context_manager
        self._bot_id = bot_id
        self._admin_id = admin_id
        self._openai_config = openai_config

        # 可选 / 惰性注入的组件
        self.http_client = http_client
        self.emoji_manager = emoji_manager
        self.media_uploader = None
        self.multimodal_service = None
        self._api_client = None

        # BotEngine 在 _on_ready 后设置昵称引用
        self.nicknames: Dict[str, str] = {}
        self.auto_nicknames: Dict[str, str] = {}

        # EverOS 长期记忆
        self.everos = everos_memory
        self._search_top_k = search_top_k

        # 自动复读检测
        self._auto_replied: Dict[str, str] = {}

        # 已处理的消息 ID 去重（防止 WS 重连重复消费）
        # 使用 OrderedDict 实现 LRU 淘汰——达上限时移除最早条目
        self._processed_ids: OrderedDict[str, bool] = OrderedDict()
        self._max_processed_ids = 1000

        # 跟踪所有运行中的消费者 Task，用于 stop() 时等待
        self._consumer_tasks: Set[asyncio.Task] = set()

        # 会话队列管理器
        self.session_manager = SessionTaskManager()

        _log.info("AgentEngine 已初始化")

    # ── 懒注入 ──

    def set_media_uploader(self, media_uploader: Any):
        """由 BotEngine 在 _on_ready 中调用。"""
        self.media_uploader = media_uploader
        _log.info("AgentEngine: MediaUploader 已注入")

    def set_api_client(self, api_client: Any):
        """设置 QQ API 客户端引用（用于发送富媒体消息）。"""
        self._api_client = api_client
        _log.info("AgentEngine: QQApiClient 已注入")

    def set_multimodal_service(self, multimodal_service: Any):
        """设置多模态服务引用。"""
        self.multimodal_service = multimodal_service

    def set_emoji_manager(self, emoji_manager: EmojiManager):
        """设置表情管理器引用。"""
        self.emoji_manager = emoji_manager

    def set_nicknames(self, nicknames: Dict[str, str], auto_nicknames: Dict[str, str]):
        """设置昵称字典引用。"""
        self.nicknames = nicknames
        self.auto_nicknames = auto_nicknames

    # ── 消息分发 ──

    async def dispatch(
        self,
        input_message: InputMessage,
        reply_callback: Callable,
        get_user_nickname: Callable[[str], str],
    ) -> None:
        """
        分发消息到对应会话的队列。

        1. 消息 ID 去重（WS 重连保护）
        2. 记录用户消息到上下文
        3. 自动复读检测
        4. 群聊非 @/非"猫猫"过滤
        5. 入队并启动消费者
        """
        # ── 消息去重（OrderedDict 实现 LRU 淘汰） ──
        if input_message.id in self._processed_ids:
            _log.debug(f"跳过重复消息: {input_message.id}")
            return
        self._processed_ids[input_message.id] = True
        self._processed_ids.move_to_end(input_message.id)
        if len(self._processed_ids) > self._max_processed_ids:
            self._processed_ids.popitem(last=False)

        chat_id = input_message.chat_id

        # ── 获取用户昵称 ──
        user_nickname = get_user_nickname(input_message.sender_id)

        # ── 格式化引用消息 ──
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

        # ── 记录用户消息到上下文 ──
        await self.context_manager.add_user_message_async(
            chat_id,
            content_with_context,
            input_message.id,
            sender_id=input_message.sender_id,
            name=user_nickname,
        )

        # ── 记录到 EverOS 长期记忆缓冲（per-session 队列保证顺序，不阻塞） ──
        if self.everos:
            await self.everos.add_message(
                session_id=chat_id,
                sender_id=input_message.sender_id,
                sender_name=user_nickname,
                content=content_with_context,
            )
            # 关键词匹配触发即时 flush
            keywords = ["我喜欢", "我讨厌", "我叫", "我是", "我的", "记住", "我不喜欢", "我有", "别忘了"]
            if any(k in input_message.content for k in keywords):
                await self.everos.flush(session_id=chat_id)

        # ── 自动复读检测（仅文本，非表情） ──
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

        # ── 群聊非 @ 且非"猫猫"开头 → 保留上下文但不进行 AI 回复 ──
        if input_message.is_group and not input_message.is_at_mention:
            if not input_message.content.startswith("猫猫"):
                _log.debug(f"跳过 AI 回复（非@且非猫猫开头）: {input_message.content[:30]}")
                return

        # ── 入队到会话队列 ──
        queue = await self.session_manager.get_queue(chat_id)
        await queue.put(input_message)

        # ── 如果该会话尚无消费者，启动一个 ──
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
        """
        单个会话的消费者循环。

        获取该会话的锁，确保同一会话串行处理。
        循环从队列取消息，调用 AI，发送回复。
        队列空超时后释放锁并标记消费者结束。
        """
        session_lock = await self.session_manager.get_lock(chat_id)

        async with session_lock:
            while True:
                try:
                    queue = await self.session_manager.get_queue(chat_id)
                    # 用超时获取，避免死等
                    input_message = await asyncio.wait_for(
                        queue.get(), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    # 队列为空超时 → 结束此会话消费者
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

        # 标记消费者结束
        await self.session_manager.mark_consumer_done(chat_id)

    # ── 单条消息处理 ──

    async def _process_message(
        self,
        input_message: InputMessage,
        reply_callback: Callable,
        get_user_nickname: Callable[[str], str],
    ) -> None:
        """处理单条消息：构建提示、调用 AI、执行工具循环。"""
        chat_id = input_message.chat_id
        is_group = input_message.is_group
        user_nickname = get_user_nickname(input_message.sender_id)

        # 从上下文管理器获取历史消息
        context_messages = await self.context_manager.get_chat_history_async(
            chat_id, max_messages=8
        )

        # 获取系统提示
        if is_group:
            system_prompt = self.template_manager.get_group_chat_prompt()
        else:
            system_prompt = self.template_manager.get_private_chat_prompt(
                user_nickname
            )

        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            *context_messages,
        ]

        # ── 确定要注入的工具 ──
        has_emojis = self.emoji_manager is not None and self.emoji_manager.count_emojis() > 0
        if is_group:
            has_users = any(
                k != self._bot_id for k in self.nicknames
            ) or any(
                k != self._bot_id for k in self.auto_nicknames
            )
        else:
            has_users = False

        tools_to_use = []
        if has_emojis:
            tools_to_use.extend(EMOJI_TOOLS)
        if has_users:
            tools_to_use.extend(SEARCH_USER_TOOL)
        # ── 如有 EverOS，注入记忆工具 ──
        if self.everos:
            tools_to_use.extend(SEARCH_MEMORY_TOOL)
            tools_to_use.extend(SEARCH_RELATION_TOOL)
            tools_to_use.extend(MARK_IMPORTANT_TOOL)
        tools_to_use = tools_to_use or None

        # 如果有工具，重新生成 system prompt 注入 flag
        if tools_to_use:
            emoji_tags = self.emoji_manager.get_all_tags() if has_emojis else []
            if is_group:
                system_prompt = self.template_manager.get_group_chat_prompt(
                    has_emojis=has_emojis,
                    has_users=has_users,
                    emoji_tags=emoji_tags,
                )
            else:
                system_prompt = self.template_manager.get_private_chat_prompt(
                    user_nickname,
                    has_emojis=has_emojis,
                    has_users=has_users,
                    emoji_tags=emoji_tags,
                )
            messages[0] = {"role": "system", "content": system_prompt}

        # ── 注入轻量 EverOS 记忆工具说明 ──
        if self.everos:
            desc = (
                "\n\n【记忆系统】\n"
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
                "- mark_important：将当前对话内容标记为重要，立即存入长期记忆\n"
            )
            messages[0]["content"] += desc

        _tz = timezone(timedelta(hours=8))
        now = datetime.now(_tz)
        weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        time_info = now.strftime(f"%Y-%m-%d %H:%M:%S ({weekday_names[now.weekday()]})")
        messages[0]["content"] += f"\n\n当前时间: {time_info}"

        # ── 自动记忆注入：透明地将当前用户的相关记忆注入到上下文 ──
        await self._auto_inject_memory_context(
            messages=messages,
            chat_id=chat_id,
            sender_id=input_message.sender_id,
            input_message=input_message,
        )

        # 打印请求消息（调试）
        _log.info(
            f"请求 AI messages:\n{json.dumps(messages, ensure_ascii=False, indent=2)}"
        )
        if tools_to_use:
            _log.info(f"本次请求注入 {len(tools_to_use)} 个工具: {[t['function']['name'] for t in tools_to_use]}")

        # 执行工具循环（内部已即时发送文本并记录上下文）
        await self._execute_tool_calls(
            messages=messages,
            tools=tools_to_use,
            chat_id=chat_id,
            is_group=is_group,
            reply_to=input_message.id,
            reply_callback=reply_callback,
            sender_id=input_message.sender_id,
        )

        _log.info(f"消息处理完成: {input_message.id}")

    # ── 自动记忆注入 ──

    async def _auto_inject_memory_context(
        self,
        messages: list,
        chat_id: str,
        sender_id: str,
        input_message: InputMessage,
    ) -> None:
        """自动检索当前用户的记忆，注入到 system prompt 作为背景知识。

        对 AI 和用户完全透明，只为 AI 提供上下文感知能力。
        如果检索结果为空或系统异常，跳过注入，不影响主流程。
        """
        if not self.everos:
            return

        query = input_message.content.strip()
        if not query:
            return

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
                return

            # 格式化记忆上下文
            parts = ["\n\n--- 相关记忆 ---"]

            if profiles:
                for p in profiles[:3]:
                    pd = p.get("profile_data", {})
                    if isinstance(pd, dict):
                        for k, v in pd.items():
                            parts.append(f"- [{k}]: {v}")

            if episodes:
                for e in episodes[:5]:
                    summary = e.get("summary", "") or e.get("episode", "")
                    if summary:
                        parts.append(f"- {summary[:300]}")

            parts.append("--- 相关记忆结束 ---")

            memory_text = "\n".join(parts)
            messages[0]["content"] += memory_text

            _log.info(
                f"自动记忆注入: chat={chat_id[:16]}.. sender={sender_id[:16]}.. "
                f"注入{len(episodes)}条经历, {len(profiles)}条画像"
            )
        except Exception as e:
            _log.warning(f"自动记忆注入失败: {e!r}")

    # ════════════════════════════════════════════════════════════
    # 工具调用
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
    ) -> bool:
        """
        执行 AI 工具调用循环。

        AI 返回 → 若有 tool_calls → 执行 → 结果追加到 messages → 继续下一轮
        直到 AI 不再返回 tool_calls 或达到最大轮数。

        每轮 AI 返回的文本会立即发送并记录到上下文。

        Returns:
            sent_emoji: 是否成功发送了表情
        """
        sent_emoji = False
        MAX_ROUNDS = 5

        for round_idx in range(MAX_ROUNDS):
            message = await self.ai_service.chat_completion_with_tools(
                messages=messages,
                tools=tools,
            )

            if message is None:
                await reply_callback(chat_id, "AI 服务异常", reply_to, is_group)
                break

            response_text = message.content or ""
            tool_calls = message.tool_calls or []

            _log.info(
                f"[工具循环 第{round_idx + 1}轮] "
                f"text={response_text[:50]!r}... "
                f"tool_calls={[tc.function.name for tc in tool_calls]}"
            )

            if response_text:
                await reply_callback(
                    chat_id=chat_id,
                    content=response_text,
                    message_id=reply_to,
                    is_group=is_group,
                )
                await self.context_manager.add_assistant_message_async(
                    chat_id, response_text, reply_to
                )

            if not tool_calls:
                break  # AI 不再调用工具，结束循环

            # 将带 tool_calls 的 assistant 消息追加到 messages
            assistant_msg = {"role": "assistant", "content": response_text or None}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
            messages.append(assistant_msg)

            # 处理每个 tool_call
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

                if tc.function.name == "search_emoji":
                    _log.info(
                        f"[工具调用] search_emoji 输入: query={args.get('query', '')!r}"
                    )
                    result = self._execute_search_emoji(args)
                    _log.info(f"[工具调用] search_emoji 输出: {result[:200]}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                elif tc.function.name == "send_emoji":
                    _log.info(
                        f"[工具调用] send_emoji 输入: "
                        f"emoji_hash={args.get('emoji_hash', '')!r}, "
                        f"reason={args.get('reason', '')!r}"
                    )
                    result_content, success = await self._execute_send_emoji(
                        args, chat_id, is_group, reply_to, reply_callback,
                    )
                    _log.info(
                        f"[工具调用] send_emoji 输出: success={success}, "
                        f"result={result_content[:200]}"
                    )
                    if success:
                        sent_emoji = True
                        await self.context_manager.add_assistant_message_async(
                            chat_id, "[助手发送了一个表情]", reply_to,
                        )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_content,
                    })

                elif tc.function.name == "search_user":
                    _log.info(
                        f"[工具调用] search_user 输入: query={args.get('query', '')!r}"
                    )
                    result = self._execute_search_user(args)
                    _log.info(f"[工具调用] search_user 输出: {result[:200]}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                elif tc.function.name == "search_memory":
                    _log.info(
                        f"[工具调用] search_memory 输入: "
                        f"query={args.get('query', '')!r} "
                        f"person_name={args.get('person_name', '')!r} "
                        f"method={args.get('method', 'hybrid')!r}"
                    )
                    result = await self._execute_search_memory(
                        args, chat_id, is_group, sender_id
                    )
                    _log.info(
                        f"[工具调用] search_memory 输出: {result[:200]}"
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                elif tc.function.name == "mark_important":
                    _log.info("[工具调用] mark_important")
                    result = await self._execute_mark_important(chat_id)
                    _log.info(
                        f"[工具调用] mark_important 输出: {result[:200]}"
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                elif tc.function.name == "search_relation":
                    _log.info(
                        f"[工具调用] search_relation 输入: "
                        f"person_a={args.get('person_a', '')!r} "
                        f"person_b={args.get('person_b', '')!r} "
                        f"query={args.get('query', '')!r} "
                        f"method={args.get('method', 'hybrid')!r}"
                    )
                    result = await self._execute_search_relation(
                        args, chat_id, is_group, sender_id
                    )
                    _log.info(
                        f"[工具调用] search_relation 输出: {result[:200]}"
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                else:
                    _log.warning(f"未知工具调用: {tc.function.name}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"未知工具: {tc.function.name}"}),
                    })

        return sent_emoji

    # ── 工具执行器 ──

    def _execute_search_emoji(self, args: dict) -> str:
        """执行 search_emoji 工具，返回 JSON 字符串。"""
        if self.emoji_manager is None:
            return json.dumps({"error": "表情管理器未就绪"}, ensure_ascii=False)

        query = args.get("query", "").strip()
        if not query:
            return json.dumps({"error": "搜索关键词为空"}, ensure_ascii=False)

        results = self.emoji_manager.find_emojis(query, max_results=5)
        if not results:
            return json.dumps(
                {"error": "未找到匹配的表情", "query": query},
                ensure_ascii=False,
            )

        result_data = []
        for r in results:
            desc = r.get("user_description") or r.get("auto_description", "") or "(无描述)"
            tags = r.get("user_tags") or r.get("auto_tags", []) or []
            result_data.append({
                "hash": r["hash"][:12],
                "description": desc,
                "tags": tags,
            })

        return json.dumps(result_data, ensure_ascii=False)

    def _execute_search_user(self, args: dict) -> str:
        """执行 search_user 工具，返回 JSON 字符串。"""
        query = args.get("query", "").strip().lower()
        if not query:
            return json.dumps({"error": "搜索关键词为空"}, ensure_ascii=False)

        results = []
        seen = set()

        for source_dict, source_name in [(self.nicknames, "手动"), (self.auto_nicknames, "自动")]:
            for uid, nickname in source_dict.items():
                if uid in seen:
                    continue
                if query in nickname.lower() or query in uid.lower():
                    seen.add(uid)
                    results.append({
                        "id": uid,
                        "nickname": nickname,
                        "source": source_name,
                    })

        if not results:
            return json.dumps(
                {"error": "未找到匹配的用户", "query": query},
                ensure_ascii=False,
            )

        return json.dumps(results[:10], ensure_ascii=False)

    # ── EverOS 工具执行器 ──

    async def _execute_search_memory(
        self, args: dict, chat_id: str, is_group: bool, sender_id: str
    ) -> str:
        """执行 search_memory 工具，统一记忆搜索。"""
        query = (args.get("query") or "").strip()
        if not query:
            return json.dumps(
                {"error": "请输入搜索内容"}, ensure_ascii=False
            )

        person_name = (args.get("person_name") or "").strip()
        method = args.get("method", "hybrid")
        target_id = sender_id
        display_name = "当前用户"

        # 如果指定了人名，解析为 user_id
        if person_name:
            if not is_group:
                return json.dumps(
                    {"error": "私聊中无法搜索其他人"}, ensure_ascii=False
                )
            # 合并昵称字典
            merged = dict(self.nicknames)
            for uid, name in self.auto_nicknames.items():
                if uid not in merged:
                    merged[uid] = name

            matched_id = None
            for uid, nickname in merged.items():
                if person_name.lower() in nickname.lower() or person_name.lower() in uid.lower():
                    matched_id = uid
                    display_name = nickname
                    break

            if not matched_id:
                return json.dumps(
                    {"error": f"在昵称列表中找不到叫「{person_name}」的人"},
                    ensure_ascii=False,
                )
            target_id = matched_id

        if not self.everos:
            return json.dumps(
                {"error": "记忆系统未就绪"}, ensure_ascii=False
            )

        result = await self.everos.search(
            user_id=target_id,
            query=query,
            top_k=10,
            include_profile=True,
            method=method,
        )
        profiles = result.get("profiles", [])
        episodes = result.get("episodes", [])

        if not episodes and not profiles:
            return json.dumps(
                {"info": f"关于「{display_name}」暂无相关记忆记录"},
                ensure_ascii=False,
            )

        lines = [f"关于「{display_name}」的记忆检索结果："]
        if profiles:
            lines.append("【人物画像】")
            for p in profiles[:3]:
                pd = p.get("profile_data", {})
                if isinstance(pd, dict):
                    for k, v in pd.items():
                        lines.append(f"- {k}: {v}")
        if episodes:
            lines.append("【相关记忆】")
            for e in episodes[:5]:
                content = e.get("summary", "") or e.get("subject", "") or e.get("episode", "")
                mem_type = e.get("memory_type", "episode")
                if content:
                    lines.append(f"- [{mem_type}] {content[:200]}")
        return "\n".join(lines)

    # ── 辅助：将人名解析为 user_id ──

    def _resolve_person(
        self, raw: str, sender_id: str, is_group: bool
    ) -> tuple[Optional[str], str]:
        """解析人名字符串 → (user_id, display_name)。

        - "我" / "自己" / 说话者自己的昵称 → 返回 sender_id
        - 群聊中模糊匹配昵称 → 返回匹配到的 user_id
        - 私聊中非"我" → 返回 None
        - 找不到 → 返回 None
        """
        raw_lower = raw.strip().lower()
        if not raw_lower:
            return None, ""

        # 说话者自己的昵称
        sender_nick = self.nicknames.get(sender_id) or self.auto_nicknames.get(sender_id, "")

        # 判断是否指代自己
        if raw_lower in ("我", "自己", "myself"):
            return sender_id, sender_nick or "当前用户"
        if sender_nick and raw_lower == sender_nick.lower():
            return sender_id, sender_nick
        if raw_lower == sender_id.lower():
            return sender_id, sender_nick or "当前用户"

        if not is_group:
            return None, ""

        # 昵称模糊匹配
        merged = dict(self.nicknames)
        for uid, name in self.auto_nicknames.items():
            if uid not in merged:
                merged[uid] = name

        for uid, nickname in merged.items():
            if raw_lower in nickname.lower() or raw_lower in uid.lower():
                return uid, nickname

        return None, ""

    # ── 关系搜索工具执行器 ──

    async def _execute_search_relation(
        self, args: dict, chat_id: str, is_group: bool, sender_id: str
    ) -> str:
        """执行 search_relation 工具 — 多维关系搜索。

        搜索维度（自动去重）:
        1. person_a 的记忆中关于 person_b 的内容
        2. person_b 的记忆中关于 person_a 的内容（若与 person_a 不同）
        3. 当前说话者的记忆中关于两人的内容（若说话者不是 a 也不是 b）

        结果会清晰标注来源。
        """
        person_a_raw = (args.get("person_a") or "").strip()
        person_b_raw = (args.get("person_b") or "").strip()
        query = (args.get("query") or "").strip()
        method = args.get("method", "hybrid")

        if not person_a_raw or not person_b_raw:
            return json.dumps(
                {"error": "请指定两个人名或昵称"}, ensure_ascii=False
            )

        if not self.everos:
            return json.dumps(
                {"error": "记忆系统未就绪"}, ensure_ascii=False
            )

        # ── 解析两个人 ──
        a_id, a_name = self._resolve_person(person_a_raw, sender_id, is_group)
        b_id, b_name = self._resolve_person(person_b_raw, sender_id, is_group)

        if a_id is None:
            return json.dumps(
                {"error": f"找不到「{person_a_raw}」对应的用户"}, ensure_ascii=False
            )
        if b_id is None:
            return json.dumps(
                {"error": f"找不到「{person_b_raw}」对应的用户"}, ensure_ascii=False
            )

        # ── 构造各维度搜索 query ──
        def make_query(base: str, target_name: str) -> str:
            return f"{base} {target_name}" if base else target_name

        query_a = make_query(query, b_name)
        query_b = make_query(query, a_name)
        query_speaker = make_query(query, f"{a_name} {b_name}")

        # ── 确定搜索维度（自动去重） ──
        tasks = []
        task_labels = {}  # 用 id 标记每个任务的角色

        # 维度 1: person_a 的记忆
        tasks.append(self.everos.search(
            user_id=a_id, query=query_a, top_k=5, include_profile=True, method=method
        ))
        task_labels[len(tasks) - 1] = ("a", a_id, a_name)

        # 维度 2: person_b 的记忆（如果和 person_a 不同）
        if b_id != a_id:
            tasks.append(self.everos.search(
                user_id=b_id, query=query_b, top_k=5, include_profile=True, method=method
            ))
            task_labels[len(tasks) - 1] = ("b", b_id, b_name)

        # 维度 3: 当前说话者的记忆（如果说话者不是 a 也不是 b）
        if sender_id not in (a_id, b_id):
            tasks.append(self.everos.search(
                user_id=sender_id, query=query_speaker, top_k=5, include_profile=False, method=method
            ))
            task_labels[len(tasks) - 1] = ("speaker", sender_id, "当前用户")

        # ── 并行执行 ──
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # ── 组装结果 ──
        lines = [f"关于「{a_name}」和「{b_name}」的关系检索结果："]

        seen_episodes: Set[str] = set()  # 去重

        for idx, r in enumerate(raw_results):
            role, uid, label = task_labels[idx]

            if isinstance(r, Exception):
                _log.warning(f"search_relation {role}({uid}) 失败: {r}")
                continue

            profiles = r.get("profiles", [])
            episodes = r.get("episodes", [])

            if role == "a":
                # person_a 的画像
                if profiles:
                    lines.append(f"【{label} 的人物画像】")
                    for p in profiles[:3]:
                        pd = p.get("profile_data", {})
                        if isinstance(pd, dict):
                            for k, v in pd.items():
                                lines.append(f"- {k}: {v}")
                # person_a 记忆中关于 person_b 的内容
                if episodes:
                    lines.append(f"【{label} 记忆中关于 {b_name} 的内容】")
                    for e in episodes[:5]:
                        content = e.get("summary", "") or e.get("subject", "") or e.get("episode", "")
                        if content:
                            dedup_key = content[:100]
                            if dedup_key not in seen_episodes:
                                seen_episodes.add(dedup_key)
                                lines.append(f"- {content[:200]}")

            elif role == "b":
                # person_b 的画像
                if profiles:
                    lines.append(f"【{label} 的人物画像】")
                    for p in profiles[:3]:
                        pd = p.get("profile_data", {})
                        if isinstance(pd, dict):
                            for k, v in pd.items():
                                lines.append(f"- {k}: {v}")
                # person_b 记忆中关于 person_a 的内容
                if episodes:
                    lines.append(f"【{label} 记忆中关于 {a_name} 的内容】")
                    for e in episodes[:5]:
                        content = e.get("summary", "") or e.get("subject", "") or e.get("episode", "")
                        if content:
                            dedup_key = content[:100]
                            if dedup_key not in seen_episodes:
                                seen_episodes.add(dedup_key)
                                lines.append(f"- {content[:200]}")

            elif role == "speaker":
                # 当前说话者记忆中关于两人的内容
                if episodes:
                    lines.append(f"【你对两人的相关记载】")
                    for e in episodes[:5]:
                        content = e.get("summary", "") or e.get("subject", "") or e.get("episode", "")
                        if content:
                            dedup_key = content[:100]
                            if dedup_key not in seen_episodes:
                                seen_episodes.add(dedup_key)
                                lines.append(f"- {content[:200]}")

        # ── 全空时的兜底 ──
        if len(lines) == 1:
            lines.append("（未找到两人相关的记忆记录）")

        return "\n".join(lines)

    async def _execute_mark_important(self, chat_id: str) -> str:
        """执行 mark_important 工具，异步触发 flush。"""
        if not self.everos:
            return json.dumps(
                {"error": "记忆系统未就绪"}, ensure_ascii=False
            )
        await self.everos.flush(session_id=chat_id)
        return json.dumps(
            {
                "success": True,
                "message": "已标记当前对话为重要，正在整理记忆中。",
            },
            ensure_ascii=False,
        )

    def _get_user_catalog_text(self, max_users: int = 30) -> str:
        """生成 AI 可读的用户目录文本。"""
        merged = dict(self.nicknames)
        for uid, name in self.auto_nicknames.items():
            if uid not in merged:
                merged[uid] = name

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

    async def _execute_send_emoji(
        self,
        args: dict,
        chat_id: str,
        is_group: bool,
        reply_to: str,
        reply_callback: Callable,
    ) -> tuple[str, bool]:
        """执行 send_emoji 工具。"""
        if self.emoji_manager is None or self.media_uploader is None:
            result = json.dumps(
                {"success": False, "reason": "表情管理器或上传器未就绪"},
                ensure_ascii=False,
            )
            return result, False

        emoji_hash = (args.get("emoji_hash") or "").strip()
        if not emoji_hash:
            result = json.dumps(
                {"success": False, "reason": "未提供表情 hash"},
                ensure_ascii=False,
            )
            return result, False

        success, description, file_name, error = await self._send_emoji_by_hash(
            chat_id=chat_id,
            emoji_hash=emoji_hash,
            is_group=is_group,
            reply_to=reply_to,
        )

        if success:
            _log.info(f"表情已发送: {description}")
            result = json.dumps({
                "success": True,
                "description": description,
                "message": f"表情「{description}」已发送到聊天中",
            }, ensure_ascii=False)
            return result, True
        else:
            _log.warning(f"表情发送失败 [{emoji_hash[:12]}..]: {error}")
            result = json.dumps({
                "success": False,
                "reason": error or "发送失败",
                "suggestion": "可以搜索其他表情试试，或直接用文字表达",
            }, ensure_ascii=False)
            return result, False

    async def _send_emoji_by_hash(
        self,
        chat_id: str,
        emoji_hash: str,
        is_group: bool,
        reply_to: str,
    ) -> tuple[bool, str, str, str]:
        """上传并发送已缓存的 emoji 图片到聊天。"""
        if self.emoji_manager is None or self.media_uploader is None:
            return False, "", "", "表情管理器或上传器未就绪"

        # 查找完整记录
        if len(emoji_hash) < 12:
            record = self.emoji_manager.get_info(emoji_hash)
            if not record:
                return False, "", "", f"未找到表情: {emoji_hash}"
        else:
            record = self.emoji_manager.find_by_hash(emoji_hash)
            if not record:
                return False, "", "", f"未找到表情: {emoji_hash[:12]}.."

        full_hash = record["hash"]
        file_name = record.get("file_name", "")
        local_path = self.emoji_manager._emoji_dir / file_name

        if not local_path.exists():
            return False, "", file_name, f"本地文件缺失: {local_path}"

        desc = record.get("user_description") or record.get("auto_description", "") or "表情"
        chat_type = "group" if is_group else "c2c"

        try:
            file_info = await self.media_uploader.upload(
                chat_type=chat_type,
                chat_id=chat_id,
                source=str(local_path),
                file_type=MEDIA_TYPE_IMAGE,
                file_name=file_name,
            )

            msg_seq = self._api_client.next_msg_seq() if self._api_client else 0
            msg = MessageToCreate(
                msg_type=QQMessageType.RICH_MEDIA,
                msg_seq=msg_seq,
                msg_id=reply_to,
                media=MediaInfo(file_info=file_info),
            )

            if is_group:
                await self._api_client.post_group_message(chat_id, msg)
            else:
                await self._api_client.post_c2c_message(chat_id, msg)

            # 更新使用次数
            record = self.emoji_manager.get_info(full_hash)
            if record:
                count = record.get("used_count", 0) + 1
                self.emoji_manager.update_emoji(full_hash, used_count=count)

            _log.info(f"表情图片已发送 [{full_hash[:12]}..]: {desc}")
            return True, desc, file_name, ""

        except Exception as e:
            _log.error(f"发送表情图片失败 [{full_hash[:12]}..]: {e}")
            return False, desc, file_name, str(e)

    # ── 自动复读检查 ──

    async def _check_auto_reply_duplicate(self, input_message: InputMessage) -> Optional[str]:
        """检查群聊重复消息，返回需要复读的内容，或 None 表示不复读。"""
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
        """返回引擎统计信息（用于状态命令）。"""
        stats: dict = {
            "queue_sizes": self.session_manager.get_queue_sizes(),
            "active_chats": self.context_manager.get_context_count(),
            "total_messages": self.context_manager.get_total_messages_count(),
        }

        # EverOS 记忆系统健康状态（缓存，非阻塞）
        if self.everos:
            health = self.everos.last_health_status
            if health:
                stats["everos_health"] = health
            else:
                # 标记为“待刷新”，实际检查在 status command 里做
                stats["everos_health"] = {"status": "unknown", "error": "待检查"}
        else:
            stats["everos_health"] = {"status": "disabled"}

        return stats

    # ── 生命周期 ──

    async def stop(self):
        """清理所有会话资源和消费者任务。"""
        # 取消并等待所有正在运行的消费者
        if self._consumer_tasks:
            for task in list(self._consumer_tasks):
                task.cancel()
            await asyncio.wait(self._consumer_tasks, timeout=5.0)
            self._consumer_tasks.clear()
        await self.session_manager.cleanup_all()

        # 关闭 EverOS 客户端
        if self.everos:
            await self.everos.close()

        _log.info("AgentEngine 已停止")
