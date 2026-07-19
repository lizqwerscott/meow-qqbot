"""Agent Engine — 核心业务引擎 (Facade)

管理所有会话的短期记忆、消息队列、AI 调用和工具执行。
内部使用 chat_id 级的锁和队列实现会话隔离。

职责委派：
- SessionTaskManager  → session_manager.py
- PromptBuilder        → prompt_builder.py
- ToolLoop             → tool_loop.py
"""

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set

from core.ai.service import AIService
from core.managers.context_manager import ChatContextManager
from core.managers.cost_tracker import CostTracker
from core.managers.emoji_manager import EmojiManager
from core.message import InputMessage, MessageType
from core.managers.nickname_manager import NicknameManager
from core.managers.template_manager import TemplateManager
from core.tools import ToolExecutor, SubAgentManager
from core.learners.base import sanitize_for_learners
from core.learners.orchestrator import LearningOrchestrator

from core.engine.prompt_builder import PromptBuilder
from core.managers.session_manager import SessionTaskManager
from core.tools.tool_loop import ToolLoop

_log = logging.getLogger(__name__)


class AgentEngine:
    """核心业务引擎 (Facade)。"""

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
        hindsight_memory: Optional[Any] = None,
        search_top_k: int = 3,
        skill_managers: Optional[Any] = None,
        learning_orchestrator: Optional[LearningOrchestrator] = None,
        max_tool_rounds: int = -1,
        cost_tracker: Optional[CostTracker] = None,
        context_window: int = 1000000,
        task_manager: Optional[Any] = None,
        cron_job_manager: Optional[Any] = None,
        rule_router: Optional[Any] = None,
        model_registry: Optional[Any] = None,
        sub_agent_manager=None,
        permission_manager=None,
        archive_manager=None,
        system_events=None,
        workspace_manager=None,
    ):
        self.ai_service = ai_service
        self.template_manager = template_manager
        self.context_manager = context_manager
        self._bot_id = bot_id
        self._admin_id = admin_id
        self._openai_config = openai_config
        self.rule_router = rule_router
        self.model_registry = model_registry

        self._nm = nickname_manager
        self.emoji_manager = emoji_manager
        self.media_uploader = None
        self.multimodal_service = None
        self._api_client = None

        self.hindsight = hindsight_memory
        self._skill_managers = skill_managers
        self.learners = learning_orchestrator
        self.cost_tracker = cost_tracker or CostTracker()
        self._task_manager = task_manager
        self._cron_job_manager = cron_job_manager
        self._archive_manager = archive_manager
        self._system_events = system_events

        self.tool_executor = ToolExecutor(
            emoji_manager=emoji_manager,
            hindsight=hindsight_memory,
            bot_id=bot_id,
            nickname_manager=nickname_manager,
            skill_managers=skill_managers,
            learning_orchestrator=learning_orchestrator,
            admin_ids=admin_id,
            permission_manager=permission_manager,
            system_events=system_events,
        )

        # ── 工作区 ──
        self._workspace_manager = workspace_manager
        if workspace_manager:
            self.tool_executor.set_workspace_manager(workspace_manager)

        # ── 子模块 ──
        self.session_manager = SessionTaskManager()

        self.prompt_builder = PromptBuilder(
            template_manager=template_manager,
            context_manager=context_manager,
            ai_service=ai_service,
            bot_id=bot_id,
            nickname_manager=nickname_manager,
            emoji_manager=emoji_manager,
            skill_managers=skill_managers,
            hindsight_memory=hindsight_memory,
            search_top_k=search_top_k,
            admin_ids=admin_id,
            learning_orchestrator=self.learners,
            has_tasks=self._task_manager is not None,
            has_sub_agents=sub_agent_manager is not None,
            permission_manager=permission_manager,
            workspace_manager=workspace_manager,
            archive_manager=archive_manager,
            system_events=system_events,
        )

        self.tool_loop = ToolLoop(
            ai_service=ai_service,
            tool_executor=self.tool_executor,
            cost_tracker=self.cost_tracker,
            context_manager=context_manager,
            session_manager=self.session_manager,
            prompt_builder=self.prompt_builder,
            hindsight_memory=hindsight_memory,
            max_rounds=max_tool_rounds,
            model_registry=model_registry,
        )

        # ── 子智能体管理器 ──
        self._sub_agent_manager = sub_agent_manager
        if sub_agent_manager:
            sub_agent_manager.set_execute_callback(self.execute_background_task)
            self.tool_executor.set_sub_agent_manager(sub_agent_manager)

        # ── 消息钩子 ──
        self._message_hooks: list = []

        # ── 消息去重 ──
        self._processed_ids: OrderedDict[str, bool] = OrderedDict()
        self._max_processed_ids = 1000
        self._dedup_lock = asyncio.Lock()

        # ── 消费者管理 ──
        self._consumer_tasks: Set[asyncio.Task] = set()

        # ── 路由模型 / 活跃追踪 ──
        self.router_model = None
        self.last_active_chat: str = ""
        self.last_active_time: float = 0.0

        self._register_builtin_hooks()

        _log.info("AgentEngine 已初始化")

    # ── 懒注入 ──

    def set_media_uploader(self, media_uploader: Any):
        self.media_uploader = media_uploader
        self.tool_executor.set_media_uploader(media_uploader)
        _log.info("AgentEngine: MediaUploader 已注入")

    def set_router_model(self, router_model: Any):
        self.router_model = router_model
        _log.info("AgentEngine: RouterModel 已注入")

    def set_api_client(self, api_client: Any):
        self._api_client = api_client
        self.tool_executor.set_api_client(api_client)
        _log.info("AgentEngine: QQApiClient 已注入")

    def set_multimodal_service(self, multimodal_service: Any):
        self.multimodal_service = multimodal_service

    def set_emoji_manager(self, emoji_manager: EmojiManager):
        self.emoji_manager = emoji_manager
        self.prompt_builder.emoji_manager = emoji_manager

    def set_nickname_manager(self, nm: NicknameManager):
        self._nm = nm
        self.tool_executor.set_nickname_manager(nm)
        self.prompt_builder._nm = nm

    # ── 消息钩子系统 ──

    def add_message_hook(self, hook, priority: int = 100) -> None:
        self._message_hooks.append((priority, hook))
        self._message_hooks.sort(key=lambda x: x[0])
        _log.debug(f"消息钩子已注册 (priority={priority}, 共 {len(self._message_hooks)} 个)")

    def remove_message_hook(self, hook) -> None:
        before = len(self._message_hooks)
        self._message_hooks[:] = [(p, h) for p, h in self._message_hooks if h is not hook]
        if len(self._message_hooks) < before:
            _log.debug(f"消息钩子已注销 ({len(self._message_hooks)} 个)")

    def _register_builtin_hooks(self) -> None:
        from core.engine.duplicate_reply import DuplicateReplyDetector
        self._duplicate_reply = DuplicateReplyDetector(self.context_manager)
        self.add_message_hook(self._duplicate_reply.handle_message, priority=100)

    async def _run_hooks(self, input_message, reply_callback, get_user_nickname) -> None:
        for _, hook in list(self._message_hooks):
            try:
                if await hook(input_message, reply_callback, get_user_nickname):
                    return
            except Exception as e:
                _log.error(f"消息钩子执行异常: {e}", exc_info=True)

    # ── 消息分发 ──

    async def dispatch(
        self,
        input_message: InputMessage,
        reply_callback: Callable,
        get_user_nickname: Callable[[str], str],
    ) -> None:
        async with self._dedup_lock:
            if input_message.id in self._processed_ids:
                _log.debug(f"跳过重复消息: {input_message.id}")
                return
            self._processed_ids[input_message.id] = True
            self._processed_ids.move_to_end(input_message.id)
            if len(self._processed_ids) > self._max_processed_ids:
                self._processed_ids.popitem(last=False)

        chat_id = input_message.chat_id

        # 自动创建工作区目录
        if self._workspace_manager:
            self._workspace_manager.sandbox_dir(input_message.is_group, chat_id)

        # 记录活跃追踪（供心跳 busy 检测用）
        self.last_active_chat = chat_id
        self.last_active_time = time.time()

        user_nickname = get_user_nickname(input_message.sender_id)

        # ── 日期边界检查：跨天后归档旧会话（仅文本消息触发） ──
        if self._archive_manager and input_message.msg_type == MessageType.TEXT:
            try:
                result = await self._archive_manager.archive_if_stale(
                    chat_id, input_message.is_group
                )
                if result:
                    _log.info(
                        "已归档会话 [%s..]: 摘要=%s 回放=%d条",
                        chat_id[:12],
                        result.summary_path or "无",
                        result.replay_count,
                    )
            except Exception as e:
                _log.warning("归档失败 [%s..]: %s", chat_id[:12], e)

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
            name=input_message.sender_id,
        )

        if self.hindsight and input_message.msg_type != MessageType.CARD:
            hs_content = self._format_hindsight_content(
                content_with_context,
                input_message.sender_id,
                input_message.mentioned_ids,
                nm=self._nm,
            )
            await self.hindsight.add_message(
                session_id=chat_id,
                content=hs_content,
                sender_id=input_message.sender_id,
                context=self.hindsight.msg_type_to_context(input_message.msg_type),
                timestamp=input_message.timestamp,
                resources=input_message.resources,
            )

        # 学习系统观察（异步，不阻塞）
        if self.learners and input_message.msg_type != MessageType.CARD:
            text_for_learners = sanitize_for_learners(content_with_context)
            if text_for_learners:
                if input_message.replied_content:
                    lines = text_for_learners.split("\n", 1)
                    for prefix in ("猫猫", f"@{self._bot_id}"):
                        if len(lines) == 2 and lines[1].strip().startswith(prefix):
                            lines[1] = lines[1].strip()[len(prefix):].lstrip()
                            text_for_learners = "\n".join(lines)
                            break
                else:
                    text_stripped = text_for_learners.strip()
                    for prefix in ("猫猫", f"@{self._bot_id}"):
                        if text_stripped.startswith(prefix):
                            text_for_learners = text_stripped[len(prefix):].lstrip()
                            break

                await self.learners.on_message(
                    message_text=text_for_learners,
                    chat_id=chat_id,
                )

        if input_message.msg_type == MessageType.EMOJI:
            return

        needs_ai = True
        if input_message.is_group and not input_message.is_at_mention:
            if not input_message.content.startswith("猫猫"):
                needs_ai = False

        # ── 规则路由智能分级（ClawRouter 风格） ──
        if needs_ai and self.rule_router and self.model_registry:
            tier = self.rule_router.classify(input_message.content)
            model_chain = self.model_registry.get_chain(tier)
            input_message.model_chain = model_chain or None

            from core.rule_router import is_simple_enough_for_direct
            if tier == "simple" and is_simple_enough_for_direct(input_message.content):
                _log.info(
                    f"规则路由直接回复 (tier={tier}): "
                    f"chat={chat_id[:12]}.. content={input_message.content[:30]}"
                )
                simple_model = model_chain[0] if model_chain else None
                if simple_model:
                    from core.rule_router import SIMPLE_SYSTEM_PROMPT as _simple_prompt
                    reply = await self.model_registry.simple_chat(
                        model_name=simple_model,
                        messages=[
                            {"role": "system", "content": _simple_prompt},
                            {"role": "user", "content": input_message.content},
                        ],
                        max_tokens=200,
                    )
                    if reply:
                        await reply_callback(
                            chat_id, reply,
                            input_message.id, input_message.is_group,
                        )
                        return
                # fallback: 走 ToolLoop
        elif needs_ai and self.router_model:
            # 旧路由模型兼容（无 rule_router 时）
            decision = await self.router_model.route(
                content=input_message.content,
                chat_id=chat_id,
            )
            if decision.action == "direct":
                _log.info(
                    f"路由模型直接回复: chat={chat_id[:12]}.. "
                    f"content={input_message.content[:30]}"
                )
                await reply_callback(
                    chat_id, decision.response,
                    input_message.id, input_message.is_group,
                )
                return
            if decision.response != input_message.content:
                _log.info(
                    f"路由 escalate: {input_message.content[:30]} -> {decision.response[:50]}"
                )
                input_message.content = decision.response

        if needs_ai:
            queue = await self.session_manager.get_queue(chat_id)
            try:
                queue.put_nowait(input_message)
            except asyncio.QueueFull:
                _log.warning(f"会话 {chat_id[:12]}.. 队列已满，丢弃最早消息")
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                queue.put_nowait(input_message)
            should_start = await self.session_manager.try_start_consumer(chat_id)
            if should_start:
                task = asyncio.create_task(
                    self._consumer(chat_id, reply_callback, get_user_nickname)
                )
                self._consumer_tasks.add(task)
                task.add_done_callback(self._consumer_tasks.discard)
                _log.debug(f"已启动会话 {chat_id[:12]}.. 的消费者")
        else:
            await self._run_hooks(input_message, reply_callback, get_user_nickname)

    # ── 辅助方法 ──

    @staticmethod
    def _format_hindsight_content(content: str, sender_id: str, mentioned_ids: list, nm=None) -> str:
        """将 ID 格式的消息格式化为 Hindsight 的 [ID(别名)] 格式。"""
        aliases = nm.get_aliases(sender_id) if nm else []
        alias_str = "，".join(aliases) if aliases else sender_id
        prefix = f"[{sender_id}({alias_str})]: "

        for uid in mentioned_ids:
            u_aliases = nm.get_aliases(uid) if nm else []
            u_name = u_aliases[-1] if u_aliases else uid
            content = content.replace(f"@{uid}", f"@{uid}({u_name})")

        return prefix + content

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
                    except Exception as e:
                        _log.warning(f"向用户发送错误回复失败 [{chat_id}]: {e}")

        await self.session_manager.mark_consumer_done(chat_id)

    # ── 消息处理 ──

    async def _process_message(
        self,
        input_message: InputMessage,
        reply_callback: Callable,
        get_user_nickname: Callable[[str], str],
    ) -> None:
        chat_id = input_message.chat_id
        is_group = input_message.is_group
        user_nickname = get_user_nickname(input_message.sender_id)

        # 1-5. Prompt 组装（含 compaction + 工具确定 + 记忆注入）
        messages, tools_to_use = await self.prompt_builder.build(
            chat_id=chat_id,
            is_group=is_group,
            user_nickname=user_nickname,
            sender_id=input_message.sender_id,
            input_message=input_message,
            cost_tracker=self.cost_tracker,
        )

        # 6. 工具调用循环
        await self.tool_loop.run(
            messages=messages,
            tools=tools_to_use,
            chat_id=chat_id,
            is_group=is_group,
            reply_to=input_message.id,
            reply_callback=reply_callback,
            sender_id=input_message.sender_id,
            get_user_nickname=get_user_nickname,
            model_chain=input_message.model_chain,
        )

        _log.info(f"消息处理完成: {input_message.id}")

    # ── 后台任务执行 ──

    async def execute_background_task(
        self,
        chat_id: str,
        prompt: str,
        sender_id: str,
        is_group: bool = True,
        delivery_channel: str = "",
        reply_to_message_id: str = "",
    ) -> tuple[Optional[str], Optional[str]]:
        """在独立会话中执行后台任务。

        创建一个合成 InputMessage 并走完整的 PromptBuilding + ToolLoop 流程，
        但回复被捕获而非实际发送。

        Args:
            chat_id: 子智能体会话 ID（如 subagent:<uuid>）
            prompt: AI 执行指令
            sender_id: 发送者（如 "system"）
            is_group: 来源聊天是否为群聊（影响工具如 send_emoji 的接口选择）
            delivery_channel: 真实聊天 ID，用于 send_emoji 等需要真实 chat_id 的工具
            reply_to_message_id: 创建任务时的原始消息 ID，用于构造 msg_id

        Returns:
            (result_text, error_text)
            - result_text: AI 的最终文本回复
            - error_text: 如果有错误则返回错误描述
        """
        _log.info(
            f"开始后台任务: chat_id={chat_id[:20]}.. prompt={prompt[:60]}"
        )

        # 用于捕获回复的容器
        captured_replies: list[str] = []

        async def capturing_reply_callback(
            chat_id: str,
            content: str,
            message_id: str,
            is_group: bool,
        ) -> None:
            captured_replies.append(content)

        # 确保会话上下文存在
        import time

        # 创建合成 InputMessage
        msg = InputMessage(
            id=f"bg_{chat_id}_{int(time.time())}",
            sender_id=sender_id,
            chat_id=chat_id,
            content=prompt,
            is_group=is_group,
            is_at_mention=False,
        )

        try:
            # 写入用户消息到上下文（便于查阅，但不作为历史注入）
            await self.context_manager.add_user_message_async(
                chat_id, prompt, msg.id,
                sender_id=sender_id, name="system",
            )

            # 规则路由分级（同 _process_message 风格）
            model_chain = None
            if self.rule_router and self.model_registry:
                tier = self.rule_router.classify(prompt)
                model_chain = self.model_registry.get_chain(tier) or None

            # 构建 task 专用 messages（system + user）
            messages, tools_to_use = await self.prompt_builder.build_task_messages(
                chat_id=chat_id,
                prompt=prompt,
            )

            # 执行工具循环
            await self.tool_loop.run(
                messages=messages,
                tools=tools_to_use,
                chat_id=chat_id,
                is_group=is_group,
                reply_to=msg.id,
                reply_callback=capturing_reply_callback,
                sender_id=sender_id,
                delivery_channel=delivery_channel,
                reply_to_message_id=reply_to_message_id,
                model_chain=model_chain,
            )

            result = "\n".join(captured_replies) if captured_replies else None
            _log.info(
                f"后台任务完成: chat_id={chat_id[:20]}.. "
                f"result_len={len(result or '')}"
            )
            return result, None

        except asyncio.CancelledError:
            _log.warning(f"后台任务被取消: chat_id={chat_id[:20]}..")
            return None, "任务被取消"
        except Exception as e:
            _log.error(
                f"后台任务异常: chat_id={chat_id[:20]}.. error={e}",
                exc_info=True,
            )
            return None, str(e)

    async def execute_heartbeat(
        self,
        prompt: str,
        session: str = "isolated",
        system_prompt_mode: str = "minimal",
        model_chain: Optional[List[str]] = None,
        chat_id: Optional[str] = None,
        system_event_key: str = "heartbeat:events",
    ) -> tuple[bool, str | None]:
        """执行心跳检查的工具调用循环。

        工具调用：搜索记忆、检查文件，最终通过 heartbeat_respond 工具回应。

        HEARTBEAT.md 内容已由 HeartbeatManager 预读并注入 prompt。
        聊天历史由 build_heartbeat_messages 按独立消息对注入。

        Args:
            prompt: 完整的 user message（HEARTBEAT.md 内容 + 时间）
            session: "isolated"（无历史）或 "main"（含最近历史）
            system_prompt_mode: "normal"（复用完整角色卡 SP）或 "minimal"（极简 SP）
            model_chain: 模型链（如 ["modelscope/ds-flash", ...]），启用 fallback
            chat_id: 心跳使用的 chat_id。为 None 时自动生成 f"heartbeat:<timestamp>"
            system_event_key: 系统事件队列的 drain key，默认 "heartbeat:events"

        Returns:
            (should_notify, notification_text)
        """
        if chat_id is None:
            chat_id = f"heartbeat:{int(time.time())}"
        _log.info(f"开始心跳检查: chat_id={chat_id} session={session} mode={system_prompt_mode} prompt={prompt[:80]}")
        prompt = prompt or ""

        captured_replies: list[str] = []

        async def capturing_reply_callback(
            chat_id: str,
            content: str,
            message_id: str,
            is_group: bool,
        ) -> None:
            captured_replies.append(content)

        msg_id = f"hb_{int(time.time())}"

        try:
            messages, tools_to_use = await self.prompt_builder.build_heartbeat_messages(
                prompt=prompt,
                system_prompt_mode=system_prompt_mode,
                session_mode=session,
                admin_chat_id=self._admin_id[0] if self._admin_id else "",
                chat_id=chat_id,
                system_event_key=system_event_key,
            )

            self.tool_executor._heartbeat_response = {}

            await self.tool_loop.run(
                messages=messages,
                tools=tools_to_use,
                chat_id=chat_id,
                is_group=False,
                reply_to=msg_id,
                reply_callback=capturing_reply_callback,
                sender_id="system",
                get_user_nickname=lambda _: "系统",
                model_chain=model_chain,
            )

            # 优先检查 heartbeat_respond 工具响应
            hb_resp = self.tool_executor.consume_heartbeat_response()
            if hb_resp.get("notify"):
                text = hb_resp.get("notification_text", "").strip()
                if text:
                    _log.info(f"心跳 heartbeat_respond: 需要通知")
                    return True, text
                else:
                    _log.info("心跳 heartbeat_respond: notify=true 但内容为空，视为不通知")
                    return False, None

            if hb_resp:
                _log.debug("心跳 heartbeat_respond: notify=false，静默")
                return False, None

            # 降级回退：解析文本 HEARTBEAT_OK
            for reply in captured_replies:
                stripped = reply.strip()
                if stripped == "HEARTBEAT_OK":
                    _log.debug("心跳 HEARTBEAT_OK（文本降级），静默")
                    return False, None

                if stripped.startswith("HEARTBEAT_OK"):
                    stripped = stripped[len("HEARTBEAT_OK"):].strip()
                if stripped.endswith("HEARTBEAT_OK"):
                    stripped = stripped[:-len("HEARTBEAT_OK")].strip()

                if stripped and len(stripped) >= 5:
                    _log.info(f"心跳文本降级: 需要通知")
                    return True, stripped

            _log.debug("心跳无任何响应，静默")
            return False, None

        except asyncio.CancelledError:
            _log.warning("心跳检查被取消")
            return False, None
        except Exception as e:
            _log.error(f"心跳检查异常: {e}", exc_info=True)
            return False, None

    # ── 统计 ──

    def get_stats(self) -> dict:
        stats: dict = {
            "queue_sizes": self.session_manager.get_queue_sizes(),
            "active_chats": self.context_manager.get_context_count(),
            "total_messages": self.context_manager.get_total_messages_count(),
        }

        if self.hindsight:
            health = self.hindsight.last_health_status
            if health:
                stats["hindsight_health"] = health
            else:
                stats["hindsight_health"] = {"status": "unknown", "error": "待检查"}
        else:
            stats["hindsight_health"] = {"status": "disabled"}

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

        if self.learners:
            stats["learners"] = self.learners.get_stats()

        return stats

    # ── 生命周期 ──

    async def stop(self):
        if self._consumer_tasks:
            for task in list(self._consumer_tasks):
                task.cancel()
            await asyncio.wait(self._consumer_tasks, timeout=5.0)
            self._consumer_tasks.clear()
        await self.session_manager.cleanup_all()

        if self.hindsight:
            await self.hindsight.close()

        _log.info("AgentEngine 已停止")
