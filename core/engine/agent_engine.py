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
from typing import Any, Callable, List, Optional, Set

from core.managers.cost_tracker import CostTracker
from core.managers.emoji_manager import EmojiManager
from core.message import InputMessage, MessageType
from core.learners.base import sanitize_for_learners

from core.engine.context import EngineContext
from core.engine.prompt_builder import PromptBuilder
from core.managers.session_manager import SessionTaskManager
from core.tools.tool_loop import ToolLoop

_log = logging.getLogger(__name__)


class AgentEngine:
    """核心业务引擎 (Facade)。"""

    def __init__(self, ctx: EngineContext):
        self.ai_service = ctx.ai.ai_service
        self.template_manager = ctx.prompt.template_manager
        self.context_manager = ctx.mgmt.context_manager
        self._bot_id = ctx.sys.bot_id
        self._admin_id = list(ctx.sys.admin_ids)
        self.rule_router = ctx.ai.rule_router
        self.model_registry = ctx.ai.model_registry

        self._nm = ctx.prompt.nickname_manager
        self.emoji_manager = ctx.prompt.emoji_manager
        self.media_uploader = None
        self.multimodal_service = ctx.ai.multimodal_service
        self._tts_service = None
        self._api_client = None

        self.hindsight = ctx.memory.hindsight_memory
        self._skill_managers = ctx.prompt.skill_managers
        self.learners = ctx.prompt.learning_orchestrator
        self.cost_tracker = ctx.mgmt.cost_tracker or CostTracker()
        self._task_manager = ctx.bg.task_manager
        self._cron_job_manager = ctx.bg.cron_job_manager
        self._archive_manager = ctx.mgmt.archive_manager
        self._system_events = ctx.mgmt.system_events
        self._workspace_manager = ctx.mgmt.workspace_manager

        # ── 子模块 ──
        self.session_manager = SessionTaskManager()
        self.prompt_builder = PromptBuilder(ctx)
        self.tool_loop = ToolLoop(
            ctx,
            prompt_builder=self.prompt_builder,
            session_manager=self.session_manager,
        )

        # ── 子智能体管理器 ──
        self._sub_agent_manager = ctx.sub.sub_agent_manager
        if self._sub_agent_manager:
            self._sub_agent_manager.set_execute_callback(self.execute_background_task)

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

        # ── reply_callback（由 bootstrap 注入） ──
        self._reply_callback: Optional[Callable] = None

        self._register_builtin_hooks()

        _log.info("AgentEngine 已初始化")

    # ── 懒注入 ──

    def set_media_uploader(self, media_uploader: Any):
        self.media_uploader = media_uploader
        if hasattr(self, '_deps') and self._deps:
            self._deps.media_uploader.value = media_uploader
        _log.info("AgentEngine: MediaUploader 已注入")

    def set_reply_callback(self, callback: Callable) -> None:
        """注入真实消息投递回调（由 BotEngine 提供）。"""
        self._reply_callback = callback
        _log.info("AgentEngine: reply_callback 已注入")

    def set_router_model(self, router_model: Any):
        self.router_model = router_model
        _log.info("AgentEngine: RouterModel 已注入")

    def set_api_client(self, api_client: Any):
        self._api_client = api_client
        if hasattr(self, '_deps') and self._deps:
            self._deps.api_client.value = api_client
        _log.info("AgentEngine: QQApiClient 已注入")

    def set_multimodal_service(self, multimodal_service: Any):
        self.multimodal_service = multimodal_service

    def set_tts_service(self, tts_service: Any):
        self._tts_service = tts_service
        self.prompt_builder._tts_service = tts_service
        if hasattr(self, '_deps') and self._deps:
            self._deps.tts_service.value = tts_service

    def set_emoji_manager(self, emoji_manager: EmojiManager):
        self.emoji_manager = emoji_manager
        self.prompt_builder.emoji_manager = emoji_manager
        if hasattr(self, '_deps') and self._deps:
            self._deps.emoji_manager = emoji_manager

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

        await self.context_manager.record_chat_type(chat_id, is_group)

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

        if self._system_events:
            self._system_events.drain_non_heartbeat(chat_id)

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
        tools_allow: Optional[List[str]] = None,
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
            tools_allow: 后台任务可用工具列表（None=默认，["*"]=全部，[]=仅announce）

        Returns:
            (result_text, error_text)
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
                tools_allow=tools_allow,
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

    async def run_wake_turn(
        self,
        *,
        source: str = "system",
        intent: str = "event",
        reason: str = "",
        session_key: str = "",
        delivery_target: str = "",
        extra_prompt: str = "",
        system_event_key: str = "",
        messages: Optional[list[dict]] = None,
        tools: Optional[list[dict]] = None,
        timeout: int = 120,
    ) -> Any:
        """统一的 wake/heartbeat AI turn。

        可接收预制 messages/tools（由 WakeRunner 传入），
        也支持向后兼容的无 messages 模式（自动构建）。

        Args:
            source: 触发来源(interval/exec-event/cron/manual/system)
            session_key: chat_id 或 heartbeat:events
            delivery_target: 投递目标 chat_id
            extra_prompt: 额外 prompt（HEARTBEAT.md 内容等）
            messages: 预制消息列表（由 WakeRunner 传入）
            tools: 预制工具列表（由 WakeRunner 传入）
            system_event_key: 系统事件队列 key
        """
        from core.engine.wake_dispatcher import WakeResult as _WakeResult
        from core.tools.impl.heartbeat import heartbeat_response as _heartbeat_response
        from core.tools.tool_loop import _is_silent_reply_text as _check_silent
        import time as _time

        result = _WakeResult()

        if not self._reply_callback:
            _log.warning("reply_callback 未注入，无法投递 AI 回应")
            return result

        is_group = (self.context_manager.get_chat_type(session_key)
                    if self.context_manager else False) or False

        chat_id = session_key
        if source in ("interval", "manual") and not chat_id.startswith("heartbeat:"):
            chat_id = f"heartbeat:{int(_time.time())}"

        msg = InputMessage(
            id=f"wake_{chat_id}_{int(_time.time())}",
            sender_id="system",
            chat_id=chat_id,
            content=extra_prompt or "[系统事件]",
            is_group=is_group,
            is_at_mention=False,
        )

        captured: list[str] = []
        wake_resp: dict = {}

        token = _heartbeat_response.set(wake_resp)
        try:
            # 向后兼容：无预制 messages 时自己构建
            if messages is None:
                if source in ("interval", "manual"):
                    messages, tools = await self.prompt_builder.build_heartbeat_messages(
                        prompt=extra_prompt,
                        system_prompt_mode="minimal" if source == "interval" else "normal",
                        session_mode="isolated",
                        admin_chat_id=self._admin_id[0] if self._admin_id else "",
                        chat_id=chat_id,
                        system_event_key=system_event_key or "heartbeat:events",
                    )
                else:
                    messages, tools = await self.prompt_builder.build(
                        chat_id=session_key,
                        is_group=is_group,
                        user_nickname="系统",
                        sender_id="system",
                        input_message=msg,
                        cost_tracker=self.cost_tracker,
                    )

            model_chain = None
            if self.rule_router and self.model_registry:
                tier = self.rule_router.classify(extra_prompt or "[系统事件]")
                model_chain = self.model_registry.get_chain(tier) or None

            async def _capture(chat_id, content, message_id, is_group):
                captured.append(content)
            await asyncio.wait_for(
                self.tool_loop.run(
                    messages=messages, tools=tools or [],
                    chat_id=chat_id, is_group=is_group,
                    reply_to=msg.id,
                    reply_callback=_capture,
                    sender_id="system",
                    model_chain=model_chain,
                ),
                timeout=timeout,
            )

            # ── 通知决策（不再区分 source，统一通知文本） ──
            if wake_resp.get("notify"):
                text = wake_resp.get("notification_text", "").strip()
                if text:
                    result.notification_text = text
                    result.should_notify = True
            else:
                for reply in captured:
                    if not _check_silent(reply):
                        result.notification_text = reply
                        result.should_notify = True
                        break

            result.captured_replies = captured
        except Exception as e:
            _log.error("run_wake_turn 异常 [%s]: %s", source, e, exc_info=True)
            result.error = str(e)
        finally:
            _heartbeat_response.reset(token)

        return result

    def is_session_active(self, session_key: str) -> bool:
        """检查指定 session 是否当前在对话中（供 WakeRunner preflight 使用）。"""
        if session_key in (self.last_active_chat, "heartbeat:events"):
            return (time.time() - self.last_active_time) < 120
        return self.session_manager.has_active_consumer(session_key)

    async def trigger_event_response(self, chat_id: str) -> None:
        """保留：由 WakeManager 调用，转向 run_wake_turn。"""
        await self.run_wake_turn(
            source="system", intent="event",
            session_key=chat_id,
        )

    async def execute_heartbeat(
        self,
        prompt: str,
        session: str = "isolated",
        system_prompt_mode: str = "minimal",
        model_chain: Optional[List[str]] = None,
        chat_id: Optional[str] = None,
        system_event_key: str = "heartbeat:events",
        timeout: int = 120,
    ) -> tuple[bool, str | None]:
        """保留兼容接口，转向 run_wake_turn。"""
        if chat_id is None:
            import time as _t
            chat_id = f"heartbeat:{int(_t.time())}"

        result = await self.run_wake_turn(
            source="interval" if session == "isolated" else "manual",
            intent="scheduled",
            reason="定时心跳",
            session_key=chat_id,
            extra_prompt=prompt,
            system_event_key=system_event_key,
            timeout=timeout,
        )
        return result.should_notify, result.notification_text or None

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
