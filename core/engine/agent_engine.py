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
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Literal, Optional, Set
from uuid import uuid4

from core.engine.admission_outbox import AdmissionOutbox
from core.engine.context import EngineContext
from core.engine.prompt_builder import PromptBuilder
from core.learners.base import sanitize_for_learners
from core.managers.cost_tracker import CostTracker
from core.managers.emoji_manager import EmojiManager
from core.managers.session_manager import PendingInbound, SessionTaskManager
from core.message import InputMessage, MessageType, ResourceMeta
from core.tasks.wake_coalescer import WakeTurnResult
from core.tools.tool_loop import ToolLoop

_log = logging.getLogger(__name__)


TurnPromptFactory = Callable[[], Awaitable[tuple[list[dict], Optional[list[dict]]]]]


class _AdmissionAlreadyCommitted(Exception):
    """Signal an inbound redelivery that is already present in local history."""


@dataclass(frozen=True)
class AdmittedMessage:
    message_id: str
    prompt_message: dict
    additional_prompt_messages: tuple[dict, ...] = ()


@dataclass(frozen=True)
class _TurnRequest:
    chat_id: str
    sender_id: str
    is_group: bool
    reply_to: str
    route_text: str
    prompt_factory: TurnPromptFactory
    reply_callback: Callable
    get_user_nickname: Optional[Callable[[str], str]] = None
    delivery_channel: str = ""
    reply_to_message_id: str = ""
    model_chain: Optional[list[str]] = None
    tier: Optional[str] = None
    timeout: Optional[float] = None
    serialize_session: bool = True
    rollback_message_id: str = ""
    stream_callback: Optional[Callable[[str], Awaitable[None]]] = None
    track_tool_delivery: bool = True
    tool_reply_callback: Optional[Callable] = None
    tool_reply_names: frozenset[str] = frozenset()
    reply_state_callback: Optional[Callable[[bool], Awaitable[None]]] = None
    rollback_after_prompt_failure_only: bool = False
    steering_enabled: bool = False
    steering_admission_callback: Optional[
        Callable[[PendingInbound], Awaitable[Optional[AdmittedMessage]]]
    ] = None


@dataclass(frozen=True)
class _TurnResult:
    replies: tuple[str, ...] = field(default_factory=tuple)
    sent_emoji: bool = False
    text_committed: bool = False
    tool_text_delivered: bool = False
    final_reply_silent: bool = False


@dataclass(frozen=True)
class BackgroundTaskResult:
    result: Optional[str] = None
    error: Optional[str] = None
    tool_delivered: bool = False
    silent: bool = False

    def __iter__(self):
        yield self.result
        yield self.error


def pick_final_notification_reply(
    captured: list[str], is_silent: Callable[[str], bool]
) -> str:
    """从工具循环捕获的回复中挑选收尾通知文本。

    取最后一轮非静默回复（AI 的收尾输出）而非第一轮：
    多轮工具循环中首轮往往是"我去查一下"之类的中间话，
    真正要投递的结果在最后（回归修复：见 run_wake_turn 通知决策）。
    """
    for reply in reversed(captured):
        if not is_silent(reply):
            return reply
    return ""


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
        self.media_service = None
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

        # ── Session 模型绑定 ──
        from core.managers.session_binding import SessionBindingManager

        self._session_binding = SessionBindingManager()

        # ── 消息去重 ──
        self._processed_ids: OrderedDict[str, bool] = OrderedDict()
        self._max_processed_ids = 1000
        self._dedup_lock = asyncio.Lock()
        self._admitted_ids: OrderedDict[tuple[str, str], bool] = OrderedDict()
        self._admission_side_effect_status: OrderedDict[
            tuple[str, str], Dict[str, str]
        ] = OrderedDict()
        self._pending_ids: Set[str] = set()
        self._admission_in_progress: set[tuple[str, str]] = set()
        self._admission_outbox = AdmissionOutbox()
        self._outbox_task: Optional[asyncio.Task] = None

        # ── 消费者管理 ──
        self._consumer_tasks: Set[asyncio.Task] = set()
        self._consumer_callbacks: dict[str, tuple[Callable, Callable]] = {}

        # ── 工具依赖容器（由 bootstrap 注入） ──
        self._deps = None

        # ── 活跃追踪 ──
        self.last_active_chat: str = ""
        self.last_active_time: float = 0.0

        # ── reply_callback（由 bootstrap 注入） ──
        self._reply_callback: Optional[Callable] = None

        self._register_builtin_hooks()

        _log.info("AgentEngine 已初始化")

    # ── 懒注入 ──

    def set_media_uploader(self, media_uploader: Any):
        self.media_uploader = media_uploader
        if self._deps:
            self._deps.media_uploader.value = media_uploader
        _log.info("AgentEngine: MediaUploader 已注入")

    def set_reply_callback(self, callback: Callable) -> None:
        """注入真实消息投递回调（由 BotEngine 提供）。"""
        self._reply_callback = callback
        _log.info("AgentEngine: reply_callback 已注入")

    def set_api_client(self, api_client: Any):
        self._api_client = api_client
        if self._deps:
            self._deps.api_client.value = api_client
        _log.info("AgentEngine: QQApiClient 已注入")

    def set_multimodal_service(self, multimodal_service: Any):
        self.multimodal_service = multimodal_service

    def set_media_service(self, media_service: Any):
        self.media_service = media_service
        self.prompt_builder.media_service = media_service
        if self._deps:
            self._deps.media_service = media_service

    def set_tts_service(self, tts_service: Any):
        self._tts_service = tts_service
        self.prompt_builder._tts_service = tts_service
        if self._deps:
            self._deps.tts_service.value = tts_service

    def set_emoji_manager(self, emoji_manager: EmojiManager):
        self.emoji_manager = emoji_manager
        self.prompt_builder.emoji_manager = emoji_manager
        if self._deps:
            self._deps.emoji_manager = emoji_manager

    # ── 消息钩子系统 ──

    def add_message_hook(self, hook, priority: int = 100) -> None:
        self._message_hooks.append((priority, hook))
        self._message_hooks.sort(key=lambda x: x[0])
        _log.debug(
            f"消息钩子已注册 (priority={priority}, 共 {len(self._message_hooks)} 个)"
        )

    def remove_message_hook(self, hook) -> None:
        before = len(self._message_hooks)
        self._message_hooks[:] = [
            (p, h) for p, h in self._message_hooks if h is not hook
        ]
        if len(self._message_hooks) < before:
            _log.debug(f"消息钩子已注销 ({len(self._message_hooks)} 个)")

    def _register_builtin_hooks(self) -> None:
        from core.engine.duplicate_reply import DuplicateReplyDetector

        self._duplicate_reply = DuplicateReplyDetector(self.context_manager)
        self.add_message_hook(self._duplicate_reply.handle_message, priority=100)

    async def _run_hooks(
        self, input_message, reply_callback, get_user_nickname
    ) -> None:
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
        await self._process_admission_outbox()
        await self._ensure_outbox_worker()
        async with self._dedup_lock:
            if (
                input_message.id in self._processed_ids
                or input_message.id in self._pending_ids
            ):
                _log.debug("跳过重复消息: %s", input_message.id)
                return
            self._pending_ids.add(input_message.id)

        chat_id = input_message.chat_id
        if self._workspace_manager:
            self._workspace_manager.sandbox_dir(input_message.is_group, chat_id)
        self.last_active_chat = chat_id
        self.last_active_time = time.time()

        triggers_ai = (
            input_message.msg_type != MessageType.EMOJI
            and self._should_dispatch_to_ai(input_message)
        )
        needs_ai = triggers_ai
        if needs_ai and self.rule_router and self.model_registry:
            tier = self.rule_router.classify(input_message.content)
            input_message.tier = tier
            input_message.model_chain = self.model_registry.get_chain(tier) or None

        try:
            pending = await self._prepare_pending_inbound(
                input_message,
                dispatch_mode="agent" if triggers_ai else "passive",
            )
            enqueued = await self.session_manager.enqueue_with_dispatch_mode(
                chat_id, pending, triggers_ai=triggers_ai
            )
        except BaseException:
            async with self._dedup_lock:
                self._pending_ids.discard(input_message.id)
            raise
        async with self._dedup_lock:
            self._processed_ids[input_message.id] = True
            self._processed_ids.move_to_end(input_message.id)
            if len(self._processed_ids) > self._max_processed_ids:
                self._processed_ids.popitem(last=False)
            self._pending_ids.discard(input_message.id)
        if enqueued.dropped:
            _log.warning(
                "会话 inbox 已满，丢弃最早未准入消息 [%s..]: %s",
                chat_id[:12],
                enqueued.dropped.message.id,
            )
            async with self._dedup_lock:
                self._processed_ids.pop(enqueued.dropped.message.id, None)
        if enqueued.accepted:
            self._consumer_callbacks[chat_id] = (
                reply_callback,
                get_user_nickname,
            )
        _log.debug("消息已入 inbox [%s..]: %s", chat_id[:12], input_message.id)
        if enqueued.should_start_consumer:
            task = asyncio.create_task(
                self._consumer(
                    chat_id,
                    reply_callback,
                    get_user_nickname,
                    enqueued.consumer_token,
                )
            )
            self._consumer_tasks.add(task)
            task.add_done_callback(self._consumer_tasks.discard)
            _log.debug("已启动会话 %s.. 的消费者", chat_id[:12])

    async def _prepare_pending_inbound(
        self,
        input_message: InputMessage,
        *,
        dispatch_mode: Literal["agent", "passive"],
    ) -> PendingInbound:
        """Build immutable media/reply context without mutating shared chat history."""
        media_history_text = ""
        if self.media_service:
            try:
                media_context = await self.media_service.prepare_for_ai(input_message)
                media_history_text = "\n\n".join(
                    [*media_context.current_blocks, *media_context.replied_blocks]
                )
            except Exception as exc:
                _log.warning(
                    "媒体历史上下文构建失败 [%s..]: %s", input_message.chat_id[:12], exc
                )

        content = input_message.content
        media_refs = [
            resource.media_uri
            for resource in (*input_message.resources, *input_message.replied_resources)
            if resource.media_uri
        ]
        if media_refs:
            content += "\n[媒体引用: " + ", ".join(media_refs) + "]"
        if media_history_text:
            content += "\n" + media_history_text
        if input_message.replied_content:
            prefix = (
                f"[正在回复 {input_message.replied_author}: {input_message.replied_content}]"
                if input_message.replied_author
                else f"[正在回复: {input_message.replied_content}]"
            )
            content = f"{prefix}\n{content}" if content else prefix
        return PendingInbound(input_message, content, dispatch_mode)

    async def _admit_pending_message(
        self,
        pending: PendingInbound,
        *,
        source: Literal["initial", "steer", "passive"],
        get_user_nickname: Callable[[str], str],
    ) -> Optional[AdmittedMessage]:
        """Commit one leased inbound message to local history exactly once."""
        message = pending.message
        chat_id = message.chat_id
        admission_key = (chat_id, message.id)
        admitted_ids = getattr(self, "_admitted_ids", OrderedDict())
        if admission_key not in admitted_ids:
            outbox = getattr(self, "_admission_outbox", None)
            if outbox and await self._is_message_admitted(chat_id, message.id):
                await self._process_admission_outbox()
                _log.info(
                    "消息已存在于本地历史，跳过准入 [%s..]: id=%s",
                    chat_id[:12],
                    message.id,
                )
                return None

            prepared_new = False
            committed = False
            if outbox:
                self._admission_in_progress.add(admission_key)
            try:
                if outbox:
                    payload = self._build_side_effect_payload(pending)
                    prepared_new = await outbox.prepare(chat_id, message.id, payload)
                nickname = get_user_nickname(message.sender_id) or message.sender_id
                await self.context_manager.record_chat_type(chat_id, message.is_group)
                committed = (
                    await self.context_manager.add_user_message_async(
                        chat_id,
                        pending.prepared_content,
                        message.id,
                        sender_id=message.sender_id,
                        name=nickname,
                        timestamp=message.timestamp,
                    )
                    is not False
                )
                if not committed:
                    if outbox:
                        if prepared_new:
                            await outbox.cancel(chat_id, message.id)
                        else:
                            await self._process_admission_outbox()
                    _log.info(
                        "消息已存在于本地历史，跳过准入 [%s..]: id=%s",
                        chat_id[:12],
                        message.id,
                    )
                    return None
                if getattr(self, "_archive_manager", None):
                    try:
                        await self._archive_manager.archive_if_stale(
                            chat_id, message.is_group
                        )
                    except Exception as exc:
                        _log.warning("归档失败 [%s..]: %s", chat_id[:12], exc)
                admitted_ids[admission_key] = True
                admitted_ids.move_to_end(admission_key)
                while len(admitted_ids) > getattr(self, "_max_processed_ids", 1000):
                    admitted_ids.popitem(last=False)
                self._admitted_ids = admitted_ids
                if outbox:
                    await outbox.mark_ready(chat_id, message.id)
                    await self._process_admission_outbox()
                else:
                    await self._run_admission_side_effects(pending, admission_key)
                _log.debug(
                    "消息已准入 [%s..]: id=%s source=%s",
                    chat_id[:12],
                    message.id,
                    source,
                )
            except BaseException:
                if outbox and not committed and prepared_new:
                    await outbox.cancel(chat_id, message.id)
                raise
            finally:
                if outbox:
                    self._admission_in_progress.discard(admission_key)
        nickname = get_user_nickname(message.sender_id) or message.sender_id
        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(message.timestamp)
        )
        return AdmittedMessage(
            message_id=message.id,
            prompt_message={
                "role": "user",
                "content": f"[{nickname} 在 {timestamp}]: {pending.prepared_content}",
            },
        )

    async def _rollback_admission(self, pending: PendingInbound) -> None:
        admission_key = (pending.message.chat_id, pending.message.id)
        try:
            await self.context_manager.remove_message_if_async(
                pending.message.chat_id, "user", pending.message.id
            )
        finally:
            if getattr(self, "_admission_outbox", None):
                await self._admission_outbox.cancel(
                    pending.message.chat_id, pending.message.id
                )
            self._admitted_ids.pop(admission_key, None)

    def _build_side_effect_payload(self, pending: PendingInbound) -> dict:
        message = pending.message
        return {
            "chat_id": message.chat_id,
            "message_id": message.id,
            "content": pending.prepared_content,
            "sender_id": message.sender_id,
            "mentioned_ids": list(message.mentioned_ids),
            "timestamp": message.timestamp,
            "msg_type": str(message.msg_type),
            "replied_content": message.replied_content,
            "resources": [
                {
                    "resource_type": resource.resource_type,
                    "resource_id": resource.resource_id,
                    "media_id": resource.media_id,
                    "media_uri": resource.media_uri,
                    "hash": resource.hash,
                    "mime_type": resource.mime_type,
                    "filename": resource.filename,
                }
                for resource in message.resources
            ],
        }

    async def _is_message_admitted(
        self, chat_id: str, message_id: str
    ) -> Optional[bool]:
        if (chat_id, message_id) in self._admission_in_progress:
            return None
        get_history = getattr(self.context_manager, "get_chat_history_async", None)
        if get_history is None:
            return False
        history = await get_history(chat_id)
        return any(
            item.get("role") == "user" and item.get("message_id") == message_id
            for item in history
        )

    async def _ensure_outbox_worker(self) -> None:
        if self._outbox_task is None or self._outbox_task.done():
            self._outbox_task = asyncio.create_task(self._admission_outbox_worker())

    async def start(self) -> None:
        await self._process_admission_outbox()
        await self._resume_preserved_consumers()

    async def _resume_preserved_consumers(self) -> None:
        claims = await self.session_manager.claim_existing_consumers(
            set(self._consumer_callbacks)
        )
        for chat_id, consumer_token in claims:
            reply_callback, get_user_nickname = self._consumer_callbacks[chat_id]
            task = asyncio.create_task(
                self._consumer(
                    chat_id,
                    reply_callback,
                    get_user_nickname,
                    consumer_token,
                )
            )
            self._consumer_tasks.add(task)
            task.add_done_callback(self._consumer_tasks.discard)
            _log.info("已恢复会话 %s.. 的消费者", chat_id[:12])

    async def _admission_outbox_worker(self) -> None:
        while True:
            await self._process_admission_outbox()
            await asyncio.sleep(5)

    async def _process_admission_outbox(self) -> None:
        outbox = getattr(self, "_admission_outbox", None)
        if outbox is None:
            return
        await self._ensure_outbox_worker()
        try:
            await outbox.recover_prepared(self._is_message_admitted)
            await outbox.process(
                {
                    "hindsight": self._run_hindsight_side_effect,
                    "learner": self._run_learner_side_effect,
                }
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning("准入副作用 outbox 处理失败: %s", exc)

    async def _run_hindsight_side_effect(self, payload: dict) -> bool:
        if not getattr(self, "hindsight", None):
            return True
        if payload["msg_type"] == str(MessageType.CARD):
            return True
        resources = [
            ResourceMeta(**resource) for resource in payload.get("resources", [])
        ]
        return await self.hindsight.add_message(
            session_id=payload["chat_id"],
            content=self._format_hindsight_content(
                payload["content"],
                payload["sender_id"],
                payload.get("mentioned_ids", []),
                nm=getattr(self, "_nm", None),
            ),
            sender_id=payload["sender_id"],
            context=self.hindsight.msg_type_to_context(
                MessageType(payload["msg_type"])
            ),
            timestamp=payload.get("timestamp"),
            resources=resources,
            idempotency_key=payload.get("idempotency_key"),
        )

    async def _run_learner_side_effect(self, payload: dict) -> bool:
        if not getattr(self, "learners", None):
            return True
        if payload["msg_type"] == str(MessageType.CARD):
            return True
        text = sanitize_for_learners(payload["content"])
        if payload.get("replied_content"):
            lines = text.split("\n", 1)
            for prefix in ("猫猫", f"@{self._bot_id}"):
                if len(lines) == 2 and lines[1].strip().startswith(prefix):
                    lines[1] = lines[1].strip()[len(prefix) :].lstrip()
                    text = "\n".join(lines)
                    break
        else:
            stripped = text.strip()
            for prefix in ("猫猫", f"@{self._bot_id}"):
                if stripped.startswith(prefix):
                    text = stripped[len(prefix) :].lstrip()
                    break
        if not text:
            return True
        return await self.learners.on_message(
            message_text=text,
            chat_id=payload["chat_id"],
            sender_id=payload["sender_id"],
            message_id=payload.get("message_id"),
            idempotency_key=payload.get("idempotency_key"),
        )

    async def _run_admission_side_effects(
        self, pending: PendingInbound, admission_key: tuple[str, str]
    ) -> None:
        payload = self._build_side_effect_payload(pending)
        handlers = {
            "hindsight": self._run_hindsight_side_effect,
            "learner": self._run_learner_side_effect,
        }
        for effect_type, handler in handlers.items():
            try:
                succeeded = await handler(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                succeeded = False
                _log.warning(
                    "%s 准入副作用失败 [%s..]: %s",
                    effect_type,
                    pending.message.chat_id[:12],
                    exc,
                )
            status_map = getattr(self, "_admission_side_effect_status", None)
            if status_map is None:
                status_map = OrderedDict()
                self._admission_side_effect_status = status_map
            status_map.setdefault(admission_key, {})[effect_type] = (
                "succeeded" if succeeded else "failed"
            )

    def _should_dispatch_to_ai(self, input_message: InputMessage) -> bool:
        if not input_message.is_group:
            return True
        if input_message.is_at_mention:
            return True
        if input_message.content.startswith("猫猫"):
            return True
        return input_message.replied_author_id == self._bot_id

    # ── 辅助方法 ──

    @staticmethod
    def _format_hindsight_content(
        content: str, sender_id: str, mentioned_ids: list, nm=None
    ) -> str:
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
        consumer_token: int,
    ) -> None:
        try:
            while True:
                lease = await self.session_manager.claim_next_for_consumer(
                    chat_id, consumer_token
                )
                if lease is None:
                    if await self.session_manager.release_consumer_if_idle(
                        chat_id, consumer_token
                    ):
                        return
                    continue

                pending = lease.items[0]
                message = pending.message
                try:
                    if pending.dispatch_mode == "agent":
                        await self._process_message(
                            pending, reply_callback, get_user_nickname
                        )
                    else:
                        session_lock = await self.session_manager.get_lock(chat_id)
                        async with session_lock:
                            await self._admit_pending_message(
                                pending,
                                source="passive",
                                get_user_nickname=get_user_nickname,
                            )
                            if message.msg_type != MessageType.EMOJI:
                                await self._run_hooks(
                                    message, reply_callback, get_user_nickname
                                )
                    await self.session_manager.commit(lease, pending)
                except asyncio.CancelledError:
                    state = self.session_manager.get_message_state(chat_id, message.id)
                    admission_key = (chat_id, message.id)
                    if state == "admitted" or admission_key in getattr(
                        self, "_admitted_ids", {}
                    ):
                        await self.session_manager.commit(lease, pending)
                    else:
                        await self.session_manager.requeue_front(lease)
                    raise
                except Exception as exc:
                    admission_key = (chat_id, message.id)
                    state = self.session_manager.get_message_state(chat_id, message.id)
                    if state == "admitted" or admission_key in getattr(
                        self, "_admitted_ids", {}
                    ):
                        await self.session_manager.commit(lease, pending)
                    else:
                        await self.session_manager.fail(lease, pending)
                        async with self._dedup_lock:
                            self._processed_ids.pop(message.id, None)
                    _log.error(
                        "消费者处理消息 %s 时出错: %s",
                        message.id,
                        exc,
                        exc_info=True,
                    )
                    try:
                        await reply_callback(
                            chat_id=chat_id,
                            content="抱歉，处理您的消息时出现了问题，请稍后再试。",
                            message_id=message.id,
                            is_group=message.is_group,
                        )
                    except Exception as reply_err:
                        _log.warning(
                            "向用户发送错误回复失败 [%s..]: %s", chat_id[:12], reply_err
                        )
                    continue
        finally:
            replacement_token = await self.session_manager.handoff_consumer(
                chat_id, consumer_token
            )
            if replacement_token is not None:
                task = asyncio.create_task(
                    self._consumer(
                        chat_id,
                        reply_callback,
                        get_user_nickname,
                        replacement_token,
                    )
                )
                self._consumer_tasks.add(task)
                task.add_done_callback(self._consumer_tasks.discard)

    # ── 消息处理 ──

    async def _run_turn(self, request: _TurnRequest) -> _TurnResult:
        """运行一次统一的 prompt、路由、工具循环编排。"""
        replies: list[str] = []
        tool_text_delivered = False
        final_reply_silent = False

        async def _tool_delivery_callback() -> None:
            nonlocal tool_text_delivered
            tool_text_delivered = True

        async def _reply_state_callback(silent: bool) -> None:
            nonlocal final_reply_silent
            final_reply_silent = silent
            if request.reply_state_callback:
                await request.reply_state_callback(silent)

        async def _capturing_reply_callback(*args, **kwargs) -> None:
            content = kwargs.get("content")
            if content is None and len(args) > 1:
                content = args[1]
            if content is not None:
                replies.append(content)
            await request.reply_callback(*args, **kwargs)

        prompt_built = False

        async def _steering_admission_callback(
            pending: PendingInbound,
        ) -> Optional[AdmittedMessage]:
            return await request.steering_admission_callback(pending)

        async def _execute() -> _TurnResult:
            nonlocal prompt_built
            messages, tools = await request.prompt_factory()
            prompt_built = True
            model_chain = request.model_chain
            tier = request.tier
            if model_chain is None and tier is None:
                if self.rule_router and self.model_registry:
                    tier = self.rule_router.classify(request.route_text)
                    model_chain = self.model_registry.get_chain(tier) or None

            run = self.tool_loop.run(
                messages=messages,
                tools=tools or [],
                chat_id=request.chat_id,
                is_group=request.is_group,
                reply_to=request.reply_to,
                reply_callback=_capturing_reply_callback,
                tool_reply_callback=request.tool_reply_callback,
                tool_reply_names=request.tool_reply_names,
                reply_state_callback=_reply_state_callback,
                steering_enabled=request.steering_enabled,
                steering_admission_callback=(
                    _steering_admission_callback
                    if request.steering_admission_callback
                    else None
                ),
                inbound_message_ids=[request.reply_to] if request.reply_to else [],
                sender_id=request.sender_id,
                get_user_nickname=request.get_user_nickname,
                delivery_channel=request.delivery_channel,
                reply_to_message_id=request.reply_to_message_id,
                model_chain=model_chain,
                binding_manager=self._session_binding,
                tier=tier,
                stream_callback=request.stream_callback,
                delivery_state_callback=(
                    _tool_delivery_callback if request.track_tool_delivery else None
                ),
            )
            if request.timeout is not None:
                sent_emoji, text_committed = await asyncio.wait_for(
                    run, timeout=request.timeout
                )
            else:
                sent_emoji, text_committed = await run
            return _TurnResult(
                replies=tuple(replies),
                sent_emoji=sent_emoji,
                text_committed=text_committed,
                tool_text_delivered=tool_text_delivered,
                final_reply_silent=final_reply_silent,
            )

        async def _execute_with_rollback() -> _TurnResult:
            try:
                return await _execute()
            except asyncio.CancelledError:
                raise
            except _AdmissionAlreadyCommitted:
                raise
            except Exception:
                if request.rollback_message_id and (
                    not request.rollback_after_prompt_failure_only or not prompt_built
                ):
                    try:
                        remove_message = getattr(
                            self.context_manager,
                            "remove_message_if_async",
                            None,
                        )
                        if remove_message is not None:
                            await remove_message(
                                request.chat_id,
                                "user",
                                request.rollback_message_id,
                            )
                        else:
                            await self.context_manager.remove_last_user_message_if_async(
                                request.chat_id,
                                request.rollback_message_id,
                            )
                        self._admitted_ids.pop(
                            (request.chat_id, request.rollback_message_id), None
                        )
                        self._processed_ids.pop(request.rollback_message_id, None)
                    except Exception as rollback_err:
                        _log.warning(
                            "回滚上下文失败 [%s..]: %s",
                            request.chat_id[:12],
                            rollback_err,
                        )
                raise

        if not request.serialize_session:
            return await _execute_with_rollback()

        session_lock = await self.session_manager.get_lock(request.chat_id)
        async with session_lock:
            return await _execute_with_rollback()

    async def _process_message(
        self,
        pending: PendingInbound,
        reply_callback: Callable,
        get_user_nickname: Callable[[str], str],
    ) -> None:
        if isinstance(pending, InputMessage):
            pending = PendingInbound(pending, pending.content, "agent")
        input_message = pending.message
        chat_id = input_message.chat_id
        is_group = input_message.is_group
        user_nickname = get_user_nickname(input_message.sender_id)
        system_event_snapshot = []

        async def _build_prompt() -> tuple[list[dict], Optional[list[dict]]]:
            admitted = await self._admit_pending_message(
                pending,
                source="initial",
                get_user_nickname=get_user_nickname,
            )
            if admitted is None:
                raise _AdmissionAlreadyCommitted
            if self._system_events:
                peek_events = getattr(self._system_events, "peek_non_heartbeat", None)
                if peek_events:
                    system_event_snapshot.extend(peek_events(chat_id))
            return await self.prompt_builder.build(
                chat_id=chat_id,
                is_group=is_group,
                user_nickname=user_nickname,
                sender_id=input_message.sender_id,
                input_message=input_message,
                cost_tracker=self.cost_tracker,
            )

        async def _admit_steering(
            steering: PendingInbound,
        ) -> Optional[AdmittedMessage]:
            admitted = await self._admit_pending_message(
                steering,
                source="steer",
                get_user_nickname=get_user_nickname,
            )
            if admitted is None:
                return None
            additional_prompt_messages: tuple[dict, ...] = ()
            if steering.message.sender_id != input_message.sender_id:
                try:
                    memory_context = await self.prompt_builder.build_memory_context(
                        steering.message.sender_id, steering.message
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    _log.warning(
                        "转向消息记忆上下文构建失败 [%s..]: %s",
                        chat_id[:12],
                        exc,
                    )
                else:
                    if memory_context:
                        additional_prompt_messages = (
                            {"role": "system", "content": memory_context},
                        )
            return AdmittedMessage(
                message_id=admitted.message_id,
                prompt_message=admitted.prompt_message,
                additional_prompt_messages=additional_prompt_messages,
            )

        async def _stream_deliver(chunk: str) -> None:
            try:
                await reply_callback(
                    chat_id=chat_id,
                    content=chunk,
                    message_id=input_message.id,
                    is_group=is_group,
                )
            except Exception as cb_err:
                _log.warning("流式转发失败 [%s..]: %s", chat_id[:12], cb_err)

        try:
            await self._run_turn(
                _TurnRequest(
                    chat_id=chat_id,
                    sender_id=input_message.sender_id,
                    is_group=is_group,
                    reply_to=input_message.id,
                    route_text=input_message.content,
                    prompt_factory=_build_prompt,
                    reply_callback=reply_callback,
                    get_user_nickname=get_user_nickname,
                    model_chain=input_message.model_chain,
                    tier=input_message.tier,
                    rollback_message_id=input_message.id,
                    stream_callback=_stream_deliver,
                    steering_enabled=True,
                    steering_admission_callback=_admit_steering,
                    rollback_after_prompt_failure_only=True,
                )
            )
        except _AdmissionAlreadyCommitted:
            _log.info("消息已在历史中，跳过重复 turn: %s", input_message.id)

        if self._system_events and system_event_snapshot:
            self._system_events.drain_non_heartbeat(
                chat_id, expected_events=system_event_snapshot
            )
        _log.info("消息处理完成: %s", input_message.id)

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
    ) -> BackgroundTaskResult:
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
            BackgroundTaskResult，支持旧式二元解包。
        """
        _log.info(f"开始后台任务: chat_id={chat_id[:20]}.. prompt={prompt[:60]}")

        async def capturing_reply_callback(
            chat_id: str,
            content: str,
            message_id: str,
            is_group: bool,
        ) -> None:
            return None

        async def deliver_tool_reply_callback(
            chat_id: str,
            content: str,
            message_id: str,
            is_group: bool,
        ) -> None:
            if not self._reply_callback or not delivery_channel:
                raise RuntimeError("后台任务没有可用的消息投递目标")
            await self._reply_callback(
                chat_id=delivery_channel,
                content=content,
                message_id="",
                is_group=is_group,
            )

        # 创建合成 InputMessage
        msg = InputMessage(
            id=f"bg_{chat_id}_{uuid4().hex}",
            sender_id=sender_id,
            chat_id=chat_id,
            content=prompt,
            is_group=is_group,
            is_at_mention=False,
        )

        try:

            async def _build_prompt() -> tuple[list[dict], Optional[list[dict]]]:
                await self._admit_pending_message(
                    PendingInbound(msg, prompt, "agent"),
                    source="initial",
                    get_user_nickname=lambda _: "system",
                )
                return await self.prompt_builder.build_task_messages(
                    chat_id=chat_id,
                    prompt=prompt,
                    tools_allow=tools_allow,
                )

            turn = await self._run_turn(
                _TurnRequest(
                    chat_id=chat_id,
                    sender_id=sender_id,
                    is_group=is_group,
                    reply_to=msg.id,
                    route_text=prompt,
                    prompt_factory=_build_prompt,
                    reply_callback=capturing_reply_callback,
                    delivery_channel=delivery_channel,
                    reply_to_message_id=reply_to_message_id,
                    timeout=300,
                    tool_reply_callback=deliver_tool_reply_callback,
                    tool_reply_names=frozenset({"send_message"}),
                    steering_enabled=False,
                )
            )

            result = (
                None
                if turn.final_reply_silent
                else (turn.replies[-1] if turn.replies else None)
            )
            _log.info(
                f"后台任务完成: chat_id={chat_id[:20]}.. "
                f"result_len={len(result or '')}"
            )
            return BackgroundTaskResult(
                result=result,
                error=None,
                tool_delivered=turn.tool_text_delivered,
                silent=turn.final_reply_silent,
            )

        except asyncio.CancelledError:
            _log.warning(f"后台任务被取消: chat_id={chat_id[:20]}..")
            return BackgroundTaskResult(error="任务被取消")
        except Exception as e:
            _log.error(
                f"后台任务异常: chat_id={chat_id[:20]}.. error={e}",
                exc_info=True,
            )
            return BackgroundTaskResult(error=str(e))

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
        import time as _time

        from core.tools.impl.heartbeat import heartbeat_response as _heartbeat_response
        from core.tools.stream_delivery import is_silent_reply_text as _check_silent

        result = WakeTurnResult()

        if not self._reply_callback:
            _log.warning("reply_callback 未注入，无法投递 AI 回应")
            return result

        is_group = (
            self.context_manager.get_chat_type(session_key)
            if self.context_manager
            else False
        ) or False

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

        wake_resp: dict = {}

        token = _heartbeat_response.set(wake_resp)
        try:

            async def _build_prompt() -> tuple[list[dict], Optional[list[dict]]]:
                # 向后兼容：无预制 messages 时自己构建
                if messages is not None:
                    return messages, tools
                if source in ("interval", "manual"):
                    return await self.prompt_builder.build_heartbeat_messages(
                        prompt=extra_prompt,
                        system_prompt_mode=(
                            "minimal" if source == "interval" else "normal"
                        ),
                        session_mode="isolated",
                        admin_chat_id=self._admin_id[0] if self._admin_id else "",
                        chat_id=chat_id,
                        system_event_key=system_event_key or "heartbeat:events",
                    )
                return await self.prompt_builder.build(
                    chat_id=session_key,
                    is_group=is_group,
                    user_nickname="系统",
                    sender_id="system",
                    input_message=msg,
                    cost_tracker=self.cost_tracker,
                )

            async def _capture(chat_id, content, message_id, is_group):
                return None

            turn = await self._run_turn(
                _TurnRequest(
                    chat_id=chat_id,
                    sender_id="system",
                    is_group=is_group,
                    reply_to=msg.id,
                    route_text=extra_prompt or "[系统事件]",
                    prompt_factory=_build_prompt,
                    reply_callback=_capture,
                    timeout=timeout,
                    steering_enabled=False,
                )
            )

            # ── 通知决策（不再区分 source，统一通知文本） ──
            if wake_resp.get("notify"):
                text = wake_resp.get("notification_text", "").strip()
                if text:
                    result.notification_text = text
                    result.should_notify = True
                    result.deliver_to_user = wake_resp.get("deliver_to_user", "")
            else:
                reply = pick_final_notification_reply(list(turn.replies), _check_silent)
                if reply:
                    result.notification_text = reply
                    result.should_notify = True

            result.captured_replies = list(turn.replies)
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
            source="system",
            intent="event",
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

    async def get_stats(self) -> dict:
        stats: dict = {
            "queue_sizes": self.session_manager.get_queue_sizes(),
            "active_chats": await self.context_manager.get_context_count_async(),
            "total_messages": await self.context_manager.get_total_messages_count_async(),
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
        if self._sub_agent_manager:
            await self._sub_agent_manager.cancel_all()

        if self._consumer_tasks:
            deadline = time.monotonic() + 5.0
            while self._consumer_tasks and time.monotonic() < deadline:
                tasks = [task for task in self._consumer_tasks if not task.done()]
                if not tasks:
                    await asyncio.sleep(0)
                    continue
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(0)
            pending = [task for task in self._consumer_tasks if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                _log.warning("尚有 %d 个消费者任务未在超时内完成", len(pending))
            self._consumer_tasks.clear()
        await self.session_manager.cleanup_all(preserve_inboxes=True)

        if self._outbox_task and not self._outbox_task.done():
            self._outbox_task.cancel()
            await asyncio.gather(self._outbox_task, return_exceptions=True)
        if self._admission_outbox:
            await self._admission_outbox.close()

        if self.hindsight:
            await self.hindsight.close()

        _log.info("AgentEngine 已停止")
