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
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, List, Literal, Optional, Set
from uuid import uuid4

from core.engine.admission_effect_policy import effect_types_for
from core.engine.admission_outbox import AdmissionOutbox
from core.engine.batch_media_context import (
    BatchMediaContext,
    BatchMediaContextBuilder,
    BatchMediaLimits,
)
from core.engine.context import EngineContext
from core.engine.conversation_scheduler import ConversationScheduler
from core.engine.conversation_timeline import ConversationTimeline
from core.engine.delivery_ledger import (
    DeliveryController,
    DeliveryLedger,
    DeliveryReceipt,
    DeliveryRecord,
    DeliveryRecoveryResult,
)
from core.engine.delivery_prompt_contract import DeliveryPromptContract
from core.engine.group_engagement import GroupEngagementManager
from core.engine.model_context_transcript import (
    ModelContextInvariantError,
    ModelContextScope,
    ModelContextTranscript,
)
from core.engine.prompt_builder import PromptBuilder, PromptBuildResult
from core.engine.turn_capabilities import TurnCapabilities
from core.engine.turn_protocol_history import TurnProtocolHistory
from core.engine.turn_state import TurnPhase, TurnState, TurnStateError
from core.learners.base import sanitize_for_learners
from core.managers.cost_tracker import CostTracker
from core.managers.emoji_manager import EmojiManager
from core.managers.session_manager import (
    AdmissionOrigin,
    InboundIntent,
    PendingInbound,
    SessionTaskManager,
)
from core.media.models import MediaTurnContext
from core.message import InputMessage, MessageType, ResourceMeta
from core.tasks.wake_coalescer import WakeTurnResult
from core.tools.policy import filter_internal_control_tools
from core.tools.tool_loop import ToolLoop

_log = logging.getLogger(__name__)


TurnPromptFactory = Callable[
    [], Awaitable[PromptBuildResult | tuple[list[dict], Optional[list[dict]]]]
]


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
    internal_control: bool = False
    steering_enabled: bool = False
    steering_intent: Optional[InboundIntent] = None
    steering_admission_callback: Optional[
        Callable[[PendingInbound], Awaitable[Optional[AdmittedMessage]]]
    ] = None
    steering_commit_callback: Optional[
        Callable[[Any, PendingInbound], Awaitable[None]]
    ] = None
    capabilities: Optional[TurnCapabilities] = None
    turn_id: str = ""
    intent: Optional[InboundIntent] = None
    model_context_commit_callback: Optional[
        Callable[[ModelContextScope], Awaitable[None]]
    ] = None
    model_context_provider_callback: Optional[Callable[..., Awaitable[Any]]] = None
    model_context_usage_callback: Optional[Callable[..., Awaitable[None]]] = None
    model_context_overflow_callback: Optional[Callable[..., Awaitable[Any]]] = None


@dataclass(frozen=True)
class _TurnResult:
    replies: tuple[str, ...] = field(default_factory=tuple)
    sent_emoji: bool = False
    text_committed: bool = False
    tool_text_delivered: bool = False
    final_reply_silent: bool = False
    model_context_scope: Optional[ModelContextScope] = None


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
        self.media_context_builder: Optional[BatchMediaContextBuilder] = None
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
        self._task_state_store = ctx.mgmt.task_state_store
        self._permission_manager = ctx.mgmt.permission_manager

        # ── 子模块 ──
        self.session_manager = SessionTaskManager()
        engagement_config = getattr(ctx.ai, "engagement_config", None)
        if engagement_config is None:
            from core.engine.engagement_config import normalize_engagement_config

            engagement_config = normalize_engagement_config({})
        self.scheduler = ConversationScheduler(
            self.session_manager,
            collect_idle_ms=engagement_config.conversation_collect_idle_ms,
            collect_max_wait_ms=engagement_config.conversation_collect_max_wait_ms,
            collect_max_messages=engagement_config.conversation_collect_max_messages,
            collect_max_chars=engagement_config.conversation_collect_max_chars,
            ambient_collect_idle_ms=engagement_config.group_ambient_idle_ms,
            direct_task_collaboration_enabled=(
                engagement_config.direct_task_collaboration_enabled
            ),
            user_role=(
                self._permission_manager.get_user_role
                if self._permission_manager is not None
                else None
            ),
            role_at_least=(
                self._permission_manager.role_at_least
                if self._permission_manager is not None
                else None
            ),
            task_state_store=ctx.mgmt.task_state_store,
        )
        self.group_engagement = GroupEngagementManager(engagement_config)
        self.engagement_config = engagement_config
        self.delivery_controller: Optional[DeliveryController] = None
        self._delivery_recovery_locks: dict[str, asyncio.Lock] = {}
        self._delivery_recovery_task: Optional[asyncio.Task] = None
        self.timeline = ConversationTimeline()
        self.protocol_history = TurnProtocolHistory()
        model_context_config = getattr(ctx.mgmt, "model_context_config", {}) or {}
        projection_enabled = bool(model_context_config.get("enabled", False))
        projection_mode = str(
            model_context_config.get(
                "mode", model_context_config.get("read_mode", "projection")
            )
        ).lower()
        self.model_context_write_enabled = projection_enabled and bool(
            model_context_config.get("write_enabled", True)
        )
        self.model_context_read_enabled = projection_enabled and bool(
            model_context_config.get("read_enabled", True)
        )
        if projection_mode in {"off", "disabled"}:
            self.model_context_write_enabled = False
            self.model_context_read_enabled = False
        elif projection_mode in {"shadow", "diagnostic", "write_only"}:
            self.model_context_read_enabled = False
        elif projection_mode == "read_only":
            self.model_context_write_enabled = False
        if bool(model_context_config.get("rollback", False)):
            self.model_context_read_enabled = False
        self.model_context_shadow = not self.model_context_read_enabled and (
            self.model_context_write_enabled
        )
        self.model_context_enabled = (
            self.model_context_write_enabled or self.model_context_read_enabled
        )
        compaction_config = model_context_config.get("compaction", {}) or {}
        if not isinstance(compaction_config, dict):
            compaction_config = {}

        def _compaction_value(name, default):
            return compaction_config.get(
                name,
                model_context_config.get(
                    f"compaction_{name}", model_context_config.get(name, default)
                ),
            )

        self.model_context = ModelContextTranscript(
            model_context_config.get("path", "data/model_context_transcript.sqlite3"),
            max_events=int(model_context_config.get("max_events", 512)),
            max_tokens=int(model_context_config.get("max_tokens", 24000)),
            compaction_enabled=bool(_compaction_value("enabled", False)),
            compaction_tier1_ratio=float(_compaction_value("tier1_ratio", 0.60)),
            compaction_tier2_ratio=float(_compaction_value("tier2_ratio", 0.80)),
            compaction_tier3_ratio=float(_compaction_value("tier3_ratio", 0.95)),
            compaction_keep_recent_tokens=int(
                _compaction_value("keep_recent_tokens", 4096)
            ),
            compaction_snip_max_chars=int(_compaction_value("snip_max_chars", 1200)),
            compaction_max_summary_tokens=int(
                _compaction_value("max_summary_tokens", 500)
            ),
        )
        self.prompt_builder = PromptBuilder(ctx)
        self.prompt_builder.timeline = self.timeline
        self.prompt_builder.model_context_transcript = self.model_context
        self.prompt_builder.model_context_read_enabled = self.model_context_read_enabled
        self.prompt_builder.model_context_write_enabled = (
            self.model_context_write_enabled
        )
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

    def _get_scheduler(self) -> ConversationScheduler:
        """Create the scheduler lazily for lightweight test/fallback engine fixtures."""
        scheduler = getattr(self, "scheduler", None)
        if scheduler is None:
            scheduler = ConversationScheduler(self.session_manager)
            self.scheduler = scheduler
        return scheduler

    async def _cancel_scheduler_turn(self, turn_id: str) -> None:
        """Best-effort terminal cleanup for a consumer work item that did not finish."""
        scheduler = self._get_scheduler()
        state = await scheduler.get_turn(turn_id)
        if state is None:
            return
        try:
            if state.phase in {TurnPhase.CANCELLED, TurnPhase.COMPLETED}:
                await scheduler.drop_turn(turn_id)
                return
            cancelled = await scheduler.transition_turn(
                turn_id,
                expected_revision=state.revision,
                phase=TurnPhase.CANCELLED,
            )
            await scheduler.drop_turn(cancelled.turn_id)
        except TurnStateError as exc:
            _log.warning("scheduler turn cancellation failed [%s]: %s", turn_id, exc)

    def _get_group_engagement(self) -> GroupEngagementManager:
        """Lazily create the gate for lightweight test/fallback engine fixtures."""
        manager = getattr(self, "group_engagement", None)
        if manager is None:
            from core.engine.engagement_config import EngagementConfig

            manager = GroupEngagementManager(EngagementConfig())
            self.group_engagement = manager
        return manager

    def _get_batch_media_context_builder(self) -> BatchMediaContextBuilder:
        builder = getattr(self, "media_context_builder", None)
        if builder is None or builder.media_service is not self.media_service:
            config = getattr(self, "engagement_config", None)
            limits = BatchMediaLimits(
                max_resources=getattr(config, "media_batch_max_resources", 8),
                max_chars=getattr(config, "media_batch_max_chars", 12000),
                max_download_bytes=getattr(
                    config, "media_batch_max_download_bytes", 100 * 1024 * 1024
                ),
                capability_timeout_seconds=getattr(
                    config, "media_batch_capability_timeout_seconds", 120.0
                ),
            )
            builder = BatchMediaContextBuilder(self.media_service, limits=limits)
            self.media_context_builder = builder
        return builder

    async def _build_batch_media_context(
        self, turn_id: str, items: tuple[PendingInbound, ...]
    ) -> BatchMediaContext:
        """Prepare media only for a batch that is about to call a provider."""
        try:
            return await self._get_batch_media_context_builder().build(
                turn_id=turn_id, items=items
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning("媒体批次上下文构建失败 [%s]: %s", turn_id, exc)
            return BatchMediaContext(turn_id, (), MediaTurnContext())

    def _turn_capabilities(
        self,
        intent: InboundIntent,
        *,
        chat_id: str,
        sender_id: str,
        reply_to: str,
        allowed_media_uris: frozenset[str] | None = None,
    ) -> TurnCapabilities:
        return TurnCapabilities.for_intent(
            intent,
            chat_id=chat_id,
            sender_id=sender_id,
            reply_to=reply_to,
            allowed_media_uris=allowed_media_uris,
        )

    def _delivery_contract(
        self, intent: InboundIntent, reply_target: str
    ) -> DeliveryPromptContract:
        config = getattr(self, "engagement_config", None)
        if config is None:
            from core.engine.engagement_config import EngagementConfig

            config = EngagementConfig()
        delivery_modes = {
            InboundIntent.PRIVATE_CONVERSATION: config.private_conversation_delivery_mode,
            InboundIntent.DIRECT_TASK: config.direct_task_delivery_mode,
            InboundIntent.GROUP_AMBIENT: config.group_ambient_delivery_mode,
        }
        return DeliveryPromptContract(
            intent=intent,
            delivery_mode=delivery_modes[intent],
            reply_target=reply_target,
        )

    def _get_timeline(self) -> ConversationTimeline:
        timeline = getattr(self, "timeline", None)
        if timeline is None:
            timeline = ConversationTimeline()
            self.timeline = timeline
        return timeline

    async def _record_timeline_user_message(self, pending: PendingInbound) -> None:
        """Write the admission projection without making it a protocol message."""
        if pending.origin is not AdmissionOrigin.USER_MESSAGE:
            return
        message = pending.message
        try:
            await self._get_timeline().append_user_message(
                chat_id=message.chat_id,
                message_id=message.id,
                content=pending.prepared_content,
                sender_id=message.sender_id,
                timestamp=message.timestamp,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # ChatContext remains the compatibility source until projection
            # migration is complete; a projection failure must be observable.
            _log.warning(
                "会话 timeline 写入失败 [%s..] message=%s: %s",
                message.chat_id[:12],
                message.id,
                exc,
            )

    async def _repair_timeline_from_legacy_history(self, chat_id: str) -> None:
        get_history = getattr(self.context_manager, "get_chat_history_async", None)
        if get_history is None:
            return
        timeline = self._get_timeline()
        repair = getattr(timeline, "repair_from_legacy_history", None)
        if repair is None:
            return
        try:
            history = await get_history(chat_id)
            migrated = await repair(chat_id, history)
            if migrated:
                _log.info(
                    "已从旧 ChatContext 回填 timeline [%s..]: %d 条",
                    chat_id[:12],
                    migrated,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning("旧历史回填 timeline 失败 [%s..]: %s", chat_id[:12], exc)

    async def get_history_migration_status(self, chat_id: str) -> dict:
        """Return non-content readiness for retiring legacy prompt history."""
        legacy = await self.context_manager.get_chat_history_async(chat_id)
        return (await self._get_timeline().migration_report(chat_id, legacy)).to_dict()

    async def get_history_migration_summary(self) -> dict:
        """Scan all known sessions and return content-free migration readiness."""
        chat_ids: set[str] = set()
        for method_name in (
            "get_all_chat_ids_async",
            "get_all_disk_chat_ids_async",
        ):
            method = getattr(self.context_manager, method_name, None)
            if method is None:
                continue
            try:
                chat_ids.update(str(chat_id) for chat_id in await method())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log.warning("枚举历史迁移会话失败 [%s]: %s", method_name, exc)

        timeline = self._get_timeline()
        try:
            chat_ids.update(str(chat_id) for chat_id in await timeline.chat_ids())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning("枚举 timeline 会话失败: %s", exc)

        reports = []
        scan_errors = 0
        for chat_id in sorted(chat_ids):
            try:
                legacy = await self.context_manager.get_chat_history_async(chat_id)
                reports.append(await timeline.migration_report(chat_id, legacy))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                scan_errors += 1
                _log.warning("扫描历史迁移状态失败 [%s..]: %s", chat_id[:12], exc)

        return timeline.migration_summary(reports, scan_errors=scan_errors).to_dict()

    def _get_protocol_history(self) -> TurnProtocolHistory:
        history = getattr(self, "protocol_history", None)
        if history is None:
            history = TurnProtocolHistory()
            self.protocol_history = history
        return history

    def _get_model_context(self) -> ModelContextTranscript:
        transcript = getattr(self, "model_context", None)
        if transcript is None:
            transcript = ModelContextTranscript()
            self.model_context = transcript
        return transcript

    async def _model_context_scope(
        self,
        pending: PendingInbound,
        input_message: InputMessage,
        *,
        capabilities: Optional[TurnCapabilities] = None,
        batch: tuple[PendingInbound, ...] = (),
    ) -> Optional[ModelContextScope]:
        if not getattr(self, "model_context_enabled", False):
            return None
        if pending.intent is InboundIntent.DIRECT_TASK:
            correlation_id = pending.message.task_correlation_id
            if not correlation_id or not getattr(self, "scheduler", None):
                return None
            if any(
                item.intent is not InboundIntent.DIRECT_TASK
                or item.message.task_correlation_id != correlation_id
                or item.message.chat_id != input_message.chat_id
                or item.message.sender_id != pending.message.sender_id
                for item in (pending, *batch)
            ):
                return None
            if not await self._get_scheduler().allows_model_context_inheritance(
                pending.message.id,
                chat_id=input_message.chat_id,
                principal_id=pending.message.sender_id,
                task_correlation_id=correlation_id,
                reply_to=input_message.id,
                capabilities=capabilities,
            ):
                return None
            return ModelContextScope.for_intent(
                chat_id=input_message.chat_id,
                principal_id=pending.message.sender_id,
                intent=pending.intent,
                task_correlation_id=correlation_id,
            )
        if pending.intent is not InboundIntent.PRIVATE_CONVERSATION:
            return None
        if any(
            item.intent is not InboundIntent.PRIVATE_CONVERSATION
            or item.message.chat_id != input_message.chat_id
            or item.message.sender_id != input_message.sender_id
            for item in (pending, *batch)
        ):
            return None
        return ModelContextScope.for_intent(
            chat_id=input_message.chat_id,
            principal_id=input_message.sender_id,
            intent=pending.intent,
        )

    def _model_context_identity(self, input_message: InputMessage) -> tuple[str, ...]:
        model_chain = input_message.model_chain
        tier = input_message.tier
        if (
            model_chain is None
            and tier is None
            and self.rule_router
            and self.model_registry
        ):
            tier = self.rule_router.classify(input_message.content)
        if model_chain is None and tier is not None and self.model_registry:
            model_chain = self.model_registry.get_chain(tier) or None
        if model_chain:
            identities = []
            for name in model_chain:
                service = self.model_registry.get(name) if self.model_registry else None
                provider = getattr(service, "provider_type", "") if service else ""
                identities.append(
                    ":".join(
                        (
                            str(name),
                            (
                                provider or type(service).__name__
                                if service
                                else "missing"
                            ),
                            str(getattr(service, "model", "")) if service else "",
                        )
                    )
                )
            return tuple(identities)
        service = self.ai_service
        return (
            f"{getattr(service, 'provider_type', '') or type(service).__name__}:"
            f"{getattr(service, 'model', '')}",
        )

    @staticmethod
    def _provider_identity(service: Any) -> str:
        return (
            f"{getattr(service, 'provider_type', '') or type(service).__name__}:"
            f"{getattr(service, 'model', '')}"
        )

    async def _model_context_snapshot(self, scope: Optional[ModelContextScope]):
        if scope is None or not getattr(self, "model_context_read_enabled", False):
            return None
        try:
            return await self._get_model_context().snapshot(scope)
        except Exception as exc:
            _log.warning(
                "模型上下文投影读取失败，回退 timeline [%s..]: %s",
                scope.chat_id[:12],
                exc,
            )
            return None

    async def _materialize_model_context(
        self,
        scope: Optional[ModelContextScope],
        *,
        turn_id: str,
        message_ids: set[str],
    ) -> None:
        if scope is None or not getattr(self, "model_context_write_enabled", False):
            return
        scope = await self._get_model_context().current_scope(scope)
        timeline_events = await self._get_timeline().snapshot(scope.chat_id)
        user_events = tuple(
            event
            for event in timeline_events
            if event.role == "user" and event.message_id in message_ids
        )
        protocol_events = await self._get_protocol_history().snapshot(turn_id)
        if not protocol_events:
            raise ModelContextInvariantError(
                f"turn has no protocol to materialize: {turn_id}"
            )
        await self._get_model_context().append_turn(
            scope,
            turn_id=turn_id,
            user_events=user_events,
            protocol_events=protocol_events,
        )
        _log.debug(
            "模型上下文投影已提交 [%s..] generation=%d turn=%s events=%d",
            scope.chat_id[:12],
            scope.generation,
            turn_id,
            len(user_events) + len(protocol_events),
        )

    async def _record_model_context_usage(
        self,
        scope: Optional[ModelContextScope],
        usage: Optional[dict[str, Any]],
        service: Any,
        model_name: str,
        *,
        turn_id: str,
    ) -> None:
        if scope is None or not getattr(self, "model_context_enabled", False):
            return
        try:
            await self._get_model_context().record_provider_usage(
                scope,
                usage,
                provider=self._provider_identity(service),
                model=model_name or getattr(service, "model", ""),
                turn_id=turn_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning(
                "记录模型上下文 provider usage 失败 [%s..]: %s",
                scope.chat_id[:12],
                exc,
            )

    async def _close_model_context_scope(self, pending: PendingInbound) -> None:
        if (
            not getattr(self, "model_context_enabled", False)
            or pending.intent is not InboundIntent.DIRECT_TASK
            or not pending.message.task_correlation_id
        ):
            return
        scope = ModelContextScope.for_intent(
            chat_id=pending.message.chat_id,
            principal_id=pending.message.sender_id,
            intent=pending.intent,
            task_correlation_id=pending.message.task_correlation_id,
        )
        if scope is None:
            return
        try:
            await self._get_model_context().close_scope(scope)
        except Exception as exc:
            _log.warning(
                "关闭 direct task 模型上下文失败 [%s..]: %s",
                pending.message.chat_id[:12],
                exc,
            )

    def _get_delivery_controller(self) -> DeliveryController:
        controller = getattr(self, "delivery_controller", None)
        if controller is None:
            config = getattr(self, "engagement_config", None)
            if config is None:
                from core.engine.engagement_config import EngagementConfig

                config = EngagementConfig()
            controller = DeliveryController(
                DeliveryLedger(),
                retry_base_seconds=config.delivery_retry_base_seconds,
                max_attempts=config.delivery_retry_max_attempts,
                timeline=self._get_timeline(),
                audit_delivery=(
                    task_state_store.record_delivery
                    if (task_state_store := getattr(self, "_task_state_store", None))
                    is not None
                    else None
                ),
            )
            self.delivery_controller = controller
        return controller

    async def _recover_ambient_deliveries(
        self,
        chat_id: str,
        reply_callback: Callable,
        *,
        allow_transport_retry: bool = False,
    ) -> DeliveryRecoveryResult:
        """Recover stale ambient text with explicit transport idempotency opt-in."""
        locks = getattr(self, "_delivery_recovery_locks", None)
        if locks is None:
            locks = {}
            self._delivery_recovery_locks = locks
        lock = locks.setdefault(chat_id, asyncio.Lock())
        async with lock:

            async def _resolve_content(record: DeliveryRecord) -> Optional[str]:
                timeline = getattr(self, "timeline", None)
                history_reader = getattr(timeline, "history", None)
                history = await history_reader(chat_id) if history_reader else []
                if not history:
                    history = await self.context_manager.get_chat_history_async(chat_id)
                for item in reversed(history):
                    if (
                        item.get("role") == "assistant"
                        and item.get("message_id") == record.reply_anchor_id
                    ):
                        content = item.get("content")
                        if isinstance(content, str) and content:
                            return content
                return None

            async def _transport(record: DeliveryRecord, content: str) -> object:
                kwargs = {
                    "chat_id": record.chat_id,
                    "content": content,
                    "message_id": record.reply_anchor_id,
                    "is_group": True,
                }
                if allow_transport_retry:
                    kwargs["delivery_id"] = record.logical_delivery_id or record.key
                return await reply_callback(
                    **kwargs,
                )

            config = getattr(self, "engagement_config", None)
            if config is None:
                from core.engine.engagement_config import EngagementConfig

                config = EngagementConfig()
            result = await self._get_delivery_controller().recover_prepared(
                chat_id=chat_id,
                key_prefix="ambient:",
                older_than=time.time() - config.delivery_recovery_after_seconds,
                content_resolver=_resolve_content,
                transport=_transport,
                allow_transport_retry=allow_transport_retry,
            )
            if result.scanned:
                _log.info(
                    "ambient delivery recovery [%s..]: sent=%d retryable=%d failed=%d",
                    chat_id[:12],
                    result.sent,
                    result.retryable,
                    result.failed,
                )
            return result

    async def _delivery_recovery_worker(self) -> None:
        config = getattr(self, "engagement_config", None)
        interval = max(
            1.0,
            getattr(config, "delivery_recovery_after_seconds", 60.0),
        )
        while True:
            try:
                for chat_id, callbacks in tuple(
                    getattr(self, "_consumer_callbacks", {}).items()
                ):
                    if not self._get_group_engagement().is_active_chat(chat_id):
                        continue
                    await self._recover_ambient_deliveries(chat_id, callbacks[0])
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log.warning("投递恢复 worker 失败: %s", exc)
                await asyncio.sleep(interval)

    async def _ensure_delivery_recovery_worker(self) -> None:
        if self._get_group_engagement().config.group_ambient_mode != "active":
            return
        task = getattr(self, "_delivery_recovery_task", None)
        if task is None or task.done():
            self._delivery_recovery_task = asyncio.create_task(
                self._delivery_recovery_worker()
            )

    async def get_engagement_status(self) -> dict:
        """Return aggregate engagement and delivery state for health surfaces."""
        metrics = self._get_group_engagement().snapshot_metrics()
        delivery = {}
        controller = getattr(self, "delivery_controller", None)
        if controller is not None:
            delivery = await controller.ledger.status_counts()
        return {"engagement": metrics, "delivery": delivery}

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
        self.media_context_builder = None
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

        self._duplicate_reply = DuplicateReplyDetector(
            self.context_manager,
            delivery_controller_getter=lambda: self.delivery_controller,
        )
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

        intent = self._classify_inbound_intent(input_message)
        needs_ai = intent is not InboundIntent.GROUP_AMBIENT
        if needs_ai and self.rule_router and self.model_registry:
            tier = self.rule_router.classify(input_message.content)
            input_message.tier = tier
            input_message.model_chain = self.model_registry.get_chain(tier) or None

        try:
            pending = await self._prepare_pending_inbound(
                input_message,
                intent=intent,
            )
            enqueued = await self._get_scheduler().enqueue(chat_id, pending)
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
        if not enqueued.accepted:
            if intent is not InboundIntent.GROUP_AMBIENT:
                try:
                    receipt = await self._get_delivery_controller().deliver_text(
                        delivery_id=f"backpressure:{chat_id}:{input_message.id}",
                        chat_id=chat_id,
                        content="当前会话任务较多，请稍后重试。",
                        callback=reply_callback,
                        message_id=input_message.id,
                        is_group=input_message.is_group,
                        reason="inbox_backpressure",
                        timeline_delivery_kind="system_fallback",
                    )
                    if receipt.status != "accepted":
                        _log.warning(
                            "会话背压提示未确认 [%s..]: %s",
                            chat_id[:12],
                            receipt.status,
                        )
                except Exception as exc:
                    _log.warning("发送会话背压提示失败 [%s..]: %s", chat_id[:12], exc)
            return
        if enqueued.accepted:
            self._consumer_callbacks[chat_id] = (
                reply_callback,
                get_user_nickname,
            )
            if self._get_group_engagement().config.group_ambient_mode == "active":
                await self._recover_ambient_deliveries(chat_id, reply_callback)
                await self._ensure_delivery_recovery_worker()
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
        intent: InboundIntent,
    ) -> PendingInbound:
        """Build lightweight immutable ingress context without media analysis."""
        content = input_message.content
        media_refs = tuple(
            resource.media_uri
            for resource in (*input_message.resources, *input_message.replied_resources)
            if resource.media_uri
        )
        if media_refs:
            content += "\n[媒体引用: " + ", ".join(media_refs) + "]"
        if input_message.replied_content:
            prefix = (
                f"[正在回复 {input_message.replied_author}: {input_message.replied_content}]"
                if input_message.replied_author
                else f"[正在回复: {input_message.replied_content}]"
            )
            content = f"{prefix}\n{content}" if content else prefix
        return PendingInbound(
            input_message,
            content,
            intent,
            AdmissionOrigin.USER_MESSAGE,
            resource_refs=media_refs,
        )

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

            effect_types = effect_types_for(pending)
            has_outbox_effects = bool(outbox and effect_types)
            prepared_new = False
            committed = False
            if has_outbox_effects:
                self._admission_in_progress.add(admission_key)
            if pending.origin is AdmissionOrigin.INTERNAL_CONTROL:
                _log.info(
                    "跳过内部控制准入副作用: origin=%s chat=%s id=%s effects=%s",
                    pending.origin,
                    chat_id[:12],
                    message.id,
                    ("hindsight", "learner"),
                )
            try:
                if has_outbox_effects:
                    payload = self._build_side_effect_payload(pending)
                    prepared_new = await outbox.prepare(
                        chat_id,
                        message.id,
                        payload,
                        effect_types=effect_types,
                    )
                nickname = get_user_nickname(message.sender_id) or message.sender_id
                await self.context_manager.record_chat_type(chat_id, message.is_group)
                if pending.origin is AdmissionOrigin.USER_MESSAGE:
                    await self._repair_timeline_from_legacy_history(chat_id)
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
                    if has_outbox_effects:
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
                if pending.origin is AdmissionOrigin.USER_MESSAGE:
                    await self._record_timeline_user_message(pending)
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
                if has_outbox_effects:
                    await outbox.mark_ready(chat_id, message.id)
                    await self._process_admission_outbox()
                elif not outbox:
                    await self._run_admission_side_effects(pending, admission_key)
                _log.debug(
                    "消息已准入 [%s..]: id=%s source=%s",
                    chat_id[:12],
                    message.id,
                    source,
                )
            except BaseException:
                if has_outbox_effects and not committed and prepared_new:
                    await outbox.cancel(chat_id, message.id)
                raise
            finally:
                if has_outbox_effects:
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
        timeline = getattr(self, "timeline", None)
        history_reader = getattr(timeline, "history", None)
        if callable(history_reader):
            try:
                timeline_history = await history_reader(chat_id)
                if any(
                    item.get("role") == "user" and item.get("message_id") == message_id
                    for item in timeline_history
                ):
                    return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log.debug("timeline 准入检查失败 [%s..]: %s", chat_id[:12], exc)
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
        await self._ensure_delivery_recovery_worker()
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
        # Legacy outbox rows created before admission-effect filtering may contain cards.
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
        # Legacy outbox rows created before admission-effect filtering may contain cards.
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
            effect_type: handler
            for effect_type, handler in {
                "hindsight": self._run_hindsight_side_effect,
                "learner": self._run_learner_side_effect,
            }.items()
            if effect_type in effect_types_for(pending)
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

    def _classify_inbound_intent(self, input_message: InputMessage) -> InboundIntent:
        """Classify once at ingress; queue and tool-loop state cannot rewrite it."""
        if not input_message.is_group:
            return InboundIntent.PRIVATE_CONVERSATION
        if self._should_dispatch_to_ai(input_message):
            return InboundIntent.DIRECT_TASK
        return InboundIntent.GROUP_AMBIENT

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
                work = await self._get_scheduler().next_work(
                    chat_id, owner_token=consumer_token
                )
                if work is None:
                    if await self._get_scheduler().release_consumer_if_idle(
                        chat_id, consumer_token
                    ):
                        return
                    continue

                pending = work.pending
                batch = work.items[1:]
                message = pending.message
                scheduler_turn_id = message.id
                turn_state: TurnState | None = None
                try:
                    turn_state = await self._get_scheduler().start_turn(
                        work,
                        turn_id=scheduler_turn_id,
                        principal_id=message.sender_id,
                    )
                    if pending.intent is not InboundIntent.GROUP_AMBIENT:
                        if batch:
                            await self._process_message(
                                pending,
                                reply_callback,
                                get_user_nickname,
                                batch=batch,
                            )
                        else:
                            await self._process_message(
                                pending,
                                reply_callback,
                                get_user_nickname,
                            )
                    else:
                        if (
                            turn_state.phase is not TurnPhase.ACTIVE
                            or work.passive_admission_only
                        ):
                            decision = None
                        else:
                            decision = await self._get_group_engagement().evaluate(
                                chat_id, batch=work.items
                            )
                            self._get_group_engagement().observe(decision)
                        if decision is not None and decision.shadow:
                            _log.info(
                                "ambient shadow candidate [%s..] reason=%s batch=%d",
                                chat_id[:12],
                                decision.reason,
                                len(work.items),
                            )
                        if (
                            decision is not None
                            and decision.allowed
                            and self.group_engagement.config.group_ambient_mode
                            == "active"
                        ):
                            await self._process_ambient_active(
                                pending,
                                batch,
                                decision,
                                reply_callback,
                                get_user_nickname,
                            )
                        else:
                            if decision is not None and decision.allowed:
                                _log.warning(
                                    "ambient active decision held passive until delivery integration [%s..]",
                                    chat_id[:12],
                                )
                            session_lock = await self.session_manager.get_lock(chat_id)
                            async with session_lock:
                                for item in work.items:
                                    await self._admit_pending_message(
                                        item,
                                        source="passive",
                                        get_user_nickname=get_user_nickname,
                                    )
                                    if item.message.msg_type != MessageType.EMOJI:
                                        await self._run_hooks(
                                            item.message,
                                            reply_callback,
                                            get_user_nickname,
                                        )
                    for item in work.items:
                        await self._get_scheduler().commit(work, item)
                    current_turn = await self._get_scheduler().get_turn(
                        scheduler_turn_id
                    )
                    if current_turn is None:
                        raise TurnStateError(
                            f"scheduler turn disappeared: {scheduler_turn_id}"
                        )
                    if current_turn.phase is TurnPhase.CANCELLED:
                        await self._close_model_context_scope(pending)
                        await self._get_scheduler().drop_turn(current_turn.turn_id)
                    else:
                        finalizing = current_turn
                        if current_turn.phase is TurnPhase.ACTIVE:
                            finalizing = await self._get_scheduler().transition_turn(
                                scheduler_turn_id,
                                expected_revision=current_turn.revision,
                                phase=TurnPhase.FINALIZING,
                            )
                        completed = await self._get_scheduler().transition_turn(
                            scheduler_turn_id,
                            expected_revision=finalizing.revision,
                            phase=TurnPhase.COMPLETED,
                        )
                        await self._close_model_context_scope(pending)
                        await self._get_scheduler().drop_turn(completed.turn_id)
                except asyncio.CancelledError:
                    await self._cancel_scheduler_turn(scheduler_turn_id)
                    await self._close_model_context_scope(pending)
                    for item in work.items:
                        state = self.session_manager.get_message_state(
                            chat_id, item.message.id
                        )
                        admission_key = (chat_id, item.message.id)
                        if state == "admitted" or admission_key in getattr(
                            self, "_admitted_ids", {}
                        ):
                            await self._get_scheduler().commit(work, item)
                        else:
                            await self._get_scheduler().requeue_front(work)
                            break
                    raise
                except Exception as exc:
                    await self._cancel_scheduler_turn(scheduler_turn_id)
                    await self._close_model_context_scope(pending)
                    first_uncommitted = True
                    for item in work.items:
                        state = self.session_manager.get_message_state(
                            chat_id, item.message.id
                        )
                        admission_key = (chat_id, item.message.id)
                        if state == "admitted" or admission_key in getattr(
                            self, "_admitted_ids", {}
                        ):
                            await self._get_scheduler().commit(work, item)
                        elif first_uncommitted:
                            await self._get_scheduler().fail(work, item)
                            first_uncommitted = False
                            async with self._dedup_lock:
                                self._processed_ids.pop(item.message.id, None)
                            await self._get_scheduler().requeue_front(work)
                            break
                        else:
                            await self._get_scheduler().requeue_front(work)
                            break
                    _log.error(
                        "消费者处理消息 %s 时出错: %s",
                        message.id,
                        exc,
                        exc_info=True,
                    )
                    try:
                        receipt = await self._get_delivery_controller().deliver_text(
                            delivery_id=f"consumer-error:{chat_id}:{message.id}",
                            chat_id=chat_id,
                            content="抱歉，处理您的消息时出现了问题，请稍后再试。",
                            callback=reply_callback,
                            message_id=message.id,
                            is_group=message.is_group,
                        )
                        if receipt.status != "accepted":
                            _log.warning(
                                "错误回复未确认 [%s..]: %s",
                                chat_id[:12],
                                receipt.status,
                            )
                    except Exception as reply_err:
                        _log.warning(
                            "向用户发送错误回复失败 [%s..]: %s", chat_id[:12], reply_err
                        )
                    continue
        finally:
            replacement_token = await self._get_scheduler().handoff_consumer(
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
            prompt_result = await request.prompt_factory()
            messages, tools = prompt_result
            model_context_scope = getattr(prompt_result, "model_context_scope", None)
            if request.internal_control:
                tools = filter_internal_control_tools(tools)
            if request.capabilities is not None:
                tools = request.capabilities.filter_tools(tools)
            prompt_built = True
            model_chain = request.model_chain
            tier = request.tier
            if model_chain is None and tier is None:
                if self.rule_router and self.model_registry:
                    tier = self.rule_router.classify(request.route_text)
                    model_chain = self.model_registry.get_chain(tier) or None

            turn_state = (
                await self._get_scheduler().get_turn(request.turn_id)
                if request.turn_id
                else None
            )
            capabilities = request.capabilities
            cancellation_generation = (
                turn_state.cancellation_generation if turn_state is not None else 0
            )
            if capabilities is not None:
                capabilities = replace(
                    capabilities,
                    cancellation_generation=cancellation_generation,
                )

            async def _transition_turn(**kwargs):
                if turn_state is None:
                    return None
                return await self._get_scheduler().transition_turn(
                    request.turn_id, **kwargs
                )

            async def _turn_is_active() -> bool:
                if turn_state is None:
                    return True
                return await self._get_scheduler().is_turn_execution_allowed(
                    request.turn_id, cancellation_generation
                )

            async def _turn_can_deliver() -> bool:
                if turn_state is None:
                    return True
                return await self._get_scheduler().is_turn_delivery_allowed(
                    request.turn_id, cancellation_generation
                )

            async def _model_context_provider(service, can_rebuild: bool):
                nonlocal model_context_scope
                if request.model_context_provider_callback is None:
                    return None
                result = await request.model_context_provider_callback(
                    service, can_rebuild
                )
                if result is not None:
                    model_context_scope = getattr(
                        result, "model_context_scope", model_context_scope
                    )
                return result

            async def _model_context_usage(usage, service, model_name):
                if request.model_context_usage_callback is None:
                    return
                await request.model_context_usage_callback(
                    model_context_scope,
                    usage,
                    service,
                    model_name,
                )

            async def _model_context_overflow(service, elapsed_ms):
                if request.model_context_overflow_callback is None:
                    return None
                return await request.model_context_overflow_callback(
                    service, elapsed_ms
                )

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
                steering_intent=request.steering_intent,
                steering_admission_callback=(
                    _steering_admission_callback
                    if request.steering_admission_callback
                    else None
                ),
                steering_claim_callback=(
                    lambda: (
                        self._get_scheduler().claim_steer(request.turn_id)
                        if request.turn_id
                        else None
                    )
                ),
                steering_commit_callback=(
                    (
                        lambda lease, item: self._get_scheduler().commit_steer(
                            request.turn_id, lease, item
                        )
                    )
                    if request.turn_id and request.steering_admission_callback
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
                internal_control=request.internal_control,
                capabilities=capabilities,
                delivery_controller=(
                    self._get_delivery_controller()
                    if (
                        request.tool_reply_callback
                        or request.capabilities is not None
                        or request.turn_id
                    )
                    else None
                ),
                turn_id=request.turn_id or request.reply_to,
                protocol_history=self._get_protocol_history(),
                transition_turn=_transition_turn if turn_state is not None else None,
                turn_active_callback=_turn_is_active,
                turn_delivery_callback=_turn_can_deliver,
                turn_revision=turn_state.revision if turn_state is not None else 0,
                delivery_state_callback=(
                    _tool_delivery_callback if request.track_tool_delivery else None
                ),
                cost_metadata=(
                    {
                        "scope_generation": model_context_scope.generation,
                        "scope_kind": str(model_context_scope.kind),
                        "tool_protocol": True,
                        "turn_intent": str(
                            request.intent
                            or request.steering_intent
                            or InboundIntent.DIRECT_TASK
                        ),
                    }
                    if model_context_scope is not None
                    else {
                        "turn_intent": str(
                            request.intent
                            or request.steering_intent
                            or InboundIntent.DIRECT_TASK
                        )
                    }
                ),
                protocol_settled_callback=(
                    (lambda: request.model_context_commit_callback(model_context_scope))
                    if request.model_context_commit_callback
                    and model_context_scope is not None
                    else None
                ),
                model_context_provider_callback=(
                    _model_context_provider
                    if request.model_context_provider_callback is not None
                    else None
                ),
                model_context_usage_callback=(
                    _model_context_usage
                    if request.model_context_usage_callback is not None
                    else None
                ),
                model_context_overflow_callback=(
                    _model_context_overflow
                    if request.model_context_overflow_callback is not None
                    else None
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
                model_context_scope=model_context_scope,
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

    async def _process_ambient_active(
        self,
        pending: PendingInbound,
        batch: tuple[PendingInbound, ...],
        decision,
        reply_callback: Callable,
        get_user_nickname: Callable[[str], str],
    ) -> None:
        """Run an opted-in ambient turn behind the durable delivery gate."""
        input_message = (batch[-1] if batch else pending).message
        chat_id = input_message.chat_id
        admitted_any = False
        final_delivered = False
        controller = self._get_delivery_controller()
        turn_id = pending.message.id
        allowed_media_uris = frozenset(
            resource.media_uri
            for item in (pending, *batch)
            for resource in (*item.message.resources, *item.message.replied_resources)
            if resource.media_uri
        )

        async def _build_prompt() -> tuple[list[dict], Optional[list[dict]]]:
            nonlocal admitted_any
            admitted_messages: list[AdmittedMessage] = []
            for item in (pending, *batch):
                admitted = await self._admit_pending_message(
                    item,
                    source="ambient",
                    get_user_nickname=get_user_nickname,
                )
                if admitted is not None:
                    admitted_any = True
                    admitted_messages.append(admitted)
            if not admitted_messages:
                raise _AdmissionAlreadyCommitted
            media_context = await self._build_batch_media_context(
                turn_id, (pending, *batch)
            )
            return await self.prompt_builder.build(
                chat_id=chat_id,
                is_group=True,
                user_nickname=get_user_nickname(input_message.sender_id),
                sender_id=input_message.sender_id,
                input_message=input_message,
                cost_tracker=self.cost_tracker,
                timeline_snapshot=await self._get_timeline().snapshot(chat_id),
                protocol_snapshot=await self._get_protocol_history().snapshot(turn_id),
                delivery_contract=self._delivery_contract(
                    pending.intent, decision.reply_anchor_id or input_message.id
                ),
                media_context=media_context,
            )

        async def _tool_reply_callback(**kwargs):
            return await reply_callback(**kwargs)

        async def _final_reply_callback(**kwargs) -> None:
            nonlocal final_delivered
            content = kwargs.get("content", "")
            delivery, record = await controller.prepare_ambient(
                chat_id=chat_id,
                turn_id=turn_id,
                content=content,
                delivery_mode=self.group_engagement.config.group_ambient_delivery_mode,
                tool_delivered=False,
                reply_anchor_id=decision.reply_anchor_id,
            )
            if not delivery.should_deliver or record is None:
                return
            receipt = await reply_callback(**kwargs)
            if isinstance(receipt, DeliveryReceipt):
                settled = await controller.settle_receipt(
                    record, receipt, content=content
                )
                final_delivered = bool(
                    settled is not None and receipt.status == "accepted"
                )
            else:
                # Keep test/dry-run callbacks compatible until all transports return receipts.
                settled = await controller.mark_sent(record)
                final_delivered = settled is not None

        if not await self.group_engagement.start(decision):
            _log.warning(
                "ambient decision expired before provider start [%s..]", chat_id[:12]
            )
            return
        try:
            turn = await self._run_turn(
                _TurnRequest(
                    chat_id=chat_id,
                    sender_id=input_message.sender_id,
                    is_group=True,
                    reply_to=decision.reply_anchor_id or input_message.id,
                    route_text=input_message.content,
                    prompt_factory=_build_prompt,
                    reply_callback=_final_reply_callback,
                    tool_reply_callback=_tool_reply_callback,
                    tool_reply_names=frozenset({"send_message"}),
                    get_user_nickname=get_user_nickname,
                    model_chain=input_message.model_chain,
                    tier=input_message.tier,
                    rollback_message_id=pending.message.id,
                    steering_enabled=False,
                    capabilities=self._turn_capabilities(
                        pending.intent,
                        chat_id=chat_id,
                        sender_id=input_message.sender_id,
                        reply_to=decision.reply_anchor_id or input_message.id,
                        allowed_media_uris=allowed_media_uris,
                    ),
                    intent=pending.intent,
                    turn_id=turn_id,
                    track_tool_delivery=True,
                )
            )
            await self.group_engagement.complete(
                decision,
                delivered=turn.sent_emoji
                or turn.tool_text_delivered
                or final_delivered,
                silent=turn.final_reply_silent
                or not (turn.sent_emoji or turn.tool_text_delivered or final_delivered),
            )
        except _AdmissionAlreadyCommitted:
            await self.group_engagement.complete(decision, delivered=False, silent=True)
        except Exception:
            await self.group_engagement.complete(decision, delivered=False, silent=True)
            raise

    async def _process_message(
        self,
        pending: PendingInbound,
        reply_callback: Callable,
        get_user_nickname: Callable[[str], str],
        *,
        batch: tuple[PendingInbound, ...] = (),
    ) -> None:
        if not isinstance(pending, PendingInbound):
            raise TypeError(
                "_process_message requires PendingInbound with explicit origin"
            )
        input_message = (batch[-1] if batch else pending).message
        chat_id = input_message.chat_id
        is_group = input_message.is_group
        user_nickname = get_user_nickname(input_message.sender_id)
        system_event_snapshot = []
        capabilities = self._turn_capabilities(
            pending.intent,
            chat_id=chat_id,
            sender_id=input_message.sender_id,
            reply_to=input_message.id,
        )
        model_context_scope = await self._model_context_scope(
            pending, input_message, capabilities=capabilities, batch=batch
        )
        message_ids = {item.message.id for item in (pending, *batch)}
        model_context_identity = (
            self._model_context_identity(input_message)
            if model_context_scope is not None
            else ()
        )
        prompt_state = {
            "scope": model_context_scope,
            "fingerprint": None,
        }

        async def _build_prompt(
            provider_identity: Optional[str] = None,
            *,
            provider_service: Any = None,
            admit: bool = True,
            force_model_context_compaction: bool = False,
        ) -> PromptBuildResult:
            if admit:
                admitted_messages: list[AdmittedMessage] = []
                for item in (pending, *batch):
                    admitted = await self._admit_pending_message(
                        item,
                        source="initial",
                        get_user_nickname=get_user_nickname,
                    )
                    if admitted is not None:
                        admitted_messages.append(admitted)
                if not admitted_messages:
                    raise _AdmissionAlreadyCommitted
            if self._system_events:
                peek_events = getattr(self._system_events, "peek_non_heartbeat", None)
                if peek_events:
                    system_event_snapshot.extend(peek_events(chat_id))
            media_context = await self._build_batch_media_context(
                pending.message.id, (pending, *batch)
            )
            built = await self.prompt_builder.build(
                chat_id=chat_id,
                is_group=is_group,
                user_nickname=user_nickname,
                sender_id=input_message.sender_id,
                input_message=input_message,
                cost_tracker=self.cost_tracker,
                timeline_snapshot=await self._get_timeline().snapshot(chat_id),
                protocol_snapshot=await self._get_protocol_history().snapshot(
                    input_message.id
                ),
                model_context_snapshot=await self._model_context_snapshot(
                    prompt_state["scope"]
                ),
                model_context_scope=prompt_state["scope"],
                model_context_identity=model_context_identity,
                model_context_provider_identity=provider_identity,
                model_context_provider_service=provider_service,
                force_model_context_compaction=force_model_context_compaction,
                delivery_contract=self._delivery_contract(
                    pending.intent, input_message.id
                ),
                media_context=media_context,
            )
            result = (
                built
                if isinstance(built, PromptBuildResult)
                else PromptBuildResult(*built)
            )
            prompt_state["scope"] = result.model_context_scope
            prompt_state["fingerprint"] = result.model_context_fingerprint
            return result

        async def _bind_model_context_provider(service, can_rebuild: bool):
            scope = prompt_state["scope"]
            fingerprint = prompt_state["fingerprint"]
            if scope is None or not fingerprint:
                return None
            provider_identity = self._provider_identity(service)
            if can_rebuild:
                return await _build_prompt(
                    provider_identity, provider_service=service, admit=False
                )
            try:
                bound_scope = await self._get_model_context().ensure_generation(
                    scope,
                    fingerprint,
                    provider_identity=provider_identity,
                )
            except Exception:
                _log.warning(
                    "provider 切换时模型上下文绑定失败 [%s..]，保留当前 turn prompt",
                    chat_id[:12],
                    exc_info=True,
                )
                prompt_state["scope"] = None
                prompt_state["fingerprint"] = None
                return None
            prompt_state["scope"] = bound_scope
            return PromptBuildResult([], [], model_context_scope=bound_scope)

        async def _compact_model_context_on_overflow(service, elapsed_ms):
            overflow_scope = prompt_state["scope"]
            if not self.model_context_write_enabled or overflow_scope is None:
                return None
            self._model_context_overflow_count = (
                getattr(self, "_model_context_overflow_count", 0) + 1
            )
            rebuilt = await _build_prompt(
                self._provider_identity(service),
                provider_service=service,
                admit=False,
                force_model_context_compaction=True,
            )
            rebuilt_scope = getattr(rebuilt, "model_context_scope", None)
            recovered = bool(
                rebuilt is not None
                and rebuilt_scope is not None
                and rebuilt_scope.generation > overflow_scope.generation
            )
            try:
                await self._get_model_context().record_incident(
                    overflow_scope,
                    "context_overflow",
                    provider=self._provider_identity(service),
                    model=getattr(service, "model", ""),
                    recovered=recovered,
                    detail="provider request exceeded context window",
                    elapsed_ms=elapsed_ms,
                )
            except Exception:
                _log.warning("持久化模型上下文 overflow incident 失败", exc_info=True)
            if not recovered:
                return None
            if rebuilt is not None:
                self._model_context_overflow_recovery_count = (
                    getattr(self, "_model_context_overflow_recovery_count", 0) + 1
                )
            return rebuilt

        async def _record_turn_model_context_usage(scope, usage, service, model_name):
            await self._record_model_context_usage(
                scope,
                usage,
                service,
                model_name,
                turn_id=input_message.id,
            )

        async def _admit_steering(
            steering: PendingInbound,
        ) -> Optional[AdmittedMessage]:
            admitted = await self._admit_pending_message(
                steering,
                source="steer",
                get_user_nickname=get_user_nickname,
            )
            message_ids.add(steering.message.id)
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

        delivery_sequence = 0

        async def _deliver_reply(*args, **kwargs):
            nonlocal delivery_sequence
            content = kwargs.get("content")
            if content is None:
                content = args[1] if len(args) > 1 else (args[0] if args else "")
            delivery_sequence += 1
            controller = self._get_delivery_controller()
            record = await controller.prepare_reply_delivery(
                chat_id=chat_id,
                turn_id=input_message.id,
                sequence=delivery_sequence,
                content=content,
                reply_anchor_id=input_message.id,
            )
            try:
                receipt = await reply_callback(
                    chat_id=chat_id,
                    content=content,
                    message_id=input_message.id,
                    is_group=is_group,
                )
            except Exception:
                receipt = DeliveryReceipt(
                    status="failed",
                    logical_delivery_id=record.logical_delivery_id,
                    error_code="transport_exception",
                    retryable=True,
                )
            await controller.settle_tool_delivery(
                record,
                receipt if isinstance(receipt, DeliveryReceipt) else None,
                content=content,
            )
            return receipt

        async def _stream_deliver(chunk: str) -> None:
            try:
                await _deliver_reply(chunk)
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
                    reply_callback=_deliver_reply,
                    get_user_nickname=get_user_nickname,
                    model_chain=input_message.model_chain,
                    tier=input_message.tier,
                    rollback_message_id=input_message.id,
                    stream_callback=_stream_deliver,
                    steering_enabled=True,
                    steering_intent=pending.intent,
                    intent=pending.intent,
                    steering_admission_callback=_admit_steering,
                    rollback_after_prompt_failure_only=True,
                    capabilities=capabilities,
                    turn_id=input_message.id,
                    model_context_commit_callback=(
                        lambda scope: self._materialize_model_context(
                            scope,
                            turn_id=input_message.id,
                            message_ids=message_ids,
                        )
                    ),
                    model_context_provider_callback=(
                        _bind_model_context_provider
                        if model_context_scope is not None
                        else None
                    ),
                    model_context_usage_callback=(
                        _record_turn_model_context_usage
                        if model_context_scope is not None
                        else None
                    ),
                    model_context_overflow_callback=(
                        _compact_model_context_on_overflow
                        if model_context_scope is not None
                        else None
                    ),
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
        auto_media_understanding: bool = False,
        media_refs: tuple[str, ...] = (),
        media_source_chat_id: str = "",
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
            auto_media_understanding: 是否请求显式授权媒体的自动理解（默认关闭）
            media_refs: 调用方显式提供的受控媒体引用，不从 prompt 推断
            media_source_chat_id: 校验媒体授权的来源聊天 ID

        Returns:
            BackgroundTaskResult，支持旧式二元解包。
        """
        _log.info(
            "开始后台任务: chat_id=%s.. prompt_chars=%d", chat_id[:20], len(prompt)
        )

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

        async def _resolve_media_resources() -> list[ResourceMeta]:
            if not auto_media_understanding or not media_refs:
                return []
            source_chat_id = media_source_chat_id or delivery_channel
            if not source_chat_id or self.media_service is None:
                return []
            store = getattr(self.media_service, "store", None)
            authorize = getattr(store, "authorize", None)
            if authorize is None:
                return []
            resources: list[ResourceMeta] = []
            seen: set[str] = set()
            for media_uri in media_refs:
                if (
                    not isinstance(media_uri, str)
                    or not media_uri.startswith("media://inbound/")
                    or "/" in media_uri.removeprefix("media://inbound/")
                    or media_uri in seen
                ):
                    continue
                seen.add(media_uri)
                try:
                    record = await authorize(source_chat_id, media_uri)
                except Exception as exc:
                    _log.warning("后台任务媒体授权失败 [%s]: %s", media_uri, exc)
                    continue
                if record is None:
                    continue
                resources.append(
                    ResourceMeta(
                        resource_type=record.resource_type,
                        media_uri=record.media_uri,
                        media_id=record.media_id,
                        hash=record.sha256,
                        mime_type=record.mime_type,
                        size=record.size,
                        filename=record.filename,
                    )
                )
            return resources

        try:

            async def _build_prompt() -> tuple[list[dict], Optional[list[dict]]]:
                await self._admit_pending_message(
                    PendingInbound(
                        msg,
                        prompt,
                        InboundIntent.PRIVATE_CONVERSATION,
                        AdmissionOrigin.INTERNAL_CONTROL,
                    ),
                    source="initial",
                    get_user_nickname=lambda _: "system",
                )
                media_context = None
                resources = await _resolve_media_resources()
                if resources:
                    source_chat_id = media_source_chat_id or delivery_channel
                    media_message = InputMessage(
                        id=msg.id,
                        sender_id=sender_id,
                        chat_id=source_chat_id,
                        content=prompt,
                        is_group=is_group,
                        resources=resources,
                    )
                    media_pending = PendingInbound(
                        media_message,
                        prompt,
                        InboundIntent.PRIVATE_CONVERSATION,
                        AdmissionOrigin.INTERNAL_CONTROL,
                        resource_refs=tuple(
                            resource.media_uri for resource in resources
                        ),
                    )
                    media_context = await self._build_batch_media_context(
                        msg.id, (media_pending,)
                    )
                return await self.prompt_builder.build_task_messages(
                    chat_id=chat_id,
                    prompt=prompt,
                    tools_allow=tools_allow,
                    media_context=media_context,
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
                    internal_control=True,
                    tool_reply_callback=deliver_tool_reply_callback,
                    tool_reply_names=frozenset({"send_message"}),
                    steering_enabled=False,
                    capabilities=self._turn_capabilities(
                        InboundIntent.PRIVATE_CONVERSATION,
                        chat_id=chat_id,
                        sender_id=sender_id,
                        reply_to=msg.id,
                    ),
                    intent=InboundIntent.DIRECT_TASK,
                    turn_id=msg.id,
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
            id=f"wake_{chat_id}_{_time.time_ns()}",
            sender_id="system",
            chat_id=chat_id,
            content=extra_prompt or "[系统事件]",
            is_group=is_group,
            is_at_mention=False,
        )
        result.turn_id = msg.id

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
                    timeline_snapshot=await self._get_timeline().snapshot(session_key),
                    protocol_snapshot=(),
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
                    internal_control=True,
                    steering_enabled=False,
                    capabilities=self._turn_capabilities(
                        InboundIntent.PRIVATE_CONVERSATION,
                        chat_id=chat_id,
                        sender_id="system",
                        reply_to=msg.id,
                    ),
                    intent=InboundIntent.DIRECT_TASK,
                    turn_id=msg.id,
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
            "cache_observation_count": g.cache_observation_count,
            "cache_usage_missing_count": g.cache_usage_missing_count,
        }

        if getattr(self, "model_context_enabled", False):
            stats["model_context"] = {
                **await self._get_model_context().status(),
                "read_enabled": self.model_context_read_enabled,
                "write_enabled": self.model_context_write_enabled,
                "shadow": self.model_context_shadow,
                "overflow_count": getattr(self, "_model_context_overflow_count", 0),
                "overflow_recovery_count": getattr(
                    self, "_model_context_overflow_recovery_count", 0
                ),
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
        recovery_task = getattr(self, "_delivery_recovery_task", None)
        if recovery_task and not recovery_task.done():
            recovery_task.cancel()
            await asyncio.gather(recovery_task, return_exceptions=True)
        if self._admission_outbox:
            await self._admission_outbox.close()

        model_context = getattr(self, "model_context", None)
        if model_context is not None:
            await model_context.close()

        if self.hindsight:
            await self.hindsight.close()

        _log.info("AgentEngine 已停止")
