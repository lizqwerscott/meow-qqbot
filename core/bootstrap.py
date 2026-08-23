"""ServiceGraph — 服务构造 + 连线 + 生命周期管理。

将 main.py 中的级联初始化封装为可测试的构造器类。
"""

import asyncio
import logging
import sys
from collections.abc import Mapping

import httpx

from core.ai.model_registry import ModelRegistry
from core.ai.multimodal import MultimodalService
from core.ai.tts_service import TtsService
from core.command_handlers import register_all_commands
from core.config_loader import ConfigLoader
from core.engine.agent_engine import AgentEngine
from core.engine.client import BotEngine
from core.engine.context import (
    AIContext,
    BgContext,
    EngineContext,
    MemoryContext,
    MgmtContext,
    PromptContext,
    SubContext,
    SysContext,
)
from core.engine.engagement_config import normalize_engagement_config
from core.engine.hindsight_memory import HindsightMemory
from core.engine.router import Router
from core.engine.system_events import SystemEventQueue
from core.engine.wake_dispatcher import WakeDispatcher
from core.learners.orchestrator import LearningOrchestrator
from core.managers.archive_manager import ArchiveManager
from core.managers.context_compactor import ContextCompactor
from core.managers.context_manager import ChatContextManager
from core.managers.context_store import JSONLContextStore, MemoryContextStore
from core.managers.cost_tracker import CostTracker
from core.managers.emoji_manager import EmojiManager
from core.managers.nickname_manager import NicknameManager
from core.managers.permission_manager import PermissionManager
from core.managers.template_manager import TemplateManager
from core.managers.workspace_manager import WorkspaceManager
from core.media.provider_factory import (
    MediaProviderBuildContext,
    MediaProviderFactory,
)
from core.media.service import MediaService
from core.plugins.manager import PluginManager
from core.rule_router import RuleRouter
from core.tasks import (
    BackgroundTaskRunner,
    CronJobManager,
    CronJobScheduler,
    TaskManager,
    TaskStateStore,
    TaskStore,
)
from core.tasks.heartbeat import HeartbeatManager
from core.tasks.heartbeat_cooldown import HeartbeatCooldown
from core.tools.aggregation import create_all_tool_entries
from core.tools.deps import ToolDeps
from core.tools.impl import registry
from core.tools.process_registry import ProcessRegistry
from core.tools.ref import Ref
from core.tools.skill_managers import SkillManagers
from core.tools.sub_agent_manager import SubAgentManager
from core.web_search.config import WebFetchConfig, WebSearchConfig
from core.web_search.service import WebService
from core.webui import create_app, start_webui

_log = logging.getLogger(__name__)


def _resolve_tts_backend_and_base_url(tts_config: Mapping) -> tuple[str, str]:
    """解析 TTS 后端及其显式或后端默认地址。"""
    backend, backend_config = TtsService.resolve_backend(
        tts_config.get("backend", "voxcpm")
    )
    return backend, tts_config.get("base_url", backend_config.default_base_url)


class ServiceGraph:
    """服务依赖图 — 单次 build() 完成所有构造 + 连线。"""

    def __init__(self, cfg: ConfigLoader):
        self.cfg = cfg
        self.http_client = None
        self.model_registry = None
        self.media_service = None
        self.bot_engine = None
        self.webui_task = None

    # ── build: 三阶段构造 ──────────────────────────────────────────

    async def build(self):
        await self._build_services()
        self._build_bot_engine()
        self._wire_callbacks()
        await self._deliver_pending_task_recovery()
        self._setup_extras()
        await self.agent_engine.start()
        return self

    # ── 阶段 1: 构造所有服务 ───────────────────────────────────────

    async def _build_services(self):

        self.http_client = httpx.AsyncClient(timeout=60.0)
        self.heartbeat_manager = None
        self._wake_runner = None

        self.template_manager = TemplateManager(self.cfg.character_card)

        providers_config = self.cfg.providers
        groups_config = self.cfg.groups
        _has_model_config = bool(providers_config and groups_config)

        self.model_registry = None
        if _has_model_config:
            self.model_registry = ModelRegistry(
                providers_config,
                groups_config,
                cooldown_config=self.cfg.cooldown,
            )
            n_models = sum(len(p.get("models", [])) for p in providers_config.values())
            _log.info(
                "模型注册表已初始化: %d 个模型, %d 个组", n_models, len(groups_config)
            )

        if not self.model_registry or not self.model_registry.default_service:
            raise RuntimeError(
                "未配置模型注册表（缺少 [providers] 或 [groups]），无法启动"
            )
        self.ai_service = self.model_registry.default_service

        # ── 多模态 ──
        multimodal_config = self.cfg.multimodal
        self.multimodal_service = None
        if multimodal_config.get("enabled", False):
            multimodal_group = multimodal_config.get("group", "multimodal")
            chain = self.model_registry.get_chain(multimodal_group)
            raw = [self.model_registry.get(n) for n in chain]
            pairs = [(s, n) for s, n in zip(raw, chain) if s is not None]
            if pairs:
                services, model_names = zip(*pairs)
                self.multimodal_service = MultimodalService(
                    list(services),
                    model_names=list(model_names),
                    cooldown_manager=self.model_registry.cooldown_manager,
                )
                _log.info(
                    "多模态服务已启用 (组 [%s]): %s",
                    multimodal_group,
                    list(model_names),
                )
            else:
                _log.warning(
                    "多模态服务未启用: 组 [%s] 找不到对应模型", multimodal_group
                )
        else:
            _log.info("多模态服务未启用（enabled=false），跳过 VLM 图片分析")

        media_config = self.cfg.media
        media_config = media_config if isinstance(media_config, Mapping) else {}
        voice_config = (
            media_config.get("voice_transcription", {})
            if isinstance(media_config, Mapping)
            else {}
        )
        voice_config = voice_config if isinstance(voice_config, Mapping) else {}
        image_config = (
            media_config.get("image_understanding", {})
            if isinstance(media_config, Mapping)
            else {}
        )
        image_config = image_config if isinstance(image_config, Mapping) else {}
        download_config = (
            media_config.get("download", {})
            if isinstance(media_config, Mapping)
            else {}
        )
        download_config = (
            download_config if isinstance(download_config, Mapping) else {}
        )
        provider_chains = MediaProviderFactory.build(
            MediaProviderBuildContext(
                model_registry=self.model_registry,
                providers_config=providers_config,
                media_config=media_config,
                http_client=self.http_client,
                multimodal=self.multimodal_service,
            )
        )
        self.media_service = MediaService(
            http_client=self.http_client,
            multimodal=self.multimodal_service,
            storage_dir=media_config.get("storage_dir", "data/media"),
            enabled=media_config.get("enabled", True),
            image_understanding=image_config,
            recent_window_seconds=media_config.get("recent_window_seconds", 600),
            recent_max_items=media_config.get("recent_max_items", 5),
            max_attachments_per_message=media_config.get(
                "max_attachments_per_message", 5
            ),
            max_image_bytes=download_config.get("max_image_bytes", 10 * 1024 * 1024),
            max_file_bytes=download_config.get("max_file_bytes", 25 * 1024 * 1024),
            download_timeout=download_config.get("timeout_seconds", 15),
            download_concurrency=download_config.get("concurrency", 4),
            max_total_bytes=media_config.get("max_total_bytes", 2 * 1024 * 1024 * 1024),
            preview_enabled=self.cfg.webui.get("media_preview", {}).get(
                "enabled", True
            ),
            preview_max_inline_bytes=self.cfg.webui.get("media_preview", {}).get(
                "max_inline_bytes", 50 * 1024 * 1024
            ),
            text_preview_max_chars=self.cfg.webui.get("media_preview", {}).get(
                "text_preview_max_chars", 20_000
            ),
            voice_transcriber=None,
            voice_transcription=voice_config,
            ai_service=self.ai_service,
            provider_chains=provider_chains,
        )

        # ── EmojiManager ──
        self.emoji_manager = EmojiManager(
            http_client=self.http_client,
            multimodal_service=self.multimodal_service,
            emoji_dir="data/emojis/",
        )

        # ── 上下文管理 ──
        ctx_mgmt = self.cfg.context_management
        _merge_ws = ctx_mgmt.get("merge_window_seconds", 15)
        _cache_cfg = ctx_mgmt.get("cache", {})
        _cache_dir = (
            (_cache_cfg.get("dir") or "data/sessions/")
            if _cache_cfg.get("enabled", True)
            else None
        )
        _store = (
            JSONLContextStore(base_dir=_cache_dir)
            if _cache_dir
            else MemoryContextStore()
        )
        self.context_compactor = ContextCompactor(
            ai_service=self.ai_service,
            compact_threshold_tokens=ctx_mgmt.get("compact_threshold_tokens", 950000),
            keep_recent_tokens=ctx_mgmt.get("keep_recent_tokens", 50000),
        )
        self.context_manager = ChatContextManager(
            store=_store,
            compactor=self.context_compactor,
            max_history_per_chat=ctx_mgmt.get("max_history", 10000),
            max_tool_results=ctx_mgmt.get("max_tool_results", 5),
            keep_last_assistants=ctx_mgmt.get("keep_last_assistants", 3),
            soft_trim=ctx_mgmt.get("soft_trim", 20000),
            hard_clear=ctx_mgmt.get("hard_clear", 180000),
            merge_window_seconds=_merge_ws,
        )

        # ── ArchiveManager ──
        archive_config = self.cfg.archive
        self.archive_manager = None
        if archive_config.get("enabled", True):
            self.archive_manager = ArchiveManager(
                context_manager=self.context_manager,
                memory_dir=archive_config.get("memory_dir", "data/archives/memory/"),
                archive_hour=archive_config.get("archive_hour", 4),
                replay_count=archive_config.get("replay_count"),
                replay_gap_seconds=archive_config.get("replay_gap_seconds", 600),
                summary_count=archive_config.get("summary_count", 15),
                summary_days=archive_config.get("summary_days", 2),
                retention_days=archive_config.get("retention_days", 30),
                merge_window_seconds=_merge_ws,
            )
            _log.info(
                "归档系统已启用 (跨天消息触发, 摘要 %d 条, 回放连续段间隔 %d 秒)",
                archive_config.get("summary_count", 15),
                archive_config.get("replay_gap_seconds", 600),
            )

        # ── CostTracker ──
        cost_tracking_config = self.cfg.cost_tracking
        self.cost_tracker = (
            CostTracker(pricing=cost_tracking_config.get("pricing"))
            if cost_tracking_config.get("enabled", True)
            else CostTracker()
        )

        # ── Hindsight 记忆 ──
        hindsight_config = self.cfg.hindsight
        self.hindsight_memory = None
        if hindsight_config.get("enabled", True):
            self.hindsight_memory = HindsightMemory(
                base_url=hindsight_config.get("base_url", "http://127.0.0.1:8888"),
                bank_id=hindsight_config.get("bank_id", "qq_bot"),
            )
            _log.info("Hindsight 记忆系统已启用: %s", hindsight_config.get("base_url"))
        else:
            _log.info("Hindsight 记忆系统未启用")

        # ── 后台任务系统 ──
        self.context_cleanup_task = None
        tasks_config = self.cfg.tasks
        self.task_manager = None
        self.task_state_store = None
        self.cron_job_manager = None
        self.background_task_runner = None
        self.cron_scheduler = None
        self._tasks_config = tasks_config

        if tasks_config.get("enabled", True):
            scheduler_cfg = tasks_config.get("scheduler", {})
            task_store = TaskStore(
                data_dir=tasks_config.get("data_dir", "data/tasks/"),
                max_tasks=tasks_config.get("max_tasks", 10000),
                terminal_ttl_hours=tasks_config.get("terminal_ttl_hours", 168),
                lost_ttl_hours=tasks_config.get("lost_ttl_hours", 24),
            )
            self.task_manager = TaskManager(store=task_store)
            self.task_state_store = TaskStateStore(
                tasks_config.get("data_dir", "data/tasks/")
            )
            interrupted_turns = (
                await self.task_state_store.mark_interrupted_on_restart()
            )
            if interrupted_turns:
                _log.warning("已封存 %d 个上次进程遗留 turn", len(interrupted_turns))
            interrupted_tasks = (
                await self.task_manager.interrupt_active_tasks_on_restart()
            )
            if interrupted_tasks:
                _log.warning(
                    "已将 %d 个上次进程遗留任务标记为 LOST，不会自动重放",
                    len(interrupted_tasks),
                )
            self.cron_job_manager = CronJobManager(store=task_store)
            self.background_task_runner = BackgroundTaskRunner(
                task_manager=self.task_manager
            )

            if scheduler_cfg.get("enabled", True):
                self.cron_scheduler = CronJobScheduler(
                    poll_interval=scheduler_cfg.get("poll_interval", 30),
                    catch_up_window=scheduler_cfg.get("catch_up_window", 3600),
                    max_concurrent=scheduler_cfg.get("max_concurrent", 3),
                )
            _log.info("后台任务系统已初始化")

        self.bot_id = self.cfg.bot_id

        # ── Permission + Workspace + Nickname ──
        self.permission_manager = PermissionManager("config/allowlist.toml")
        self.admin_ids = self.permission_manager.get_role_ids("admin")
        if self.background_task_runner:
            # cron 命令载荷的 login shell 开关与 exec 工具实时一致
            self.background_task_runner.set_permission_manager(self.permission_manager)
        workspace_config = self.cfg.workspace
        self.workspace_manager = WorkspaceManager(
            root=workspace_config.get("root", "workspaces"),
        )
        self.nickname_manager = NicknameManager(bot_id=self.bot_id)

        # ── Skills ──
        self.skill_managers = SkillManagers(
            project_skill_dir="./.agents/skills/",
            permission_manager=self.permission_manager,
        )

        # ── RuleRouter ──
        self.rule_router = None
        routing_enabled = self.cfg.routing.get("enabled", False)
        if routing_enabled and self.model_registry:
            tier_config = self.cfg.routing.get("tiers", {})
            self.model_registry.configure_tiers(tier_config)
            self.rule_router = RuleRouter()
            _log.info("ClawRouter 规则路由已初始化")

        # ── Learning ──
        learners_config = self.cfg.learners
        self.learning_orchestrator = None
        if learners_config.get("enabled", True):
            self.learning_orchestrator = LearningOrchestrator(
                config=learners_config,
                ai_service=self.ai_service,
                data_dir=learners_config.get("data_dir", "data/learners/"),
                emoji_manager=self.emoji_manager,
            )
            _log.info("学习系统已启用")
        else:
            _log.info("学习系统未启用")

        # ── 系统事件队列 ──
        self.system_events = SystemEventQueue()
        _log.info("SystemEventQueue 已初始化")

        # ── 网页搜索 / 抓取 ──
        self.web_service = None
        web_search_cfg = WebSearchConfig.from_dict(self.cfg.web_search)
        web_fetch_cfg = WebFetchConfig.from_dict(self.cfg.web_fetch)
        if web_search_cfg.enabled or web_fetch_cfg.enabled:
            self.web_service = WebService(
                search_cfg=web_search_cfg,
                fetch_cfg=web_fetch_cfg,
                http_client=self.http_client,
            )
            _log.info(
                "WebService 已初始化: search链=%s, fetch链=%s",
                web_search_cfg.providers if web_search_cfg.enabled else "(disabled)",
                web_fetch_cfg.providers if web_fetch_cfg.enabled else "(disabled)",
            )
        else:
            _log.info("网页搜索/抓取未启用（[web_search]/[web_fetch] 均未开启）")

        # ── 后台进程注册表 ──
        self.process_registry = ProcessRegistry()
        _log.info("ProcessRegistry 已初始化")

        # ── 子智能体 ──
        sub_agent_config = self.cfg.sub_agents
        self.sub_agent_manager = None
        if sub_agent_config.get("enabled", True):
            self.sub_agent_manager = SubAgentManager(
                max_concurrent=sub_agent_config.get("max_concurrent", 4),
                max_children=sub_agent_config.get("max_children", 5),
                run_timeout=sub_agent_config.get("run_timeout", 900),
                system_events=self.system_events,
            )
            _log.info(
                "子智能体系统已启用 (max_concurrent=%d, max_children=%d, run_timeout=%ds)",
                sub_agent_config.get("max_concurrent", 4),
                sub_agent_config.get("max_children", 5),
                sub_agent_config.get("run_timeout", 900),
            )
        else:
            _log.info("子智能体系统未启用")

        # ── TTS ──
        tts_config = self.cfg.tts
        self.tts_service = None
        if tts_config.get("enabled", False):
            tts_backend, tts_base_url = _resolve_tts_backend_and_base_url(tts_config)
            self.tts_service = TtsService(
                base_url=tts_base_url,
                http_client=self.http_client,
                model=tts_config.get("model", "voxcpm"),
                temp_dir=tts_config.get("temp_dir", "data/tts_temp/"),
                ref_audio=tts_config.get("ref_audio"),
                ref_text=tts_config.get("ref_text"),
                cfg_value=tts_config.get("cfg_value"),
                inference_timesteps=tts_config.get("inference_timesteps"),
                temperature=tts_config.get("temperature"),
                seed=tts_config.get("seed"),
                max_steps=tts_config.get("max_steps"),
                backend=tts_backend,
                s2_params=tts_config.get("s2_params"),
            )
            _log.info(
                "TTS 语音服务已初始化 (backend=%s, base_url=%s, model=%s)",
                tts_backend,
                tts_base_url,
                tts_config.get("model", "voxcpm"),
            )

        # ── EngineContext ──
        ai_config = self.cfg.ai or {}
        engagement_config = normalize_engagement_config(ai_config)
        ctx = EngineContext(
            ai=AIContext(
                ai_service=self.ai_service,
                model_registry=self.model_registry,
                rule_router=self.rule_router,
                multimodal_service=self.multimodal_service,
                max_tool_rounds=self.cfg.max_tool_rounds,
                stream_reply=ai_config.get("stream_reply", False),
                stream_block_chars=ai_config.get("stream_block_chars", 800),
                stream_block_idle_ms=ai_config.get("stream_block_idle_ms", 1000),
                engagement_config=engagement_config,
            ),
            prompt=PromptContext(
                template_manager=self.template_manager,
                nickname_manager=self.nickname_manager,
                emoji_manager=self.emoji_manager,
                skill_managers=self.skill_managers,
                learning_orchestrator=self.learning_orchestrator,
            ),
            memory=MemoryContext(
                hindsight_memory=self.hindsight_memory,
                search_top_k=hindsight_config.get("search_top_k", 5),
            ),
            mgmt=MgmtContext(
                context_manager=self.context_manager,
                permission_manager=self.permission_manager,
                cost_tracker=self.cost_tracker,
                workspace_manager=self.workspace_manager,
                archive_manager=self.archive_manager,
                system_events=self.system_events,
                task_state_store=self.task_state_store,
                model_context_config=self.cfg.model_context_projection,
            ),
            bg=BgContext(
                task_manager=self.task_manager,
                cron_job_manager=self.cron_job_manager,
            ),
            sub=SubContext(sub_agent_manager=self.sub_agent_manager),
            sys=SysContext(bot_id=self.bot_id, admin_ids=tuple(self.admin_ids)),
        )

        # ── AgentEngine ──
        self.agent_engine = AgentEngine(ctx)
        self.agent_engine.set_media_service(self.media_service)
        self.context_manager.set_timeline(self.agent_engine.timeline)
        if self.archive_manager is not None:
            self.archive_manager.set_timeline(self.agent_engine.timeline)

        # ── 注入 TTS ──
        if self.tts_service:
            self.agent_engine.set_tts_service(self.tts_service)

        # ── 构造 ToolDeps 并注册工具 ──
        self.tool_deps = ToolDeps(
            emoji_manager=self.emoji_manager,
            nickname_manager=self.nickname_manager,
            skill_managers=self.skill_managers,
            hindsight=self.hindsight_memory,
            learning_orchestrator=self.learning_orchestrator,
            permission_manager=self.permission_manager,
            workspace_manager=self.workspace_manager,
            sub_agent_manager=self.sub_agent_manager,
            system_events=self.system_events,
            search_top_k=hindsight_config.get("search_top_k", 5),
            admin_ids=list(self.admin_ids),
            bot_id=self.bot_id,
            media_service=self.media_service,
            web=self.web_service,
            media_uploader=Ref(),
            bot_engine=Ref(),
            api_client=Ref(),
            tts_service=Ref(self.tts_service),
            process_registry=Ref(self.process_registry),
            approval_manager=Ref(),
            task_manager=Ref(self.task_manager),
            cron_job_manager=Ref(self.cron_job_manager),
            background_task_runner=Ref(self.background_task_runner),
        )
        self.agent_engine._deps = self.tool_deps
        self.agent_engine.prompt_builder._deps = self.tool_deps

        entries = create_all_tool_entries(self.tool_deps)
        for entry in entries:
            registry.register(entry)
        _log.info("工具系统已初始化: %d 个工具", len(entries))

    # ── 阶段 2: 构造 BotEngine ─────────────────────────────────────

    def _build_bot_engine(self):
        router = Router(agent_engine=self.agent_engine)
        self.bot_engine = BotEngine(
            app_id=self.cfg.appid,
            client_secret=self.cfg.secret,
            bot_id=self.bot_id,
            agent_engine=self.agent_engine,
            router=router,
            admin_id=self.admin_ids,
            permission_manager=self.permission_manager,
            nickname_manager=self.nickname_manager,
            emoji_manager=self.emoji_manager,
            multimodal_service=self.multimodal_service,
            media_service=self.media_service,
        )

    async def _deliver_pending_task_recovery(self) -> None:
        if not self.task_manager or not self.background_task_runner:
            return
        for task in self.task_manager.list_restart_recovery_tasks():
            is_group = bool(
                self.context_manager.get_chat_type(task.delivery_channel or "")
            )
            await self.background_task_runner.deliver_restart_recovery(
                task, is_group=is_group
            )

    # ── 阶段 3: 交叉连线 ────────────────────────────────────────────

    def _wire_callbacks(self):
        self.agent_engine.set_reply_callback(self.bot_engine._send_reply)

        # ── 更新后台任务 Ref ──
        if self.task_manager or self.cron_job_manager or self.background_task_runner:
            self.tool_deps.task_manager.value = self.task_manager
            self.tool_deps.cron_job_manager.value = self.cron_job_manager
            self.tool_deps.background_task_runner.value = self.background_task_runner
            self.tool_deps.process_registry.value = self.process_registry
            _log.info("任务管理器 + 进程注册表已注入工具系统")
        else:
            self.tool_deps.process_registry.value = self.process_registry

        # ── 后台任务执行器连线 ──
        if self.background_task_runner:
            self.background_task_runner.set_system_events(self.system_events)

        if self.background_task_runner and self.task_manager:
            self.background_task_runner.set_execute_callback(
                self.agent_engine.execute_background_task
            )

            async def _deliver(chat_id, content, message_id, is_group, delivery_id=""):
                actual = self.context_manager.get_chat_type(chat_id)
                if actual is not None:
                    is_group = actual

                async def transport(**kwargs):
                    return await self.bot_engine.send_proactive(
                        kwargs["chat_id"],
                        kwargs["content"],
                        is_group=kwargs["is_group"],
                    )

                return await self.agent_engine._get_delivery_controller().deliver_text(
                    delivery_id=delivery_id or f"background:{chat_id}",
                    chat_id=chat_id,
                    content=content,
                    callback=transport,
                    message_id=message_id,
                    is_group=is_group,
                )

            self.background_task_runner.set_delivery_callback(_deliver)

        # ── Cron 调度器连线 ──
        if self.cron_scheduler and self.cron_job_manager:
            self.cron_scheduler.set_callbacks(
                on_trigger=lambda job: self.background_task_runner.run_cron_job(
                    job=job,
                    timeout=self.cfg.tasks.get("scheduler", {}).get(
                        "task_timeout", 300
                    ),
                ),
                get_jobs=self.cron_job_manager.list_jobs,
                update_job=self.cron_job_manager.update_job,
                delete_job=self.cron_job_manager.delete_job,
            )

        # ── 注入 BotEngine 到工具系统 ──
        self.tool_deps.bot_engine.value = self.bot_engine

        # ── 审批系统 ──
        from core.approval.approval_manager import ApprovalManager

        # 2.3：审批卡可转发到其他会话（[approval].forward_to，格式 c2c:<id>/group:<id>）
        approval_cfg = getattr(self.cfg, "approval", {}) or {}
        self.approval_manager = ApprovalManager(
            api_client=self.bot_engine.api,
            admin_ids=self.admin_ids,
            forward_to=list(approval_cfg.get("forward_to") or ()),
            delivery_controller=self.agent_engine._get_delivery_controller(),
        )
        self.tool_deps.approval_manager.value = self.approval_manager
        self.bot_engine.approval_manager = self.approval_manager
        _log.info("ApprovalManager 已初始化")

        # ── exec auto-reviewer（[exec].auto_reviewer 启用时，mode=auto 用）──
        self._setup_exec_reviewer()

        # ── 兼容层 WakeDispatcher ──
        min_spacing = self.cfg.heartbeat.get("min_spacing_seconds", 30)
        flood_window = self.cfg.heartbeat.get("cooldown_flood_window_seconds", 60)
        flood_threshold = self.cfg.heartbeat.get("cooldown_flood_threshold", 5)
        self._cooldown = HeartbeatCooldown(
            min_spacing_seconds=min_spacing,
            flood_window_seconds=flood_window,
            flood_threshold=flood_threshold,
        )
        self.wake_dispatcher = WakeDispatcher(
            system_events=self.system_events,
        )
        # BackgroundTaskRunner 需保留 wake_dispatcher 引用用于 NOW 模式
        if self.background_task_runner:
            self.background_task_runner.set_wake_dispatcher(self.wake_dispatcher)

    # ── exec auto-reviewer 注入（mode=auto 的生产接线）──

    def _setup_exec_reviewer(self):
        """按 [exec].auto_reviewer 配置创建 ExecAutoReviewer 并注入 tool_deps。

        review_fn 用 ModelRegistry.simple_chat 走轻量模型判定（allow/ask）；
        未启用或模型不可用时保持 exec_reviewer=None，mode=auto 降级为人工审批。
        """
        from core.approval.auto_reviewer import REVIEW_PROMPT, ExecAutoReviewer

        ar_cfg = {}
        if self.permission_manager:
            ar_cfg = (
                self.permission_manager.get_exec_policy().get("auto_reviewer") or {}
            )
        if not ar_cfg.get("enabled") or not self.model_registry:
            _log.info("exec auto-reviewer 未启用（mode=auto 时降级人工审批）")
            return
        model_name = ar_cfg.get("model") or ""
        if not model_name:
            _log.warning("auto_reviewer 未配置 model，跳过注入")
            return

        async def _review_fn(plan: dict) -> str:
            prompt = REVIEW_PROMPT.format(
                command=plan.get("command", ""),
                cwd=plan.get("cwd", "") or "",
                resolved_path=plan.get("resolved_path", "") or "",
                role=plan.get("role", ""),
            )
            reply = await self.model_registry.simple_chat(
                model_name,
                [{"role": "user", "content": prompt}],
                max_tokens=16,
            )
            text = (reply or "").strip().lower()
            if text.startswith("allow"):
                return "allow"
            return "ask"

        self.exec_reviewer = ExecAutoReviewer(review_fn=_review_fn)
        self.tool_deps.exec_reviewer.value = self.exec_reviewer
        _log.info("exec auto-reviewer 已注入 (model=%s)", model_name)

    # ── 阶段 4: 心跳 / 命令 / 插件 / WebUI ─────────────────────────

    def _setup_extras(self):
        # ── 心跳 ──
        if self.cfg.heartbeat.get("enabled", False):
            self.heartbeat_manager = HeartbeatManager(
                config=self.cfg.heartbeat,
                api_client=self.bot_engine.api,
                admin_ids=self.admin_ids,
                context_manager=self.context_manager,
                agent_engine=self.agent_engine,
                wake_dispatcher=self.wake_dispatcher,
                heartbeat_path=str(self.workspace_manager.heartbeat_path()),
                cooldown=self._cooldown,
            )

        # ── WakeCoalescer + WakeRunner + Delivery ──
        from core.tasks.delivery_strategy import (
            ChatReplyDeliveryStrategy,
            HeartbeatDeliveryStrategy,
        )
        from core.tasks.wake_coalescer import set_wake_handler
        from core.tasks.wake_runner import WakeRunner

        chat_delivery = ChatReplyDeliveryStrategy(
            reply_callback=self.bot_engine._send_reply,
            context_manager=self.context_manager,
            delivery_controller=self.agent_engine._get_delivery_controller(),
        )

        hb_delivery = None
        if self.heartbeat_manager is not None:
            show_ok = self.cfg.heartbeat.get("show_ok", False)
            show_alerts = self.cfg.heartbeat.get("show_alerts", True)
            hb_delivery = HeartbeatDeliveryStrategy(
                self.heartbeat_manager,
                reply_callback=self.bot_engine._send_reply,
                context_manager=self.context_manager,
                show_ok=show_ok,
                show_alerts=show_alerts,
                delivery_controller=self.agent_engine._get_delivery_controller(),
            )

        active_hours_cfg = self.cfg.heartbeat.get("active_hours", {})
        ah = (
            active_hours_cfg.get("start"),
            active_hours_cfg.get("end"),
            active_hours_cfg.get("timezone", "Asia/Shanghai"),
        )

        delivery_strategies: dict = {}
        cron_hb_delivery = hb_delivery or chat_delivery  # 心跳禁用时回退到 chat 投递
        if hb_delivery:
            delivery_strategies["interval"] = hb_delivery
            delivery_strategies["manual"] = hb_delivery
        delivery_strategies["cron-heartbeat"] = cron_hb_delivery
        delivery_strategies["cron"] = chat_delivery

        hb_isolated_key_fn = (
            self.heartbeat_manager.resolve_isolated_session_key
            if self.heartbeat_manager
            else None
        )

        self._wake_runner = WakeRunner(
            agent_engine=self.agent_engine,
            system_events=self.system_events,
            cooldown=self._cooldown,
            delivery_strategies=delivery_strategies,
            active_hours=ah,
            session_active_check=self.agent_engine.is_session_active,
            has_cron_check=lambda: (
                bool(self.cron_job_manager and self.cron_job_manager.list_jobs())
                if hasattr(self, "cron_job_manager")
                else False
            ),
            main_lane_busy_check=lambda: False,  # 可扩展：从 BotEngine 获取队列状态
            agent_busy_check=lambda: False,  # 可扩展：从 AgentEngine 获取
            skip_when_busy=self.cfg.heartbeat.get("skip_when_busy", False),
            session_lane_busy_check=lambda _: False,
            isolated_session_key_fn=hb_isolated_key_fn,
            delivery_pending_check=(
                self.heartbeat_manager.is_delivery_pending
                if self.heartbeat_manager
                else lambda: False
            ),
        )
        set_wake_handler(self._wake_runner)
        _log.info("WakeCoalescer + WakeRunner 已初始化")

        # ── exec 进程退出回调 ──
        async def _on_exec_exit(session):
            chat_id = session.delivery_channel or session.chat_id
            exit_code = session.exit_code
            status = "成功" if exit_code == 0 else f"失败 (exit={exit_code})"
            stdout = "".join(session.stdout_lines[-5:]) if session.stdout_lines else ""
            extra = f"后台进程 [{session.id[:8]}..] 已退出: {status}\n命令: {session.command[:100]}"
            if stdout:
                extra += f"\n最后输出:\n{stdout[:500]}"
            import core.tasks.wake_coalescer as _coalescer

            _coalescer.request_wake(
                source="exec-event",
                intent="event",
                session_key=chat_id,
                delivery_target=chat_id,
                extra_prompt=extra,
                reason=f"后台进程完成: {session.command[:80]}",
            )

        self.process_registry.on_exit(_on_exec_exit)

        # ── 命令注册 ──
        register_all_commands(
            self.bot_engine.command_manager,
            context_manager=self.context_manager,
            timeline=self.agent_engine.timeline,
            emoji_manager=self.emoji_manager,
            agent_engine=self.agent_engine,
            skill_managers=self.skill_managers,
            learning_orchestrator=self.learning_orchestrator,
            api_client=self.bot_engine.api,
            bot_engine=self.bot_engine,
            ai_service=self.ai_service,
            task_manager=self.task_manager,
            cron_job_manager=self.cron_job_manager,
            background_task_runner=self.background_task_runner,
            heartbeat_manager=self.heartbeat_manager,
            archive_manager=self.archive_manager,
            tts_service=self.tts_service,
            delivery_controller=self.agent_engine._get_delivery_controller(),
            approval_manager=self.approval_manager,
            media_service=self.media_service,
        )

        # ── 插件加载 ──
        plugin_manager = PluginManager(plugin_dir="plugins")
        plugin_manager.load_all(
            command_manager=self.bot_engine.command_manager,
            context_manager=self.context_manager,
            emoji_manager=self.emoji_manager,
            agent_engine=self.agent_engine,
            skill_managers=self.skill_managers,
            api_client=self.bot_engine.api,
            bot_engine=self.bot_engine,
        )

        # ── WebUI ──
        webui_config = self.cfg.webui
        if webui_config.get("enabled", False):
            webui_app = create_app(
                managers={
                    "emoji_manager": self.emoji_manager,
                    "nickname_manager": self.nickname_manager,
                    "context_manager": self.context_manager,
                    "conversation_timeline": self.agent_engine.timeline,
                    "cost_tracker": self.cost_tracker,
                    "agent_engine": self.agent_engine,
                    "learning_orchestrator": self.learning_orchestrator,
                    "archive_manager": self.archive_manager,
                    "media_service": self.media_service,
                },
                webui_config=webui_config,
            )
            _webui_host = webui_config.get("host", "127.0.0.1")
            _webui_port = webui_config.get("port", 8080)
            _log.info("WebUI 管理面板将在 http://%s:%d 启动", _webui_host, _webui_port)
            if _webui_host in ("0.0.0.0", "::"):
                _log.info("局域网内可通过 http://<本机IP>:%d 访问", _webui_port)
            self.webui_task = asyncio.create_task(start_webui(webui_app, webui_config))

    # ── Hindsight 健康检查（async，需单独调用） ────────────────────

    async def check_hindsight_health(self):
        if self.hindsight_memory:
            health_result = await self.hindsight_memory.health()
            if health_result.get("status") == "ok":
                _log.info(
                    "Hindsight 健康检查通过 (%sms)", health_result.get("latency_ms")
                )
            else:
                _log.warning(
                    "Hindsight 健康检查失败: %s — 记忆功能将降级运行",
                    health_result.get("error"),
                )

    # ── start / stop ───────────────────────────────────────────────

    async def start(self):
        """启动所有后台服务。"""
        await self.check_hindsight_health()
        await self.media_service.open()

        gateway_url = await self.bot_engine.api.get_gateway_url()
        loop = asyncio.get_running_loop()
        self.bot_engine.start(gateway_url, loop)
        print(f"机器人已启动，WebSocket 网关: {gateway_url}")

        await self.process_registry.start()

        if self.cron_scheduler:
            self.cron_scheduler.start()

        if self.heartbeat_manager:
            await self.heartbeat_manager.start()

        self.task_cleanup_task = None
        if self.task_manager:
            self._start_task_cleanup()

        self._start_context_cleanup()

    def _start_context_cleanup(self):
        async def _periodic_cleanup():
            while True:
                await asyncio.sleep(3600)
                try:
                    removed = (
                        await self.context_manager.cleanup_inactive_contexts_async(
                            max_inactivity=7200
                        )
                    )
                    if removed:
                        _log.info("上下文清理: 移除了 %d 个不活跃会话", len(removed))
                except Exception as e:
                    _log.warning("上下文清理异常: %s", e)

        self.context_cleanup_task = asyncio.create_task(_periodic_cleanup())
        _log.info("定期上下文清理任务已启动 (周期 3600s)")

    def _start_task_cleanup(self):
        tasks_config = self._tasks_config
        lost_detection_minutes = tasks_config.get("lost_detection_minutes", 30)
        max_terminal_per_job = tasks_config.get("max_terminal_per_job", 2000)

        async def _run_cycle():
            lost = await self.task_manager.detect_lost_tasks(lost_detection_minutes)
            cleaned = await self.task_manager.cleanup_old_tasks()
            capped = await self.task_manager.enforce_per_job_terminal_limit(
                max_terminal_per_job
            )
            if lost or cleaned or capped:
                _log.info(
                    "任务清理周期: %d 丢失标记, %d TTL 清理, %d job 上限裁剪",
                    lost,
                    cleaned,
                    capped,
                )

        async def _periodic_cleanup():
            try:
                await _run_cycle()
            except Exception as e:
                _log.warning("定时清理任务异常(首次): %s", e)
            while True:
                await asyncio.sleep(3600)
                try:
                    await _run_cycle()
                except Exception as e:
                    _log.warning("定时清理任务异常: %s", e)

        self.task_cleanup_task = asyncio.create_task(_periodic_cleanup())

    async def _safe_cleanup(self, name, cleanup):
        try:
            await cleanup()
        except asyncio.CancelledError:
            _log.warning(
                "服务关闭步骤取消 step=%s category=cancelled",
                name,
            )
        except Exception as exc:
            _log.warning(
                "服务关闭步骤失败 step=%s category=%s",
                name,
                type(exc).__name__,
            )

    async def _cancel_task(self, name):
        task = getattr(self, name, None)
        setattr(self, name, None)
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            _log.warning(
                "后台任务关闭失败 step=%s category=%s",
                name,
                type(exc).__name__,
            )

    async def stop(self):
        """优雅关闭；单个步骤失败不阻断其他资源清理。"""
        process_registry = getattr(self, "process_registry", None)
        if process_registry:
            await self._safe_cleanup("process_registry", process_registry.stop)

        cron_scheduler = getattr(self, "cron_scheduler", None)
        if cron_scheduler:
            await self._safe_cleanup("cron_scheduler", cron_scheduler.stop)

        heartbeat_manager = getattr(self, "heartbeat_manager", None)
        if heartbeat_manager:
            await self._safe_cleanup("heartbeat_manager", heartbeat_manager.stop)

        bot_engine = getattr(self, "bot_engine", None)
        self.bot_engine = None
        if bot_engine:
            await self._safe_cleanup("bot_engine", bot_engine.stop)

        media_service = getattr(self, "media_service", None)
        self.media_service = None
        if media_service:
            await self._safe_cleanup("media_service", media_service.close)

        await self._cancel_task("task_cleanup_task")
        await self._cancel_task("context_cleanup_task")
        await self._cancel_task("webui_task")

        tts_service = getattr(self, "tts_service", None)
        self.tts_service = None
        if tts_service:
            await self._safe_cleanup("tts_service", tts_service.close)

        model_registry = getattr(self, "model_registry", None)
        self.model_registry = None
        if model_registry:
            await self._safe_cleanup("model_registry", model_registry.close)

        http_client = getattr(self, "http_client", None)
        self.http_client = None
        if http_client is not None:
            await self._safe_cleanup("http_client", http_client.aclose)
