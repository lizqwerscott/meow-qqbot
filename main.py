import asyncio
import os

import yaml
import logging
import httpx
from colorlog import ColoredFormatter

from core.ai.service import AIService
from core.engine.agent_engine import AgentEngine
from core.engine.client import BotEngine
from core.managers.context_manager import ChatContextManager
from core.managers.cost_tracker import CostTracker
from core.managers.emoji_manager import EmojiManager
from core.ai.multimodal import MultimodalService
from core.managers.nickname_manager import NicknameManager
from core.engine.router import Router
from core.managers.template_manager import TemplateManager
from core.engine.everos_memory import EverOSMemory
from core.command_handlers import register_all_commands
from core.tools.skill_managers import SkillManagers
from core.plugins.manager import PluginManager
from core.learners.orchestrator import LearningOrchestrator
from core.tasks import TaskStore, TaskManager, CronJobManager, BackgroundTaskRunner, CronJobScheduler
from core.tasks.heartbeat import HeartbeatManager
from core.router_model import RouterModel

from core.webui import create_app, start_webui


async def main() -> None:
    print("Hello from meow-qqbot!")

    # --- 彩色日志配置 ---
    _colors = {
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    }

    _formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)-8s] %(blue)s%(name)s%(reset)s: %(message)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors=_colors,
    )

    _handler = logging.StreamHandler()
    _handler.setFormatter(_formatter)
    logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)

    _log = logging.getLogger(__name__)

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ── 1. 初始化全局单例服务 ──
    # 共享 HTTP 客户端
    http_client = httpx.AsyncClient(timeout=60.0)

    # 模板管理器
    template_manager = TemplateManager(config)

    # AI 服务
    openai_config = config.get("openai", {})
    ai_service = AIService(
        api_key=openai_config.get("api_key"),
        base_url=openai_config.get("base_url"),
        model=openai_config.get("model", "gpt-3.5-turbo"),
        timeout=openai_config.get("timeout", 30),
        max_retries=openai_config.get("max_retries", 3),
        temperature=openai_config.get("temperature", 0.7),
        max_tokens=openai_config.get("max_tokens", 8192),
        reasoning_effort=openai_config.get("reasoning_effort"),
    )

    # 多模态配置
    multimodal_config = config.get("multimodal", {})

    # 多模态服务（如果启用）
    multimodal_service = None
    if multimodal_config.get("enabled", False):
        multimodal_service = MultimodalService(
            api_key=multimodal_config.get("api_key", ""),
            base_url=multimodal_config.get("base_url"),
            model=multimodal_config.get("model", "deepseek-v4-flash"),
        )
        _log.info(f"多模态服务已启用，模型: {multimodal_config.get('model')}")
    else:
        _log.info("多模态服务未启用（enabled=false），跳过 VLM 图片分析")

    # 全局 EmojiManager（依赖 http_client + multimodal_service）
    emoji_manager = EmojiManager(
        http_client=http_client,
        multimodal_service=multimodal_service,
        emoji_dir="data/emojis/",
    )

    # 上下文管理配置
    ctx_mgmt = config.get("context_management", {})

    # 全局 ChatContextManager（短期记忆，append-only，token 阈值触发 compaction）
    cache_cfg = ctx_mgmt.get("cache", {})
    context_manager = ChatContextManager(
        max_history_per_chat=ctx_mgmt.get("max_history", 10000),
        compact_threshold_tokens=ctx_mgmt.get("compact_threshold_tokens", 950000),
        keep_recent_tokens=ctx_mgmt.get("keep_recent_tokens", 50000),
        max_tool_results=ctx_mgmt.get("max_tool_results", 5),
        keep_last_assistants=ctx_mgmt.get("keep_last_assistants", 3),
        soft_trim=ctx_mgmt.get("soft_trim", 20000),
        hard_clear=ctx_mgmt.get("hard_clear", 180000),
        cache_dir=(cache_cfg.get("dir") or "data/sessions/") if cache_cfg.get("enabled", True) else None,
    )

    # 管理员 ID 列表
    admin_ids = config.get("admin_id", [])

    # ── CostTracker（AI 消耗追踪） ──
    cost_tracking_config = config.get("cost_tracking", {})
    cost_tracker = CostTracker(
        pricing=cost_tracking_config.get("pricing"),
    ) if cost_tracking_config.get("enabled", True) else CostTracker()

    # ── EverOS 长期记忆系统 ──
    everos_config = config.get("everos", {})
    everos_memory = None
    if everos_config.get("enabled", True):
        everos_memory = EverOSMemory(
            base_url=everos_config.get("base_url", "http://127.0.0.1:8000"),
            app_id=everos_config.get("app_id", "qq_bot"),
            project_id=everos_config.get("project_id", "production"),
            flush_threshold=everos_config.get("flush_threshold", 20),
        )
        _log.info(
            f"EverOS 记忆系统已启用: {everos_config.get('base_url', 'http://127.0.0.1:8000')}"
        )
    else:
        _log.info("EverOS 记忆系统未启用")

    # ── EverOS 启动健康检查 ──
    if everos_memory:
        health_result = await everos_memory.health()
        if health_result.get("status") == "ok":
            _log.info(
                f"EverOS 健康检查通过 ({health_result.get('latency_ms')}ms)"
            )
        else:
            _log.warning(
                f"EverOS 健康检查失败: {health_result.get('error')}"
                " — 记忆功能将降级运行"
            )

    # ── 后台任务系统（Tasks + Cron） ──
    tasks_config = config.get("tasks", {})
    task_manager = None
    cron_job_manager = None
    background_task_runner = None
    cron_scheduler = None

    if tasks_config.get("enabled", True):
        scheduler_cfg = tasks_config.get("scheduler", {})
        task_store = TaskStore(
            data_dir=tasks_config.get("data_dir", "data/tasks/"),
            max_tasks=tasks_config.get("max_tasks", 1000),
            task_ttl_days=tasks_config.get("task_ttl_days", 30),
        )
        task_manager = TaskManager(store=task_store)
        cron_job_manager = CronJobManager(store=task_store)
        background_task_runner = BackgroundTaskRunner(task_manager=task_manager)

        if scheduler_cfg.get("enabled", True):
            cron_scheduler = CronJobScheduler(
                poll_interval=scheduler_cfg.get("poll_interval", 30),
                catch_up_window=scheduler_cfg.get("catch_up_window", 3600),
                max_concurrent=scheduler_cfg.get("max_concurrent", 3),
            )
        _log.info("后台任务系统已初始化")

    bot_id = config.get("bot_id", "")

    # ── NicknameManager（统一昵称管理，全局单例） ──
    nickname_manager = NicknameManager(bot_id=bot_id)

    # ── SkillManagers（技能系统，全局单例，仅从项目本地加载） ──
    skill_managers = SkillManagers(project_skill_dir="./.agents/skills/")

    # ── LearningOrchestrator（学习系统） ──
    learners_config = config.get("learners", {})
    learning_orchestrator = None
    if learners_config.get("enabled", True):
        learning_orchestrator = LearningOrchestrator(
            config=learners_config,
            ai_service=ai_service,
            data_dir=learners_config.get("data_dir", "data/learners/"),
            emoji_manager=emoji_manager,
        )
        _log.info("学习系统已启用")
    else:
        _log.info("学习系统未启用")

    # ── 路由模型（智能分级） ──
    router_model = None
    if config.get("router_model", {}).get("enabled", False):
        character_card = getattr(template_manager, "character_card", "")
        router_model = RouterModel(config["router_model"], character_card=character_card)

    # ── 2. 创建 AgentEngine（全局单例） ──
    agent_engine = AgentEngine(
        ai_service=ai_service,
        template_manager=template_manager,
        context_manager=context_manager,
        bot_id=bot_id,
        admin_id=admin_ids,
        openai_config=openai_config,
        nickname_manager=nickname_manager,
        emoji_manager=emoji_manager,
        everos_memory=everos_memory,
        search_top_k=everos_config.get("search_top_k", 3),
        skill_managers=skill_managers,
        learning_orchestrator=learning_orchestrator,
        max_tool_rounds=config.get("max_tool_rounds", -1),
        cost_tracker=cost_tracker,
        task_manager=task_manager,
        cron_job_manager=cron_job_manager,
    )

    # ── 将后台任务执行器注入 ToolExecutor（供 AI 工具调用） ──
    if task_manager or cron_job_manager or background_task_runner:
        agent_engine.tool_executor.set_task_managers(
            task_manager=task_manager,
            cron_job_manager=cron_job_manager,
            background_task_runner=background_task_runner,
        )
        _log.info("任务管理器已注入 ToolExecutor")

    # ── 注入路由模型 ──
    if router_model:
        agent_engine.set_router_model(router_model)

    # ── 连接后台任务执行器与 AgentEngine ──
    if background_task_runner and task_manager:
        background_task_runner.set_execute_callback(
            agent_engine.execute_background_task
        )
        # 投递回调（将任务执行结果发回 QQ 聊天）
        async def _deliver(chat_id, content, message_id, is_group):
            try:
                chat_type = "group" if is_group else "c2c"
                await engine.api.send_text(chat_type, chat_id, content, reply_to=None)
            except Exception as e:
                _log.error(f"投递任务结果失败: {e}")
        background_task_runner.set_delivery_callback(_deliver)

    # ── 连接 Cron 调度器 ──
    if cron_scheduler and cron_job_manager:
        cron_scheduler.set_callbacks(
            on_trigger=lambda job: background_task_runner.run_cron_job(
                job=job,
                timeout=config.get("tasks", {}).get("scheduler", {}).get("task_timeout", 300),
            ),
            get_jobs=cron_job_manager.list_jobs,
            update_job=cron_job_manager.update_job,
            delete_job=cron_job_manager.delete_job,
        )

    # ── 3. 创建 BotEngine ──
    router = Router(agent_engine=agent_engine)

    engine = BotEngine(
        app_id=config["appid"],
        client_secret=config["secret"],
        bot_id=bot_id,
        agent_engine=agent_engine,
        router=router,
        admin_id=admin_ids,
        nickname_manager=nickname_manager,
        emoji_manager=emoji_manager,
        multimodal_service=multimodal_service,
    )

    # ── 心跳系统 ──
    heartbeat_manager = None
    if config.get("heartbeat", {}).get("enabled", False):
        heartbeat_manager = HeartbeatManager(
            config=config["heartbeat"],
            router_model=router_model,
            bot_id=bot_id,
            admin_ids=admin_ids,
            api_client=engine.api,
            agent_engine=agent_engine,
        )

    # 注册命令处理器（从 core/command_handlers/ 自动发现）
    register_all_commands(
        engine.command_manager,
        context_manager=context_manager,
        emoji_manager=emoji_manager,
        agent_engine=agent_engine,
        skill_managers=skill_managers,
        learning_orchestrator=learning_orchestrator,
        api_client=engine.api,
        bot_engine=engine,
        ai_service=ai_service,
        task_manager=task_manager,
        cron_job_manager=cron_job_manager,
        background_task_runner=background_task_runner,
    )

    # ── 4. 加载插件 ──
    plugin_manager = PluginManager(plugin_dir="plugins")
    plugin_manager.load_all(
        command_manager=engine.command_manager,
        context_manager=context_manager,
        emoji_manager=emoji_manager,
        agent_engine=agent_engine,
        skill_managers=skill_managers,
        api_client=engine.api,
        bot_engine=engine,
    )

    # ── 5. 启动 WebUI（如果启用） ──
    webui_config = config.get("webui", {})
    if webui_config.get("enabled", False):
        webui_app = create_app(
            managers={
                "emoji_manager": emoji_manager,
                "nickname_manager": nickname_manager,
                "context_manager": context_manager,
                "cost_tracker": cost_tracker,
                "agent_engine": agent_engine,
                "learning_orchestrator": learning_orchestrator,
            },
            webui_config=webui_config,
        )
        _webui_host = webui_config.get("host", "127.0.0.1")
        _webui_port = webui_config.get("port", 8080)
        _log.info(f"WebUI 管理面板将在 http://{_webui_host}:{_webui_port} 启动")
        if _webui_host in ("0.0.0.0", "::"):
            _log.info("局域网内可通过 http://<本机IP>:%d 访问", _webui_port)
        asyncio.create_task(start_webui(webui_app, webui_config))

    # ── 6. 启动 WebSocket ──
    gateway_url = await engine.api.get_gateway_url()
    loop = asyncio.get_running_loop()

    engine.start(gateway_url, loop)
    print(f"机器人已启动，WebSocket 网关: {gateway_url}")

    # 启动 Cron 调度器（在 WS 就绪后）
    if cron_scheduler:
        cron_scheduler.start()

    # 启动心跳
    if heartbeat_manager:
        await heartbeat_manager.start()

    task_cleanup_task = None
    if task_manager:
        # 定期清理过期任务记录（每 1 小时）
        async def _periodic_task_cleanup():
            while True:
                await asyncio.sleep(3600)
                try:
                    cleaned = task_manager.cleanup_old_tasks()
                    if cleaned:
                        _log.info(f"定时清理了 {cleaned} 条过期任务")
                except Exception:
                    pass
        task_cleanup_task = asyncio.create_task(_periodic_task_cleanup())

    try:
        await asyncio.Event().wait()
    finally:
        if cron_scheduler:
            await cron_scheduler.stop()
        if heartbeat_manager:
            await heartbeat_manager.stop()
        if task_cleanup_task:
            task_cleanup_task.cancel()
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
