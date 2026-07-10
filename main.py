import asyncio
import os

import yaml
import logging
import httpx
from colorlog import ColoredFormatter

from core.ai_service import AIService
from core.agent_engine import AgentEngine
from core.client import BotEngine
from core.context_manager import ChatContextManager
from core.emoji import EmojiManager
from core.multimodal_service import MultimodalService
from core.nickname_manager import NicknameManager
from core.router import Router
from core.template_manager import TemplateManager
from core.everos_memory import EverOSMemory
from core.command_handlers import register_all_commands
from core.skill_managers import SkillManagers


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
        max_tokens=openai_config.get("max_tokens", 1000),
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
        _log = logging.getLogger(__name__)
        _log.info(f"多模态服务已启用，模型: {multimodal_config.get('model')}")
    else:
        _log = logging.getLogger(__name__)
        _log.info("多模态服务未启用（enabled=false），跳过 VLM 图片分析")

    # 全局 EmojiManager（依赖 http_client + multimodal_service）
    emoji_manager = EmojiManager(
        http_client=http_client,
        multimodal_service=multimodal_service,
        emoji_dir="data/emojis/",
    )

    # 全局 ChatContextManager（短期记忆，不依赖 WS 生命周期）
    context_manager = ChatContextManager()

    # 管理员 ID 列表
    admin_ids = config.get("admin_id", [])

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
        _log = logging.getLogger(__name__)
        _log.info(
            f"EverOS 记忆系统已启用: {everos_config.get('base_url', 'http://127.0.0.1:8000')}"
        )
    else:
        _log = logging.getLogger(__name__)
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

    bot_id = config.get("bot_id", "")

    # ── NicknameManager（统一昵称管理，全局单例） ──
    nickname_manager = NicknameManager(bot_id=bot_id)

    # ── SkillManagers（技能系统，全局单例，仅从项目本地加载） ──
    skill_managers = SkillManagers(project_skill_dir="./.agents/skills/")

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

    # 注册命令处理器（从 core/command_handlers/ 自动发现）
    register_all_commands(
        engine.command_manager,
        context_manager=context_manager,
        emoji_manager=emoji_manager,
        agent_engine=agent_engine,
        skill_managers=skill_managers,
    )

    # ── 4. 启动 WebSocket ──
    gateway_url = await engine.api.get_gateway_url()
    loop = asyncio.get_running_loop()

    engine.start(gateway_url, loop)
    print(f"机器人已启动，WebSocket 网关: {gateway_url}")

    try:
        await asyncio.Event().wait()
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
