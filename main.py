import asyncio
import os

import yaml
import logging
from colorlog import ColoredFormatter

from core.ai_service import AIService
from core.client import BotEngine
from core.template_manager import TemplateManager


async def main() -> None:
    print("Hello from meow-qqbot!")

    # 在程序最开始的地方进行配置
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

    # 初始化模板管理器
    template_manager = TemplateManager(config)

    # 初始化 AI 服务
    openai_config = config.get("openai", {})
    ai_service = AIService(
        api_key=openai_config.get("api_key"),
        base_url=openai_config.get("base_url"),
        model=openai_config.get("model", "gpt-3.5-turbo"),
        timeout=openai_config.get("timeout", 30),
        max_retries=openai_config.get("max_retries", 3),
        temperature=openai_config.get("temperature", 0.7),
        max_tokens=openai_config.get("max_tokens", 1000),
    )

    # 读取多模态配置
    multimodal_config = config.get("multimodal", {})

    engine = BotEngine(
        app_id=config["appid"],
        client_secret=config["secret"],
        bot_id=config["bot_id"],
        template_manager=template_manager,
        ai_service=ai_service,
        admin_id=config.get("admin_id", []),
        openai_config=openai_config,
        multimodal_config=multimodal_config,
    )

    # 获取 WebSocket 网关 URL
    gateway_url = await engine.api.get_gateway_url()
    loop = asyncio.get_running_loop()

    # 启动机器人
    engine.start(gateway_url, loop)
    print(f"机器人已启动，WebSocket 网关: {gateway_url}")

    try:
        await asyncio.Event().wait()
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
