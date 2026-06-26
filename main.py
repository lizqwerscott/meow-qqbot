import asyncio
import os

import yaml
import logging

from core.ai_service import AIService
from core.client import BotEngine
from core.template_manager import TemplateManager


async def main() -> None:
    print("Hello from meow-qqbot!")

    # 在程序最开始的地方进行配置
    logging.basicConfig(level=logging.INFO)  # 将级别设为 DEBUG 以显示所有信息

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    engine = BotEngine(app_id=config["appid"], client_secret=config["secret"], bot_id=config["bot_id"])

    # 初始化模板管理器
    template_manager = TemplateManager(config)
    engine.template_manager = template_manager
    engine.admin_id = config.get("admin_id", [])

    # 初始化 AI 服务
    openai_config = config.get("openai", {})
    engine.ai_service = AIService(
        api_key=openai_config.get("api_key"),
        base_url=openai_config.get("base_url"),
        model=openai_config.get("model", "gpt-3.5-turbo"),
        timeout=openai_config.get("timeout", 30),
        max_retries=openai_config.get("max_retries", 3),
        temperature=openai_config.get("temperature", 0.7),
        max_tokens=openai_config.get("max_tokens", 1000),
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
