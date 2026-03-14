import asyncio
import os
from dataclasses import dataclass

import botpy
from botpy.ext.cog_yaml import read

from core.ai_service import AIService
from core.client import MyClient


def main():
    print("Hello from meow-qqbot!")

    config = read(os.path.join(os.path.dirname(__file__), "config.yaml"))

    intents = botpy.Intents(
        direct_message=True, public_guild_messages=True, public_messages=True
    )
    client = MyClient(intents=intents)

    system_prompt = config["ai_system_prompt"]

    if system_prompt is None:
        print("Not find system prompt in config")
        return

    client.system_prompt = system_prompt

    client.admin_id = config.get("admin_id", [])

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

    client.ai_service = ai_service

    client.run(appid=config["appid"], secret=config["secret"])


if __name__ == "__main__":
    main()
