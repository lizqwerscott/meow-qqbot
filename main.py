import asyncio
import os
from dataclasses import dataclass

import botpy
from botpy.ext.cog_yaml import read

from core.client import MyClient

from core.ai_service import AIService

def main():
    print("Hello from meow-qqbot!")

    config = read(os.path.join(os.path.dirname(__file__), "config.yaml"))

    ai_service = AIServiceFactory.create_from_env()


    # 通过kwargs，设置需要监听的事件通道
    intents = botpy.Intents(
        direct_message=True, public_guild_messages=True, public_messages=True
    )
    client = MyClient(intents=intents)
    client.run(appid=config["appid"], secret=config["secret"])


if __name__ == "__main__":
    main()
