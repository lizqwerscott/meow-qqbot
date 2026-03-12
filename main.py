import asyncio
import os
from dataclasses import dataclass

import botpy
from botpy import logging
from botpy.ext.cog_yaml import read
from botpy.message import C2CMessage, DirectMessage, GroupMessage, Message

_log = logging.get_logger()


@dataclass
class ReciveMessage:
    id: str
    sender_id: str
    chat_id: str
    content: str
    is_group: bool


class MyClient(botpy.Client):

    _msg_seq: int = 1

    async def on_ready(self):
        _log.info(f"robot 「{self.robot.name}」 on_ready!")

    async def on_c2c_message_create(self, message: C2CMessage):
        print(f"{message.author} 发送私信: {message.content}")
        _log.info(f"{message.author} 发送私信: {message.content}")

        chat_id = str(
            getattr(message.author, "id", None)
            or getattr(message.author, "user_openid", "unknown")
        )
        user_id = str(
            getattr(message.author, "id", None)
            or getattr(message.author, "user_openid", "unknown")
        )

    async def on_direct_message_create(self, message: DirectMessage):
        print(f"{message.author} 发送私信: {message.content}")
        _log.info(f"{message.author} 发送私信: {message.content}")
        await message.reply(
            content=f"机器人{self.robot.name}收到你的消息了: {message.content}"
        )

    async def on_at_message_create(self, message: Message):
        await message.reply(
            content=f"机器人{self.robot.name}收到你的@消息了: {message.content}"
        )

    async def on_group_at_message_create(self, message: GroupMessage):
        chat_id: str = str(message.group_openid)
        user_id = message.author.member_openid

        _log.info(f"群聊@消息: {user_id} -> {chat_id}: {message.content}")

    async def _send_reply(
        self, chat_id: str, content: str, message_id: str, is_group: bool = False
    ):
        """
        发送回复消息
        """
        self._msg_seq += 1

        if is_group:
            await self.api.post_group_message(
                group_openid=chat_id,
                msg_type=2,
                markdown={"content": content},
                msg_id=message_id,
                msg_seq=self._msg_seq,
            )
        else:
            await self.api.post_c2c_message(
                openid=chat_id,
                msg_type=2,
                markdown={"content": content},
                msg_id=message_id,
                msg_seq=str(self._msg_seq),
            )

        _log.info(f"已发送回复: {chat_id}, 消息ID: {message_id}")


def main():
    print("Hello from meow-qqbot!")

    config = read(os.path.join(os.path.dirname(__file__), "config.yaml"))

    # 通过kwargs，设置需要监听的事件通道
    intents = botpy.Intents(
        direct_message=True, public_guild_messages=True, public_messages=True
    )
    client = MyClient(intents=intents)
    client.run(appid=config["appid"], secret=config["secret"])


if __name__ == "__main__":
    main()
