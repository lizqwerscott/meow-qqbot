import asyncio

import botpy
from botpy import logging
from botpy.message import C2CMessage, DirectMessage, GroupMessage, Message

from core.context_manager import ChatContextManager
from core.message_queue import InputMessage, MessageQueue, ProcessedMessage

_log = logging.get_logger()


class MyClient(botpy.Client):

    _msg_seq: int = 1

    async def on_ready(self):
        _log.info(f"robot 「{self.robot.name}」 on_ready!")

        self.message_queue = MessageQueue()
        self.context_manager = ChatContextManager()

        # 启动消息处理循环
        asyncio.create_task(self._process_messages_loop())

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

        input_message = InputMessage(
            id=message.id,
            sender_id=user_id,
            chat_id=chat_id,
            content=message.content,
            is_group=False,
        )

        # 记录用户消息到上下文
        await self.context_manager.add_user_message_async(
            chat_id, message.content, message.id
        )

        await self.message_queue.put_input_message(input_message)

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

        input_message = InputMessage(
            id=message.id,
            sender_id=user_id,
            chat_id=chat_id,
            content=message.content,
            is_group=True,
        )

        # 记录用户消息到上下文
        await self.context_manager.add_user_message_async(
            chat_id, message.content, message.id
        )

        await self.message_queue.put_input_message(input_message)

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

        # 记录助手回复到上下文
        await self.context_manager.add_assistant_message_async(
            chat_id, content, message_id
        )

        _log.info(f"已发送回复: {chat_id}, 消息ID: {message_id}")

    async def _process_messages_loop(self):
        """处理消息队列的循环"""
        _log.info("启动消息处理循环")

        while True:
            try:
                # 从队列获取消息
                input_message = await self.message_queue.get_input_message(timeout=1.0)
                if input_message is None:
                    continue

                # 处理消息
                await self._process_message(input_message)

            except asyncio.CancelledError:
                _log.info("消息处理循环被取消")
                break
            except Exception as e:
                _log.error(f"处理消息时发生错误: {e}")
                await asyncio.sleep(1)

    async def _process_message(self, input_message: InputMessage):
        """处理单个消息"""
        try:
            _log.info(
                f"开始处理消息: {input_message.id}, 聊天ID: {input_message.chat_id}"
            )

            # 初始化 AI 服务
            if not hasattr(self, "ai_service"):
                self.ai_service = AIServiceFactory.create_from_env()

            # 生成 AI 响应
            response = await self.ai_service.generate_with_context(
                chat_id=input_message.chat_id,
                user_message=input_message.content,
                context_manager=self.context_manager,
                system_prompt="你是一个友好的QQ机器人助手，请用中文回答用户的问题。保持回答简洁、有帮助，避免冗长。",
                max_context_messages=8,
            )

            # 发送回复
            await self._send_reply(
                chat_id=input_message.chat_id,
                content=response,
                message_id=input_message.id,
                is_group=input_message.is_group,
            )

            _log.info(f"消息处理完成: {input_message.id}")

        except Exception as e:
            _log.error(f"处理消息 {input_message.id} 时发生错误: {e}")

            # 发送错误提示
            error_message = "抱歉，处理您的消息时出现了问题，请稍后再试。"
            await self._send_reply(
                chat_id=input_message.chat_id,
                content=error_message,
                message_id=input_message.id,
                is_group=input_message.is_group,
            )
