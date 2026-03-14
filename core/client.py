import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List

import botpy
import psutil
from botpy import logging
from botpy.message import C2CMessage, DirectMessage, GroupMessage, Message
from openai.resources.conversations import AsyncItemsWithStreamingResponse

from core.ai_service import AIService
from core.command_manager import CommandManager
from core.commands import Command, CommandRegistry, PermissionLevel
from core.context_manager import ChatContextManager
from core.message_queue import InputMessage, MessageQueue, ProcessedMessage

_log = logging.get_logger()


class MyClient(botpy.Client):

    _msg_seq: int = 1
    ai_service: AIService

    system_prompt: str

    admin_id: list[str]

    async def on_ready(self):
        _log.info(f"robot 「{self.robot.name}」 on_ready!")

        self.message_queue = MessageQueue()
        self.context_manager = ChatContextManager()

        self.command_manager = CommandManager(self)

        self.command_manager.register_default_commands()

        self.context_manager.register_default_command(self.command_manager)

        # 状态命令（管理员专用）
        self.command_manager.register_command(
            Command(
                name="状态",
                handler=self.handle_status_command,
                aliases=["status"],
                permission=PermissionLevel.ADMIN,
                description="查看系统状态（管理员专用）",
            )
        )

        _log.info(f"已注册 {self.command_manager.registry.count()} 个命令")

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

            # 使用命令管理器处理消息（检查是否为命令）
            command_messages = self.command_manager.process_message(input_message)

            # 如果有命令消息返回，则发送这些消息并返回
            if command_messages:
                for msg in command_messages:
                    await self._send_reply(
                        chat_id=msg["chat_id"],
                        content=msg["content"],
                        message_id=msg["message_id"],
                        is_group=msg["is_group"],
                    )
                return

            # 记录用户消息到上下文
            await self.context_manager.add_user_message_async(
                input_message.chat_id, input_message.content, input_message.id
            )

            # 检查 AI 服务是否已初始化
            if not hasattr(self, "ai_service") or self.ai_service is None:
                _log.error("AI 服务未初始化")
                raise RuntimeError("AI 服务未初始化")

            # 从上下文管理器获取历史消息
            context_messages = await self.context_manager.get_chat_history_async(
                input_message.chat_id, max_messages=8
            )

            # 构建消息列表
            messages = [
                {"role": "system", "content": self.system_prompt},
                *context_messages,
                {"role": "user", "content": input_message.content},
            ]

            # 调用 AI 服务生成响应
            response = await self.ai_service.chat_completion(messages=messages)

            _log.info(f"Res: {response}")

            if response is None:
                response = "AI 服务异常"

            # 发送回复
            await self._send_reply(
                chat_id=input_message.chat_id,
                content=response,
                message_id=input_message.id,
                is_group=input_message.is_group,
            )

            # 记录助手回复到上下文
            await self.context_manager.add_assistant_message_async(
                input_message.chat_id, response, input_message.id
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

    def handle_status_command(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        """处理状态命令，显示系统状态（管理员专用）"""
        try:
            chat_id = input_message.chat_id

            # 内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used / (1024**3)  # GB
            memory_total = memory.total / (1024**3)  # GB

            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # 磁盘使用情况
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent
            disk_used = disk.used / (1024**3)  # GB
            disk_total = disk.total / (1024**3)  # GB

            # 进程信息
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**2)  # MB
            process_cpu = process.cpu_percent(interval=0.1)

            # 消息队列状态
            input_queue_size = self.message_queue.input_queue.qsize()
            processed_queue_size = self.message_queue.processed_queue.qsize()

            # 上下文管理器状态
            active_chats = self.context_manager.get_context_count()

            # 格式化状态信息
            status_text = [
                "=== 系统状态 ===",
                f"系统时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "=== 系统资源 ===",
                f"CPU使用率: {cpu_percent:.1f}%",
                f"内存使用: {memory_percent:.1f}% ({memory_used:.1f}GB / {memory_total:.1f}GB)",
                f"磁盘使用: {disk_percent:.1f}% ({disk_used:.1f}GB / {disk_total:.1f}GB)",
                "",
                "=== 进程状态 ===",
                f"进程内存: {process_memory:.1f}MB",
                f"进程CPU: {process_cpu:.1f}%",
                "",
                "=== 机器人状态 ===",
                f"消息队列: 输入队列 {input_queue_size} 条，处理队列 {processed_queue_size} 条",
                f"活跃聊天: {active_chats} 个",
                f"管理员ID: {', '.join(self.admin_id) if self.admin_id else '未设置'}",
            ]

            reply_content = "\n".join(status_text)

            # 返回消息列表
            return [
                {
                    "chat_id": chat_id,
                    "content": reply_content,
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]

        except ImportError:
            reply_content = "无法获取系统状态信息，请安装psutil库。"
            return [
                {
                    "chat_id": chat_id,
                    "content": reply_content,
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]
        except Exception as e:
            _log.error(f"处理状态命令时出错: {e}")
            return []
