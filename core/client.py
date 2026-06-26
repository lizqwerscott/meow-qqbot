import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx
import psutil

from qqbot_agent_sdk import (
    QQApiClient,
    QQWebSocket,
    WSCallbacks,
    EventParser,
    InboundEvent,
)
from qqbot_agent_sdk.dto import WSReadyData

from core.ai_service import AIService
from core.command_manager import CommandManager
from core.commands import Command, CommandRegistry, PermissionLevel
from core.context_manager import ChatContextManager
from core.message_queue import InputMessage, MessageQueue, ProcessedMessage
from core.template_manager import TemplateManager

_log = logging.getLogger(__name__)


class BotEngine:
    """使用 qqbot_agent_sdk 的独立 QQ 机器人引擎。"""

    def __init__(self, app_id: str, client_secret: str, bot_id: str):
        self._app_id = app_id
        self._client_secret = client_secret
        self._bot_id = bot_id
        self._http_client = httpx.AsyncClient(timeout=60.0)
        self.api = QQApiClient(app_id=app_id, client_secret=client_secret)
        self.api.setup(self._http_client)
        self.ws: Optional[QQWebSocket] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        # 业务组件（在 start 前由 main.py 注入，或在 on_ready 中初始化）
        self.ai_service: Optional[AIService] = None
        self.template_manager: Optional[TemplateManager] = None
        self.admin_id: List[str] = []
        self.nicknames: Dict[str, str] = {}
        self.message_queue: Optional[MessageQueue] = None
        self.context_manager: Optional[ChatContextManager] = None
        self.command_manager: Optional[CommandManager] = None
        self._bot_name: str = "机器人"

    # ── 生命周期 ──

    def _build_callbacks(self) -> WSCallbacks:
        """构造 WSCallbacks，全部回调绑定到 BotEngine 方法。"""
        return WSCallbacks(
            on_message_event=self._on_message_event,
            get_token=self.api.ensure_token_sync,
            get_gateway_url=self.api.get_gateway_url_sync,
            on_connected=lambda: _log.info("WebSocket 已连接"),
            on_disconnected=lambda: _log.info("WebSocket 已断开"),
            on_fatal_error=lambda code, msg: _log.error(f"致命错误 [{code}]: {msg}"),
            get_session=lambda: (None, None),
            set_session=lambda sid, seq: None,
            set_heartbeat_interval=lambda interval: _log.info(f"心跳间隔: {interval}s"),
            clear_token=lambda: self.api.clear_token(),
            fail_pending=lambda reason: _log.warning(f"挂起请求失败: {reason}"),
            on_ready=self._on_ready,
        )

    def _on_ready(self, ready: WSReadyData) -> None:
        """同步回调——WS 就绪时初始化所有组件。"""
        if ready.user:
            self._bot_name = ready.user.username or "机器人"
        _log.info(f"机器人「{self._bot_name}」on_ready!")

        # 初始化组件
        self.message_queue = MessageQueue()
        self.context_manager = ChatContextManager()
        self.command_manager = CommandManager(self)
        self.command_manager.register_default_commands()
        self.context_manager.register_default_command(self.command_manager)
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
        self.nicknames = self._load_nicknames()
        _log.info(f"已加载 {len(self.nicknames)} 个用户昵称")

        # 在主循环调度消息处理循环
        if self._main_loop:
            asyncio.run_coroutine_threadsafe(
                self._process_messages_loop(), self._main_loop
            )

    def start(self, gateway_url: str, main_loop: asyncio.AbstractEventLoop) -> None:
        """启动 WebSocket 连接。"""
        self._main_loop = main_loop
        self.ws = QQWebSocket(callbacks=self._build_callbacks())
        self.ws.start(gateway_url, main_loop)

    async def stop(self) -> None:
        """安全关闭。"""
        if self.ws:
            await self.ws.async_stop()
        await self._http_client.aclose()

    # ── 事件处理 ──

    async def _on_message_event(self, event_type: str, raw: dict) -> None:
        """处理所有入站消息事件。"""
        event = EventParser().parse(event_type, raw)
        if event is None:
            return

        _log.info(f"[{event.chat_scope}][({event_type})] {event.user_id}: {event.content}")

        # DM（频道直发消息）→ 简单回复，不进入 AI 流程
        if event.chat_scope == "dm":
            await self.api.send_text(
                "guild", event.chat_id,
                f"机器人{self._bot_name}收到你的消息了: {event.content}",
                reply_to=event.message_id,
            )
            return

        # 解析消息中所有 @提及的 ID（含机器人自身），并从内容中移除所有 @标记
        mentioned_ids = re.findall(r'<@([^>]+)>', event.content)
        _log.info(f"mentioned_ids: {mentioned_ids}")
        event.content = re.sub(r'<@[^>]+>', '', event.content).strip()

        # 通过 mentioned_ids 判断是否被 @
        is_at_mention = self._bot_id in mentioned_ids

        # 所有消息都排入 AI 处理队列
        # 群聊消息全部入队以保留上下文，但仅在 @机器人 或 "猫猫" 开头时回复
        input_message = InputMessage(
            id=event.message_id,
            sender_id=event.user_id,
            chat_id=event.chat_id,
            content=event.content,
            is_group=(event.chat_scope == "group"),
            is_at_mention=is_at_mention,
            mentioned_ids=mentioned_ids,
        )

        await self.message_queue.put_input_message(input_message)

    # ── 发送回复 ──

    async def _send_reply(
        self, chat_id: str, content: str, message_id: str, is_group: bool = False
    ) -> None:
        """发送回复——使用 SDK send_text 自动路由 + 重试。"""
        chat_type = "group" if is_group else "c2c"
        await self.api.send_text(chat_type, chat_id, content, reply_to=message_id)
        _log.info(f"已发送回复: {chat_id}, 消息ID: {message_id}")

    # ── 消息处理循环 ──

    async def _process_messages_loop(self) -> None:
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

    async def _process_message(self, input_message: InputMessage) -> None:
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

            # 获取用户昵称
            user_nickname = self._get_user_nickname(input_message.sender_id)

            # 记录用户消息到上下文（携带发送者ID和昵称）
            await self.context_manager.add_user_message_async(
                input_message.chat_id,
                input_message.content,
                input_message.id,
                sender_id=input_message.sender_id,
                name=user_nickname,
            )

            # 群聊非 @且非猫猫开头 → 保留上下文，但不进行 AI 回复
            if input_message.is_group and not input_message.is_at_mention:
                if not input_message.content.startswith("猫猫"):
                    _log.debug(f"跳过 AI 回复（非@且非猫猫开头）: {input_message.content[:30]}")
                    return

            # 检查 AI 服务是否已初始化
            if self.ai_service is None:
                _log.error("AI 服务未初始化")
                raise RuntimeError("AI 服务未初始化")

            # 从上下文管理器获取历史消息
            context_messages = await self.context_manager.get_chat_history_async(
                input_message.chat_id, max_messages=8
            )

            # 使用模板管理器获取系统提示
            if input_message.is_group:
                system_prompt = self.template_manager.get_group_chat_prompt()
            else:
                system_prompt = self.template_manager.get_private_chat_prompt(
                    user_nickname
                )

            # 构建消息列表，history 已包含当前消息，无需重复添加
            messages = [
                {"role": "system", "content": system_prompt},
                *context_messages,
            ]

            # 打印请求消息（格式化，便于调试）
            _log.info(
                f"请求 AI messages:\n{json.dumps(messages, ensure_ascii=False, indent=2)}"
            )

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

    def _load_nicknames(self) -> Dict[str, str]:
        """
        加载昵称映射文件

        Returns:
            用户ID -> 昵称的字典映射
        """
        nicknames_file = "nicknames.json"
        if os.path.exists(nicknames_file):
            try:
                with open(nicknames_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                _log.error(f"加载昵称文件失败: {e}")
                return {}
        else:
            _log.warning(f"昵称文件 {nicknames_file} 不存在")
            return {}

    def _get_user_nickname(self, user_id: str) -> str:
        """
        获取用户昵称，如果未找到则返回用户ID

        Args:
            user_id: 用户ID

        Returns:
            用户昵称或用户ID
        """
        return self.nicknames.get(user_id, user_id)
