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
from core.emoji import EmojiManager, is_custom_emoji
from core.message_queue import InputMessage, MessageQueue, ProcessedMessage
from core.multimodal_service import MultimodalService
from core.template_manager import TemplateManager

_log = logging.getLogger(__name__)


class BotEngine:
    """使用 qqbot_agent_sdk 的独立 QQ 机器人引擎。"""

    def __init__(
        self,
        app_id: str,
        client_secret: str,
        bot_id: str,
        template_manager: TemplateManager,
        ai_service: AIService,
        admin_id: list[str],
        openai_config: Optional[dict] = None,
        multimodal_config: Optional[dict] = None,
    ):
        self._app_id = app_id
        self._client_secret = client_secret
        self._bot_id = bot_id
        self._http_client = httpx.AsyncClient(timeout=60.0)
        self.api = QQApiClient(app_id=app_id, client_secret=client_secret)
        self.api.setup(self._http_client)
        self.ws: Optional[QQWebSocket] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        # 业务组件（构造器注入）
        self.ai_service: AIService = ai_service
        self.template_manager: TemplateManager = template_manager
        self.admin_id: list[str] = admin_id
        self.nicknames: Dict[str, str] = {}
        self.message_queue: MessageQueue = MessageQueue()
        self.context_manager: ChatContextManager = ChatContextManager()
        self.command_manager: CommandManager = CommandManager(self)
        self._bot_name: str = "机器人"
        self._openai_config: dict = openai_config or {}
        self._multimodal_config: dict = multimodal_config or {}
        self._auto_replied: dict[str, str] = {}  # {chat_id: content} — 已复读的内容追踪

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

        # 初始化多模态（视觉）模型服务
        # 开关：multimodal.enabled == true 时激活 VLM 分析
        mm_cfg = self._multimodal_config
        if mm_cfg.get("enabled", False):
            self.multimodal_service = MultimodalService(
                api_key=mm_cfg.get("api_key", ""),
                base_url=mm_cfg.get("base_url"),
                model=mm_cfg.get("model", "deepseek-v4-flash"),
            )
            _log.info(f"多模态服务已启用，模型: {mm_cfg.get('model')}")
        else:
            self.multimodal_service = None
            _log.info("多模态服务未启用（enabled=false），跳过 VLM 图片分析")

        # 初始化表情管理器
        self.emoji_manager = EmojiManager(
            http_client=self._http_client,
            multimodal_service=self.multimodal_service,
            emoji_dir="data/emojis/",
        )

        # 注册表情相关命令
        self._register_emoji_commands()

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
        _log.info(f"raw: {raw}")

        # ── 检测自定义表情（faceType=6 + attachments）──
        if is_custom_emoji(event.content, event.attachments):
            _log.info(f"检测到自定义表情，用户: {event.user_id}")
            try:
                desc, tags = await self.emoji_manager.get_or_build(
                    event.attachments[0]
                )
                tag_str = " ".join(tags) if tags else ""
                event.content = f"[表情: {desc}]"
                if tag_str:
                    event.content += f" [情绪: {tag_str}]"
                _log.info(f"自定义表情解析结果: {event.content}")
            except Exception as e:
                _log.error(f"自定义表情处理失败: {e}")
                event.content = "[自定义表情]"
            # 跳过后续的空内容/表情过滤，直接进入流程
        else:
            # ── 跳过空内容或仅包含内置 QQ 表情（无附件）的消息 ──
            stripped = event.content.strip()
            if not stripped:
                _log.debug("跳过空内容消息")
                return
            cleaned = re.sub(r'<faceType=\d+,[^>]+>', '', stripped).strip()
            if not cleaned:
                _log.debug("跳过仅包含 QQ 内置表情的消息")
                return

        # DM（频道直发消息）→ 简单回复，不进入 AI 流程
        if event.chat_scope == "dm":
            await self.api.send_text(
                "guild", event.chat_id,
                f"机器人{self._bot_name}收到你的消息了: {event.content}",
                reply_to=event.message_id,
            )
            return

        # 解析消息中所有 @提及的 ID（含机器人自身），并从内容中移除 @机器人的标记
        mentioned_ids = re.findall(r'<@([^>]+)>', event.content)
        _log.info(f"mentioned_ids: {mentioned_ids}")
        # 只移除 @机器人自身的标记，保留 @其他人的
        event.content = re.sub(rf'<@{re.escape(self._bot_id)}>', '', event.content).strip()

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

    # ── 自动复读检查 ──

    async def _check_auto_reply_duplicate(self, input_message: InputMessage) -> Optional[str]:
        """检查群聊重复消息，返回需要复读的内容，或 None 表示不复读。"""
        if not input_message.is_group:
            return None

        context = await self.context_manager.get_context_async(input_message.chat_id)
        user_msgs = [m for m in context.history if m.role == "user"]
        if len(user_msgs) < 2:
            return None

        last_content = user_msgs[-1].content
        prev_content = user_msgs[-2].content
        if last_content != prev_content:
            return None

        # 已复读过相同内容 → 不复读
        if self._auto_replied.get(input_message.chat_id) == last_content:
            return None

        return last_content

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

            # ── 群聊重复消息自动复读（仅文本消息，表情消息不重复）──
            if input_message.content.startswith("[表情:") or input_message.content == "[自定义表情]":
                reply_content = None
            else:
                reply_content = await self._check_auto_reply_duplicate(input_message)
            if reply_content is not None:
                _log.info(
                    f"检测到重复消息 [{input_message.chat_id}]，自动复读: {reply_content[:30]}"
                )
                await self._send_reply(
                    chat_id=input_message.chat_id,
                    content=reply_content,
                    message_id=input_message.id,
                    is_group=True,
                )
                await self.context_manager.add_assistant_message_async(
                    input_message.chat_id, reply_content, input_message.id,
                )
                self._auto_replied[input_message.chat_id] = reply_content
                return

            # 群聊非 @且非猫猫开头 → 保留上下文，但不进行 AI 回复
            if input_message.is_group and not input_message.is_at_mention:
                if not input_message.content.startswith("猫猫"):
                    _log.debug(f"跳过 AI 回复（非@且非猫猫开头）: {input_message.content[:30]}")
                    return

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
        self, input_message: InputMessage, _: str
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

    # ════════════════════════════════════════════════════════════
    # 表情命令
    # ════════════════════════════════════════════════════════════

    def _register_emoji_commands(self) -> None:
        """注册表情相关命令"""
        from core.commands import Command, PermissionLevel

        self.command_manager.register_command(
            Command(
                name="表情列表",
                handler=self._handle_emoji_list,
                aliases=["emojis"],
                permission=PermissionLevel.DEFAULT,
                description="查看所有已知自定义表情",
            )
        )
        self.command_manager.register_command(
            Command(
                name="表情查看",
                handler=self._handle_emoji_info,
                aliases=["emoji"],
                permission=PermissionLevel.DEFAULT,
                description="查看指定表情的详细信息。用法：猫猫表情查看 <hash>",
            )
        )
        self.command_manager.register_command(
            Command(
                name="表情编辑",
                handler=self._handle_emoji_edit,
                aliases=[],
                permission=PermissionLevel.ADMIN,
                description="自定义表情描述和标签。用法：猫猫表情编辑 <hash> 描述=xxx 标签=xxx",
            )
        )
        self.command_manager.register_command(
            Command(
                name="表情重置",
                handler=self._handle_emoji_reset,
                aliases=[],
                permission=PermissionLevel.ADMIN,
                description="恢复表情为 AI 自动识别结果。用法：猫猫表情重置 <hash>",
            )
        )

    def _handle_emoji_list(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        """猫猫表情列表 — 分页显示所有已知表情"""
        try:
            page = 1
            if args.strip():
                try:
                    page = max(1, int(args.strip()))
                except ValueError:
                    pass

            result = self.emoji_manager.list_emojis(page=page, page_size=10)
            if result["total"] == 0:
                return [
                    {
                        "chat_id": input_message.chat_id,
                        "content": "还没有记录任何自定义表情。",
                        "message_id": input_message.id,
                        "is_group": input_message.is_group,
                    }
                ]

            lines = [
                f"已知自定义表情（共 {result['total']} 个，第 {result['page']} 页）："
            ]
            for emoji in result["emojis"]:
                short_hash = emoji["hash"][:12]
                desc = emoji.get("user_description") or emoji.get("auto_description", "")
                tags = emoji.get("user_tags") or emoji.get("auto_tags", [])
                tag_str = f" [{', '.join(tags[:3])}]" if tags else ""
                marker = " ★" if (emoji.get("user_description") is not None or emoji.get("user_tags")) else ""
                count = emoji.get("used_count", 0)
                lines.append(f"  {short_hash}: {desc}{tag_str}{marker} (x{count})")

            if result["total"] > page * result["page_size"]:
                lines.append(f"输入「猫猫表情列表 {page + 1}」查看下一页")

            return [
                {
                    "chat_id": input_message.chat_id,
                    "content": "\n".join(lines),
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]
        except Exception as e:
            _log.error(f"表情列表命令失败: {e}")
            return []

    def _handle_emoji_info(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        """猫猫表情查看 <hash> — 查看指定表情的详细信息"""
        emoji_hash = args.strip()
        if not emoji_hash:
            return [
                {
                    "chat_id": input_message.chat_id,
                    "content": "请提供表情 hash。用法：猫猫表情查看 <hash>",
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]

        # 支持短 hash 匹配
        record = self._find_emoji(emoji_hash)
        if record is None:
            return [
                {
                    "chat_id": input_message.chat_id,
                    "content": f"未找到表情「{emoji_hash}」。",
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]

        lines = [
            f"=== 表情详情 ===",
            f"Hash: {record['hash']}",
            f"文件名: {record.get('file_name', 'N/A')}",
            f"使用次数: {record.get('used_count', 0)}",
            f"",
            f"AI 描述: {record.get('auto_description', '(无)')}",
            f"AI 标签: {', '.join(record.get('auto_tags', [])) or '(无)'}",
        ]
        has_custom = record.get("user_description") is not None or record.get("user_tags")
        if has_custom:
            lines.append(f"")
            lines.append(
                f"★ 用户自定义描述: {record.get('user_description', '(无)')}"
            )
            lines.append(
                f"★ 用户自定义标签: {', '.join(record.get('user_tags', [])) or '(无)'}"
            )
        lines.append(f"")
        lines.append(f"创建时间: {record.get('created_at', 'N/A')}")
        lines.append(f"最后更新: {record.get('updated_at', 'N/A')}")
        lines.append(
            f"URL: {record.get('url', 'N/A')[:60]}..."
        )

        return [
            {
                "chat_id": input_message.chat_id,
                "content": "\n".join(lines),
                "message_id": input_message.id,
                "is_group": input_message.is_group,
            }
        ]

    def _handle_emoji_edit(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        """猫猫表情编辑 <hash> 描述=xxx 标签=A、B — 自定义表情描述和标签"""
        parts = args.strip().split()
        if len(parts) < 2:
            return [
                {
                    "chat_id": input_message.chat_id,
                    "content": "格式：猫猫表情编辑 <hash> 描述=xxx 标签=A、B",
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]

        emoji_hash = parts[0]
        desc = None
        tags = None

        for p in parts[1:]:
            if p.startswith("描述="):
                desc = p[3:]
            elif p.startswith("标签="):
                raw = p[3:]
                tags = [t.strip() for t in raw.replace("、", ",").split(",") if t.strip()]

        if desc is None and tags is None:
            return [
                {
                    "chat_id": input_message.chat_id,
                    "content": "至少要提供描述或标签中的一个。\n格式：猫猫表情编辑 <hash> 描述=xxx 标签=A、B",
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]

        # 支持短 hash 匹配
        record = self._find_emoji(emoji_hash)
        if record is None:
            return [
                {
                    "chat_id": input_message.chat_id,
                    "content": f"未找到表情「{emoji_hash}」。「猫猫表情列表」查看所有。",
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]

        ok = self.emoji_manager.set_custom(record["hash"], description=desc, tags=tags)
        if ok:
            changes = []
            if desc is not None:
                changes.append(f"描述 → {desc}")
            if tags is not None:
                changes.append(f"标签 → {', '.join(tags)}")
            return [
                {
                    "chat_id": input_message.chat_id,
                    "content": f"表情 {record['hash'][:12]}.. 已更新：{'；'.join(changes)}",
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]
        else:
            return [
                {
                    "chat_id": input_message.chat_id,
                    "content": f"更新失败，请重试。",
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]

    def _handle_emoji_reset(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        """猫猫表情重置 <hash> — 恢复为 AI 自动识别结果"""
        emoji_hash = args.strip()
        if not emoji_hash:
            return [
                {
                    "chat_id": input_message.chat_id,
                    "content": "请提供表情 hash。用法：猫猫表情重置 <hash>",
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]

        record = self._find_emoji(emoji_hash)
        if record is None:
            return [
                {
                    "chat_id": input_message.chat_id,
                    "content": f"未找到表情「{emoji_hash}」。",
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]

        ok = self.emoji_manager.reset_to_auto(record["hash"])
        if ok:
            return [
                {
                    "chat_id": input_message.chat_id,
                    "content": f"表情 {record['hash'][:12]}.. 已恢复为 AI 自动识别结果。",
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]
        else:
            return [
                {
                    "chat_id": input_message.chat_id,
                    "content": f"重置失败，请重试。",
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]

    def _find_emoji(self, partial_hash: str) -> Optional[dict]:
        """
        根据 hash（支持短前缀）查找 emoji 记录。
        """
        emojis = self.emoji_manager._storage.list_all()
        # 先精确匹配
        for e in emojis:
            if e["hash"] == partial_hash:
                return e
        # 再前缀匹配
        matches = [e for e in emojis if e["hash"].startswith(partial_hash)]
        if len(matches) == 1:
            return matches[0]
        return None

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
