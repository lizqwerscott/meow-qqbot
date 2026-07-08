import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx

from qqbot_agent_sdk import (
    QQApiClient,
    QQWebSocket,
    WSCallbacks,
    EventParser,
    InboundEvent,
)
from qqbot_agent_sdk.constants import MEDIA_TYPE_IMAGE
from qqbot_agent_sdk.dto import MediaInfo, MessageToCreate, QQMessageType, WSReadyData
from qqbot_agent_sdk.media_loader import MediaUploader

from core.agent_engine import AgentEngine
from core.command_manager import CommandManager
from core.commands import Command, PermissionLevel
from core.emoji import EmojiManager, is_custom_emoji
from core.message import InputMessage
from core.multimodal_service import MultimodalService
from core.router import Router

_log = logging.getLogger(__name__)


class BotEngine:
    """
    使用 qqbot_agent_sdk 的独立 QQ 机器人引擎。

    职责仅限：
    - WebSocket 连接管理
    - 消息接收与解析（→ InputMessage）
    - 发送回复（API 调用）
    - 命令管理（表情、状态、帮助等）
    - 昵称管理

    AI 编排、会话管理、工具执行由 AgentEngine 负责。
    消息路由由 Router 负责。
    """

    def __init__(
        self,
        app_id: str,
        client_secret: str,
        bot_id: str,
        agent_engine: AgentEngine,
        router: Router,
        admin_id: List[str],
        emoji_manager: Optional[EmojiManager] = None,
        multimodal_service: Optional[MultimodalService] = None,
    ):
        self._app_id = app_id
        self._client_secret = client_secret
        self._bot_id = bot_id
        self._http_client = httpx.AsyncClient(timeout=60.0)
        self.api = QQApiClient(app_id=app_id, client_secret=client_secret)
        self.api.setup(self._http_client)
        self.ws: Optional[QQWebSocket] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        # 全局单例注入
        self.agent_engine = agent_engine
        self.router = router
        self.admin_id: List[str] = admin_id

        # BotEngine 自有组件
        self.emoji_manager = emoji_manager
        self.multimodal_service = multimodal_service
        self.command_manager: CommandManager = CommandManager(admin_id=admin_id)
        self.router.command_manager = self.command_manager
        self.media_uploader = None
        self._bot_name: str = "机器人"
        self._commands_registered = False
        self._deps_injected = False

        # 昵称
        self.nicknames: Dict[str, str] = {}
        self.auto_nicknames: Dict[str, str] = {}
        self._nickname_save_task: Optional[asyncio.Task] = None

        _log.info("BotEngine 已初始化")

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
        """
        同步回调——WS 就绪时初始化。

        ★ 不再启动 _process_messages_loop — AgentEngine 按需启动会话消费者。
        """
        if ready.user:
            self._bot_name = ready.user.username or "机器人"
        _log.info(f"机器人「{self._bot_name}」on_ready!")

        # 注册命令（仅在首次 on_ready 时注册，WS 重连跳过）
        if not self._commands_registered:
            self.command_manager.register_default_commands()
            self.command_manager.register_command(
                Command(
                    name="状态",
                    handler=self.handle_status_command,
                    aliases=["status"],
                    permission=PermissionLevel.ADMIN,
                    description="查看系统状态（管理员专用）",
                )
            )
            self._register_emoji_commands()
            self._commands_registered = True
            _log.info(f"已注册 {self.command_manager.registry.count()} 个命令")
        else:
            _log.debug("命令已在首次 on_ready 注册，跳过")

        # ★ 注册 ChatContextManager 提供的命令（历史/清空/聊天列表）
        self.agent_engine.context_manager.register_default_command(
            self.command_manager
        )

        # 加载昵称
        self.nicknames = self._load_nicknames()
        self.auto_nicknames = self._load_auto_nicknames()
        _log.info(f"已加载 {len(self.nicknames)} 个手动昵称 + {len(self.auto_nicknames)} 个自动昵称")

        # 初始化多模态（如尚未由外部注入）
        if self.multimodal_service is None:
            # 如果外部没有传入，尝试创建一个（兼容旧配置）
            _log.info("多模态服务未由外部注入，将使用外部创建或保持 None")

        # 初始化 MediaUploader（依赖 self.api，必须在 WS 就绪后）
        self.media_uploader = MediaUploader(
            api_client=self.api,
            http_client=self._http_client,
            log_tag="MeowQQ",
        )
        _log.info("MediaUploader 已初始化")

        # ★ 将延迟初始化的组件注入到 AgentEngine
        # 由于 _on_ready 是同步的，用 run_coroutine_threadsafe 来设置
        if self._main_loop:
            if not self._deps_injected:
                # 首次注入：全部组件
                future = asyncio.run_coroutine_threadsafe(
                    self._inject_agent_engine_deps(),
                    self._main_loop,
                )
                future.add_done_callback(
                    lambda f: _log.error(f"注入 AgentEngine 依赖失败: {f.exception()}")
                    if f.exception() else None
                )
                self._deps_injected = True
            else:
                # 重连注入：仅更新可能变化的组件
                future = asyncio.run_coroutine_threadsafe(
                    self._reconnect_update_agent_engine(),
                    self._main_loop,
                )
                future.add_done_callback(
                    lambda f: _log.error(f"重连更新 AgentEngine 失败: {f.exception()}")
                    if f.exception() else None
                )

    async def _inject_agent_engine_deps(self):
        """将 BotEngine 拥有的依赖注入到 AgentEngine。"""
        self.agent_engine.set_media_uploader(self.media_uploader)
        self.agent_engine.set_api_client(self.api)
        if self.multimodal_service:
            self.agent_engine.set_multimodal_service(self.multimodal_service)
        if self.emoji_manager:
            self.agent_engine.set_emoji_manager(self.emoji_manager)
        self.agent_engine.set_nicknames(self.nicknames, self.auto_nicknames)

    async def _reconnect_update_agent_engine(self):
        """WS 重连时仅更新可能变化的组件。"""
        self.agent_engine.set_media_uploader(self.media_uploader)
        self.agent_engine.set_nicknames(self.nicknames, self.auto_nicknames)

    def start(self, gateway_url: str, main_loop: asyncio.AbstractEventLoop) -> None:
        """启动 WebSocket 连接。"""
        self._main_loop = main_loop
        self.ws = QQWebSocket(callbacks=self._build_callbacks())
        self.ws.start(gateway_url, main_loop)

    async def stop(self) -> None:
        """安全关闭。"""
        # 等待防抖持久化完成，然后兜底写入一次
        if self._nickname_save_task and not self._nickname_save_task.done():
            try:
                await asyncio.wait_for(self._nickname_save_task, timeout=3.0)
            except asyncio.TimeoutError:
                pass
        self._save_auto_nicknames()
        await self.agent_engine.stop()
        if self.ws:
            await self.ws.async_stop()
        await self._http_client.aclose()

    # ── 事件处理 ──

    async def _on_message_event(self, event_type: str, raw: dict) -> None:
        """处理所有入站消息事件，解析后交由 Router 分发。"""
        event = EventParser().parse(event_type, raw)
        if event is None:
            return

        _log.info(f"[{event.chat_scope}][({event_type})] {event.user_id}: {event.content}")

        # ── 检测自定义表情（faceType=6 + attachments）──
        if is_custom_emoji(event.content, event.attachments):
            _log.info(f"检测到自定义表情，用户: {event.user_id}")
            try:
                if self.emoji_manager:
                    desc, tags = await self.emoji_manager.get_or_build(
                        event.attachments[0]
                    )
                    tag_str = " ".join(tags) if tags else ""
                    event.content = f"[表情: {desc}]"
                    if tag_str:
                        event.content += f" [情绪: {tag_str}]"
                else:
                    event.content = "[自定义表情]"
            except Exception as e:
                _log.error(f"自定义表情处理失败: {e}")
                event.content = "[自定义表情]"
        else:
            # 跳过空内容或仅 QQ 内置表情
            stripped = event.content.strip()
            if not stripped:
                return
            cleaned = re.sub(r'<faceType=\d+,[^>]+>', '', stripped).strip()
            if not cleaned:
                return

        # DM（频道直发消息）→ 简单回复
        if event.chat_scope == "dm":
            await self.api.send_text(
                "guild", event.chat_id,
                f"机器人{self._bot_name}收到你的消息了: {event.content}",
                reply_to=event.message_id,
            )
            return

        # 解析 @提及
        mentioned_ids = []
        mentions_data = raw.get("mentions", [])
        for m in mentions_data:
            uid = m.get("id")
            if uid:
                mentioned_ids.append(uid)
                nickname = m.get("username") or uid
                if m.get("is_you"):
                    nickname = self._bot_name
                event.content = event.content.replace(f"<@{uid}>", f"@{nickname}")
        event.content = event.content.strip()

        is_at_mention = any(m.get("is_you") for m in mentions_data)

        # 提取引用消息
        replied_content = ""
        replied_author = ""
        if event.msg_elements:
            elem = event.msg_elements[0]
            raw_elems = raw.get("msg_elements", [])
            if raw_elems:
                replied_author = raw_elems[0].get("author", {}).get("username", "")
            if elem.attachments and is_custom_emoji(elem.content or "", elem.attachments):
                try:
                    if self.emoji_manager:
                        desc, tags = await self.emoji_manager.get_or_build(
                            elem.attachments[0]
                        )
                        tag_str = " ".join(tags) if tags else ""
                        replied_content = f"[表情: {desc}]"
                        if tag_str:
                            replied_content += f" [情绪: {tag_str}]"
                except Exception as e:
                    _log.error(f"解析引用消息中的自定义表情失败: {e}")
                    replied_content = "[引用消息: 自定义表情]"
            elif elem.attachments:
                replied_content = (elem.content or "") + " [含附件]"
            else:
                replied_content = elem.content or ""

        # 采集昵称
        self._collect_nickname(
            raw.get("author", {}).get("id", ""),
            raw.get("author", {}).get("username", ""),
        )
        for m in mentions_data:
            self._collect_nickname(m.get("id", ""), m.get("username", ""))
        for raw_elem in raw.get("msg_elements", []):
            elem_author = raw_elem.get("author", {})
            self._collect_nickname(elem_author.get("id", ""), elem_author.get("username", ""))

        # 构造 InputMessage 并交给 Router
        input_message = InputMessage(
            id=event.message_id,
            sender_id=event.user_id,
            chat_id=event.chat_id,
            content=event.content,
            is_group=(event.chat_scope == "group"),
            is_at_mention=is_at_mention,
            mentioned_ids=mentioned_ids,
            replied_content=replied_content,
            replied_author=replied_author,
        )

        await self.router.route(
            input_message=input_message,
            reply_callback=self._send_reply,
            get_user_nickname=self._get_user_nickname,
        )

    # ── 发送回复 ──

    async def _send_reply(
        self, chat_id: str, content: str, message_id: str, is_group: bool = False
    ) -> None:
        """发送回复——使用 SDK send_text 自动路由 + 重试。"""
        chat_type = "group" if is_group else "c2c"
        await self.api.send_text(chat_type, chat_id, content, reply_to=message_id)
        _log.info(f"已发送回复: {chat_id}, 消息ID: {message_id}")

    # ── 状态命令 ──

    def handle_status_command(
        self, input_message: InputMessage, _: str
    ) -> List[Dict[str, Any]]:
        """处理状态命令，显示系统状态（管理员专用）"""
        try:
            chat_id = input_message.chat_id

            # 系统资源
            import psutil
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used / (1024**3)
            memory_total = memory.total / (1024**3)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent
            disk_used = disk.used / (1024**3)
            disk_total = disk.total / (1024**3)
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**2)
            process_cpu = process.cpu_percent(interval=0.1)

            # 从 AgentEngine 获取统计信息
            stats = self.agent_engine.get_stats()
            queue_sizes = stats.get("queue_sizes", {})
            total_queue = sum(queue_sizes.values())
            active_chats = stats.get("active_chats", 0)

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
                f"消息队列: {total_queue} 条（{len(queue_sizes)} 个活跃会话）",
                f"活跃聊天: {active_chats} 个",
                f"管理员ID: {', '.join(self.admin_id) if self.admin_id else '未设置'}",
            ]

            reply_content = "\n".join(status_text)
            return [
                {
                    "chat_id": chat_id,
                    "content": reply_content,
                    "message_id": input_message.id,
                    "is_group": input_message.is_group,
                }
            ]

        except ImportError:
            return [
                {
                    "chat_id": chat_id,
                    "content": "无法获取系统状态信息，请安装psutil库。",
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
        if self.emoji_manager is None:
            return [{"chat_id": input_message.chat_id, "content": "表情管理器未就绪。",
                     "message_id": input_message.id, "is_group": input_message.is_group}]
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
        if self.emoji_manager is None:
            return [{"chat_id": input_message.chat_id, "content": "表情管理器未就绪。",
                     "message_id": input_message.id, "is_group": input_message.is_group}]
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

        record = self.emoji_manager.find_by_hash(emoji_hash)
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
            lines.append(f"★ 用户自定义描述: {record.get('user_description', '(无)')}")
            lines.append(f"★ 用户自定义标签: {', '.join(record.get('user_tags', [])) or '(无)'}")
        lines.append(f"")
        lines.append(f"创建时间: {record.get('created_at', 'N/A')}")
        lines.append(f"最后更新: {record.get('updated_at', 'N/A')}")
        lines.append(f"URL: {record.get('url', 'N/A')[:60]}...")

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
        if self.emoji_manager is None:
            return [{"chat_id": input_message.chat_id, "content": "表情管理器未就绪。",
                     "message_id": input_message.id, "is_group": input_message.is_group}]
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

        record = self.emoji_manager.find_by_hash(emoji_hash)
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
        if self.emoji_manager is None:
            return [{"chat_id": input_message.chat_id, "content": "表情管理器未就绪。",
                     "message_id": input_message.id, "is_group": input_message.is_group}]
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

        record = self.emoji_manager.find_by_hash(emoji_hash)
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

    # ════════════════════════════════════════════════════════════
    # 昵称管理
    # ════════════════════════════════════════════════════════════

    def _load_nicknames(self) -> Dict[str, str]:
        """加载昵称映射文件（nicknames.json）"""
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
        """获取用户昵称，手动优先，自动兜底。"""
        if user_id in self.nicknames:
            return self.nicknames[user_id]
        if user_id in self.auto_nicknames:
            return self.auto_nicknames[user_id]
        return user_id

    def _load_auto_nicknames(self) -> Dict[str, str]:
        """加载自动采集的昵称文件（data/nicknames.json）"""
        path = "data/nicknames.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                _log.error(f"加载自动昵称文件失败: {e}")
        return {}

    def _save_auto_nicknames(self) -> None:
        """将自动采集的昵称持久化到 data/nicknames.json"""
        path = "data/nicknames.json"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.auto_nicknames, f, ensure_ascii=False, indent=2)
        except Exception as e:
            _log.error(f"保存自动昵称失败: {e}")

    def _collect_nickname(self, user_id: str, username: str) -> None:
        """采集一个用户昵称（防抖持久化，10 秒内多次写入合并为一次）。"""
        if not user_id or not username:
            return
        if user_id == self._bot_id:
            return
        if user_id in self.nicknames:
            return
        if self.auto_nicknames.get(user_id) == username:
            return
        self.auto_nicknames[user_id] = username
        _log.debug(f"已采集昵称: {username} ({user_id[:12]}..)")
        # 防抖持久化
        if self._nickname_save_task is None or self._nickname_save_task.done():
            self._nickname_save_task = asyncio.create_task(
                self._debounced_save_nicknames()
            )

    async def _debounced_save_nicknames(self):
        """10 秒防抖后批量持久化昵称。"""
        await asyncio.sleep(10)
        self._save_auto_nicknames()
