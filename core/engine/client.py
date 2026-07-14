import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from qqbot_agent_sdk import (
    QQApiClient,
    QQWebSocket,
    WSCallbacks,
    EventParser,
    InboundEvent,
    parse_interaction_event,
    parse_approval_button_data,
)
from qqbot_agent_sdk.constants import MEDIA_TYPE_IMAGE
from qqbot_agent_sdk.dto import InlineKeyboard, MediaInfo, MessageToCreate, QQMessageType, WSReadyData
from qqbot_agent_sdk.media_loader import MediaUploader

from core.card_parser import parse_card
from core.engine.agent_engine import AgentEngine
from core.managers.command_manager import CommandManager
from core.managers.emoji_manager import EmojiManager, is_custom_emoji
from core.message import InputMessage
from core.ai.multimodal import MultimodalService
from core.managers.nickname_manager import NicknameManager
from core.engine.router import Router

_log = logging.getLogger(__name__)


class BotEngine:
    """
    使用 qqbot_agent_sdk 的独立 QQ 机器人引擎。

    职责仅限：
    - WebSocket 连接管理
    - 消息接收与解析（→ InputMessage）
    - 发送回复（API 调用）

    AI 编排、会话管理、工具执行由 AgentEngine 负责。
    消息路由/命令分发由 Router 负责。
    昵称管理由 NicknameManager 负责。
    命令处理器在 core/command_handlers/ 中各独立文件实现。
    """

    def __init__(
        self,
        app_id: str,
        client_secret: str,
        bot_id: str,
        agent_engine: AgentEngine,
        router: Router,
        admin_id: List[str],
        nickname_manager: NicknameManager,
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
        self.nickname_manager = nickname_manager
        self.emoji_manager = emoji_manager
        self.multimodal_service = multimodal_service
        self.command_manager: CommandManager = CommandManager(admin_id=admin_id)
        self.router.command_manager = self.command_manager
        self.media_uploader = None
        self._bot_name: str = "机器人"
        self._deps_injected = False
        self.pending_approvals: Dict[str, tuple] = {}

        _log.info("BotEngine 已初始化")

    # ── 生命周期 ──

    def _build_callbacks(self) -> WSCallbacks:
        """构造 WSCallbacks，全部回调绑定到 BotEngine 方法。"""
        return WSCallbacks(
            on_message_event=self._on_message_event,
            on_interaction_event=self._on_interaction_event,
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
        """
        if ready.user:
            self._bot_name = ready.user.username or "机器人"
        _log.info(f"机器人「{self._bot_name}」on_ready!")

        self.nickname_manager.load_all()

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
        self.agent_engine.set_media_uploader(self.media_uploader)
        self.agent_engine.set_api_client(self.api)
        if self.multimodal_service:
            self.agent_engine.set_multimodal_service(self.multimodal_service)
        if self.emoji_manager:
            self.agent_engine.set_emoji_manager(self.emoji_manager)

    async def _reconnect_update_agent_engine(self):
        self.agent_engine.set_media_uploader(self.media_uploader)

    def start(self, gateway_url: str, main_loop: asyncio.AbstractEventLoop) -> None:
        """启动 WebSocket 连接。"""
        self._main_loop = main_loop
        self.ws = QQWebSocket(callbacks=self._build_callbacks())
        self.ws.start(gateway_url, main_loop)

    async def stop(self) -> None:
        await self.nickname_manager.flush_save()
        self.nickname_manager.save_auto()
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

        # ── 检测卡片消息（ARK/EMBED）──
        elif event.raw and ("ark" in event.raw or "ark_data" in event.raw or "embed" in event.raw or event.message_type in (3, 4)):
            _log.info(f"Card raw: {event.raw}")
            card_text = parse_card(event.raw or {}, event.message_type)
            if card_text:
                _log.info(f"检测到卡片消息，解析为: {card_text}")
                event.content = card_text
            else:
                _log.info("卡片消息解析失败，跳过")
                return

        else:
            # 跳过空内容或仅 QQ 内置表情
            stripped = event.content.strip()
            if not stripped:
                return
            cleaned = re.sub(r'<faceType=\d+,[^>]+>', '', stripped).strip()
            if not cleaned:
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

        # 如果机器人被 @，剥离 @bot_name 前缀并替换为 猫猫 前缀
        if is_at_mention:
            at_prefix = f"@{self._bot_name}"
            if event.content.startswith(at_prefix):
                rest = event.content[len(at_prefix):].strip()
                event.content = f"猫猫 {rest}" if rest else "猫猫"

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
        self.nickname_manager.collect(
            raw.get("author", {}).get("id", ""),
            raw.get("author", {}).get("username", ""),
        )
        for m in mentions_data:
            self.nickname_manager.collect(m.get("id", ""), m.get("username", ""))
        for raw_elem in raw.get("msg_elements", []):
            elem_author = raw_elem.get("author", {})
            self.nickname_manager.collect(elem_author.get("id", ""), elem_author.get("username", ""))

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
            get_user_nickname=self.nickname_manager.get,
        )

    # ── 交互事件（按钮点击） ──

    async def _on_interaction_event(self, event_type: str, raw: dict) -> None:
        """处理按钮交互事件（审批按钮和自定义键盘）。"""
        interaction = parse_interaction_event(raw)
        _log.info(f"收到按钮交互: {interaction.data.resolved.button_data}")

        # ACK 交互
        await self.api.acknowledge_interaction(interaction.id)

        chat_type = "c2c" if interaction.is_c2c else "group"
        chat_id = interaction.chat_id

        # 审批按钮
        parsed = parse_approval_button_data(interaction.data.resolved.button_data)
        if parsed:
            session_key, decision = parsed
            if session_key in self.pending_approvals:
                self.pending_approvals.pop(session_key)
                responses = {
                    "allow-once": "✅ 已允许一次",
                    "allow-always": "⭐ 已始终允许",
                    "deny": "❌ 已拒绝",
                }
                await self.send_proactive(
                    chat_id,
                    responses.get(decision, f"❓ 审批结果: {decision}"),
                    is_group=(chat_type == "group"),
                )
                _log.info(f"审批响应: {decision}")
            return

        # 自定义键盘
        if interaction.data.resolved.button_data.startswith("test_keyboard:"):
            choice = interaction.data.resolved.button_data.split(":")[-1]
            await self.send_proactive(
                chat_id,
                f"✓ 你选择了: {choice.upper()}",
                is_group=(chat_type == "group"),
            )
            _log.info(f"自定义键盘选择: {choice}")

    # ── 统一发送接口 ──

    async def send_reply(
        self,
        chat_id: str,
        content: str = "",
        *,
        message_id: str,
        is_group: bool = False,
        media_file_info: Optional[str] = None,
        markdown: bool = True,
        keyboard: Optional[InlineKeyboard] = None,
    ) -> Dict[str, Any]:
        """发送回复消息（被动消息，带 msg_id 上下文）。

        统一处理文本 / markdown / 富媒体。根据参数自动选择：
        - media_file_info 非空 → 富媒体 msg_type=7
        - keyboard 非空 → 附加内联键盘
        - 其他 → 文本 / markdown，走 SDK 内置重试
        """
        chat_type = "group" if is_group else "c2c"

        if media_file_info:
            msg = MessageToCreate(
                msg_type=QQMessageType.RICH_MEDIA,
                msg_seq=self.api.next_msg_seq(),
                msg_id=message_id,
                media=MediaInfo(file_info=media_file_info),
            )
            if is_group:
                return await self.api.post_group_message(chat_id, msg, keyboard=keyboard)
            return await self.api.post_c2c_message(chat_id, msg, keyboard=keyboard)

        if keyboard:
            msg = self.api.build_text_body(content, reply_to=message_id, markdown=markdown)
            if is_group:
                return await self.api.post_group_message(chat_id, msg, keyboard=keyboard)
            return await self.api.post_c2c_message(chat_id, msg, keyboard=keyboard)

        return await self.api.send_text(
            chat_type, chat_id, content,
            reply_to=message_id,
            markdown=markdown,
        )

    async def send_proactive(
        self,
        chat_id: str,
        content: str = "",
        *,
        is_group: bool = False,
        media_file_info: Optional[str] = None,
        markdown: bool = True,
        keyboard: Optional[InlineKeyboard] = None,
    ) -> Dict[str, Any]:
        """发送主动消息（无被动消息上下文，不含 msg_id）。"""
        chat_type = "group" if is_group else "c2c"

        if media_file_info:
            msg = MessageToCreate(
                msg_type=QQMessageType.RICH_MEDIA,
                msg_seq=self.api.next_msg_seq(),
                media=MediaInfo(file_info=media_file_info),
            )
            if is_group:
                return await self.api.post_group_message(chat_id, msg, keyboard=keyboard)
            return await self.api.post_c2c_message(chat_id, msg, keyboard=keyboard)

        if keyboard:
            msg = self.api.build_text_body(content, reply_to=None, markdown=markdown)
            if is_group:
                return await self.api.post_group_message(chat_id, msg, keyboard=keyboard)
            return await self.api.post_c2c_message(chat_id, msg, keyboard=keyboard)

        return await self.api.send_text(
            chat_type, chat_id, content,
            reply_to=None,
            markdown=markdown,
        )

    async def _send_reply(
        self, chat_id: str, content: str, message_id: str, is_group: bool = False
    ) -> None:
        """发送回复——委托给 send_reply。"""
        await self.send_reply(chat_id, content, message_id=message_id, is_group=is_group)
