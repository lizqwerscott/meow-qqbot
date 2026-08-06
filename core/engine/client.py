import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from qqbot_agent_sdk import (
    QQApiClient,
    QQWebSocket,
    WSCallbacks,
    parse_approval_button_data,
    parse_interaction_event,
)
from qqbot_agent_sdk.constants import MEDIA_TYPE_IMAGE
from qqbot_agent_sdk.dto import (
    InlineKeyboard,
    MediaInfo,
    MessageToCreate,
    QQMessageType,
    WSReadyData,
)
from qqbot_agent_sdk.media_loader import MediaUploader

from core.ai.multimodal import MultimodalService
from core.engine.agent_engine import AgentEngine
from core.engine.message_parser import MessageParser, MessageParserDeps
from core.engine.router import Router
from core.managers.command_manager import CommandManager
from core.managers.emoji_manager import EmojiManager
from core.managers.nickname_manager import NicknameManager
from core.markdown_split import split_markdown
from core.message import InputMessage

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
        permission_manager=None,
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
        self.permission_manager = permission_manager
        self.parser = MessageParser(MessageParserDeps(emoji_manager=emoji_manager))
        self.command_manager: CommandManager = CommandManager(
            admin_id=admin_id,
            permission_manager=permission_manager,
        )
        self.router.command_manager = self.command_manager
        self.media_uploader = None
        self._bot_name: str = "机器人"
        self._deps_injected = False
        self.approval_manager: Optional[Any] = None
        self._session_id: Optional[str] = None
        self._last_seq: Optional[int] = None

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
            get_session=lambda: (self._session_id, self._last_seq),
            set_session=lambda sid, seq: setattr(self, "_session_id", sid)
            or setattr(self, "_last_seq", seq),
            set_heartbeat_interval=lambda interval: _log.info(
                f"WS 心跳间隔: {interval}s"
            ),
            on_heartbeat_ack=lambda: _log.debug("WS 心跳 ACK 已确认"),
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
                    lambda f: (
                        _log.error(f"注入 AgentEngine 依赖失败: {f.exception()}")
                        if f.exception()
                        else None
                    )
                )
                self._deps_injected = True
            else:
                # 重连注入：仅更新可能变化的组件
                future = asyncio.run_coroutine_threadsafe(
                    self._reconnect_update_agent_engine(),
                    self._main_loop,
                )
                future.add_done_callback(
                    lambda f: (
                        _log.error(f"重连更新 AgentEngine 失败: {f.exception()}")
                        if f.exception()
                        else None
                    )
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
        await self.nickname_manager.save_auto()
        # 审批白名单使用计数落盘（防抖窗口内最后一次，避免关机丢失）
        if self.approval_manager is not None:
            try:
                self.approval_manager.flush()
            except Exception as e:
                _log.warning("审批白名单 flush 失败: %s", e)
        await self.agent_engine.stop()
        if self.ws:
            await self.ws.async_stop()
        await self._http_client.aclose()

    # ── 事件处理 ──

    async def _on_message_event(self, event_type: str, raw: dict) -> None:
        parsed = await self.parser.parse(event_type, raw)
        if parsed is None:
            return

        _log.info(
            f"[{parsed.chat_scope}][({event_type})] {parsed.sender_id}: {parsed.content}"
        )

        # 采集昵称（副作用）
        await self.nickname_manager.collect(parsed.author_id, parsed.author_username)
        for uid, name in parsed.mention_entries:
            await self.nickname_manager.collect(uid, name)
        for uid, name in parsed.reply_author_entries:
            await self.nickname_manager.collect(uid, name)

        input_message = InputMessage(
            id=parsed.id,
            sender_id=parsed.sender_id,
            chat_id=parsed.chat_id,
            content=parsed.content,
            is_group=(parsed.chat_scope == "group"),
            is_at_mention=parsed.is_at_mention,
            bot_id=self._bot_id,
            mentioned_ids=parsed.mentioned_ids,
            replied_content=parsed.replied_content,
            replied_author=parsed.replied_author,
            replied_author_id=parsed.replied_author_id,
            msg_type=parsed.msg_type,
            resources=parsed.resources,
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
            approver_id = interaction.operator_openid
            resolved = self.approval_manager and self.approval_manager.resolve(
                session_key,
                decision,
                approver_id,
            )
            if resolved:
                responses = {
                    "allow-once": "✅ 已允许一次",
                    "allow-always": "⭐ 已始终允许（已保存到白名单）",
                    "deny": "❌ 已拒绝",
                }
                await self.send_proactive(
                    chat_id,
                    responses.get(decision, f"❓ 审批结果: {decision}"),
                    is_group=(chat_type == "group"),
                )
                _log.info("审批响应: %s (by %s..)", decision, approver_id[:12])
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

    # ── 统一发送 ──

    async def _send(
        self,
        chat_id: str,
        content: str = "",
        *,
        reply_to: str | None = None,
        is_group: bool = False,
        media_file_info: Optional[str] = None,
        markdown: bool = True,
        keyboard: Optional[InlineKeyboard] = None,
    ) -> Dict[str, Any]:
        chat_type = "group" if is_group else "c2c"

        if media_file_info:
            msg = MessageToCreate(
                msg_type=QQMessageType.RICH_MEDIA,
                msg_seq=self.api.next_msg_seq(),
                media=MediaInfo(file_info=media_file_info),
            )
            if reply_to is not None:
                msg.msg_id = reply_to
            if is_group:
                return await self.api.post_group_message(
                    chat_id, msg, keyboard=keyboard
                )
            return await self.api.post_c2c_message(chat_id, msg, keyboard=keyboard)

        if keyboard:
            msg = self.api.build_text_body(
                content, reply_to=reply_to, markdown=markdown
            )
            if is_group:
                return await self.api.post_group_message(
                    chat_id, msg, keyboard=keyboard
                )
            return await self.api.post_c2c_message(chat_id, msg, keyboard=keyboard)

        chunks = split_markdown(content)
        last_result: Dict[str, Any] = {}
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                chunk = f"[{i + 1}/{len(chunks)}]\n{chunk}"
            try:
                last_result = await self.api.send_text(
                    chat_type,
                    chat_id,
                    chunk,
                    reply_to=reply_to,
                    markdown=markdown,
                )
            except Exception:
                if markdown:
                    _log.warning("Chunk %d Markdown 发送失败，降级为纯文本重试", i)
                    last_result = await self.api.send_text(
                        chat_type,
                        chat_id,
                        chunk,
                        reply_to=reply_to,
                        markdown=False,
                    )
                else:
                    raise
            if i < len(chunks) - 1:
                await asyncio.sleep(0.3)
        return last_result

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
        return await self._send(
            chat_id,
            content,
            reply_to=message_id,
            is_group=is_group,
            media_file_info=media_file_info,
            markdown=markdown,
            keyboard=keyboard,
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
        return await self._send(
            chat_id,
            content,
            reply_to=None,
            is_group=is_group,
            media_file_info=media_file_info,
            markdown=markdown,
            keyboard=keyboard,
        )

    async def _send_reply(
        self, chat_id: str, content: str, message_id: str, is_group: bool = False
    ) -> None:
        try:
            reply_to = message_id if message_id else None
            await self._send(chat_id, content, reply_to=reply_to, is_group=is_group)
        except Exception as e:
            _log.error("发送回复失败 [%s]: %s", chat_id, e)
