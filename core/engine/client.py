import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from qqbot_agent_sdk import (
    QQApiClient,
    QQWebSocket,
    WSCallbacks,
    parse_interaction_event,
    parse_approval_button_data,
)
from qqbot_agent_sdk.constants import MEDIA_TYPE_IMAGE
from qqbot_agent_sdk.dto import InlineKeyboard, MediaInfo, MessageToCreate, QQMessageType, WSReadyData
from qqbot_agent_sdk.media_loader import MediaUploader

from core.engine.agent_engine import AgentEngine
from core.engine.message_parser import MessageParser, MessageParserDeps
from core.engine.router import Router
from core.managers.command_manager import CommandManager
from core.managers.emoji_manager import EmojiManager
from core.managers.nickname_manager import NicknameManager
from core.message import InputMessage
from core.ai.multimodal import MultimodalService

_log = logging.getLogger(__name__)

QQBOT_MARKDOWN_SAFE_CHUNK_BYTE_LIMIT = 3600


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

    # ── Markdown 文本拆分 ──

    @staticmethod
    def _utf8len(text: str) -> int:
        return len(text.encode('utf-8'))

    @staticmethod
    def _is_fence_line(line: str) -> str | None:
        m = re.match(r'^(\s*)(`{3,}|~{3,})', line)
        return m.group(2) if m else None

    @staticmethod
    def _is_closing_fence_line(line: str, marker: str) -> bool:
        marker_char = marker[0]
        m = re.match(r'^\s*(' + re.escape(marker_char) + r'{3,})\s*$', line)
        return bool(m and len(m.group(1)) >= len(marker))

    @staticmethod
    def _is_table_separator(line: str) -> bool:
        line = line.strip()
        if not line.startswith('|') or not line.endswith('|'):
            return False
        cells = [c.strip() for c in line[1:-1].split('|')]
        return len(cells) >= 2 and all(re.match(r'^:?-+:?$', c) for c in cells)

    @staticmethod
    def _is_table_row(line: str) -> bool:
        line = line.strip()
        if not line.startswith('|') or not line.endswith('|'):
            return False
        return len([c.strip() for c in line[1:-1].split('|')]) >= 2

    @staticmethod
    def _append_or_flush(chunks: list[str], text: str, max_bytes: int, spacer: str = '\n\n'):
        if not text:
            return
        if chunks:
            last = chunks[-1]
            cand = last + spacer + text
            if BotEngine._utf8len(cand) <= max_bytes:
                chunks[-1] = cand
                return
        chunks.append(text)

    @staticmethod
    def _chunk_text(t: str, max_bytes: int, chunks: list[str]):
        if BotEngine._utf8len(t) <= max_bytes:
            BotEngine._append_or_flush(chunks, t, max_bytes)
            return
        for para in t.split('\n\n'):
            para = para.strip()
            if not para:
                continue
            if BotEngine._utf8len(para) <= max_bytes:
                BotEngine._append_or_flush(chunks, para, max_bytes)
            else:
                for sentence in re.split(r'(?<=[。！？!?\n])', para):
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if BotEngine._utf8len(sentence) <= max_bytes:
                        BotEngine._append_or_flush(chunks, sentence, max_bytes, spacer='\n')
                    else:
                        buf = ''
                        for char in sentence:
                            cand = buf + char
                            if BotEngine._utf8len(cand) <= max_bytes:
                                buf = cand
                            else:
                                chunks.append(buf)
                                buf = char
                        if buf:
                            chunks.append(buf)

    @staticmethod
    def _flush_text(chunks: list[str], text_lines: list[str], max_bytes: int):
        if not text_lines:
            return
        t = '\n'.join(text_lines)
        text_lines.clear()
        BotEngine._chunk_text(t, max_bytes, chunks)

    @staticmethod
    def _flush_table(chunks: list[str], table_lines: list[str], max_bytes: int):
        if len(table_lines) < 3:
            return
        full = '\n'.join(table_lines)
        if BotEngine._utf8len(full) <= max_bytes:
            chunks.append(full)
            table_lines.clear()
            return
        header = table_lines[0:2]
        rows = table_lines[2:]
        out_lines = list(header)
        for row in rows:
            cand = '\n'.join(out_lines + [row])
            if BotEngine._utf8len(cand) <= max_bytes:
                out_lines.append(row)
            else:
                if len(out_lines) > 2:
                    chunks.append('\n'.join(out_lines))
                out_lines = list(header) + [row]
        if len(out_lines) > 2:
            chunks.append('\n'.join(out_lines))
        table_lines.clear()

    @staticmethod
    def _split_markdown(text: str, max_bytes: int = QQBOT_MARKDOWN_SAFE_CHUNK_BYTE_LIMIT) -> list[str]:
        if not text:
            return []
        if BotEngine._utf8len(text) <= max_bytes:
            return [text]

        chunks: list[str] = []
        text_lines: list[str] = []
        fence_body: list[str] = []
        active_fence: tuple[str, str] | None = None  # (open_line, marker)

        pending_header: str | None = None
        pending_header_cells: list[str] | None = None
        table_lines: list[str] = []
        in_table = False

        def _ct(t: str):
            BotEngine._chunk_text(t, max_bytes, chunks)

        def _flush_text():
            nonlocal text_lines
            if active_fence:
                return
            BotEngine._flush_text(chunks, text_lines, max_bytes)

        def _flush_fence_and_close():
            nonlocal fence_body, active_fence
            if not active_fence:
                return
            open_line, marker = active_fence
            close = marker
            if not fence_body:
                chunks.append(f"{open_line}\n{close}")
            else:
                body = '\n'.join(fence_body)
                full = f"{open_line}\n{body}\n{close}"
                if BotEngine._utf8len(full) <= max_bytes:
                    chunks.append(full)
                else:
                    lines = list(fence_body)
                    cur = [open_line]
                    for line in lines:
                        cand = '\n'.join(cur + [line, close])
                        if BotEngine._utf8len(cand) <= max_bytes:
                            cur.append(line)
                        else:
                            chunks.append('\n'.join(cur + [close]))
                            cur = [open_line, line]
                    if len(cur) > 1:
                        chunks.append('\n'.join(cur + [close]))
            fence_body.clear()
            active_fence = None

        def _flush_table_lines():
            nonlocal table_lines, in_table, pending_header, pending_header_cells
            if in_table or (pending_header and table_lines):
                BotEngine._flush_table(chunks, table_lines, max_bytes)
                in_table = False
                pending_header = None
                pending_header_cells = None
            elif pending_header:
                text_lines.append(pending_header)
                pending_header = None
                pending_header_cells = None

        lines = text.split('\n')

        for line in lines:
            marker = BotEngine._is_fence_line(line)
            if marker:
                if active_fence is None:
                    _flush_table_lines()
                    _flush_text()
                    active_fence = (line, marker)
                elif BotEngine._is_closing_fence_line(line, active_fence[1]):
                    _flush_fence_and_close()
                continue

            if active_fence:
                fence_body.append(line)
                continue

            if in_table and BotEngine._is_table_row(line):
                table_lines.append(line)
                continue

            if BotEngine._is_table_separator(line):
                if pending_header is not None:
                    _flush_text()
                    table_lines = [pending_header, line]
                    in_table = True
                    pending_header = None
                    pending_header_cells = None
                else:
                    text_lines.append(line)
                continue

            if BotEngine._is_table_row(line):
                if pending_header is not None:
                    text_lines.append(pending_header)
                    pending_header = None
                    pending_header_cells = None
                    text_lines.append(line)
                elif in_table:
                    table_lines.append(line)
                else:
                    pending_header = line
                    pending_header_cells = [c.strip() for c in line.strip()[1:-1].split('|')]
                continue

            _flush_table_lines()
            text_lines.append(line)

        _flush_table_lines()
        if active_fence:
            _flush_fence_and_close()
        _flush_text()

        return chunks

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
            set_session=lambda sid, seq: setattr(self, "_session_id", sid) or setattr(self, "_last_seq", seq),
            set_heartbeat_interval=lambda interval: _log.info(f"WS 心跳间隔: {interval}s"),
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
        await self.nickname_manager.save_auto()
        await self.agent_engine.stop()
        if self.ws:
            await self.ws.async_stop()
        await self._http_client.aclose()

    # ── 事件处理 ──

    async def _on_message_event(self, event_type: str, raw: dict) -> None:
        parsed = await self.parser.parse(event_type, raw)
        if parsed is None:
            return

        _log.info(f"[{parsed.chat_scope}][({event_type})] {parsed.sender_id}: {parsed.content}")

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
                session_key, decision, approver_id,
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
                return await self.api.post_group_message(chat_id, msg, keyboard=keyboard)
            return await self.api.post_c2c_message(chat_id, msg, keyboard=keyboard)

        if keyboard:
            msg = self.api.build_text_body(content, reply_to=reply_to, markdown=markdown)
            if is_group:
                return await self.api.post_group_message(chat_id, msg, keyboard=keyboard)
            return await self.api.post_c2c_message(chat_id, msg, keyboard=keyboard)

        chunks = BotEngine._split_markdown(content)
        last_result: Dict[str, Any] = {}
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                chunk = f"[{i + 1}/{len(chunks)}]\n{chunk}"
            try:
                last_result = await self.api.send_text(
                    chat_type, chat_id, chunk,
                    reply_to=reply_to,
                    markdown=markdown,
                )
            except Exception:
                if markdown:
                    _log.warning("Chunk %d Markdown 发送失败，降级为纯文本重试", i)
                    last_result = await self.api.send_text(
                        chat_type, chat_id, chunk,
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
            chat_id, content, reply_to=message_id, is_group=is_group,
            media_file_info=media_file_info, markdown=markdown, keyboard=keyboard,
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
            chat_id, content, reply_to=None, is_group=is_group,
            media_file_info=media_file_info, markdown=markdown, keyboard=keyboard,
        )

    async def _send_reply(
        self, chat_id: str, content: str, message_id: str, is_group: bool = False
    ) -> None:
        try:
            reply_to = message_id if message_id else None
            await self._send(chat_id, content, reply_to=reply_to, is_group=is_group)
        except Exception as e:
            _log.error("发送回复失败 [%s]: %s", chat_id, e)
