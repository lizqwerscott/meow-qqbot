"""Router — 轻量级消息路由器

判断消息类型（命令、普通对话），分别投递到不同队列或直接执行。
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from core.message import InputMessage

if TYPE_CHECKING:
    from core.engine.agent_engine import AgentEngine
    from core.managers.command_manager import CommandManager

_log = logging.getLogger(__name__)


class Router:
    """
    轻量级消息路由器。

    持有 CommandManager 和 AgentEngine 引用。
    收到消息后：
    1. 检查命令 → 执行并回复
    2. 普通对话 → 交给 AgentEngine.dispatch()
    """

    def __init__(
        self,
        agent_engine: "AgentEngine",
    ):
        self.command_manager: Optional["CommandManager"] = None
        self.agent_engine = agent_engine

    async def route(
        self,
        input_message: InputMessage,
        reply_callback: Callable,
        get_user_nickname: Callable[[str], str],
        tool_event_callback: Optional[Callable] = None,
    ) -> None:
        """
        分发消息。

        Args:
            input_message: 已经解析的 InputMessage
            reply_callback: 发送回复的回调 (chat_id, content, message_id, is_group) -> None
            get_user_nickname: 获取用户昵称的回调 (user_id) -> str
        """
        if self.command_manager is None:
            _log.error("command_manager 未初始化，无法处理命令")
            return

        resolve_identity = getattr(self.agent_engine, "resolve_message_identity", None)
        if callable(resolve_identity):
            resolve_identity(input_message)

        # ── 1. 命令检测 ──
        command_messages = await self.command_manager.process_message(input_message)
        if command_messages:
            delivery_controller = None
            get_delivery_controller = getattr(
                self.agent_engine, "_get_delivery_controller", None
            )
            if callable(get_delivery_controller):
                try:
                    delivery_controller = get_delivery_controller()
                except Exception as exc:
                    _log.warning("命令回复 ledger 初始化失败: %s", exc)
            for index, msg in enumerate(command_messages):
                try:
                    delivery_chat_id = msg["chat_id"]
                    session_key = getattr(input_message, "session_key", "")
                    if session_key and msg.get("chat_id") == input_message.chat_id:
                        delivery_chat_id = session_key

                    async def _transport(**kwargs):
                        if msg.get("chat_id") == input_message.chat_id:
                            kwargs["chat_id"] = msg["chat_id"]
                            kwargs["is_group"] = input_message.is_group
                        return await reply_callback(**kwargs)

                    if delivery_controller is None:
                        await _transport(
                            chat_id=msg["chat_id"],
                            content=msg["content"],
                            message_id=msg["message_id"],
                            is_group=msg["is_group"],
                        )
                    else:
                        receipt = await delivery_controller.deliver_text(
                            delivery_id=(
                                f"command:{input_message.chat_id}:"
                                f"{input_message.id}:{index}"
                            ),
                            chat_id=delivery_chat_id,
                            content=msg["content"],
                            callback=_transport,
                            message_id=msg["message_id"],
                            is_group=msg["is_group"],
                            reason="command_reply",
                            timeline_delivery_kind=None,
                        )
                        if receipt.status not in {"accepted", "partial"}:
                            _log.warning(
                                "命令回复未确认 [%s..]: %s",
                                msg["chat_id"][:12],
                                receipt.status,
                            )
                except Exception as cb_err:
                    _log.warning(
                        "命令回复发送失败 [%s..]: %s", msg["chat_id"][:12], cb_err
                    )
            _log.debug(f"命令已处理: {input_message.content[:30]}")
            return

        _log.debug(f"非命令消息，转 AI: {input_message.content[:50]}")

        # ── 2. 非命令 → AI 对话处理 ──
        await self.agent_engine.dispatch(
            input_message=input_message,
            reply_callback=reply_callback,
            get_user_nickname=get_user_nickname,
            tool_event_callback=tool_event_callback,
        )
