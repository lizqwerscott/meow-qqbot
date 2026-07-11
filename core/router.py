"""Router — 轻量级消息路由器

判断消息类型（命令、普通对话），分别投递到不同队列或直接执行。
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from core.message import InputMessage

if TYPE_CHECKING:
    from core.agent_engine import AgentEngine
    from core.command_manager import CommandManager

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

        # ── 1. 命令检测 ──
        command_messages = await self.command_manager.process_message(input_message)
        if command_messages:
            for msg in command_messages:
                await reply_callback(
                    chat_id=msg["chat_id"],
                    content=msg["content"],
                    message_id=msg["message_id"],
                    is_group=msg["is_group"],
                )
            _log.debug(f"命令已处理: {input_message.content[:30]}")
            return

        _log.debug(f"非命令消息，转 AI: {input_message.content[:50]}")

        # ── 2. 非命令 → AI 对话处理 ──
        await self.agent_engine.dispatch(
            input_message=input_message,
            reply_callback=reply_callback,
            get_user_nickname=get_user_nickname,
        )
