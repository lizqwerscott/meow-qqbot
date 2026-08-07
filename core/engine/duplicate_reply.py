"""消息钩子：群聊自动复读检测器"""

import asyncio
import logging
from collections import OrderedDict

from core.message import MessageType

_log = logging.getLogger(__name__)

_MAX_CACHED_CHATS = 500


class DuplicateReplyDetector:
    """检测群聊中连续相同内容并自动复读"""

    def __init__(self, context_manager):
        self._cm = context_manager
        self._replied: "OrderedDict[str, str]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def handle_message(
        self, input_message, reply_callback, get_user_nickname
    ) -> bool:
        if not input_message.is_group:
            return False

        if input_message.msg_type != MessageType.TEXT:
            return False

        user_contents = await self._cm.get_recent_user_contents_async(
            input_message.chat_id
        )
        if len(user_contents) < 2:
            return False

        last_content = user_contents[-1].strip()
        prev_content = user_contents[-2].strip()
        if not last_content:
            return False
        if last_content != prev_content:
            return False

        async with self._lock:
            if self._replied.get(input_message.chat_id) == last_content:
                return False

        _log.info(
            "检测到重复消息 [%s..]，自动复读: %s",
            input_message.chat_id[:12],
            last_content[:30],
        )
        reply_id = f"dupe_{input_message.id}"
        try:
            await reply_callback(
                chat_id=input_message.chat_id,
                content=last_content,
                message_id=input_message.id,
                is_group=True,
            )
        except Exception as cb_err:
            _log.warning("复读发送失败 [%s..]: %s", input_message.chat_id[:12], cb_err)
            return False

        async with self._lock:
            self._replied[input_message.chat_id] = last_content
            self._replied.move_to_end(input_message.chat_id)
            if len(self._replied) > _MAX_CACHED_CHATS:
                self._replied.popitem(last=False)

        await self._cm.add_assistant_message_async(
            input_message.chat_id,
            last_content,
            reply_id,
        )
        return True
