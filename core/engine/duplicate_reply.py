"""消息钩子：群聊自动复读检测器"""

import logging
from collections import OrderedDict

_log = logging.getLogger(__name__)

_MAX_CACHED_CHATS = 500


class DuplicateReplyDetector:
    """检测群聊中连续相同内容并自动复读"""

    def __init__(self, context_manager):
        self._cm = context_manager
        self._replied: "OrderedDict[str, str]" = OrderedDict()

    async def handle_message(self, input_message, reply_callback, get_user_nickname) -> bool:
        if not input_message.is_group:
            return False

        context = await self._cm.get_context_async(input_message.chat_id)
        user_msgs = [m for m in context.history if m.role == "user"]
        if len(user_msgs) < 2:
            return False

        last_content = user_msgs[-1].content
        prev_content = user_msgs[-2].content
        if last_content != prev_content:
            return False
        if self._replied.get(input_message.chat_id) == last_content:
            return False

        _log.info(f"检测到重复消息 [{input_message.chat_id}]，自动复读: {last_content[:30]}")
        await reply_callback(
            chat_id=input_message.chat_id,
            content=last_content,
            message_id=input_message.id,
            is_group=True,
        )
        await self._cm.add_assistant_message_async(
            input_message.chat_id, last_content, input_message.id,
        )
        self._replied[input_message.chat_id] = last_content
        self._replied.move_to_end(input_message.chat_id)
        if len(self._replied) > _MAX_CACHED_CHATS:
            self._replied.popitem(last=False)
        return True
