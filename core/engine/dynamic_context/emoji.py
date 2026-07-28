import logging
from typing import Optional

_log = logging.getLogger(__name__)


class EmojiBlockBuilder:
    """构建表情标签动态块。"""

    def __init__(self, emoji_manager) -> None:
        self.emoji_manager = emoji_manager

    async def build(self, *, has_emojis: bool) -> Optional[str]:
        if not has_emojis or not self.emoji_manager:
            return None
        try:
            tags = self.emoji_manager.get_all_tags()
            if tags:
                return "可用表情标签：" + "、".join(tags)
        except Exception as e:
            _log.warning("表情标签获取失败: %s", e)
        return None
