import logging
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.emoji_manager import EmojiManager
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(name="表情查看", aliases=["emoji"], description="查看指定表情的详细信息。用法：猫猫 /表情查看 <hash>")
class EmojiInfoCommand:
    def __init__(self, emoji_manager: EmojiManager):
        self.emoji_manager = emoji_manager

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        emoji_hash = args.strip()
        if not emoji_hash:
            return make_reply(input_message, "请提供表情 hash。用法：猫猫 /表情查看 <hash>")

        record = self.emoji_manager.find_by_hash(emoji_hash)
        if record is None:
            return make_reply(input_message, f"未找到表情「{emoji_hash}」。")

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

        return make_reply(input_message, "\n".join(lines))
