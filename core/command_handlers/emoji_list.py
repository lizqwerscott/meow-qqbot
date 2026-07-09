import logging
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.emoji import EmojiManager
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(name="表情列表", aliases=["emojis"], description="查看所有已知自定义表情")
class EmojiListCommand:
    def __init__(self, emoji_manager: EmojiManager):
        self.emoji_manager = emoji_manager

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        try:
            page = 1
            if args.strip():
                try:
                    page = max(1, int(args.strip()))
                except ValueError:
                    pass

            result = self.emoji_manager.list_emojis(page=page, page_size=10)
            if result["total"] == 0:
                return make_reply(input_message, "还没有记录任何自定义表情。")

            lines = [f"已知自定义表情（共 {result['total']} 个，第 {result['page']} 页）："]
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

            return make_reply(input_message, "\n".join(lines))
        except Exception as e:
            _log.error(f"表情列表命令失败: {e}")
            return []
