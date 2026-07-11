import logging
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.emoji import EmojiManager
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(name="表情重置", aliases=[], permission="admin", description="恢复表情为 AI 自动识别结果。用法：猫猫 /表情重置 <hash>")
class EmojiResetCommand:
    def __init__(self, emoji_manager: EmojiManager):
        self.emoji_manager = emoji_manager

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        emoji_hash = args.strip()
        if not emoji_hash:
            return make_reply(input_message, "请提供表情 hash。用法：猫猫 /表情重置 <hash>")

        record = self.emoji_manager.find_by_hash(emoji_hash)
        if record is None:
            return make_reply(input_message, f"未找到表情「{emoji_hash}」。")

        ok = self.emoji_manager.reset_to_auto(record["hash"])
        if ok:
            return make_reply(input_message, f"表情 {record['hash'][:12]}.. 已恢复为 AI 自动识别结果。")
        else:
            return make_reply(input_message, f"重置失败，请重试。")
