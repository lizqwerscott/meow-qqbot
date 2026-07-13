import logging
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.emoji import EmojiManager
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(name="表情编辑", aliases=[], permission="admin", description="自定义表情描述和标签。用法：猫猫 /表情编辑 <hash> 描述=xxx 标签=xxx")
class EmojiEditCommand:
    def __init__(self, emoji_manager: EmojiManager):
        self.emoji_manager = emoji_manager

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        parts = args.strip().split()
        if len(parts) < 2:
            return make_reply(input_message, "格式：猫猫 /表情编辑 <hash> 描述=xxx 标签=A、B")

        emoji_hash = parts[0]
        desc = None
        tags = None

        for p in parts[1:]:
            if p.startswith("描述="):
                desc = p[3:]
            elif p.startswith("标签="):
                raw = p[3:]
                tags = [t.strip() for t in raw.replace("、", ",").split(",") if t.strip()]

        if desc is None and tags is None:
            return make_reply(input_message, "至少要提供描述或标签中的一个。\n格式：猫猫 /表情编辑 <hash> 描述=xxx 标签=A、B")

        record = self.emoji_manager.find_by_hash(emoji_hash)
        if record is None:
            return make_reply(input_message, f"未找到表情「{emoji_hash}」。「猫猫 /表情列表」查看所有。")

        ok = await self.emoji_manager.set_custom(record["hash"], description=desc, tags=tags)
        if ok:
            changes = []
            if desc is not None:
                changes.append(f"描述 → {desc}")
            if tags is not None:
                changes.append(f"标签 → {', '.join(tags)}")
            return make_reply(input_message, f"表情 {record['hash'][:12]}.. 已更新：{'；'.join(changes)}")
        else:
            return make_reply(input_message, f"更新失败，请重试。")
