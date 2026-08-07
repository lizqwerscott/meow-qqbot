from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.message import InputMessage


@command(
    name="媒体清理",
    aliases=["media-cleanup", "media_cleanup"],
    permission="admin",
    description="清理未被消息引用的媒体（管理员专用）",
)
class MediaCleanupCommand:
    def __init__(self, media_service=None):
        self.media_service = media_service

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        if self.media_service is None:
            return make_reply(input_message, "媒体服务未启用。")
        option = args.strip().lower()
        used, count = await self.media_service.usage()
        limit = self.media_service.max_total_bytes
        if option in {"状态", "usage", "status"}:
            limit_text = f"{limit / 1024 / 1024:.1f} MiB" if limit else "未设置"
            return make_reply(
                input_message,
                f"媒体占用：{used / 1024 / 1024:.1f} MiB，共 {count} 个对象。\n"
                f"容量提醒上限：{limit_text}。",
            )
        clear_all = option in {"全部", "all", "!"}
        removed = await self.media_service.cleanup(clear_all=clear_all)
        used, count = await self.media_service.usage()
        action = "全部媒体" if clear_all else "未引用媒体"
        return make_reply(
            input_message,
            f"媒体清理完成（{action}）：移除 {removed} 个对象。\n"
            f"当前占用：{used / 1024 / 1024:.1f} MiB，共 {count} 个对象。",
        )
