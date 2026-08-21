import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.command_handlers.base import command, make_reply
from core.managers.archive_manager import ArchiveManager
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(
    name="归档", aliases=["archive"], permission="admin", description="会话归档管理"
)
class ArchiveCommand:
    def __init__(self, archive_manager: Optional[ArchiveManager] = None):
        self.archive_manager = archive_manager

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        if not self.archive_manager:
            return make_reply(input_message, "归档系统未启用。")

        parts = args.strip().split(maxsplit=1) if args.strip() else []
        subcmd = parts[0] if parts else ""
        subargs = parts[1] if len(parts) > 1 else ""

        if not subcmd or subcmd in ("当前", "this", "."):
            return await self._show_status(input_message)
        if subcmd in ("查看", "list", "ls"):
            return await self._list_archives(input_message, subargs)
        if subcmd in ("执行", "run", "do"):
            return await self._run_archive(input_message, subargs)
        if subcmd in ("摘要", "summary"):
            return await self._show_summary(input_message, subargs)
        if subcmd in ("清理", "clean"):
            return await self._clean_archives(input_message)
        return make_reply(
            input_message,
            "未知子命令。可用: 当前, 查看, 执行, 摘要, 清理",
        )

    async def _show_status(self, input_message: InputMessage) -> List[Dict[str, Any]]:
        chat_id = input_message.chat_id
        status = await self.archive_manager.get_session_status_async(chat_id)
        count = status["message_count"]
        last_act = (
            time.strftime("%H:%M", time.localtime(status["last_activity"]))
            if status["last_activity"]
            else "无"
        )
        return make_reply(
            input_message,
            f"会话: {chat_id[:24]}…\n"
            f"当前消息: {count} 条\n"
            f"最后活跃: {last_act}\n"
            f"归档摘要: {status['archive_count']} 个\n"
            f"归档触发: 跨天首条消息（按消息时间戳）\n"
            f"回放: 昨天最后一个连续片段（间隔 {self.archive_manager.replay_gap_seconds} 秒）\n"
            f"摘要: {self.archive_manager.summary_count} 条",
        )

    async def _list_archives(
        self, input_message: InputMessage, chat_id: str
    ) -> List[Dict[str, Any]]:
        target = chat_id or input_message.chat_id
        files = await self.archive_manager.list_archives_async(target)
        if not files:
            return make_reply(input_message, f"会话 {target[:24]}… 没有归档记录。")
        files = sorted(files, key=lambda item: item.get("path", ""), reverse=True)
        lines = [
            f"{Path(item['path']).stem} ({item.get('size', 0)} 字节)"
            for item in files[:20]
        ]
        reply = f"归档摘要 ({len(files)} 个):\n" + "\n".join(lines)
        if len(files) > 20:
            reply += f"\n... (还有 {len(files) - 20} 个)"
        return make_reply(input_message, reply)

    async def _show_summary(
        self, input_message: InputMessage, chat_id: str
    ) -> List[Dict[str, Any]]:
        target = chat_id or input_message.chat_id
        text = await self.archive_manager.load_recent_summaries_async(target)
        if not text:
            return make_reply(
                input_message, f"会话 {target[:24]}… 没有可用的归档摘要。"
            )
        preview = text[:1500]
        if len(text) > 1500:
            preview += "\n…(过长已截断)"
        return make_reply(input_message, f"最近摘要:\n{preview}")

    async def _run_archive(
        self, input_message: InputMessage, chat_id: str
    ) -> List[Dict[str, Any]]:
        target = chat_id or input_message.chat_id
        try:
            target_is_group = (
                input_message.is_group if target == input_message.chat_id else None
            )
            result = await self.archive_manager.archive_manual(target, target_is_group)
            return make_reply(
                input_message,
                f"会话 {target[:24]}… 归档完成。\n"
                f"保留: {result.replay_count} 条（含今天全部）\n"
                f"摘要: {'已生成' if result.summary_path else '无'}\n"
                f"归档: {'已写入' if result.archive_path else '无文件'}",
            )
        except Exception as e:
            return make_reply(input_message, f"归档失败: {e}")

    async def _clean_archives(
        self, input_message: InputMessage
    ) -> List[Dict[str, Any]]:
        try:
            removed = await self.archive_manager.cleanup_old_archives_async()
            return make_reply(input_message, f"归档清理完成，移除了 {removed} 个文件。")
        except Exception as e:
            return make_reply(input_message, f"清理失败: {e}")
