import logging
import time
from typing import Any, Dict, List, Optional

from core.command_handlers.base import command, make_reply
from core.managers.context_manager import ChatContextManager
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(
    name="历史",
    aliases=["history", "context", "ctx"],
    permission="admin",
    description="上下文管理",
)
class HistoryCommand:
    def __init__(self, context_manager: ChatContextManager):
        self.context_manager = context_manager

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        parts = args.strip().split(maxsplit=1) if args.strip() else []
        subcmd = parts[0] if parts else ""
        subargs = parts[1] if len(parts) > 1 else ""

        try:
            if not subcmd or subcmd in ("当前", "this", "."):
                return await self._show_status(input_message)
            if subcmd in ("查看", "view", "show"):
                return await self._view_chat(input_message, subargs)
            if subcmd in ("压缩", "compact"):
                return await self._compact(input_message, subargs)
            if subcmd in ("清空", "clear", "cls"):
                return await self._clear(input_message, subargs)
            if subcmd in ("列表", "list", "ls"):
                return await self._list_sessions(input_message)
            return make_reply(
                input_message,
                f"未知子命令: {subcmd}\n可用: 当前, 查看, 压缩, 清空, 列表",
            )
        except Exception as e:
            _log.error(f"历史命令处理失败: {e}")
            return make_reply(input_message, f"处理失败: {e}")

    async def _show_status(self, input_message: InputMessage) -> List[Dict[str, Any]]:
        chat_id = input_message.chat_id
        history = await self.context_manager.get_chat_history_async(chat_id)
        count = len(history)
        last = history[-1] if history else None
        last_time = (
            time.strftime(
                "%H:%M:%S", time.localtime(last.get("timestamp", time.time()))
            )
            if last
            else "无"
        )
        last_preview = (
            (last.get("content", "")[:80] + "…")
            if last and len(last.get("content", "")) > 80
            else (last.get("content", "") if last else "无")
        )
        role_counts = {}
        for message in history:
            role = message.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
        parts = [
            f"会话: {chat_id[:24]}…" if len(chat_id) > 24 else f"会话: {chat_id}",
            f"消息数: {count} (用户 {role_counts.get('user', 0)}, 助手 {role_counts.get('assistant', 0)}, 工具 {role_counts.get('tool', 0)})",
            f"最近活动: {last_time}",
            f"最近消息: {last_preview}",
            f"最大历史: {self.context_manager.max_history_per_chat} | 压缩阈值: {self.context_manager.compaction_threshold_tokens} tokens",
        ]
        return make_reply(input_message, "\n".join(parts))

    async def _view_chat(
        self, input_message: InputMessage, chat_id: str
    ) -> List[Dict[str, Any]]:
        target = chat_id or input_message.chat_id
        try:
            history = await self.context_manager.get_chat_history_async(target)
            lines = []
            for i, msg in enumerate(history, 1):
                role = (
                    "用户"
                    if msg["role"] == "user"
                    else ("助手" if msg["role"] == "assistant" else "工具")
                )
                content = msg.get("content", "") or ""
                preview = content[:100].replace("\n", " ")
                if len(content) > 100:
                    preview += "…"
                lines.append(f"{i}. [{role}] {preview}")
            if not lines:
                return make_reply(input_message, f"会话 {target[:24]}… 没有历史记录。")
            reply = f"会话 {target[:24]}… 的历史 ({len(lines)} 条):\n" + "\n".join(
                lines
            )
            if len(reply) > 2000:
                reply = reply[:2000] + "\n…(过长已截断)"
            return make_reply(input_message, reply)
        except Exception as e:
            return make_reply(input_message, f"查看失败: {e}")

    async def _compact(
        self, input_message: InputMessage, chat_id: str
    ) -> List[Dict[str, Any]]:
        try:
            target = chat_id or input_message.chat_id
            old_count = len(await self.context_manager.get_chat_history_async(target))
            compacted, _, new_count = (
                await self.context_manager.compact_history_if_needed(target, force=True)
            )
            if compacted:
                return make_reply(
                    input_message,
                    f"会话 {target[:24]}… 压缩完成: {old_count} → {new_count} 条",
                )
            return make_reply(
                input_message,
                f"会话 {target[:24]}… 无需压缩 ({old_count} 条, 阈值 {self.context_manager.compaction_threshold_tokens} tokens)",
            )
        except Exception as e:
            return make_reply(input_message, f"压缩失败: {e}")

    async def _clear(
        self, input_message: InputMessage, chat_id: str
    ) -> List[Dict[str, Any]]:
        target = chat_id or input_message.chat_id
        try:
            await self.context_manager.clear_chat_history_async(target)
            return make_reply(input_message, f"会话 {target[:24]}… 历史已清空。")
        except Exception as e:
            return make_reply(input_message, f"清空失败: {e}")

    async def _list_sessions(self, input_message: InputMessage) -> List[Dict[str, Any]]:
        all_ids = await self.context_manager.get_all_chat_ids_async()
        if not all_ids:
            return make_reply(input_message, "没有活跃的会话。")
        lines = []
        for cid in all_ids:
            history = await self.context_manager.get_chat_history_async(cid)
            count = len(history)
            last_act = (
                time.strftime(
                    "%H:%M", time.localtime(history[-1].get("timestamp", time.time()))
                )
                if history
                else "无"
            )
            short = cid[:16] + "…" if len(cid) > 16 else cid
            lines.append(f"{short} ({count} 条, {last_act})")
        reply = f"活跃会话 ({len(all_ids)}):\n" + "\n".join(lines)
        return make_reply(input_message, reply)
