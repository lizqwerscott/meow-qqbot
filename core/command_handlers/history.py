import logging
import time
from typing import Any, Dict, List, Optional

from core.command_handlers.base import command, make_reply
from core.engine.history_projection import visible_legacy_history
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
    def __init__(
        self,
        context_manager: ChatContextManager,
        timeline=None,
        protocol_history=None,
        agent_engine=None,
        event_log=None,
        model_context_transcript=None,
        prompt_history_projection=None,
        turn_summary_store=None,
        prompt_context_reports=None,
        archive_index=None,
    ):
        self.context_manager = context_manager
        self.timeline = timeline
        self.protocol_history = protocol_history
        self.agent_engine = agent_engine
        self.event_log = event_log
        self.model_context_transcript = model_context_transcript
        self.prompt_history_projection = prompt_history_projection
        self.turn_summary_store = turn_summary_store
        self.prompt_context_reports = prompt_context_reports
        self.archive_index = archive_index

    async def _get_visible_history(self, chat_id: str) -> List[Dict[str, Any]]:
        if self.event_log is not None:
            return await self.event_log.history(chat_id)
        if self.timeline is not None:
            projected = await self.timeline.history(chat_id)
            legacy = await self.context_manager.get_chat_history_async(chat_id)
            await self.timeline.repair_from_legacy_history(chat_id, legacy)
            if projected or legacy:
                return await self.timeline.history(chat_id)
            return projected
        return visible_legacy_history(
            await self.context_manager.get_chat_history_async(chat_id)
        )

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
        history = await self._get_visible_history(chat_id)
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
            history = await self._get_visible_history(target)
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
            if self.model_context_transcript is not None:
                scopes = await self.model_context_transcript.scopes_for_chat(target)
                changed = 0
                for scope in scopes:
                    result = await self.model_context_transcript.compact_if_needed(
                        scope, force=True
                    )
                    changed += int(result.changed)
                if not scopes:
                    return make_reply(
                        input_message,
                        f"会话 {target[:24]}… 使用 bounded Prompt projection，无独立模型上下文可压缩。",
                    )
                return make_reply(
                    input_message,
                    f"会话 {target[:24]}… 已执行增量模型上下文压缩：{changed} 个 scope，按完整 turn/checkpoint 处理。",
                )
            if self.event_log is None:
                compact = getattr(
                    self.context_manager, "compact_history_if_needed", None
                )
                if callable(compact):
                    changed, _usage, count = await compact(target, force=True)
                    return make_reply(
                        input_message,
                        f"会话 {target[:24]}… {'压缩完成' if changed else '无需压缩'}（{count} 条）。",
                    )
            return make_reply(
                input_message,
                f"会话 {target[:24]}… 没有可压缩的模型上下文 scope；当前使用 bounded Prompt projection。",
            )
        except Exception as e:
            return make_reply(input_message, f"压缩失败: {e}")

    async def _clear(
        self, input_message: InputMessage, chat_id: str
    ) -> List[Dict[str, Any]]:
        target = chat_id or input_message.chat_id
        try:
            clear_session = getattr(self.agent_engine, "clear_session_async", None)
            if callable(clear_session):
                await clear_session(target)
                return make_reply(input_message, f"会话 {target[:24]}… 历史已清空。")
            await self.context_manager.clear_chat_history_async(target)
            if self.event_log is not None:
                await self.event_log.clear_chat(target)
            if self.prompt_history_projection is not None:
                await self.prompt_history_projection.clear_chat(target)
            if self.turn_summary_store is not None:
                await self.turn_summary_store.clear_chat(target)
            if self.prompt_context_reports is not None:
                await self.prompt_context_reports.clear_chat(target)
            if self.model_context_transcript is not None:
                await self.model_context_transcript.clear_chat(target)
            if self.archive_index is not None:
                await self.archive_index.clear_chat(target)
            if self.timeline is not None:
                await self.timeline.clear_chat(target)
            if self.protocol_history is not None:
                await self.protocol_history.delete_chat(target)
            return make_reply(input_message, f"会话 {target[:24]}… 历史已清空。")
        except Exception as e:
            return make_reply(input_message, f"清空失败: {e}")

    async def _list_sessions(self, input_message: InputMessage) -> List[Dict[str, Any]]:
        if self.event_log is not None:
            try:
                all_ids = await self.event_log.chat_ids()
                if self.archive_index is not None:
                    all_ids = list(
                        dict.fromkeys(all_ids + await self.archive_index.chat_ids())
                    )
            except Exception:
                all_ids = []
        else:
            all_ids = await self.context_manager.get_all_chat_ids_async()
        if self.event_log is None and self.timeline is not None:
            try:
                all_ids = list(dict.fromkeys(all_ids + await self.timeline.chat_ids()))
            except Exception:
                pass
        if not all_ids:
            return make_reply(input_message, "没有活跃的会话。")
        lines = []
        for cid in all_ids:
            summary = None
            if self.event_log is not None:
                try:
                    candidate = await self.event_log.session_summary(cid)
                    if candidate.get("event_count", 0):
                        summary = candidate
                except Exception:
                    summary = None
            if self.event_log is None and summary is None and self.timeline is not None:
                try:
                    candidate = await self.timeline.session_summary(cid)
                    if not candidate.get("message_count", 0):
                        legacy = await self.context_manager.get_chat_history_async(cid)
                        await self.timeline.repair_from_legacy_history(cid, legacy)
                        candidate = await self.timeline.session_summary(cid)
                    summary = candidate
                except Exception:
                    summary = None
            if summary is not None:
                count = summary["message_count"]
                last_act = time.strftime(
                    "%H:%M", time.localtime(summary["last_activity"])
                )
            elif self.event_log is None:
                history = await self.context_manager.get_chat_history_async(cid)
                visible_history = visible_legacy_history(history)
                count = len(visible_history)
                last_act = (
                    time.strftime(
                        "%H:%M",
                        time.localtime(
                            visible_history[-1].get("timestamp", time.time())
                        ),
                    )
                    if visible_history
                    else "无"
                )
            else:
                count = 0
                last_act = "无"
            short = cid[:16] + "…" if len(cid) > 16 else cid
            lines.append(f"{short} ({count} 条, {last_act})")
        reply = f"活跃会话 ({len(all_ids)}):\n" + "\n".join(lines)
        return make_reply(input_message, reply)
