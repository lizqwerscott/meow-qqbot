import logging
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


class ChatContextManager:
    """Ledger-backed conversation facade.

    Message history, deduplication and summaries all read/write the
    ``ConversationEventLog``; the retired active store, its in-memory
    ``ChatContext`` objects and the legacy JSONL archive readers were removed.
    """

    def __init__(
        self,
        max_history_per_chat: int = 10000,
        cleanup_interval: int = 3600,
    ):
        self.max_history_per_chat = max_history_per_chat
        self.cleanup_interval = cleanup_interval
        self._timeline = None
        self._protocol_history = None
        self._event_log = None
        self._prompt_projection = None
        self._chat_type_index = None

    def set_timeline(self, timeline: Any) -> None:
        self._timeline = timeline

    def set_protocol_history(self, protocol_history: Any) -> None:
        self._protocol_history = protocol_history

    def set_event_log(self, event_log: Any) -> None:
        """Use the core ledger for all normal history reads."""
        self._event_log = event_log

    def set_prompt_projection(self, projection: Any) -> None:
        self._prompt_projection = projection

    def set_chat_type_index(self, index: Any) -> None:
        """Use the delivery target catalog for group/private lookups."""
        self._chat_type_index = index

    # ── 聊天类型 ──

    def get_chat_type(self, chat_id: str) -> Optional[bool]:
        if self._chat_type_index is not None:
            return self._chat_type_index.get(chat_id)
        return None

    # ── 消息添加 ──

    async def add_user_message_async(
        self,
        chat_id: str,
        content: str,
        message_id: Optional[str] = None,
        sender_id: Optional[str] = None,
        name: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Return True when the ledger does not already hold this user message."""
        if self._event_log is None:
            return False
        has_user_message = getattr(self._event_log, "has_user_message", None)
        if callable(has_user_message):
            return not await has_user_message(chat_id, str(message_id or ""))
        history = await self._event_log.history(
            chat_id,
            include_internal=False,
            max_events=100,
        )
        return not any(
            event.get("role") == "user" and event.get("message_id") == message_id
            for event in history
        )

    async def add_assistant_message_async(
        self,
        chat_id: str,
        content: str,
        message_id: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        reasoning_content: Optional[str] = None,
    ) -> None:
        """Assistant messages are recorded by the ledger/tool loop, not here."""
        return None

    async def add_tool_result_async(
        self,
        chat_id: str,
        tool_name: str,
        content: str,
        tool_call_id: str,
    ) -> None:
        """Tool results are recorded by the ledger/tool loop, not here."""
        return None

    # ── 历史读取 ──

    async def get_chat_history_async(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> List[Dict]:
        if self._event_log is None:
            return []
        bounded_limit = max(0, int(max_messages)) if max_messages is not None else 100
        history = await self._event_log.history(
            chat_id,
            max_events=bounded_limit,
        )
        return (
            history[-max_messages:]
            if max_messages and max_messages > 0
            else ([] if max_messages == 0 else history)
        )

    async def get_session_summary_async(self, chat_id: str) -> Dict[str, Any]:
        """获取完整会话摘要（用于详情页面）。"""
        if self._event_log is None:
            return {"message_count": 0, "last_activity": None, "estimated_tokens": 0}
        return await self._event_log.session_summary(chat_id)

    async def get_session_summary_light(self, chat_id: str) -> Dict[str, Any]:
        """获取轻量级会话摘要（用于列表页面）。"""
        summary = await self.get_session_summary_async(chat_id)
        return {
            "message_count": summary["message_count"],
            "last_activity": summary["last_activity"],
            "estimated_tokens": summary.get("estimated_tokens", 0),
        }

    async def remove_orphaned_tool_calls_async(self, chat_id: str) -> int:
        return 0

    async def get_recent_user_contents_async(
        self, chat_id: str, count: int = 2
    ) -> List[str]:
        if self._event_log is None:
            return []
        if self._prompt_projection is not None:
            snapshot = await self._prompt_projection.snapshot_for_prompt(chat_id)
            history = [event.to_history_dict() for event in snapshot.events]
        else:
            bounded_events = max(20, max(1, int(count)) * 4)
            history = await self._event_log.history(
                chat_id,
                max_events=bounded_events,
            )
        return [
            message.get("content", "")
            for message in history
            if message.get("role") == "user"
        ][-count:]

    async def get_recent_user_messages_async(
        self, chat_id: str, count: int = 2
    ) -> List[Dict[str, Any]]:
        if self._event_log is None:
            return []
        bounded_events = max(20, max(1, int(count)) * 4)
        history = await self._event_log.history(
            chat_id,
            max_events=bounded_events,
        )
        return [message for message in history if message.get("role") == "user"][
            -count:
        ]

    async def get_pruned_history_async(
        self,
        chat_id: str,
        max_messages: Optional[int] = None,
    ) -> List[Dict]:
        if self._event_log is None:
            return []
        if self._prompt_projection is not None:
            snapshot = await self._prompt_projection.snapshot_for_prompt(chat_id)
            return [event.to_history_dict() for event in snapshot.events]
        bounded_limit = max_messages if max_messages is not None else 100
        return await self._event_log.history(
            chat_id,
            max_events=max(0, int(bounded_limit)),
        )

    # ── 历史管理 ──

    async def clear_chat_history_async(self, chat_id: str) -> None:
        clear_ledger = getattr(self._event_log, "clear_chat", None)
        if callable(clear_ledger):
            await clear_ledger(chat_id)
        if self._prompt_projection is not None:
            clear_projection = getattr(self._prompt_projection, "clear_chat", None)
            if callable(clear_projection):
                await clear_projection(chat_id)

    async def remove_message_if_async(
        self, chat_id: str, role: str, message_id: str
    ) -> bool:
        return False

    async def remove_last_user_message_if_async(
        self, chat_id: str, message_id: str
    ) -> bool:
        return False

    # ── 上下文生命周期（账本模式下无内存上下文） ──

    async def remove_context_async(self, chat_id: str) -> None:
        return None

    async def cleanup_inactive_contexts_async(
        self, max_inactivity: int = 7200
    ) -> List[str]:
        return []

    # ── 统计与查询 ──

    async def get_all_chat_ids_async(self) -> List[str]:
        if self._event_log is None:
            return []
        return sorted(await self._event_log.chat_ids())

    async def get_all_disk_chat_ids_async(self) -> List[str]:
        return await self.get_all_chat_ids_async()

    async def get_total_messages_count_async(self) -> int:
        if self._event_log is None:
            return 0
        total = 0
        for chat_id in await self._event_log.chat_ids():
            total += (await self._event_log.session_summary(chat_id))["message_count"]
        return total

    async def get_context_count_async(self) -> int:
        return 0

    async def get_archived_sessions_summary_async(self) -> Dict[str, int]:
        return {}

    async def get_archived_files_async(self, chat_id: str) -> List[dict]:
        return []

    async def read_archived_messages_async(
        self, file_path: str, max_messages: int = 200
    ) -> List[Dict]:
        return []
