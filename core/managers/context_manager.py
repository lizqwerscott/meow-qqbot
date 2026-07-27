import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional

from core.managers.chat_context import ChatContext
from core.managers.context_store import ContextStore

_log = logging.getLogger(__name__)


class ChatContextManager:

    def __init__(
        self,
        store: ContextStore,
        max_history_per_chat: int = 10000,
        cleanup_interval: int = 3600,
        compact_threshold_tokens: int = 950000,
        keep_recent_tokens: int = 50000,
        max_tool_results: int = 5,
        keep_last_assistants: int = 3,
        soft_trim: int = 20000,
        hard_clear: int = 180000,
        merge_window_seconds: int = 15,
    ):
        self._store = store
        self.max_history_per_chat = max_history_per_chat
        self.cleanup_interval = cleanup_interval
        self.compact_threshold_tokens = compact_threshold_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.max_tool_results = max_tool_results
        self.keep_last_assistants = keep_last_assistants
        self.soft_trim = soft_trim
        self.hard_clear = hard_clear
        self.merge_window_seconds = merge_window_seconds
        self.contexts: Dict[str, ChatContext] = {}
        self._ctx_lock = asyncio.Lock()
        self._ctx_sync_lock = threading.Lock()
        self._chat_locks: Dict[str, asyncio.Lock] = {}

    @property
    def store(self) -> ContextStore:
        return self._store

    # ── 上下文获取 ──

    async def _get_chat_lock(self, chat_id: str) -> asyncio.Lock:
        async with self._ctx_lock:
            if chat_id not in self._chat_locks:
                self._chat_locks[chat_id] = asyncio.Lock()
            return self._chat_locks[chat_id]

    def get_context(self, chat_id: str) -> ChatContext:
        with self._ctx_sync_lock:
            if chat_id not in self.contexts:
                ctx = ChatContext(
                    chat_id=chat_id,
                    store=self._store,
                    max_history=self.max_history_per_chat,
                    compact_threshold_tokens=self.compact_threshold_tokens,
                    keep_recent_tokens=self.keep_recent_tokens,
                    merge_window_seconds=self.merge_window_seconds,
                )
                ctx.restore_from_store()
                self.contexts[chat_id] = ctx
            return self.contexts[chat_id]

    async def get_context_async(self, chat_id: str) -> ChatContext:
        return self.get_context(chat_id)

    # ── 聊天类型 ──

    async def record_chat_type(self, chat_id: str, is_group: bool) -> None:
        await self._store.set_chat_type(chat_id, is_group)

    def get_chat_type(self, chat_id: str) -> Optional[bool]:
        return self._store.get_chat_type(chat_id)

    # ── 消息添加 ──

    def add_user_message(
        self,
        chat_id: str,
        content: str,
        message_id: Optional[str] = None,
        sender_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        context = self.get_context(chat_id)
        context.add_user_message(content, message_id, sender_id=sender_id, name=name)

    def add_assistant_message(
        self,
        chat_id: str,
        content: str,
        message_id: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        reasoning_content: Optional[str] = None,
    ) -> None:
        context = self.get_context(chat_id)
        context.add_assistant_message(
            content, message_id, tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )

    def add_tool_result(
        self,
        chat_id: str,
        tool_name: str,
        content: str,
        tool_call_id: str,
    ) -> None:
        context = self.get_context(chat_id)
        context.add_tool_result(tool_name, content, tool_call_id)

    async def add_user_message_async(
        self,
        chat_id: str,
        content: str,
        message_id: Optional[str] = None,
        sender_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            self.add_user_message(
                chat_id, content, message_id, sender_id=sender_id, name=name
            )

    async def add_assistant_message_async(
        self,
        chat_id: str,
        content: str,
        message_id: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        reasoning_content: Optional[str] = None,
    ) -> None:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            self.add_assistant_message(
                chat_id, content, message_id, tool_calls=tool_calls,
                reasoning_content=reasoning_content,
            )

    async def add_tool_result_async(
        self,
        chat_id: str,
        tool_name: str,
        content: str,
        tool_call_id: str,
    ) -> None:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            self.add_tool_result(chat_id, tool_name, content, tool_call_id)

    # ── 历史读取 ──

    def get_history(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> List[Dict]:
        if chat_id not in self.contexts:
            return []
        return self.contexts[chat_id].get_history_as_dicts(max_messages)

    def get_chat_history(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> List[Dict]:
        return self.get_history(chat_id, max_messages)

    async def get_chat_history_async(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> List[Dict]:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            return self.get_chat_history(chat_id, max_messages)

    async def get_pruned_history_async(
        self,
        chat_id: str,
        max_messages: Optional[int] = None,
    ) -> List[Dict]:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            if chat_id not in self.contexts:
                return []
            ctx = self.contexts[chat_id]
            return ctx.get_pruned_history(
                max_messages=max_messages or len(ctx.history),
                max_tool_results=self.max_tool_results,
                keep_last_assistants=self.keep_last_assistants,
                soft_trim=self.soft_trim,
                hard_clear=self.hard_clear,
            )

    def get_conversation_context(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> str:
        if chat_id not in self.contexts:
            return ""
        return self.contexts[chat_id].get_conversation_context(max_messages)

    # ── 历史管理 ──

    def clear_history(self, chat_id: str) -> None:
        if chat_id in self.contexts:
            self.contexts[chat_id].clear_history()

    def clear_chat_history(self, chat_id: str) -> None:
        self.clear_history(chat_id)

    async def clear_chat_history_async(self, chat_id: str) -> None:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            self.clear_chat_history(chat_id)

    async def with_chat_lock(self, chat_id: str, func):
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            return await func()

    async def compact_history_if_needed(
        self, chat_id: str, ai_service, force: bool = False
    ) -> tuple[bool, Optional[Dict], "ChatContext"]:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            context = self.get_context(chat_id)
            compacted, usage = await context.compact_history_if_needed(
                ai_service, force=force
            )
        return compacted, usage, context

    # ── 上下文生命周期 ──

    def remove_context(self, chat_id: str) -> None:
        with self._ctx_sync_lock:
            if chat_id in self.contexts:
                del self.contexts[chat_id]
        self._chat_locks.pop(chat_id, None)
        self._store.release_file_lock(chat_id)

    def cleanup_inactive_contexts(self, max_inactivity: int = 7200) -> List[str]:
        removed = []
        with self._ctx_sync_lock:
            for chat_id, context in list(self.contexts.items()):
                if context.get_inactivity_time() > max_inactivity:
                    removed.append(chat_id)
                    del self.contexts[chat_id]
                    self._chat_locks.pop(chat_id, None)
        for cid in removed:
            self._store.release_file_lock(cid)
        return removed

    async def cleanup_inactive_contexts_async(
        self, max_inactivity: int = 7200
    ) -> List[str]:
        async with self._ctx_lock:
            return self.cleanup_inactive_contexts(max_inactivity)

    # ── 统计与查询 ──

    def get_all_chat_ids(self) -> List[str]:
        return list(self.contexts.keys())

    def get_all_disk_chat_ids(self) -> List[str]:
        disk_ids = self._store.get_all_disk_ids()
        memory_ids = set(self.contexts.keys())
        return sorted(set(disk_ids) | memory_ids)

    def get_all_chats(self) -> Dict[str, ChatContext]:
        return self.contexts.copy()

    def get_context_count(self) -> int:
        return len(self.contexts)

    def get_total_messages_count(self) -> int:
        total = 0
        for context in self.contexts.values():
            total += len(context.history)
        return total

    def get_archived_sessions_summary(self) -> Dict[str, int]:
        return self._store.get_archived_summary()

    def get_archived_files(self, chat_id: str) -> List[dict]:
        return self._store.list_archives(chat_id)

    def read_archived_messages(
        self, file_path: str, max_messages: int = 200
    ) -> List[Dict]:
        return self._store.read_archive(file_path, max_messages)
