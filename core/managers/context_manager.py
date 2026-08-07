import asyncio
import logging
from typing import Any, Dict, List, Optional

from core.managers.chat_context import ChatContext
from core.managers.context_compactor import ContextCompactor
from core.managers.context_store import ContextStore

_log = logging.getLogger(__name__)


class ChatContextManager:

    def __init__(
        self,
        store: ContextStore,
        compactor: ContextCompactor,
        max_history_per_chat: int = 10000,
        cleanup_interval: int = 3600,
        max_tool_results: int = 5,
        keep_last_assistants: int = 3,
        soft_trim: int = 20000,
        hard_clear: int = 180000,
        merge_window_seconds: int = 15,
    ):
        self._store = store
        self._compactor = compactor
        self.max_history_per_chat = max_history_per_chat
        self.cleanup_interval = cleanup_interval
        self.max_tool_results = max_tool_results
        self.keep_last_assistants = keep_last_assistants
        self.soft_trim = soft_trim
        self.hard_clear = hard_clear
        self.merge_window_seconds = merge_window_seconds
        self.contexts: Dict[str, ChatContext] = {}
        self._ctx_lock = asyncio.Lock()
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

    async def _get_or_restore_context_locked(self, chat_id: str) -> ChatContext:
        async with self._ctx_lock:
            context = self.contexts.get(chat_id)
        if context is None:
            context = ChatContext(
                chat_id=chat_id,
                store=self._store,
                max_history=self.max_history_per_chat,
                merge_window_seconds=self.merge_window_seconds,
            )
            await context.restore_from_store_async()
            async with self._ctx_lock:
                existing = self.contexts.get(chat_id)
                if existing is None:
                    self.contexts[chat_id] = context
                    return context
                return existing
        return context

    # ── 聊天类型 ──

    async def record_chat_type(self, chat_id: str, is_group: bool) -> None:
        await self._store.set_chat_type(chat_id, is_group)

    def get_chat_type(self, chat_id: str) -> Optional[bool]:
        return self._store.get_chat_type(chat_id)

    # ── 消息添加 ──

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
            context = await self._get_or_restore_context_locked(chat_id)
            context.add_user_message(
                content, message_id, sender_id=sender_id, name=name
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
            context = await self._get_or_restore_context_locked(chat_id)
            context.add_assistant_message(
                content,
                message_id,
                tool_calls=tool_calls,
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
            context = await self._get_or_restore_context_locked(chat_id)
            context.add_tool_result(tool_name, content, tool_call_id)

    # ── 历史读取 ──

    async def get_chat_history_async(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> List[Dict]:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            context = await self._get_or_restore_context_locked(chat_id)
            return context.get_history_as_dicts(max_messages)

    async def get_history_as_dicts_merged_async(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> List[Dict]:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            context = await self._get_or_restore_context_locked(chat_id)
            return context.get_history_as_dicts_merged(max_messages)

    async def get_session_summary_async(self, chat_id: str) -> Dict[str, Any]:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            context = await self._get_or_restore_context_locked(chat_id)
            history = context.get_history_as_dicts()
            return {
                "message_count": len(history),
                "last_activity": context.last_activity,
                "estimated_tokens": context.estimate_tokens_for_history(),
            }

    async def remove_orphaned_tool_calls_async(self, chat_id: str) -> int:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            context = await self._get_or_restore_context_locked(chat_id)
            return context.remove_orphaned_tool_calls()

    async def get_recent_user_contents_async(
        self, chat_id: str, count: int = 2
    ) -> List[str]:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            context = await self._get_or_restore_context_locked(chat_id)
            return [
                message.content for message in context.history if message.role == "user"
            ][-count:]

    async def get_pruned_history_async(
        self,
        chat_id: str,
        max_messages: Optional[int] = None,
    ) -> List[Dict]:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            ctx = await self._get_or_restore_context_locked(chat_id)
            return ctx.get_pruned_history(
                max_messages=max_messages or len(ctx.history),
                max_tool_results=self.max_tool_results,
                keep_last_assistants=self.keep_last_assistants,
                soft_trim=self.soft_trim,
                hard_clear=self.hard_clear,
            )

    # ── 历史管理 ──

    async def clear_chat_history_async(self, chat_id: str) -> None:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            context = await self._get_or_restore_context_locked(chat_id)
            await context.clear_history_async()

    async def remove_last_user_message_if_async(
        self, chat_id: str, message_id: str
    ) -> bool:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            async with self._ctx_lock:
                context = self.contexts.get(chat_id)
            if context is None:
                return False
            return context.remove_last_message_if("user", message_id)

    async def with_chat_lock(self, chat_id: str, func):
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            return await func()

    async def _with_context_locked(self, chat_id: str, func):
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            context = await self._get_or_restore_context_locked(chat_id)
            return await func(context)

    @property
    def compaction_threshold_tokens(self) -> int:
        return self._compactor.compact_threshold_tokens

    async def compact_history_if_needed(
        self, chat_id: str, force: bool = False
    ) -> tuple[bool, Optional[Dict], int]:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            context = await self._get_or_restore_context_locked(chat_id)
            result = await self._compactor.compact(context.get_history(), force=force)
            if result.compacted:
                context.set_messages(result.messages)
            return result.compacted, result.usage, len(result.messages)

    # ── 上下文生命周期 ──

    async def remove_context_async(self, chat_id: str) -> None:
        lock = await self._get_chat_lock(chat_id)
        try:
            async with lock:
                async with self._ctx_lock:
                    context = self.contexts.pop(chat_id, None)
                if context is not None:
                    await context.wait_for_save_async()
        finally:
            self._store.release_file_lock(chat_id)

    async def cleanup_inactive_contexts_async(
        self, max_inactivity: int = 7200
    ) -> List[str]:
        removed = []
        async with self._ctx_lock:
            chat_ids = list(self.contexts)
        for chat_id in chat_ids:
            lock = await self._get_chat_lock(chat_id)
            removed_context = False
            try:
                async with lock:
                    async with self._ctx_lock:
                        context = self.contexts.get(chat_id)
                    if (
                        context is None
                        or context.get_inactivity_time() <= max_inactivity
                    ):
                        continue
                    async with self._ctx_lock:
                        removed_context = self.contexts.pop(chat_id, None) is not None
                    if not removed_context:
                        continue
                    await context.wait_for_save_async()
                    removed.append(chat_id)
            finally:
                if removed_context:
                    self._store.release_file_lock(chat_id)
        return removed

    # ── 统计与查询 ──

    async def get_all_chat_ids_async(self) -> List[str]:
        async with self._ctx_lock:
            return list(self.contexts.keys())

    async def get_all_disk_chat_ids_async(self) -> List[str]:
        disk_ids = await asyncio.to_thread(self._store.get_all_disk_ids)
        async with self._ctx_lock:
            memory_ids = set(self.contexts)
        return sorted(set(disk_ids) | memory_ids)

    async def get_total_messages_count_async(self) -> int:
        async with self._ctx_lock:
            chat_ids = list(self.contexts)
        total = 0
        for chat_id in chat_ids:
            history = await self.get_chat_history_async(chat_id)
            total += len(history)
        return total

    async def get_context_count_async(self) -> int:
        async with self._ctx_lock:
            return len(self.contexts)

    async def get_archived_sessions_summary_async(self) -> Dict[str, int]:
        return await asyncio.to_thread(self._store.get_archived_summary)

    async def get_archived_files_async(self, chat_id: str) -> List[dict]:
        return await asyncio.to_thread(self._store.list_archives, chat_id)

    async def read_archived_messages_async(
        self, file_path: str, max_messages: int = 200
    ) -> List[Dict]:
        return await asyncio.to_thread(
            self._store.read_archive, file_path, max_messages
        )
