import asyncio
import logging
import time
from collections import OrderedDict
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
        timestamp: Optional[float] = None,
    ) -> bool:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            context = await self._get_or_restore_context_locked(chat_id)
            return context.add_user_message(
                content,
                message_id,
                sender_id=sender_id,
                name=name,
                timestamp=timestamp,
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

    # Token 估算缓存：避免重复计算，key=chat_id, value=token_count
    # 使用 OrderedDict 实现 LRU，自动维护访问顺序
    _token_cache: OrderedDict[str, int] = OrderedDict()
    _token_cache_time: Dict[str, float] = {}  # 记录每个缓存条目的时间戳
    _token_cache_lock = asyncio.Lock()
    _token_cache_max_size: int = 1000  # 最大缓存会话数
    
    # 会话 ID 缓存：避免重复扫描磁盘文件
    _chat_ids_cache: Optional[List[str]] = None
    _chat_ids_cache_time: float = 0
    _chat_ids_cache_ttl: float = 60  # 缓存有效期 60 秒

    async def get_session_summary_async(self, chat_id: str) -> Dict[str, Any]:
        """获取完整会话摘要（用于详情页面）。"""
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            context = await self._get_or_restore_context_locked(chat_id)
            history = context.get_history_as_dicts()
            message_count = len(history)
            last_activity = context.last_activity
            
            # 使用缓存的 token 估算，避免每次重新计算
            estimated_tokens = await self._get_cached_tokens(chat_id)
            
            return {
                "message_count": message_count,
                "last_activity": last_activity,
                "estimated_tokens": estimated_tokens,
            }
    
    async def get_session_summary_light(self, chat_id: str) -> Dict[str, Any]:
        """获取轻量级会话摘要（用于列表页面）。
        
        只读取最后一条消息，避免加载完整历史，大幅提升列表页加载速度。
        """
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            context = await self._get_or_restore_context_locked(chat_id)
            
            # 只获取最后一条消息，不加载完整历史
            last_message = context.history[-1] if context.history else None
            message_count = len(context.history)
            last_activity = context.last_activity
            
            # 只估算最后一条消息的 token（轻量级）
            estimated_tokens = 0
            if last_message:
                try:
                    # 使用简单的字符估算（1 个 token ≈ 4 个字符）
                    estimated_tokens = len(last_message.content) // 4
                except Exception:
                    estimated_tokens = 0
            
            return {
                "message_count": message_count,
                "last_activity": last_activity,
                "estimated_tokens": estimated_tokens,
            }

    async def _get_cached_tokens(self, chat_id: str, context=None) -> int:
        """获取或计算 token 估算值，使用缓存避免重复计算。
        
        使用基于时间戳的失效检查和 LRU 策略。
        
        Args:
            chat_id: 会话 ID
            context: 可选的上下文对象（用于计算新值）
            
        Returns:
            估算的 token 数
        """
        async with self._token_cache_lock:
            # 检查缓存是否有效
            cached_tokens = self._token_cache.get(chat_id)
            cached_time = self._token_cache_time.get(chat_id, 0)
            now = time.time()
            
            # 缓存存在且未过期（1 秒内），直接返回
            if cached_tokens is not None and (now - cached_time) < 1.0:
                # 访问缓存，更新为最近使用
                self._token_cache.move_to_end(chat_id)
                return cached_tokens
            
            # 缓存失效或首次访问，需要重新计算
            if context is None:
                try:
                    context = await self._get_or_restore_context_locked(chat_id)
                except Exception:
                    return 0
            
            # 计算 token
            tokens = context.estimate_tokens_for_history()
            
            # 写入缓存（移除旧条目，添加新条目）
            if chat_id in self._token_cache:
                # 更新现有条目
                self._token_cache[chat_id] = tokens
                self._token_cache_time[chat_id] = now
                self._token_cache.move_to_end(chat_id)
            else:
                # 添加新条目
                self._token_cache[chat_id] = tokens
                self._token_cache_time[chat_id] = now
            
            # 如果超过最大缓存大小，清理最旧的条目
            if len(self._token_cache) > self._token_cache_max_size:
                self._prune_token_cache()
            
            return tokens

    async def _invalidate_token_cache(self, chat_id: str) -> None:
        """统一清理 token 缓存。
        
        集中处理以下场景：
        - 会话移除：删除对应的缓存条目
        - 清空历史：将 token 重置为 0
        
        Args:
            chat_id: 会话 ID
        """
        async with self._token_cache_lock:
            if chat_id in self._token_cache:
                del self._token_cache[chat_id]
                self._token_cache_time.pop(chat_id, None)
    
    def _prune_token_cache(self):
        """清理最旧的缓存条目（保留最近访问的）。
        
        使用 OrderedDict 的 popitem(last=False) 实现 O(1) LRU 清理。
        """
        if len(self._token_cache) <= self._token_cache_max_size:
            return
        
        # 计算需要移除的条目数
        to_remove = len(self._token_cache) - self._token_cache_max_size
        
        # 移除最旧的条目（O(1) 操作）
        for _ in range(to_remove):
            if self._token_cache:
                self._token_cache.popitem(last=False)  # 移除最旧的
                # 同时删除对应的时间戳
                self._token_cache_time.pop(next(iter(self._token_cache_time)), None)

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
            # 清空 token 缓存（历史已清空，token 应为 0）
            await self._invalidate_token_cache(chat_id)

    async def remove_message_if_async(
        self, chat_id: str, role: str, message_id: str
    ) -> bool:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            context = await self._get_or_restore_context_locked(chat_id)
            return context.remove_message_if(role, message_id)

    async def remove_last_user_message_if_async(
        self, chat_id: str, message_id: str
    ) -> bool:
        return await self.remove_message_if_async(chat_id, "user", message_id)

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
        
        # 清理 token 缓存
        await self._invalidate_token_cache(chat_id)

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
        # 使用缓存，减少磁盘 I/O
        async with self._ctx_lock:
            memory_ids = set(self.contexts)
        
        # 检查缓存是否过期（在锁外检查，避免死锁）
        now = time.time()
        if self._chat_ids_cache is not None and (now - self._chat_ids_cache_time) < self._chat_ids_cache_ttl:
            disk_ids = set(self._chat_ids_cache)
        else:
            disk_ids = set(await asyncio.to_thread(self._store.get_all_disk_ids))
            # 更新缓存（在锁外操作）
            self._chat_ids_cache = sorted(disk_ids)
            self._chat_ids_cache_time = now
        
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
