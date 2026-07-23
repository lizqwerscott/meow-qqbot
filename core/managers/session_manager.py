"""SessionTaskManager — 每会话队列 + 锁，实现会话级隔离。"""

import asyncio
from typing import Dict, Set


class SessionTaskManager:
    """管理每个 chat_id 的异步队列和锁，实现会话级隔离。"""

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._running: Set[str] = set()
        self._lock = asyncio.Lock()

    async def get_queue(self, chat_id: str) -> asyncio.Queue:
        async with self._lock:
            if chat_id not in self._queues:
                self._queues[chat_id] = asyncio.Queue(maxsize=256)
            return self._queues[chat_id]

    async def get_lock(self, chat_id: str) -> asyncio.Lock:
        async with self._lock:
            if chat_id not in self._locks:
                self._locks[chat_id] = asyncio.Lock()
            return self._locks[chat_id]

    async def try_start_consumer(self, chat_id: str) -> bool:
        async with self._lock:
            if chat_id in self._running:
                return False
            self._running.add(chat_id)
            return True

    async def mark_consumer_done(self, chat_id: str):
        async with self._lock:
            self._running.discard(chat_id)

    def get_queue_sizes(self) -> Dict[str, int]:
        return {cid: q.qsize() for cid, q in self._queues.items() if q.qsize() > 0}

    def has_active_consumer(self, chat_id: str) -> bool:
        """检查指定 session 当前是否有活跃的 consumer。"""
        return chat_id in self._running

    async def cleanup_session(self, chat_id: str):
        async with self._lock:
            self._queues.pop(chat_id, None)
            self._locks.pop(chat_id, None)
            self._running.discard(chat_id)

    async def cleanup_all(self):
        async with self._lock:
            self._queues.clear()
            self._locks.clear()
            self._running.clear()
