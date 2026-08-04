"""SessionBindingManager — session 模型绑定管理器。

每个 (chat_id, tier) 绑定一个模型，支持请求数预算和 TTL 释放。
键格式: "{chat_id}:{tier}"
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionBinding:
    """单次 session 绑定状态。"""

    model_name: str
    tier: str
    bound_at: float
    request_count: int = 0

    def is_expired(self, budget: int, ttl: float) -> bool:
        if self.request_count >= budget:
            return True
        if time.monotonic() - self.bound_at >= ttl:
            return True
        return False


class SessionBindingManager:
    """(chat_id, tier) → SessionBinding 管理。

    线程安全（asyncio.Lock），纯内存存储。
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._bindings: dict[str, SessionBinding] = {}

    @staticmethod
    def _key(chat_id: str, tier: str) -> str:
        return f"{chat_id}:{tier}"

    async def get_if_valid(
        self, chat_id: str, tier: str, budget: int, ttl: float
    ) -> Optional[SessionBinding]:
        """原子获取并验证绑定。若过期则自动解绑。

        消除外部 get() + is_expired() + unbind() 之间的竞态窗口。
        """
        key = self._key(chat_id, tier)
        async with self._lock:
            binding = self._bindings.get(key)
            if binding is None:
                return None
            if binding.is_expired(budget, ttl):
                self._bindings.pop(key, None)
                return None
            return binding

    async def get(self, chat_id: str, tier: str) -> Optional[SessionBinding]:
        key = self._key(chat_id, tier)
        async with self._lock:
            return self._bindings.get(key)

    async def bind(self, chat_id: str, tier: str, model_name: str):
        key = self._key(chat_id, tier)
        binding = SessionBinding(
            model_name=model_name,
            tier=tier,
            bound_at=time.monotonic(),
            request_count=0,
        )
        async with self._lock:
            self._bindings[key] = binding

    async def unbind(self, chat_id: str, tier: str):
        key = self._key(chat_id, tier)
        async with self._lock:
            self._bindings.pop(key, None)

    async def tick(self, chat_id: str, tier: str):
        """请求计数 +1。"""
        key = self._key(chat_id, tier)
        async with self._lock:
            binding = self._bindings.get(key)
            if binding:
                binding.request_count += 1

    async def get_all(self) -> dict:
        async with self._lock:
            return dict(self._bindings)
