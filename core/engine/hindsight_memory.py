"""Hindsight 记忆系统适配层。

单 bank + tag 模式，每条消息 retain 到统一记忆库：
- 同 session（chat_id）的消息共享 document_id，通过 append 模式持续追加
- 用户隔离靠 tag user:{sender_id}，recall 时 tags_match=all_strict
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hindsight_client import Hindsight

_log = logging.getLogger(__name__)


class HindsightMemory:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8888",
        bank_id: str = "qq_bot",
    ):
        self._client = Hindsight(base_url=base_url, timeout=30.0)
        self._bank_id = bank_id
        self._health_cache: Optional[Dict[str, Any]] = None
        self._health_cache_time: float = 0.0
        self._health_cache_ttl: float = 10.0

        _log.info(f"HindsightMemory 已初始化 (bank={bank_id}, url={base_url})")

    def _cache_health(self, result: Dict[str, Any]):
        self._health_cache = result
        self._health_cache_time = time.monotonic()

    @property
    def last_health_status(self) -> Optional[Dict[str, Any]]:
        if self._health_cache is None:
            return None
        if time.monotonic() - self._health_cache_time > self._health_cache_ttl:
            return None
        return dict(self._health_cache)

    @staticmethod
    def _to_datetime(ts: Optional[float]) -> Optional[datetime]:
        if ts is None:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    async def add_message(
        self,
        session_id: str,
        sender_id: str,
        content: str,
        sender_name: Optional[str] = None,
        role: str = "user",
        timestamp: Optional[float] = None,
    ) -> None:
        """保留一条消息到记忆库。同一 session 共享 document_id 持续追加。"""
        try:
            prefix = f"[{sender_name}]: " if sender_name else ""
            await self._client.aretain(
                bank_id=self._bank_id,
                content=f"{prefix}{content}",
                document_id=f"session-{session_id}",
                update_mode="append",
                tags=[f"user:{sender_id}", f"chat:{session_id}"],
                timestamp=self._to_datetime(timestamp),
                retain_async=True,
            )
            self._cache_health({"status": "ok"})
        except Exception as e:
            self._cache_health({"status": "unreachable", "error": str(e)})
            _log.warning(f"Hindsight add_message 失败: {e!r}")

    async def flush(self, session_id: str) -> None:
        """无操作 — Hindsight 在 retain 时自动提取事实。"""

    async def search(
        self,
        user_id: str,
        query: str = "",
        top_k: int = 10,
        include_profile: bool = True,
        method: str = "hybrid",
    ) -> Dict[str, Any]:
        """搜索记忆。适配返回 {episodes, profiles} 格式。"""
        try:
            response = await self._client.arecall(
                bank_id=self._bank_id,
                query=query,
                tags=[f"user:{user_id}"],
                tags_match="all_strict",
                max_tokens=top_k * 500,
            )
            episodes: List[Dict] = []
            profiles: List[Dict] = []
            for r in response.results:
                text = r.text
                if r.type in ("experience", "observation"):
                    episodes.append({"summary": text, "memory_type": r.type})
                else:
                    profiles.append({"profile_data": {"info": text}})
            self._cache_health({"status": "ok"})
            return {"episodes": episodes, "profiles": profiles}
        except Exception as e:
            self._cache_health({"status": "unreachable", "error": str(e)})
            _log.warning(f"Hindsight search 失败 (user={user_id[:16]}..): {e!r}")
            return {"episodes": [], "profiles": []}

    async def health(self) -> Dict[str, Any]:
        cached = self.last_health_status
        if cached is not None:
            result = dict(cached)
            result["_from_cache"] = True
            return result

        start = time.monotonic()
        try:
            self._client.get_version()   # 同步 API 在独立调用中没问题（不在事件循环内）
            latency = (time.monotonic() - start) * 1000
            result = {"status": "ok", "latency_ms": round(latency, 1)}
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            result = {
                "status": "unreachable",
                "error": str(e),
                "latency_ms": round(latency, 1),
            }

        self._cache_health(result)
        return result

    async def close(self) -> None:
        self._client.close()
        _log.info("HindsightMemory: 客户端已关闭")
