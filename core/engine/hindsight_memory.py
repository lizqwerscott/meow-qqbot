"""Hindsight 记忆系统适配层。

单 bank + tag 模式，每条消息 retain 到统一记忆库：
- 同 session（chat_id）的消息共享 document_id，通过 append 模式持续追加
- 用户隔离靠 tag user:{sender_id}，recall 时 tags_match=all_strict
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hindsight_client import Hindsight

from core.message import MessageType, ResourceMeta

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
        content: str,
        sender_id: str = "",
        timestamp: Optional[float] = None,
        context: Optional[str] = None,
        resources: Optional[List[ResourceMeta]] = None,
    ) -> None:
        """保留一条消息到记忆库。同一 session 共享 document_id 持续追加。
        content 已由调用方（agent_engine）预格式化，格式为 [ID(别名)]: 消息正文。
        """
        try:
            kwargs: dict = dict(
                bank_id=self._bank_id,
                content=content,
                document_id=f"session-{session_id}",
                update_mode="append",
                tags=[f"user:{sender_id}", f"chat:{session_id}"],
                timestamp=self._to_datetime(timestamp),
                retain_async=True,
            )
            if context:
                kwargs["context"] = context
            if resources:
                r = resources[0]
                meta: Dict[str, str] = {}
                if r.resource_type:
                    meta["res_type"] = r.resource_type
                if r.hash:
                    meta["res_hash"] = r.hash
                if r.resource_id:
                    meta["res_id"] = r.resource_id
                if r.filename:
                    meta["res_filename"] = r.filename
                if r.mime_type:
                    meta["res_mime"] = r.mime_type
                kwargs["metadata"] = meta
            await self._client.aretain(**kwargs)
            self._cache_health({"status": "ok"})
        except Exception as e:
            self._cache_health({"status": "unreachable", "error": str(e)})
            _log.warning(f"Hindsight add_message 失败: {e!r}")

    @staticmethod
    def msg_type_to_context(msg_type: MessageType) -> Optional[str]:
        """根据消息类型返回 Hindsight context 标签。

        context 注入 LLM 提取提示词，让提取器知道内容来源场景。
        纯文本不需要额外 context（返回 None 即不传 context 参数）。
        """
        mapping = {
            MessageType.EMOJI: "用户发送了一张表情",
            MessageType.IMAGE: "用户发送了一张图片",
            MessageType.VOICE: "用户发送了一条语音消息",
            MessageType.VIDEO: "用户发送了一个视频",
            MessageType.FILE: "用户发送了一个文件",
        }
        return mapping.get(msg_type)

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
            await self._client.aget_version()
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
        try:
            await asyncio.to_thread(self._client.close)
        except Exception as e:
            _log.warning("HindsightMemory 关闭失败: %s", e)
        _log.info("HindsightMemory: 客户端已关闭")
