"""Hindsight 记忆系统适配层。

单 bank + tag 模式，每条消息 retain 到统一记忆库：
- 同 session（chat_id）的消息共享 document_id，通过 append 模式持续追加
- 用户隔离靠 tag user:{sender_id}，recall 时 tags_match=all_strict
"""

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hindsight_client import Hindsight

from core.message import MessageType, ResourceMeta

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HindsightDocumentRef:
    """Stable remote append identity and tags for one conversation."""

    document_id: str
    canonical_chat_tag: str
    legacy_chat_tags: tuple[str, ...] = ()

    @property
    def chat_tags(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.canonical_chat_tag, *self.legacy_chat_tags)))


class HindsightMemory:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8888",
        bank_id: str = "qq_bot",
        session_identity_resolver: Any = None,
    ):
        self._client = Hindsight(base_url=base_url, timeout=30.0)
        self._bank_id = bank_id
        self._session_identity_resolver = session_identity_resolver
        self._health_cache: Optional[Dict[str, Any]] = None
        self._health_cache_time: float = 0.0
        self._health_cache_ttl: float = 10.0
        self._seen_idempotency_keys: OrderedDict[str, None] = OrderedDict()
        self._seen_idempotency_limit = 10_000
        self._idempotency_lock = asyncio.Lock()

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
        session_id: str = "",
        content: str = "",
        sender_id: str = "",
        timestamp: Optional[float] = None,
        context: Optional[str] = None,
        resources: Optional[List[ResourceMeta]] = None,
        idempotency_key: Optional[str] = None,
        document: Optional[HindsightDocumentRef] = None,
        document_id: Optional[str] = None,
        chat_tags: Optional[List[str]] = None,
    ) -> bool:
        """保留一条消息到记忆库，并返回是否成功提交。同一 session 共享 document_id 持续追加。
        content 已由调用方（agent_engine）预格式化，格式为 [ID(别名)]: 消息正文。
        """
        if idempotency_key:
            async with self._idempotency_lock:
                if idempotency_key in self._seen_idempotency_keys:
                    return True
                self._seen_idempotency_keys[idempotency_key] = None
                self._seen_idempotency_keys.move_to_end(idempotency_key)
                while len(self._seen_idempotency_keys) > self._seen_idempotency_limit:
                    self._seen_idempotency_keys.popitem(last=False)
        try:
            if document is not None and document_id is not None:
                raise ValueError("pass document or document_id, not both")
            if document is not None:
                resolved_document_id = document.document_id
                resolved_chat_tags = list(document.chat_tags)
            else:
                resolved_document_id = document_id or f"session-{session_id}"
                resolved_chat_tags = chat_tags or [f"chat:{session_id}"]
                resolver = self._session_identity_resolver
                if resolver is not None and session_id:
                    try:
                        ref = resolver.resolve_legacy(session_id)
                    except (TypeError, ValueError):
                        ref = None
                    if ref is not None:
                        resolved_document_id = (
                            ref.hindsight_document_id or resolved_document_id
                        )
                        resolved_chat_tags = list(resolver.recall_aliases(ref))
            if not resolved_document_id:
                raise ValueError("document_id or session_id is required")
            tags = list(dict.fromkeys([f"user:{sender_id}", *resolved_chat_tags]))
            kwargs: dict = dict(
                bank_id=self._bank_id,
                content=content,
                document_id=resolved_document_id,
                update_mode="append",
                tags=tags,
                timestamp=self._to_datetime(timestamp),
                retain_async=True,
            )
            metadata: Dict[str, str] = {}
            if idempotency_key:
                metadata["idempotency_key"] = idempotency_key
            if context:
                kwargs["context"] = context
            if resources:
                r = resources[0]
                if r.resource_type:
                    metadata["res_type"] = r.resource_type
                if r.hash:
                    metadata["res_hash"] = r.hash
                if r.resource_id:
                    metadata["res_id"] = r.resource_id
                if r.filename:
                    metadata["res_filename"] = r.filename
                if r.mime_type:
                    metadata["res_mime"] = r.mime_type
            if metadata:
                kwargs["metadata"] = metadata
            await self._client.aretain(**kwargs)
            self._cache_health({"status": "ok"})
            return True
        except asyncio.CancelledError:
            if idempotency_key:
                async with self._idempotency_lock:
                    self._seen_idempotency_keys.pop(idempotency_key, None)
            raise
        except Exception as e:
            if idempotency_key:
                async with self._idempotency_lock:
                    self._seen_idempotency_keys.pop(idempotency_key, None)
            self._cache_health({"status": "unreachable", "error": str(e)})
            _log.warning(f"Hindsight add_message 失败: {e!r}")
            return False

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

    async def search_shared(
        self,
        chat_id: str,
        query: str = "",
        top_k: int = 10,
        method: str = "hybrid",
        aliases: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Recall only shared events retained for one chat, never a user profile."""
        try:
            resolved_aliases = list(aliases or [])
            resolver = self._session_identity_resolver
            if resolver is not None and not resolved_aliases:
                try:
                    ref = resolver.resolve_legacy(chat_id)
                except (TypeError, ValueError):
                    ref = None
                if ref is not None:
                    resolved_aliases = list(resolver.recall_aliases(ref))
            tags = list(dict.fromkeys([f"chat:{chat_id}", *resolved_aliases]))
            responses = await asyncio.gather(
                *(
                    self._client.arecall(
                        bank_id=self._bank_id,
                        query=query,
                        tags=[tag if tag.startswith("chat:") else f"chat:{tag}"],
                        tags_match="all_strict",
                        max_tokens=top_k * 500,
                    )
                    for tag in tags
                )
            )
            episodes: List[Dict] = []
            seen: set[tuple[str, str]] = set()
            for response in responses:
                for result in response.results:
                    if result.type not in ("experience", "observation"):
                        continue
                    identity = (result.type, result.text)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    episodes.append(
                        {"summary": result.text, "memory_type": result.type}
                    )
                    if len(episodes) >= top_k:
                        break
                if len(episodes) >= top_k:
                    break
            self._cache_health({"status": "ok"})
            return {"episodes": episodes, "profiles": []}
        except Exception as exc:
            self._cache_health({"status": "unreachable", "error": str(exc)})
            _log.warning(
                "Hindsight shared search failed (chat=%s..): %r", chat_id[:16], exc
            )
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
            await self._client.aclose()
        except Exception as e:
            _log.warning("HindsightMemory 关闭失败: %s", e)
        _log.info("HindsightMemory: 客户端已关闭")
