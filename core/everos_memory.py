"""EverOS 记忆系统异步 HTTP 客户端封装。"""

import time
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

_log = logging.getLogger(__name__)


class EverOSMemory:
    """EverOS 记忆系统异步 HTTP 客户端。

    提供 add_message / flush / search / health / close 五个核心方法。
    所有异常仅打日志，绝不阻断主流程。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        app_id: str = "qq_bot",
        project_id: str = "production",
        timeout: float = 15.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._app_id = app_id
        self._project_id = project_id
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout

        # 健康缓存：避免频繁请求 /health
        self._health_cache: Optional[Dict[str, Any]] = None
        self._health_cache_time: float = 0.0
        self._health_cache_ttl: float = 10.0  # 10 秒 TTL

    # ── 内部 ──

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    # ── 核心 API ──

    async def add_message(
        self,
        session_id: str,
        sender_id: str,
        content: str,
        sender_name: Optional[str] = None,
        role: str = "user",
    ) -> None:
        """写入一条消息到会话缓冲区。异常仅打日志，不阻断。"""
        try:
            client = await self._get_client()
            ts = int(time.time() * 1000)  # 毫秒
            msg: Dict[str, Any] = {
                "sender_id": sender_id,
                "role": role,
                "timestamp": ts,
                "content": content,
            }
            if sender_name:
                msg["sender_name"] = sender_name
            payload = {
                "app_id": self._app_id,
                "project_id": self._project_id,
                "session_id": session_id,
                "messages": [msg],
            }
            await client.post(f"{self._base_url}/api/v1/memory/add", json=payload)
            self._infer_health_from_result(error=None)
        except Exception as e:
            self._infer_health_from_result(error=e)
            _log.warning(f"EverOS add_message 失败 (session={session_id[:16]}..): {e}")

    async def flush(self, session_id: str) -> None:
        """强制提炼会话缓冲区，生成画像和经历。异常仅打日志。"""
        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self._base_url}/api/v1/memory/flush",
                json={
                    "app_id": self._app_id,
                    "project_id": self._project_id,
                    "session_id": session_id,
                },
            )
            data = resp.json()
            status = data.get("data", {}).get("status", "unknown")
            self._infer_health_from_result(error=None)
            _log.info(f"EverOS flush: session={session_id[:16]}.. status={status}")
        except Exception as e:
            self._infer_health_from_result(error=e)
            _log.warning(f"EverOS flush 失败: {e}")

    async def search(
        self,
        user_id: str,
        query: str = "",
        top_k: int = 10,
        include_profile: bool = True,
        method: str = "hybrid",
    ) -> Dict[str, Any]:
        """混合检索用户记忆。

        返回格式: {episodes: [...], profiles: [...]}
        method 可选: hybrid（默认）, keyword, vector, agentic
        失败时返回空字典。
        """
        try:
            client = await self._get_client()
            payload: Dict[str, Any] = {
                "app_id": self._app_id,
                "project_id": self._project_id,
                "user_id": user_id,
                "query": query,
                "top_k": top_k,
                "include_profile": include_profile,
                "method": method,
            }
            resp = await client.post(
                f"{self._base_url}/api/v1/memory/search", json=payload
            )
            resp.raise_for_status()
            body = resp.json()
            data = body.get("data", {})
            self._infer_health_from_result(error=None)
            return {
                "episodes": data.get("episodes", []),
                "profiles": data.get("profiles", []),
            }
        except Exception as e:
            self._infer_health_from_result(error=e)
            _log.warning(f"EverOS search 失败 (user={user_id[:16]}..): {e}")
            return {"episodes": [], "profiles": []}

    # ── 内部辅助：健康缓存 ──

    def _cache_health(self, result: Dict[str, Any]):
        """写入健康缓存。"""
        self._health_cache = result
        self._health_cache_time = time.monotonic()

    @property
    def last_health_status(self) -> Optional[Dict[str, Any]]:
        """
        返回缓存的健康状态（同步读取，无需 await）。

        如果缓存未过期，返回缓存值；否则返回 None
        （调用方应调 health() 主动刷新）。
        从不抛出异常。
        """
        if self._health_cache is None:
            return None
        if time.monotonic() - self._health_cache_time > self._health_cache_ttl:
            return None
        return dict(self._health_cache)

    def _infer_health_from_result(self, error: Optional[Exception] = None):
        """根据其他 API 调用的结果推断健康状态。"""
        if error is None:
            # 成功调用说明服务可达
            self._cache_health({
                "status": "ok",
                "latency_ms": None,
            })
        else:
            self._cache_health({
                "status": "unreachable",
                "error": str(error),
                "latency_ms": None,
            })

    # ── 健康检查 ──

    async def health(self) -> Dict[str, Any]:
        """
        对 EverOS 服务发起主动健康检查。

        结果缓存 10 秒，避免高频请求。读取缓存请用 last_health_status 属性。

        返回: {"status": "ok", "latency_ms": float} 或
             {"status": "unreachable", "error": str, "latency_ms": float}
        从不抛出异常。
        """
        # 缓存有效？直接返回
        cached = self.last_health_status
        if cached is not None:
            # 追加标记表示是缓存
            result = dict(cached)
            result["_from_cache"] = True
            return result

        start = time.monotonic()
        try:
            client = await self._get_client()
            resp = await client.get(f"{self._base_url}/health", timeout=5.0)
            latency = (time.monotonic() - start) * 1000
            if resp.status_code == 200:
                result = {
                    "status": "ok",
                    "latency_ms": round(latency, 1),
                }
            else:
                body = resp.text[:200]
                result = {
                    "status": "unreachable",
                    "error": f"HTTP {resp.status_code}: {body}",
                    "latency_ms": round(latency, 1),
                }
        except httpx.TimeoutException:
            elapsed = (time.monotonic() - start) * 1000
            result = {
                "status": "unreachable",
                "error": "连接超时 (5s)",
                "latency_ms": round(elapsed, 1),
            }
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            result = {
                "status": "unreachable",
                "error": str(e),
                "latency_ms": round(elapsed, 1),
            }

        self._cache_health(result)
        return result

    async def close(self) -> None:
        """关闭底层 HTTP 客户端。"""
        if self._client:
            await self._client.aclose()
            self._client = None
