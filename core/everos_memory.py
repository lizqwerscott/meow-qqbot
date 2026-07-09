"""EverOS 记忆系统异步 HTTP 客户端封装。

每个 session_id 拥有独立的任务队列和工作协程，确保同一个会话内的
add_message 和 flush 按提交顺序串行执行，消除竞态。
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional

import httpx

_log = logging.getLogger(__name__)


class EverOSMemory:
    """EverOS 记忆系统异步 HTTP 客户端。

    提供 add_message / flush / search / health / close 五个核心方法。
    所有异常仅打日志，绝不阻断主流程。

    add_message 和 flush 内部使用 per-session 任务队列，
    确保同一个 session 内的操作严格按序执行。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        app_id: str = "qq_bot",
        project_id: str = "production",
        timeout: float = 15.0,
        flush_threshold: int = 20,
    ):
        self._base_url = base_url.rstrip("/")
        self._app_id = app_id
        self._project_id = project_id
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout
        self._flush_threshold = flush_threshold

        # 健康缓存
        self._health_cache: Optional[Dict[str, Any]] = None
        self._health_cache_time: float = 0.0
        self._health_cache_ttl: float = 10.0

        # per-session 任务队列系统
        self._queues: Dict[str, "asyncio.Queue"] = {}
        self._workers: Dict[str, asyncio.Task] = {}
        self._pending_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._running = True

        _log.info(f"EverOSMemory 已初始化 (threshold={flush_threshold})")

    # ── 内部 ──

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _get_or_create_queue(self, session_id: str) -> "asyncio.Queue":
        """获取或创建 session 的任务队列及 worker。"""
        async with self._lock:
            queue = self._queues.get(session_id)
            if queue is None:
                queue = asyncio.Queue()
                self._queues[session_id] = queue
                self._pending_counts[session_id] = 0
                worker = asyncio.create_task(self._worker_loop(session_id))
                self._workers[session_id] = worker
                _log.debug(f"EverOS 创建队列 & worker: session={session_id[:16]}..")
            return queue

    # ── 外部方法（入队列，立即返回） ──

    async def add_message(
        self,
        session_id: str,
        sender_id: str,
        content: str,
        sender_name: Optional[str] = None,
        role: str = "user",
    ) -> None:
        """写入一条消息到缓冲区（入队后立即返回，不等待 HTTP）。"""
        if not self._running:
            _log.warning(
                f"EverOSMemory 已关闭，忽略 add_message (session={session_id[:16]}..)"
            )
            return
        queue = await self._get_or_create_queue(session_id)
        await queue.put({
            "type": "add",
            "sender_id": sender_id,
            "sender_name": sender_name,
            "content": content,
            "role": role,
        })
        _log.debug(f"EverOS 入队 add_message: session={session_id[:16]}..")

    async def flush(self, session_id: str) -> None:
        """强制提炼缓冲区（入队后立即返回，不等待 HTTP）。"""
        if not self._running:
            return
        queue = await self._get_or_create_queue(session_id)
        await queue.put({"type": "flush"})
        _log.debug(f"EverOS 入队 flush: session={session_id[:16]}..")

    # ── Worker 循环 ──

    async def _worker_loop(self, session_id: str):
        """单 session 任务循环：FIFO 顺序消费，确保有序性。"""
        queue = self._queues[session_id]
        try:
            while True:
                task = await queue.get()
                if task["type"] == "shutdown":
                    queue.task_done()
                    break

                try:
                    await self._process_task(session_id, task)
                except asyncio.CancelledError:
                    queue.task_done()
                    raise
                except Exception as e:
                    _log.error(
                        f"EverOS worker 异常 (session={session_id[:16]}..): {e!r}"
                    )
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            pass
        finally:
            # worker 退出时清理资源
            async with self._lock:
                self._queues.pop(session_id, None)
                self._workers.pop(session_id, None)
                self._pending_counts.pop(session_id, None)
            _log.debug(f"EverOS worker 已退出: session={session_id[:16]}..")

    async def _process_task(self, session_id: str, task: dict):
        """处理单个任务（在 worker 协程内执行）。"""
        if task["type"] == "add":
            status = await self._http_add_message(
                session_id=session_id,
                sender_id=task["sender_id"],
                sender_name=task.get("sender_name"),
                content=task["content"],
                role=task.get("role", "user"),
            )
            if status is None:
                return  # HTTP 失败，计数不变

            if status == "extracted":
                # 服务端自动提取（缓冲区已满），计数器归零
                self._pending_counts[session_id] = 0
                _log.info(
                    f"EverOS 自动提取: session={session_id[:16]}.. pending=0"
                )
            else:  # "accumulated"
                count = self._pending_counts.get(session_id, 0) + 1
                self._pending_counts[session_id] = count
                _log.debug(
                    f"EverOS 累积: session={session_id[:16]}.. pending={count}/{self._flush_threshold}"
                )
                # 达到阈值 → 自动 flush
                if count >= self._flush_threshold:
                    _log.info(
                        f"EverOS 计数触发 flush: session={session_id[:16]}.. "
                        f"count={count}"
                    )
                    await self._http_flush(session_id)
                    self._pending_counts[session_id] = 0

        elif task["type"] == "flush":
            status = await self._http_flush(session_id)
            if status is not None:
                # flush 成功，缓冲区清空
                self._pending_counts[session_id] = 0
                _log.info(
                    f"EverOS flush 完成: session={session_id[:16]}.. status={status}"
                )

    # ── HTTP 核心方法 ──

    async def _http_add_message(
        self,
        session_id: str,
        sender_id: str,
        content: str,
        sender_name: Optional[str] = None,
        role: str = "user",
    ) -> Optional[str]:
        """真实的 HTTP POST /add。返回 status 或 None（失败）。"""
        try:
            client = await self._get_client()
            ts = int(time.time() * 1000)
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
            resp = await client.post(f"{self._base_url}/api/v1/memory/add", json=payload)
            resp.raise_for_status()
            body = resp.json()
            status = body.get("data", {}).get("status", "unknown")
            self._infer_health_from_result(error=None)
            _log.info(f"EverOS add_message: session={session_id[:16]}.. status={status}")
            return status
        except Exception as e:
            self._infer_health_from_result(error=e)
            _log.warning(f"EverOS add_message 失败 (session={session_id[:16]}..): {e!r}")
            return None

    async def _http_flush(self, session_id: str) -> Optional[str]:
        """真实的 HTTP POST /flush。返回 status 或 None（失败）。"""
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
            resp.raise_for_status()
            body = resp.json()
            status = body.get("data", {}).get("status", "unknown")
            self._infer_health_from_result(error=None)
            _log.info(f"EverOS flush: session={session_id[:16]}.. status={status}")
            return status
        except Exception as e:
            self._infer_health_from_result(error=e)
            _log.warning(f"EverOS flush 失败: {e!r}")
            return None

    # ── 内部辅助：健康缓存 ──

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

    def _infer_health_from_result(self, error: Optional[Exception] = None):
        if error is None:
            self._cache_health({"status": "ok", "latency_ms": None})
        else:
            self._cache_health({
                "status": "unreachable",
                "error": str(error),
                "latency_ms": None,
            })

    # ── 健康检查 ──

    async def health(self) -> Dict[str, Any]:
        cached = self.last_health_status
        if cached is not None:
            result = dict(cached)
            result["_from_cache"] = True
            return result

        start = time.monotonic()
        try:
            client = await self._get_client()
            resp = await client.get(f"{self._base_url}/health", timeout=5.0)
            latency = (time.monotonic() - start) * 1000
            if resp.status_code == 200:
                result = {"status": "ok", "latency_ms": round(latency, 1)}
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

    # ── 检索 ──

    async def search(
        self,
        user_id: str,
        query: str = "",
        top_k: int = 10,
        include_profile: bool = True,
        method: str = "hybrid",
    ) -> Dict[str, Any]:
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
            _log.warning(f"EverOS search 失败 (user={user_id[:16]}..): {e!r}")
            return {"episodes": [], "profiles": []}

    # ── 关闭 ──

    async def close(self) -> None:
        """关闭所有 worker 和底层 HTTP 客户端。"""
        self._running = False

        # 向所有队列发送 shutdown 信号
        async with self._lock:
            session_ids = list(self._queues.keys())
        for sid in session_ids:
            queue = self._queues.get(sid)
            if queue is not None:
                await queue.put({"type": "shutdown"})

        # 等待所有 worker 结束
        async with self._lock:
            workers = list(self._workers.values())
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)

        # 关闭 HTTP 客户端
        if self._client:
            await self._client.aclose()
            self._client = None
            _log.info("EverOSMemory: HTTP 客户端已关闭")
