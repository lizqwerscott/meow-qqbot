"""SubAgentManager — 子智能体生命周期管理

追踪子智能体运行状态，完成后通过 SystemEventQueue 通知父 session。
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_log = logging.getLogger(__name__)


@dataclass
class SubAgentRecord:
    id: str
    parent_chat_id: str
    task: str
    status: str  # running | completed | failed | timeout | cancelled
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    context: str = "isolated"


class SubAgentManager:
    def __init__(
        self,
        max_concurrent: int = 4,
        max_children: int = 5,
        run_timeout: int = 900,
        system_events: Any = None,
    ):
        self._max_concurrent = max_concurrent
        self._max_children = max_children
        self._run_timeout = run_timeout
        self._records: dict[str, SubAgentRecord] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._execute_callback: Optional[Callable] = None
        self._system_events = system_events

    def set_execute_callback(self, callback: Callable):
        self._execute_callback = callback

    async def spawn(
        self,
        parent_chat_id: str,
        task: str,
        context: str = "isolated",
    ) -> dict:
        async with self._lock:
            active = sum(
                1 for r in self._records.values()
                if r.parent_chat_id == parent_chat_id and r.status == "running"
            )
            if active >= self._max_children:
                return {
                    "error": f"该会话已有 {active} 个运行中的子智能体（上限 {self._max_children}）",
                }

            total_active = sum(
                1 for r in self._records.values() if r.status == "running"
            )
            if total_active >= self._max_concurrent:
                return {
                    "error": f"全局子智能体并发已满（{total_active}/{self._max_concurrent}）",
                }

            sub_id = uuid.uuid4().hex[:16]
            record = SubAgentRecord(
                id=sub_id,
                parent_chat_id=parent_chat_id,
                task=task,
                status="running",
                context=context,
            )
            self._records[sub_id] = record

        run_task = asyncio.create_task(self._run(sub_id))
        self._tasks[sub_id] = run_task
        run_task.add_done_callback(lambda t: self._tasks.pop(sub_id, None))

        return {"subagent_id": sub_id, "status": "accepted"}

    async def _run(self, sub_id: str):
        record = self._records.get(sub_id)
        if not record or not self._execute_callback:
            return
        if record.status != "running":
            return

        try:
            result, error = await asyncio.wait_for(
                self._execute_callback(
                    chat_id=f"subagent:{sub_id}",
                    prompt=record.task,
                    sender_id="system",
                    is_group=False,
                    delivery_channel="",
                    reply_to_message_id="",
                ),
                timeout=self._run_timeout,
            )
            async with self._lock:
                if record.status != "running":
                    return
                record.status = "completed" if not error else "failed"
                record.result = result
                record.error = error
                record.finished_at = time.time()
        except asyncio.TimeoutError:
            async with self._lock:
                if record.status != "running":
                    return
                record.status = "timeout"
                record.error = "子智能体执行超时"
                record.finished_at = time.time()
        except Exception as e:
            async with self._lock:
                if record.status != "running":
                    return
                record.status = "failed"
                record.error = str(e)
                record.finished_at = time.time()

        _log.info(
            "子智能体 [%s..] %s (%.1fs)",
            sub_id[:8],
            record.status,
            (record.finished_at or time.time()) - record.created_at,
        )

        self._notify_parent(record)

    async def cancel(self, subagent_id: str) -> dict:
        async with self._lock:
            record = self._records.get(subagent_id)
            if not record:
                return {"error": "未找到子智能体", "found": False}
            if record.status != "running":
                return {
                    "found": True,
                    "cancelled": False,
                    "error": f"子智能体状态为 {record.status}，无法取消",
                }
            record.status = "cancelled"
            record.error = "已取消"
            record.finished_at = time.time()

        run_task = self._tasks.get(subagent_id)
        if run_task and not run_task.done():
            run_task.cancel()

        _log.info("子智能体 [%s..] 已取消", subagent_id[:8])
        self._notify_parent(record)
        return {"found": True, "cancelled": True, "subagent_id": subagent_id}

    def _notify_parent(self, record: SubAgentRecord):
        if not self._system_events or not record.parent_chat_id:
            return
        status_zh = {"completed": "完成", "failed": "失败", "timeout": "超时", "cancelled": "已取消"}.get(
            record.status, record.status
        )
        preview = (record.result or record.error or "无输出")
        self._system_events.enqueue(
            session_key=record.parent_chat_id,
            text=f"子智能体 [{record.id[:8]}..] {status_zh}: {preview}",
            context_key=f"subagent:{record.id}",
            replace=True,
        )

    async def get_records(
        self,
        parent_chat_id: str,
        status: Optional[str] = None,
    ) -> list[dict]:
        async with self._lock:
            out = []
            for r in self._records.values():
                if r.parent_chat_id != parent_chat_id:
                    continue
                if status and r.status != status:
                    continue
                out.append(self._record_to_dict(r))
            return out

    async def get_record_by_id(self, sub_id: str) -> Optional[dict]:
        async with self._lock:
            r = self._records.get(sub_id)
            return self._record_to_dict(r) if r else None

    async def cancel_all(self):
        """取消所有运行中的子智能体，用于引擎关闭时清理。"""
        async with self._lock:
            for sub_id, r in self._records.items():
                if r.status == "running":
                    r.status = "cancelled"
                    r.error = "引擎关闭"
                    r.finished_at = time.time()
            tasks = list(self._tasks.values())
            self._tasks.clear()
        for t in tasks:
            if not t.done():
                t.cancel()

    async def cleanup_stale(self, max_age: float = 3600):
        async with self._lock:
            now = time.time()
            to_del = []
            for rid, r in self._records.items():
                if r.status in ("completed", "failed", "timeout", "cancelled"):
                    if r.finished_at and (now - r.finished_at) > max_age:
                        to_del.append(rid)
            for rid in to_del:
                del self._records[rid]

    @staticmethod
    def _record_to_dict(r: SubAgentRecord) -> dict:
        return {
            "id": r.id,
            "parent_chat_id": r.parent_chat_id,
            "task": r.task[:100],
            "status": r.status,
            "result": r.result[:500] if r.result else None,
            "error": r.error[:200] if r.error else None,
            "created_at": r.created_at,
            "finished_at": r.finished_at,
            "context": r.context,
        }
