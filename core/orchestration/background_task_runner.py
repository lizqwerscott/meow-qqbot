"""Durable delegation runner using the existing isolated subagent callback."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from core.orchestration.work_plan_store import (
    BackgroundTask,
    BackgroundTaskStatus,
    WorkPlanStatus,
    WorkPlanStore,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StructuredTaskResult:
    status: str
    result: Any = None
    error: str = ""


class BackgroundTaskRunner:
    """Executes only durable task records and writes structured terminal results."""

    VALID_RESULTS = frozenset({"completed", "failed", "needs_input", "cancelled"})

    def __init__(
        self,
        store: WorkPlanStore,
        executor: Callable[[BackgroundTask], Awaitable[Any]],
        *,
        max_retries: int = 2,
        on_result: (
            Callable[[BackgroundTask, StructuredTaskResult], Awaitable[None]] | None
        ) = None,
        routing_metrics: Any = None,
        max_running_per_plan: int = 2,
        max_running_per_chat: int = 4,
    ):
        self.store = store
        self.executor = executor
        self.on_result = on_result
        self.routing_metrics = routing_metrics
        self.max_retries = max(0, max_retries)
        self.max_running_per_plan = max(1, max_running_per_plan)
        self.max_running_per_chat = max(1, max_running_per_chat)
        self._active: set[asyncio.Task[StructuredTaskResult]] = set()
        self._active_by_task_id: dict[str, asyncio.Task[StructuredTaskResult]] = {}

    def start(self, task: BackgroundTask) -> asyncio.Task[StructuredTaskResult]:
        """Own an execution task so shutdown can settle it before closing storage."""
        existing = self._active_by_task_id.get(task.id)
        if existing is not None and not existing.done():
            return existing
        running = asyncio.create_task(self.run(task))
        self._active.add(running)
        self._active_by_task_id[task.id] = running
        running.add_done_callback(self._active.discard)

        def _forget(done: asyncio.Task[StructuredTaskResult]) -> None:
            if self._active_by_task_id.get(task.id) is done:
                self._active_by_task_id.pop(task.id, None)

        running.add_done_callback(_forget)
        return running

    async def resume(self) -> int:
        tasks = await self.store.list_runnable_background_tasks()
        for task in tasks:
            self.start(task)
        notified = await self._dispatch_pending_notifications()
        await self._dispatch_pending_inbox(skip_task_ids=notified)
        return len(tasks)

    async def stop(self) -> None:
        tasks = tuple(self._active)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def cancel_plan(self, work_plan_id: str) -> int:
        """Interrupt locally owned runners after their durable cancellation commits."""
        cancelled = 0
        for task_id, running in tuple(self._active_by_task_id.items()):
            if running.done():
                continue
            task = await self.store.get_background_task(task_id)
            if task is not None and task.work_plan_id == work_plan_id:
                running.cancel()
                cancelled += 1
        return cancelled

    async def run(self, task: BackgroundTask) -> StructuredTaskResult:
        started = asyncio.get_running_loop().time()
        claimed = await self.store.claim_background_task(
            task.id,
            max_retries=self.max_retries,
            max_running_per_plan=self.max_running_per_plan,
            max_running_per_chat=self.max_running_per_chat,
        )
        if claimed is None:
            return StructuredTaskResult(
                "deferred",
                error="background task was already claimed or is not runnable",
            )
        task = claimed
        cancelled_by_runner = False
        try:
            raw = await self.executor(task)
            if isinstance(raw, StructuredTaskResult):
                result = raw
            elif isinstance(raw, dict):
                result = StructuredTaskResult(
                    str(raw.get("status", "completed")),
                    raw.get("result"),
                    str(raw.get("error", "")),
                )
            else:
                result = StructuredTaskResult("completed", raw)
            if result.status not in self.VALID_RESULTS:
                result = StructuredTaskResult(
                    "failed", error="invalid subagent result status"
                )
        except asyncio.CancelledError:
            cancelled_by_runner = True
            result = StructuredTaskResult("cancelled", error="cancelled")
        except Exception as exc:
            result = StructuredTaskResult("failed", error=str(exc))
        if cancelled_by_runner:
            # Cancellation of the local runner is shutdown/interruption, not a
            # durable task decision. Leave RUNNING for startup reconcile so no
            # result event, notification, or wake can be emitted here.
            _log.info("Background task runner cancelled: task=%s", task.id)
            return result
        settled = await self.store.settle_background_task(
            task.id,
            result.status.upper(),
            {"status": result.status, "result": result.result, "error": result.error},
            expected_status=BackgroundTaskStatus.RUNNING,
        )
        if settled.status != result.status.upper():
            _log.info(
                "Background task settlement lost a race: task=%s stored=%s attempted=%s",
                task.id,
                settled.status,
                result.status.upper(),
            )
            return StructuredTaskResult(
                settled.status.lower(), error="settlement superseded by another state"
            )
        if self.routing_metrics is not None:
            self.routing_metrics.record_background_terminal(
                status=result.status,
                latency_ms=(asyncio.get_running_loop().time() - started) * 1000,
            )
            plan = await self.store.get_plan(task.work_plan_id)
            if plan is not None and plan.status in {
                WorkPlanStatus.COMPLETED,
                WorkPlanStatus.FAILED,
                WorkPlanStatus.CANCELLED,
            }:
                self.routing_metrics.record_work_plan_terminal(
                    status=plan.status,
                    latency_ms=(time.time() - plan.created_at) * 1000,
                )
        await self._notify_result(task, result)
        await self._start_newly_runnable(task.id)
        return result

    async def _start_newly_runnable(self, settled_task_id: str) -> None:
        for candidate in await self.store.list_runnable_background_tasks():
            if (
                candidate.id != settled_task_id
                and candidate.id not in self._active_by_task_id
            ):
                self.start(candidate)

    async def _dispatch_pending_notifications(self) -> set[str]:
        notified: set[str] = set()
        if self.on_result is None:
            return notified
        for task in await self.store.list_pending_background_notifications():
            try:
                payload = json.loads(task.result_json)
                result = StructuredTaskResult(
                    str(payload.get("status", task.status.lower())),
                    payload.get("result"),
                    str(payload.get("error", "")),
                )
                await self._notify_result(task, result)
                notified.add(task.id)
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.exception("WorkPlan notification recovery failed: %s", task.id)
        return notified

    async def _dispatch_pending_inbox(self, *, skip_task_ids: set[str]) -> None:
        if self.on_result is None:
            return
        for task in await self.store.list_background_tasks_with_pending_inbox():
            if task.id in skip_task_ids:
                continue
            try:
                payload = json.loads(task.result_json)
                result = StructuredTaskResult(
                    str(payload.get("status", task.status.lower())),
                    payload.get("result"),
                    str(payload.get("error", "")),
                )
                await self.on_result(task, result)
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.exception("WorkPlan inbox recovery failed: %s", task.id)

    async def _notify_result(
        self, task: BackgroundTask, result: StructuredTaskResult
    ) -> None:
        if self.on_result is None:
            return
        try:
            await self.on_result(task, result)
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.exception("WorkPlan background notification failed: %s", task.id)
            return
        await self.store.mark_background_notification_delivered(task.id)

    async def reconcile(self) -> dict[str, int]:
        result = await self.store.reconcile()
        if self.routing_metrics is not None:
            self.routing_metrics.record_reconcile(result)
        return result
