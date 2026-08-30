"""Durable orchestration state for foreground WorkPlans and background tasks.

The store owns transactional persistence and invariant projections. User-facing
policy, ACL and planner authorization belong to WorkPlanService.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


class WorkPlanStatus(StrEnum):
    QUEUED = "QUEUED"
    ACTIVE = "ACTIVE"
    WAITING_USER = "WAITING_USER"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    WAITING_BACKGROUND = "WAITING_BACKGROUND"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"


class BackgroundTaskStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NEEDS_INPUT = "NEEDS_INPUT"
    CANCELLED = "CANCELLED"


class PlanStepStatus(StrEnum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    BLOCKED = "BLOCKED"
    DONE = "DONE"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class WorkPlan:
    id: str
    chat_id: str
    owner_id: str
    title: str
    status: str
    revision: int
    created_at: float
    updated_at: float
    short_handle: str


@dataclass(frozen=True)
class BackgroundTask:
    id: str
    work_plan_id: str
    status: str
    brief_json: str
    required: bool
    attempts: int
    allow_parallel: bool = False
    result_json: str = ""


@dataclass(frozen=True)
class PlanStep:
    id: str
    work_plan_id: str
    title: str
    description: str
    status: str
    depends_json: str
    execution_mode: str
    background_task_id: str | None
    result_summary: str

    @property
    def depends_on(self) -> list[str]:
        return list(json.loads(self.depends_json))


@dataclass(frozen=True)
class WorkPlanInboxItem:
    id: str
    work_plan_id: str
    event_id: str
    background_task_id: str | None
    coalesce_key: str
    payload_json: str
    created_at: float
    lease_id: str | None = None
    lease_expires_at: float | None = None


class WorkPlanConflict(RuntimeError):
    pass


class WorkPlanStore:
    """SQLite-backed repository with transactional CAS updates."""

    def __init__(
        self,
        path: str = "data/orchestration.sqlite",
        *,
        max_pending_events_per_plan: int = 100,
        terminal_retention_seconds: float = 30 * 24 * 60 * 60,
    ) -> None:
        self.path = str(path)
        self.max_pending_events_per_plan = max(1, max_pending_events_per_plan)
        self.terminal_retention_seconds = max(0.0, terminal_retention_seconds)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def _open(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS work_plans (
                    id TEXT PRIMARY KEY, chat_id TEXT NOT NULL, owner_id TEXT NOT NULL,
                    title TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL, short_handle TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_work_plans_chat ON work_plans(chat_id, status);
                CREATE TABLE IF NOT EXISTS plan_steps (
                    id TEXT PRIMARY KEY, work_plan_id TEXT NOT NULL, title TEXT NOT NULL,
                    description TEXT NOT NULL, status TEXT NOT NULL, depends_json TEXT NOT NULL,
                    execution_mode TEXT NOT NULL, background_task_id TEXT, result_summary TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS work_plan_events (
                    id TEXT PRIMARY KEY, work_plan_id TEXT NOT NULL, event_type TEXT NOT NULL,
                    event_key TEXT NOT NULL UNIQUE, payload_json TEXT NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS work_plan_inbox (
                    id TEXT PRIMARY KEY, work_plan_id TEXT NOT NULL, event_id TEXT NOT NULL UNIQUE,
                    background_task_id TEXT, coalesce_key TEXT NOT NULL, payload_json TEXT NOT NULL, created_at REAL NOT NULL,
                    lease_id TEXT, lease_expires_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_work_plan_inbox_claim
                    ON work_plan_inbox(work_plan_id, lease_expires_at, created_at);
                CREATE TABLE IF NOT EXISTS work_plan_event_summaries (
                    work_plan_id TEXT PRIMARY KEY, summary_json TEXT NOT NULL,
                    compacted_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS planner_leases (
                    work_plan_id TEXT PRIMARY KEY, lease_id TEXT NOT NULL, expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS background_tasks (
                    id TEXT PRIMARY KEY, work_plan_id TEXT NOT NULL, status TEXT NOT NULL,
                    brief_json TEXT NOT NULL, required INTEGER NOT NULL, attempts INTEGER NOT NULL,
                    allow_parallel INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS background_task_notifications (
                    background_task_id TEXT PRIMARY KEY, work_plan_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL, created_at REAL NOT NULL,
                    delivered_at REAL
                );
                CREATE TABLE IF NOT EXISTS plan_task_links (
                    work_plan_id TEXT NOT NULL, background_task_id TEXT NOT NULL,
                    PRIMARY KEY(work_plan_id, background_task_id)
                );
                CREATE TABLE IF NOT EXISTS background_task_dependencies (
                    background_task_id TEXT NOT NULL, depends_on_task_id TEXT NOT NULL,
                    PRIMARY KEY(background_task_id, depends_on_task_id)
                );
                CREATE TABLE IF NOT EXISTS plan_acl (
                    work_plan_id TEXT NOT NULL, principal_id TEXT NOT NULL,
                    acl_role TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(work_plan_id, principal_id)
                );
                CREATE TABLE IF NOT EXISTS delegation_intents (
                    id TEXT PRIMARY KEY, work_plan_id TEXT NOT NULL, idempotency_key TEXT NOT NULL UNIQUE,
                    background_task_id TEXT NOT NULL, payload_json TEXT NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_agent_handoffs (
                    handoff_key TEXT PRIMARY KEY, source_message_id TEXT NOT NULL,
                    chat_turn_id TEXT NOT NULL, chat_id TEXT NOT NULL, sender_id TEXT NOT NULL,
                    task_summary TEXT NOT NULL, reason TEXT NOT NULL, created_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'COMPLETED'
                );
                """)
            try:
                self._conn.execute(
                    "ALTER TABLE work_plan_inbox ADD COLUMN background_task_id TEXT"
                )
            except sqlite3.OperationalError:
                pass
            try:
                self._conn.execute(
                    "ALTER TABLE background_tasks ADD COLUMN allow_parallel INTEGER NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass
            self._conn.commit()
        return self._conn

    @staticmethod
    def _plan(row: sqlite3.Row) -> WorkPlan:
        return WorkPlan(**dict(row))

    @staticmethod
    def _task(row: sqlite3.Row) -> BackgroundTask:
        values = dict(row)
        values.pop("created_at", None)
        values.pop("updated_at", None)
        values["required"] = bool(values["required"])
        values["allow_parallel"] = bool(values.get("allow_parallel", 0))
        return BackgroundTask(**values)

    @staticmethod
    def _step(row: sqlite3.Row) -> PlanStep:
        return PlanStep(**dict(row))

    async def update_step(
        self,
        plan_id: str,
        expected_revision: int,
        step_id: str,
        *,
        status: str,
        result_summary: str = "",
    ) -> PlanStep:
        try:
            target = PlanStepStatus(status)
        except ValueError as exc:
            raise ValueError("invalid step status") from exc
        async with self._lock:
            conn = await self._open()
            plan = conn.execute(
                "SELECT revision FROM work_plans WHERE id=?", (plan_id,)
            ).fetchone()
            if plan is None:
                raise KeyError(plan_id)
            if plan["revision"] != expected_revision:
                raise WorkPlanConflict(f"stale work plan revision: {plan_id}")
            row = conn.execute(
                "SELECT * FROM plan_steps WHERE id=? AND work_plan_id=?",
                (step_id, plan_id),
            ).fetchone()
            if row is None:
                raise KeyError(step_id)
            step = self._step(row)
            allowed = {
                PlanStepStatus.PENDING: {
                    PlanStepStatus.ACTIVE,
                    PlanStepStatus.BLOCKED,
                    PlanStepStatus.DONE,
                    PlanStepStatus.SKIPPED,
                },
                PlanStepStatus.ACTIVE: {
                    PlanStepStatus.BLOCKED,
                    PlanStepStatus.DONE,
                    PlanStepStatus.SKIPPED,
                },
                PlanStepStatus.BLOCKED: {PlanStepStatus.ACTIVE, PlanStepStatus.SKIPPED},
            }
            if target not in allowed.get(step.status, set()):
                raise ValueError("invalid step state transition")
            if target in {PlanStepStatus.ACTIVE, PlanStepStatus.DONE}:
                dependencies = self._step_dependencies_ready(
                    conn, plan_id, step.depends_on
                )
                if not dependencies:
                    raise WorkPlanConflict("step dependencies are not complete")
            if target is PlanStepStatus.DONE and step.background_task_id:
                task = conn.execute(
                    "SELECT status FROM background_tasks WHERE id=?",
                    (step.background_task_id,),
                ).fetchone()
                if task is None or task["status"] != BackgroundTaskStatus.COMPLETED:
                    raise WorkPlanConflict("background step task is not complete")
            conn.execute(
                "UPDATE plan_steps SET status=?, result_summary=? WHERE id=?",
                (target, result_summary[:2000], step_id),
            )
            conn.execute(
                "UPDATE work_plans SET revision=revision+1, updated_at=? WHERE id=?",
                (time.time(), plan_id),
            )
            conn.commit()
            return self._step(
                conn.execute(
                    "SELECT * FROM plan_steps WHERE id=?", (step_id,)
                ).fetchone()
            )

    @staticmethod
    def _step_dependencies_ready(
        conn: sqlite3.Connection, plan_id: str, dependencies: Iterable[str]
    ) -> bool:
        dependency_ids = tuple(dependencies)
        if not dependency_ids:
            return True
        marks = ",".join("?" for _ in dependency_ids)
        rows = conn.execute(
            f"SELECT id, status FROM plan_steps WHERE work_plan_id=? AND id IN ({marks})",
            (plan_id, *dependency_ids),
        ).fetchall()
        return len(rows) == len(dependency_ids) and all(
            row["status"] in {PlanStepStatus.DONE, PlanStepStatus.SKIPPED}
            for row in rows
        )

    async def activate_plan(self, plan_id: str, expected_revision: int) -> WorkPlan:
        """Acquire the one foreground WorkPlan slot for a chat atomically."""
        async with self._lock:
            conn = await self._open()
            plan = conn.execute(
                "SELECT * FROM work_plans WHERE id=?", (plan_id,)
            ).fetchone()
            if plan is None:
                raise KeyError(plan_id)
            if plan["revision"] != expected_revision:
                raise WorkPlanConflict(f"stale work plan revision: {plan_id}")
            active = conn.execute(
                "SELECT id FROM work_plans WHERE chat_id=? AND status=? AND id<>?",
                (plan["chat_id"], WorkPlanStatus.ACTIVE, plan_id),
            ).fetchone()
            if active is not None:
                raise WorkPlanConflict("foreground WorkPlan slot is occupied")
            updated = conn.execute(
                "UPDATE work_plans SET status=?, revision=revision+1, updated_at=? "
                "WHERE id=? AND revision=?",
                (WorkPlanStatus.ACTIVE, time.time(), plan_id, expected_revision),
            )
            if updated.rowcount != 1:
                conn.rollback()
                raise WorkPlanConflict(f"stale work plan revision: {plan_id}")
            conn.commit()
            return self._plan(
                conn.execute(
                    "SELECT * FROM work_plans WHERE id=?", (plan_id,)
                ).fetchone()
            )

    async def required_tasks_complete(self, plan_id: str) -> bool:
        async with self._lock:
            conn = await self._open()
            pending = conn.execute(
                "SELECT COUNT(*) FROM background_tasks WHERE work_plan_id=? "
                "AND required=1 AND status NOT IN (?, ?)",
                (
                    plan_id,
                    BackgroundTaskStatus.COMPLETED,
                    BackgroundTaskStatus.CANCELLED,
                ),
            ).fetchone()[0]
            return pending == 0

    async def tasks_settled_for_completion(self, plan_id: str) -> bool:
        async with self._lock:
            conn = await self._open()
            active = conn.execute(
                "SELECT COUNT(*) FROM background_tasks WHERE work_plan_id=? "
                "AND status IN (?, ?)",
                (plan_id, BackgroundTaskStatus.QUEUED, BackgroundTaskStatus.RUNNING),
            ).fetchone()[0]
            return active == 0

    async def cancel_plan(self, plan_id: str, expected_revision: int) -> WorkPlan:
        """Cancel linked work atomically before exposing a terminal plan state."""
        async with self._lock:
            conn = await self._open()
            current = conn.execute(
                "SELECT * FROM work_plans WHERE id=?", (plan_id,)
            ).fetchone()
            if current is None:
                raise KeyError(plan_id)
            if current["revision"] != expected_revision:
                raise WorkPlanConflict(f"stale work plan revision: {plan_id}")
            if current["status"] in {
                WorkPlanStatus.COMPLETED,
                WorkPlanStatus.FAILED,
                WorkPlanStatus.CANCELLED,
            }:
                raise WorkPlanConflict("terminal WorkPlan cannot be cancelled")
            now = time.time()
            conn.execute(
                "UPDATE work_plans SET status=?, revision=revision+1, updated_at=? WHERE id=?",
                (WorkPlanStatus.CANCELLING, now, plan_id),
            )
            conn.execute(
                "UPDATE background_tasks SET status=?, updated_at=? WHERE work_plan_id=? "
                "AND status IN (?, ?)",
                (
                    BackgroundTaskStatus.CANCELLED,
                    now,
                    plan_id,
                    BackgroundTaskStatus.QUEUED,
                    BackgroundTaskStatus.RUNNING,
                ),
            )
            conn.execute(
                "UPDATE plan_steps SET status=? WHERE work_plan_id=? "
                "AND status IN (?, ?, ?)",
                (
                    PlanStepStatus.SKIPPED,
                    plan_id,
                    PlanStepStatus.PENDING,
                    PlanStepStatus.ACTIVE,
                    PlanStepStatus.BLOCKED,
                ),
            )
            conn.execute(
                "UPDATE work_plans SET status=?, revision=revision+1, updated_at=? WHERE id=?",
                (WorkPlanStatus.CANCELLED, now, plan_id),
            )
            conn.commit()
            return self._plan(
                conn.execute(
                    "SELECT * FROM work_plans WHERE id=?", (plan_id,)
                ).fetchone()
            )

    @staticmethod
    def _inbox_item(row: sqlite3.Row) -> WorkPlanInboxItem:
        return WorkPlanInboxItem(**dict(row))

    async def create_plan(
        self,
        chat_id: str,
        owner_id: str,
        title: str,
        *,
        max_open_per_chat: int | None = None,
        max_open_per_owner: int | None = None,
    ) -> WorkPlan:
        async with self._lock:
            conn = await self._open()
            terminal = (
                WorkPlanStatus.COMPLETED,
                WorkPlanStatus.FAILED,
                WorkPlanStatus.CANCELLED,
            )
            if max_open_per_chat is not None:
                count = conn.execute(
                    "SELECT COUNT(*) FROM work_plans WHERE chat_id=? "
                    "AND status NOT IN (?, ?, ?)",
                    (chat_id, *terminal),
                ).fetchone()[0]
                if count >= max(1, max_open_per_chat):
                    raise WorkPlanConflict("open WorkPlan chat limit reached")
            if max_open_per_owner is not None:
                count = conn.execute(
                    "SELECT COUNT(*) FROM work_plans WHERE owner_id=? "
                    "AND status NOT IN (?, ?, ?)",
                    (owner_id, *terminal),
                ).fetchone()[0]
                if count >= max(1, max_open_per_owner):
                    raise WorkPlanConflict("open WorkPlan owner limit reached")
            now = time.time()
            plan_id = str(uuid4())
            handle = plan_id.split("-")[0]
            conn.execute(
                "INSERT INTO work_plans VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)",
                (
                    plan_id,
                    chat_id,
                    owner_id,
                    title[:400],
                    WorkPlanStatus.QUEUED,
                    now,
                    now,
                    handle,
                ),
            )
            conn.commit()
            return self._plan(
                conn.execute(
                    "SELECT * FROM work_plans WHERE id=?", (plan_id,)
                ).fetchone()
            )

    async def create_step(
        self,
        plan_id: str,
        expected_revision: int,
        *,
        title: str,
        description: str = "",
        depends_on: Iterable[str] = (),
        execution_mode: str = "foreground",
    ) -> PlanStep:
        dependency_ids = tuple(dict.fromkeys(depends_on))
        if execution_mode not in {"foreground", "background"}:
            raise ValueError("invalid step execution mode")
        async with self._lock:
            conn = await self._open()
            plan = conn.execute(
                "SELECT revision FROM work_plans WHERE id=?", (plan_id,)
            ).fetchone()
            if plan is None:
                raise KeyError(plan_id)
            if plan["revision"] != expected_revision:
                raise WorkPlanConflict(f"stale work plan revision: {plan_id}")
            if dependency_ids:
                marks = ",".join("?" for _ in dependency_ids)
                rows = conn.execute(
                    f"SELECT id FROM plan_steps WHERE work_plan_id=? AND id IN ({marks})",
                    (plan_id, *dependency_ids),
                ).fetchall()
                if {row["id"] for row in rows} != set(dependency_ids):
                    raise ValueError("step dependencies must belong to the WorkPlan")
            step_id = str(uuid4())
            conn.execute(
                "INSERT INTO plan_steps VALUES (?, ?, ?, ?, ?, ?, ?, NULL, '')",
                (
                    step_id,
                    plan_id,
                    title[:400],
                    description[:2000],
                    PlanStepStatus.PENDING,
                    json.dumps(dependency_ids),
                    execution_mode,
                ),
            )
            conn.execute(
                "UPDATE work_plans SET revision=revision+1, updated_at=? WHERE id=?",
                (time.time(), plan_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM plan_steps WHERE id=?", (step_id,)
            ).fetchone()
            return self._step(row)

    async def list_steps(self, plan_id: str) -> list[PlanStep]:
        async with self._lock:
            conn = await self._open()
            rows = conn.execute(
                "SELECT * FROM plan_steps WHERE work_plan_id=? ORDER BY rowid",
                (plan_id,),
            ).fetchall()
            return [self._step(row) for row in rows]

    async def list_ready_steps(self, plan_id: str) -> list[PlanStep]:
        steps = await self.list_steps(plan_id)
        statuses = {step.id: step.status for step in steps}
        return [
            step
            for step in steps
            if step.status == PlanStepStatus.PENDING
            and all(
                statuses.get(dep) in {PlanStepStatus.DONE, PlanStepStatus.SKIPPED}
                for dep in step.depends_on
            )
        ]

    async def get_plan(self, plan_id: str) -> WorkPlan | None:
        async with self._lock:
            conn = await self._open()
            row = conn.execute(
                "SELECT * FROM work_plans WHERE id=?", (plan_id,)
            ).fetchone()
            return self._plan(row) if row else None

    async def list_plans(
        self, chat_id: str, owner_id: str | None = None
    ) -> list[WorkPlan]:
        async with self._lock:
            conn = await self._open()
            if owner_id is None:
                rows = conn.execute(
                    "SELECT * FROM work_plans WHERE chat_id=? ORDER BY updated_at DESC",
                    (chat_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM work_plans WHERE chat_id=? AND owner_id=? ORDER BY updated_at DESC",
                    (chat_id, owner_id),
                ).fetchall()
            return [self._plan(row) for row in rows]

    async def list_plans_for_owner(self, owner_id: str) -> list[WorkPlan]:
        async with self._lock:
            conn = await self._open()
            rows = conn.execute(
                "SELECT * FROM work_plans WHERE owner_id=? ORDER BY updated_at DESC",
                (owner_id,),
            ).fetchall()
            return [self._plan(row) for row in rows]

    async def list_visible_plans(
        self, chat_id: str, principal_id: str
    ) -> list[WorkPlan]:
        async with self._lock:
            conn = await self._open()
            rows = conn.execute(
                "SELECT DISTINCT p.* FROM work_plans p LEFT JOIN plan_acl a ON a.work_plan_id=p.id WHERE p.chat_id=? AND (p.owner_id=? OR a.principal_id=?) ORDER BY p.updated_at DESC",
                (chat_id, principal_id, principal_id),
            ).fetchall()
            return [self._plan(row) for row in rows]

    async def update_plan(
        self,
        plan_id: str,
        expected_revision: int,
        *,
        status: str | None = None,
        title: str | None = None,
    ) -> WorkPlan:
        async with self._lock:
            conn = await self._open()
            current = conn.execute(
                "SELECT * FROM work_plans WHERE id=?", (plan_id,)
            ).fetchone()
            if current is None:
                raise KeyError(plan_id)
            now = time.time()
            assignments = ["revision=revision+1", "updated_at=?"]
            params: list[Any] = [now]
            if status is not None:
                assignments.append("status=?")
                params.append(status)
            if title is not None:
                assignments.append("title=?")
                params.append(title[:400])
            params.extend([plan_id, expected_revision])
            result = conn.execute(
                f"UPDATE work_plans SET {', '.join(assignments)} WHERE id=? AND revision=?",
                params,
            )
            if result.rowcount != 1:
                conn.rollback()
                raise WorkPlanConflict(f"stale work plan revision: {plan_id}")
            conn.commit()
            return self._plan(
                conn.execute(
                    "SELECT * FROM work_plans WHERE id=?", (plan_id,)
                ).fetchone()
            )

    async def append_event(
        self,
        work_plan_id: str,
        event_type: str,
        event_key: str,
        payload: dict[str, Any],
    ) -> bool:
        async with self._lock:
            conn = await self._open()
            try:
                conn.execute(
                    "INSERT INTO work_plan_events VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid4()),
                        work_plan_id,
                        event_type,
                        event_key,
                        json.dumps(payload, ensure_ascii=False),
                        time.time(),
                    ),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                conn.rollback()
                return False

    async def acquire_lease(
        self, work_plan_id: str, lease_id: str, seconds: float = 60
    ) -> bool:
        async with self._lock:
            conn = await self._open()
            now = time.time()
            conn.execute("DELETE FROM planner_leases WHERE expires_at <= ?", (now,))
            result = conn.execute(
                "INSERT INTO planner_leases VALUES (?, ?, ?) "
                "ON CONFLICT(work_plan_id) DO UPDATE SET "
                "lease_id=excluded.lease_id, expires_at=excluded.expires_at "
                "WHERE planner_leases.lease_id=excluded.lease_id",
                (work_plan_id, lease_id, now + seconds),
            )
            conn.commit()
            return result.rowcount == 1

    async def release_leases_by_id(self, lease_id: str) -> int:
        """Release all plan leases owned by a completed internal turn."""
        if not lease_id:
            return 0
        async with self._lock:
            conn = await self._open()
            released = conn.execute(
                "DELETE FROM planner_leases WHERE lease_id=?", (lease_id,)
            ).rowcount
            conn.commit()
            return released

    async def release_lease(self, work_plan_id: str, lease_id: str) -> bool:
        async with self._lock:
            conn = await self._open()
            result = conn.execute(
                "DELETE FROM planner_leases WHERE work_plan_id=? AND lease_id=?",
                (work_plan_id, lease_id),
            )
            conn.commit()
            return result.rowcount == 1

    async def set_acl(
        self, work_plan_id: str, principal_id: str, acl_role: str
    ) -> bool:
        if acl_role not in {"viewer", "contributor", "operator"}:
            raise ValueError("invalid WorkPlan ACL role")
        async with self._lock:
            conn = await self._open()
            conn.execute(
                "INSERT INTO plan_acl VALUES (?, ?, ?, ?) ON CONFLICT(work_plan_id, principal_id) DO UPDATE SET acl_role=excluded.acl_role",
                (work_plan_id, principal_id, acl_role, time.time()),
            )
            conn.commit()
            return True

    async def get_acl_role(self, work_plan_id: str, principal_id: str) -> str | None:
        async with self._lock:
            conn = await self._open()
            row = conn.execute(
                "SELECT acl_role FROM plan_acl WHERE work_plan_id=? AND principal_id=?",
                (work_plan_id, principal_id),
            ).fetchone()
            return str(row[0]) if row else None

    async def list_events(self, work_plan_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            conn = await self._open()
            rows = conn.execute(
                "SELECT event_type, event_key, payload_json, created_at FROM work_plan_events WHERE work_plan_id=? ORDER BY created_at",
                (work_plan_id,),
            ).fetchall()
            return [
                {
                    "event_type": row[0],
                    "event_key": row[1],
                    "payload": json.loads(row[2]),
                    "created_at": row[3],
                }
                for row in rows
            ]

    async def record_handoff(
        self,
        *,
        handoff_key: str,
        source_message_id: str,
        chat_turn_id: str,
        chat_id: str,
        sender_id: str,
        task_summary: str,
        reason: str,
    ) -> bool:
        """Persist one Chat-to-Agent ownership transfer before Agent execution."""
        async with self._lock:
            conn = await self._open()
            try:
                conn.execute(
                    "INSERT INTO chat_agent_handoffs "
                    "(handoff_key, source_message_id, chat_turn_id, chat_id, sender_id, task_summary, reason, created_at, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'RESERVED')",
                    (
                        handoff_key,
                        source_message_id,
                        chat_turn_id,
                        chat_id,
                        sender_id,
                        task_summary[:600],
                        reason[:240],
                        time.time(),
                    ),
                )
            except sqlite3.IntegrityError:
                conn.rollback()
                return False
            conn.commit()
            return True

    async def get_handoff_status(self, handoff_key: str) -> str | None:
        async with self._lock:
            conn = await self._open()
            row = conn.execute(
                "SELECT status FROM chat_agent_handoffs WHERE handoff_key=?",
                (handoff_key,),
            ).fetchone()
            return str(row[0]) if row is not None else None

    async def complete_handoff(self, handoff_key: str) -> bool:
        async with self._lock:
            conn = await self._open()
            updated = conn.execute(
                "UPDATE chat_agent_handoffs SET status='COMPLETED' "
                "WHERE handoff_key=? AND status='RESERVED'",
                (handoff_key,),
            ).rowcount
            conn.commit()
            return updated == 1

    async def delete_handoff(self, handoff_key: str) -> None:
        """Release a handoff reservation when its Agent continuation fails."""
        async with self._lock:
            conn = await self._open()
            conn.execute(
                "DELETE FROM chat_agent_handoffs WHERE handoff_key=?", (handoff_key,)
            )
            conn.commit()

    async def delegate_background(
        self,
        work_plan_id: str,
        *,
        expected_revision: int,
        brief: dict[str, Any],
        idempotency_key: str,
        required: bool = True,
        step_id: str | None = None,
        allow_parallel: bool = False,
        max_running: int = 1,
        depends_on: Iterable[str] = (),
    ) -> BackgroundTask:
        """Atomically transition a plan and create its durable delegation outbox."""
        async with self._lock:
            conn = await self._open()
            existing = conn.execute(
                "SELECT background_task_id FROM delegation_intents WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                row = conn.execute(
                    "SELECT * FROM background_tasks WHERE id=?", (existing[0],)
                ).fetchone()
                if row is None:
                    raise RuntimeError("delegation intent references a missing task")
                return self._task(row)
            now = time.time()
            dependency_ids = tuple(dict.fromkeys(depends_on))
            if dependency_ids:
                marks = ",".join("?" for _ in dependency_ids)
                dependencies = conn.execute(
                    f"SELECT id, status FROM background_tasks WHERE work_plan_id=? AND id IN ({marks})",
                    (work_plan_id, *dependency_ids),
                ).fetchall()
                if {item["id"] for item in dependencies} != set(dependency_ids):
                    raise ValueError(
                        "background dependencies must belong to the WorkPlan"
                    )
                if any(
                    item["status"]
                    in {
                        BackgroundTaskStatus.FAILED,
                        BackgroundTaskStatus.CANCELLED,
                        BackgroundTaskStatus.NEEDS_INPUT,
                    }
                    for item in dependencies
                ):
                    raise WorkPlanConflict("background dependency is terminal")
                dependencies_ready = all(
                    item["status"] == BackgroundTaskStatus.COMPLETED
                    for item in dependencies
                )
            else:
                dependencies_ready = True
            active_count = conn.execute(
                "SELECT COUNT(*) FROM background_tasks t "
                "WHERE t.work_plan_id=? AND (t.status=? OR (t.status=? AND NOT EXISTS ("
                "SELECT 1 FROM background_task_dependencies d "
                "JOIN background_tasks parent ON parent.id=d.depends_on_task_id "
                "WHERE d.background_task_id=t.id AND parent.status<>?"
                ")))",
                (
                    work_plan_id,
                    BackgroundTaskStatus.RUNNING,
                    BackgroundTaskStatus.QUEUED,
                    BackgroundTaskStatus.COMPLETED,
                ),
            ).fetchone()[0]
            allowed_running = max(1, max_running) if allow_parallel else 1
            if dependencies_ready and active_count >= allowed_running:
                raise WorkPlanConflict("background task concurrency limit reached")
            if step_id is not None:
                step_row = conn.execute(
                    "SELECT * FROM plan_steps WHERE id=? AND work_plan_id=?",
                    (step_id, work_plan_id),
                ).fetchone()
                if step_row is None:
                    raise KeyError(step_id)
                step = self._step(step_row)
                if step.status != PlanStepStatus.PENDING:
                    raise WorkPlanConflict("background step is not pending")
                if not self._step_dependencies_ready(
                    conn, work_plan_id, step.depends_on
                ):
                    raise WorkPlanConflict("step dependencies are not complete")
            updated = conn.execute(
                "UPDATE work_plans SET status=?, revision=revision+1, updated_at=? WHERE id=? AND revision=?",
                (
                    WorkPlanStatus.WAITING_BACKGROUND,
                    now,
                    work_plan_id,
                    expected_revision,
                ),
            )
            if updated.rowcount != 1:
                conn.rollback()
                raise WorkPlanConflict(f"stale work plan revision: {work_plan_id}")
            task_id = str(uuid4())
            payload = json.dumps(brief, ensure_ascii=False)
            conn.execute(
                "INSERT INTO background_tasks "
                "(id, work_plan_id, status, brief_json, required, attempts, allow_parallel, result_json, created_at, updated_at) "
                "VALUES (?, ?, 'QUEUED', ?, ?, 0, ?, '', ?, ?)",
                (
                    task_id,
                    work_plan_id,
                    payload,
                    int(required),
                    int(allow_parallel),
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO plan_task_links VALUES (?, ?)", (work_plan_id, task_id)
            )
            for dependency_id in dependency_ids:
                conn.execute(
                    "INSERT INTO background_task_dependencies VALUES (?, ?)",
                    (task_id, dependency_id),
                )
            conn.execute(
                "INSERT INTO delegation_intents VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid4()), work_plan_id, idempotency_key, task_id, payload, now),
            )
            if step_id is not None:
                conn.execute(
                    "UPDATE plan_steps SET status=?, background_task_id=? WHERE id=?",
                    (PlanStepStatus.ACTIVE, task_id, step_id),
                )
            conn.commit()
            return self._task(
                conn.execute(
                    "SELECT * FROM background_tasks WHERE id=?", (task_id,)
                ).fetchone()
            )

    async def retry_background(
        self,
        work_plan_id: str,
        *,
        expected_revision: int,
        previous_task_id: str,
        brief: dict[str, Any],
        idempotency_key: str,
    ) -> BackgroundTask:
        """Create an explicitly requested retry without replaying old execution."""
        async with self._lock:
            conn = await self._open()
            existing = conn.execute(
                "SELECT background_task_id FROM delegation_intents WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                row = conn.execute(
                    "SELECT * FROM background_tasks WHERE id=?", (existing[0],)
                ).fetchone()
                if row is None:
                    raise RuntimeError("delegation intent references a missing task")
                return self._task(row)
            previous = conn.execute(
                "SELECT * FROM background_tasks WHERE id=? AND work_plan_id=?",
                (previous_task_id, work_plan_id),
            ).fetchone()
            if previous is None:
                raise KeyError(previous_task_id)
            if previous["status"] not in {
                BackgroundTaskStatus.FAILED,
                BackgroundTaskStatus.NEEDS_INPUT,
                BackgroundTaskStatus.INTERRUPTED,
            }:
                raise WorkPlanConflict(
                    "only failed, interrupted, or input-blocked tasks can retry"
                )
            plan = conn.execute(
                "SELECT revision FROM work_plans WHERE id=?", (work_plan_id,)
            ).fetchone()
            if plan is None:
                raise KeyError(work_plan_id)
            if plan["revision"] != expected_revision:
                raise WorkPlanConflict(f"stale work plan revision: {work_plan_id}")
            running = conn.execute(
                "SELECT COUNT(*) FROM background_tasks WHERE work_plan_id=? AND status IN (?, ?)",
                (
                    work_plan_id,
                    BackgroundTaskStatus.QUEUED,
                    BackgroundTaskStatus.RUNNING,
                ),
            ).fetchone()[0]
            if running:
                raise WorkPlanConflict("background task concurrency limit reached")
            now = time.time()
            task_id = str(uuid4())
            payload = json.dumps(
                {**brief, "retry_of": previous_task_id}, ensure_ascii=False
            )
            updated = conn.execute(
                "UPDATE work_plans SET status=?, revision=revision+1, updated_at=? WHERE id=? AND revision=?",
                (
                    WorkPlanStatus.WAITING_BACKGROUND,
                    now,
                    work_plan_id,
                    expected_revision,
                ),
            )
            if updated.rowcount != 1:
                conn.rollback()
                raise WorkPlanConflict(f"stale work plan revision: {work_plan_id}")
            conn.execute(
                "INSERT INTO background_tasks "
                "(id, work_plan_id, status, brief_json, required, attempts, allow_parallel, result_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 0, ?, '', ?, ?)",
                (
                    task_id,
                    work_plan_id,
                    BackgroundTaskStatus.QUEUED,
                    payload,
                    previous["required"],
                    previous["allow_parallel"],
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO plan_task_links VALUES (?, ?)", (work_plan_id, task_id)
            )
            conn.execute(
                "INSERT INTO delegation_intents VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid4()), work_plan_id, idempotency_key, task_id, payload, now),
            )
            step = conn.execute(
                "SELECT id FROM plan_steps WHERE background_task_id=?",
                (previous_task_id,),
            ).fetchone()
            if step is not None:
                conn.execute(
                    "UPDATE plan_steps SET status=?, background_task_id=? WHERE id=?",
                    (PlanStepStatus.ACTIVE, task_id, step["id"]),
                )
            conn.execute(
                "INSERT INTO work_plan_events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    work_plan_id,
                    "background_retry",
                    f"retry:{previous_task_id}:{task_id}",
                    json.dumps(
                        {
                            "previous_task_id": previous_task_id,
                            "background_task_id": task_id,
                        }
                    ),
                    now,
                ),
            )
            conn.commit()
            return self._task(
                conn.execute(
                    "SELECT * FROM background_tasks WHERE id=?", (task_id,)
                ).fetchone()
            )

    async def create_background_task(
        self,
        work_plan_id: str,
        brief: dict[str, Any],
        *,
        idempotency_key: str,
        required: bool = True,
        allow_parallel: bool = False,
    ) -> BackgroundTask:
        async with self._lock:
            conn = await self._open()
            existing = conn.execute(
                "SELECT background_task_id FROM delegation_intents WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                return self._task(
                    conn.execute(
                        "SELECT * FROM background_tasks WHERE id=?", (existing[0],)
                    ).fetchone()
                )
            task_id = str(uuid4())
            now = time.time()
            payload = json.dumps(brief, ensure_ascii=False)
            conn.execute(
                "INSERT INTO background_tasks "
                "(id, work_plan_id, status, brief_json, required, attempts, allow_parallel, result_json, created_at, updated_at) "
                "VALUES (?, ?, 'QUEUED', ?, ?, 0, ?, '', ?, ?)",
                (
                    task_id,
                    work_plan_id,
                    payload,
                    int(required),
                    int(allow_parallel),
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO plan_task_links VALUES (?, ?)", (work_plan_id, task_id)
            )
            conn.execute(
                "INSERT INTO delegation_intents VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid4()), work_plan_id, idempotency_key, task_id, payload, now),
            )
            conn.commit()
            return self._task(
                conn.execute(
                    "SELECT * FROM background_tasks WHERE id=?", (task_id,)
                ).fetchone()
            )

    async def get_background_task(self, task_id: str) -> BackgroundTask | None:
        async with self._lock:
            conn = await self._open()
            row = conn.execute(
                "SELECT * FROM background_tasks WHERE id=?", (task_id,)
            ).fetchone()
            return self._task(row) if row is not None else None

    async def list_background_tasks(
        self, work_plan_id: str, *, statuses: Iterable[str] | None = None
    ) -> list[BackgroundTask]:
        async with self._lock:
            conn = await self._open()
            if statuses:
                values = tuple(statuses)
                placeholders = ", ".join("?" for _ in values)
                rows = conn.execute(
                    f"SELECT * FROM background_tasks WHERE work_plan_id=? AND status IN ({placeholders}) ORDER BY created_at",
                    (work_plan_id, *values),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM background_tasks WHERE work_plan_id=? ORDER BY created_at",
                    (work_plan_id,),
                ).fetchall()
            return [self._task(row) for row in rows]

    async def list_runnable_background_tasks(self) -> list[BackgroundTask]:
        """Return work that can be resumed after startup reconciliation."""
        async with self._lock:
            conn = await self._open()
            rows = conn.execute(
                "SELECT t.* FROM background_tasks t WHERE t.status=? AND NOT EXISTS ("
                "SELECT 1 FROM background_task_dependencies d "
                "JOIN background_tasks parent ON parent.id=d.depends_on_task_id "
                "WHERE d.background_task_id=t.id AND parent.status<>?"
                ") ORDER BY t.created_at",
                (BackgroundTaskStatus.QUEUED, BackgroundTaskStatus.COMPLETED),
            ).fetchall()
            return [self._task(row) for row in rows]

    async def background_dependencies_satisfied(self, task_id: str) -> bool:
        async with self._lock:
            conn = await self._open()
            pending = conn.execute(
                "SELECT COUNT(*) FROM background_task_dependencies d "
                "JOIN background_tasks parent ON parent.id=d.depends_on_task_id "
                "WHERE d.background_task_id=? AND parent.status<>?",
                (task_id, BackgroundTaskStatus.COMPLETED),
            ).fetchone()[0]
            return pending == 0

    async def claim_background_task(
        self,
        task_id: str,
        *,
        max_retries: int,
        max_running_per_plan: int = 1,
        max_running_per_chat: int = 1,
    ) -> BackgroundTask | None:
        """Atomically claim one runnable queued task for execution."""
        async with self._lock:
            conn = await self._open()
            now = time.time()
            cursor = conn.execute(
                "UPDATE background_tasks SET status=?, attempts=attempts + 1, updated_at=? "
                "WHERE id=? AND status=? AND attempts<=? AND ("
                "(allow_parallel=1 AND (SELECT COUNT(*) FROM background_tasks running "
                "WHERE running.work_plan_id=background_tasks.work_plan_id AND running.status=?) < ?) "
                "OR (allow_parallel=0 AND NOT EXISTS (SELECT 1 FROM background_tasks running "
                "WHERE running.work_plan_id=background_tasks.work_plan_id AND running.status=?))"
                ") AND (SELECT COUNT(*) FROM background_tasks running "
                "JOIN work_plans plan ON plan.id=running.work_plan_id "
                "WHERE plan.chat_id=(SELECT chat_id FROM work_plans WHERE id=background_tasks.work_plan_id) "
                "AND running.status=?) < ? AND NOT EXISTS ("
                "SELECT 1 FROM background_task_dependencies d "
                "JOIN background_tasks parent ON parent.id=d.depends_on_task_id "
                "WHERE d.background_task_id=? AND parent.status<>?"
                ")",
                (
                    BackgroundTaskStatus.RUNNING,
                    now,
                    task_id,
                    BackgroundTaskStatus.QUEUED,
                    max(0, max_retries),
                    BackgroundTaskStatus.RUNNING,
                    max(1, max_running_per_plan),
                    BackgroundTaskStatus.RUNNING,
                    BackgroundTaskStatus.RUNNING,
                    max(1, max_running_per_chat),
                    task_id,
                    BackgroundTaskStatus.COMPLETED,
                ),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                return None
            conn.commit()
            row = conn.execute(
                "SELECT * FROM background_tasks WHERE id=?", (task_id,)
            ).fetchone()
            return self._task(row) if row is not None else None

    async def update_background_task(
        self, task_id: str, status: str, result: dict[str, Any] | None = None
    ) -> BackgroundTask:
        async with self._lock:
            conn = await self._open()
            now = time.time()
            conn.execute(
                "UPDATE background_tasks SET status=?, result_json=?, attempts=attempts + CASE WHEN ? = 'RUNNING' THEN 1 ELSE 0 END, updated_at=? "
                "WHERE id=? AND status<>?",
                (
                    status,
                    json.dumps(result or {}, ensure_ascii=False),
                    status,
                    now,
                    task_id,
                    BackgroundTaskStatus.CANCELLED,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM background_tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            return self._task(row)

    async def claim_inbox(
        self,
        work_plan_id: str,
        lease_id: str,
        *,
        limit: int = 16,
        lease_seconds: int = 60,
    ) -> list[WorkPlanInboxItem]:
        """Lease pending or expired events for one WorkPlan consumer."""
        async with self._lock:
            conn = await self._open()
            now = time.time()
            rows = conn.execute(
                "SELECT * FROM work_plan_inbox WHERE work_plan_id=? "
                "AND (lease_id IS NULL OR lease_expires_at <= ?) ORDER BY created_at LIMIT ?",
                (work_plan_id, now, max(1, limit)),
            ).fetchall()
            if not rows:
                return []
            expires = now + max(1, lease_seconds)
            ids = tuple(row["id"] for row in rows)
            marks = ",".join("?" for _ in ids)
            conn.execute(
                f"UPDATE work_plan_inbox SET lease_id=?, lease_expires_at=? WHERE id IN ({marks})",
                (lease_id, expires, *ids),
            )
            conn.commit()
            leased = conn.execute(
                f"SELECT * FROM work_plan_inbox WHERE id IN ({marks}) ORDER BY created_at",
                ids,
            ).fetchall()
            return [self._inbox_item(row) for row in leased]

    async def record_consumer_evidence(
        self,
        work_plan_id: str,
        lease_id: str,
        event_ids: Iterable[str],
        action: str,
    ) -> bool:
        """Record an explicit consumer decision while its inbox lease is active."""
        ids = tuple(dict.fromkeys(event_ids))
        if not ids:
            return False
        async with self._lock:
            conn = await self._open()
            lease = conn.execute(
                "SELECT 1 FROM planner_leases WHERE work_plan_id=? AND lease_id=? AND expires_at>?",
                (work_plan_id, lease_id, time.time()),
            ).fetchone()
            if lease is None:
                raise WorkPlanConflict("consumer planner lease is missing or expired")
            marks = ",".join("?" for _ in ids)
            leased = conn.execute(
                f"SELECT COUNT(*) FROM work_plan_inbox WHERE work_plan_id=? AND lease_id=? AND event_id IN ({marks})",
                (work_plan_id, lease_id, *ids),
            ).fetchone()[0]
            if leased != len(ids):
                raise WorkPlanConflict("consumer evidence does not match inbox lease")
            payload = json.dumps(
                {"lease_id": lease_id, "event_ids": ids, "action": action[:80]},
                ensure_ascii=False,
            )
            event_key = f"consumer-ack:{lease_id}:{','.join(ids)}"
            inserted = conn.execute(
                "INSERT OR IGNORE INTO work_plan_events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    work_plan_id,
                    "consumer_ack",
                    event_key,
                    payload,
                    time.time(),
                ),
            ).rowcount
            conn.commit()
            return inserted == 1

    async def has_consumer_evidence(
        self, work_plan_id: str, lease_id: str, event_ids: Iterable[str]
    ) -> bool:
        ids = set(event_ids)
        if not ids:
            return False
        async with self._lock:
            conn = await self._open()
            rows = conn.execute(
                "SELECT payload_json FROM work_plan_events "
                "WHERE work_plan_id=? AND event_type=? AND event_key LIKE ?",
                (work_plan_id, "consumer_ack", f"consumer-ack:{lease_id}:%"),
            ).fetchall()
            evidenced: set[str] = set()
            for row in rows:
                try:
                    payload = json.loads(row["payload_json"])
                    if payload.get("lease_id") == lease_id:
                        evidenced.update(payload.get("event_ids", ()))
                except (TypeError, ValueError, AttributeError):
                    continue
            return ids.issubset(evidenced)

    async def acknowledge_inbox(self, lease_id: str, event_ids: Iterable[str]) -> int:
        ids = tuple(event_ids)
        if not ids:
            return 0
        async with self._lock:
            conn = await self._open()
            marks = ",".join("?" for _ in ids)
            deleted = conn.execute(
                f"DELETE FROM work_plan_inbox WHERE lease_id=? AND event_id IN ({marks})",
                (lease_id, *ids),
            ).rowcount
            conn.commit()
            return deleted

    async def release_inbox(self, lease_id: str) -> int:
        """Release event leases after an unsuccessful consumer attempt."""
        async with self._lock:
            conn = await self._open()
            released = conn.execute(
                "UPDATE work_plan_inbox SET lease_id=NULL, lease_expires_at=NULL "
                "WHERE lease_id=?",
                (lease_id,),
            ).rowcount
            conn.commit()
            return released

    def _enqueue_background_inbox(
        self,
        conn: sqlite3.Connection,
        *,
        work_plan_id: str,
        event_id: str,
        task_id: str,
        payload: str,
        now: float,
    ) -> None:
        """Bound pending projections without discarding durable task/result facts."""
        pending = conn.execute(
            "SELECT COUNT(*) FROM work_plan_inbox WHERE work_plan_id=?", (work_plan_id,)
        ).fetchone()[0]
        if pending < self.max_pending_events_per_plan:
            conn.execute(
                "INSERT INTO work_plan_inbox "
                "(id, work_plan_id, event_id, background_task_id, coalesce_key, payload_json, created_at, lease_id, lease_expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                (
                    str(uuid4()),
                    work_plan_id,
                    event_id,
                    task_id,
                    f"background:{task_id}",
                    payload,
                    now,
                ),
            )
            return

        # Prefer coalescing into an event that has not yet been handed to a
        # consumer. The authoritative task/result and event rows remain intact.
        overflow = conn.execute(
            "SELECT * FROM work_plan_inbox WHERE work_plan_id=? "
            "AND (lease_id IS NULL OR lease_expires_at <= ?) ORDER BY created_at LIMIT 1",
            (work_plan_id, now),
        ).fetchone()
        if overflow is not None:
            try:
                overflow_payload = json.loads(overflow["payload_json"])
            except (TypeError, ValueError):
                overflow_payload = {}
            task_ids = list(
                dict.fromkeys(
                    [
                        *overflow_payload.get("overflow_task_ids", []),
                        overflow_payload.get("background_task_id", ""),
                        task_id,
                    ]
                )
            )
            merged = {
                "overflow": True,
                "overflow_task_ids": [item for item in task_ids if item],
                "latest_background_task_id": task_id,
                "latest_status": json.loads(payload).get("status", "unknown"),
            }
            conn.execute(
                "UPDATE work_plan_inbox SET coalesce_key=?, payload_json=?, created_at=? WHERE id=?",
                (
                    f"overflow:{work_plan_id}",
                    json.dumps(merged, ensure_ascii=False),
                    now,
                    overflow["id"],
                ),
            )
            conn.execute(
                "INSERT INTO work_plan_events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    work_plan_id,
                    "inbox_overflow",
                    f"inbox-overflow:{work_plan_id}:{task_id}",
                    json.dumps({"background_task_id": task_id}, ensure_ascii=False),
                    now,
                ),
            )
            return

        # All existing events are in an active consumer decision. Keep one
        # additional durable record rather than silently losing a result.
        conn.execute(
            "INSERT INTO work_plan_inbox "
            "(id, work_plan_id, event_id, background_task_id, coalesce_key, payload_json, created_at, lease_id, lease_expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
            (
                str(uuid4()),
                work_plan_id,
                event_id,
                task_id,
                f"background:{task_id}",
                payload,
                now,
            ),
        )

    async def settle_background_task(
        self,
        task_id: str,
        status: str,
        result: dict[str, Any],
        *,
        expected_status: str | None = None,
    ) -> BackgroundTask:
        """Atomically persist a terminal result, event, and notification outbox row."""
        async with self._lock:
            conn = await self._open()
            row = conn.execute(
                "SELECT * FROM background_tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            if expected_status is not None and row["status"] != expected_status:
                return self._task(row)
            try:
                target_status = BackgroundTaskStatus(status)
            except ValueError as exc:
                raise ValueError("invalid background task status") from exc
            now = time.time()
            result_payload = {**result, "background_task_id": task_id}
            payload = json.dumps(result_payload, ensure_ascii=False)
            conn.execute(
                "UPDATE background_tasks SET status=?, result_json=?, updated_at=? WHERE id=?",
                (target_status, payload, now, task_id),
            )
            step_status = {
                BackgroundTaskStatus.COMPLETED: PlanStepStatus.DONE,
                BackgroundTaskStatus.NEEDS_INPUT: PlanStepStatus.BLOCKED,
                BackgroundTaskStatus.FAILED: PlanStepStatus.BLOCKED,
                BackgroundTaskStatus.CANCELLED: PlanStepStatus.SKIPPED,
            }[target_status]
            summary = str(
                result_payload.get("result")
                or result_payload.get("error")
                or target_status.value.lower()
            )[:2000]
            if target_status is BackgroundTaskStatus.NEEDS_INPUT:
                next_plan_status = WorkPlanStatus.WAITING_USER
            elif target_status in {
                BackgroundTaskStatus.FAILED,
                BackgroundTaskStatus.CANCELLED,
            }:
                next_plan_status = WorkPlanStatus.PAUSED
            else:
                next_plan_status = None
            event_key = (
                f"background:{task_id}:{result_payload.get('status', status.lower())}"
            )
            event_id = str(uuid4())
            inserted = conn.execute(
                "INSERT OR IGNORE INTO work_plan_events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    row["work_plan_id"],
                    "background_result",
                    event_key,
                    payload,
                    now,
                ),
            ).rowcount
            if inserted:
                step_updated = conn.execute(
                    "UPDATE plan_steps SET status=?, result_summary=? "
                    "WHERE work_plan_id=? AND background_task_id=? AND status=?",
                    (
                        step_status,
                        summary,
                        row["work_plan_id"],
                        task_id,
                        PlanStepStatus.ACTIVE,
                    ),
                ).rowcount
                if next_plan_status is not None:
                    conn.execute(
                        "UPDATE work_plans SET status=?, revision=revision+1, updated_at=? "
                        "WHERE id=? AND status NOT IN (?, ?, ?, ?)",
                        (
                            next_plan_status,
                            now,
                            row["work_plan_id"],
                            WorkPlanStatus.COMPLETED,
                            WorkPlanStatus.FAILED,
                            WorkPlanStatus.CANCELLING,
                            WorkPlanStatus.CANCELLED,
                        ),
                    )
                if step_updated:
                    conn.execute(
                        "INSERT INTO work_plan_events VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            str(uuid4()),
                            row["work_plan_id"],
                            "background_step_projected",
                            f"step-result:{task_id}",
                            json.dumps(
                                {
                                    "background_task_id": task_id,
                                    "step_status": step_status,
                                }
                            ),
                            now,
                        ),
                    )
                self._enqueue_background_inbox(
                    conn,
                    work_plan_id=row["work_plan_id"],
                    event_id=event_id,
                    task_id=task_id,
                    payload=payload,
                    now=now,
                )
            if target_status in {
                BackgroundTaskStatus.FAILED,
                BackgroundTaskStatus.CANCELLED,
                BackgroundTaskStatus.NEEDS_INPUT,
            }:
                blocked_children = conn.execute(
                    "SELECT t.id, t.work_plan_id FROM background_tasks t "
                    "JOIN background_task_dependencies d ON d.background_task_id=t.id "
                    "WHERE d.depends_on_task_id=? AND t.status=?",
                    (task_id, BackgroundTaskStatus.QUEUED),
                ).fetchall()
                for child in blocked_children:
                    child_payload = json.dumps(
                        {
                            "status": "cancelled",
                            "error": f"dependency {task_id} ended as {target_status.value}",
                            "background_task_id": child["id"],
                        },
                        ensure_ascii=False,
                    )
                    conn.execute(
                        "UPDATE background_tasks SET status=?, result_json=?, updated_at=? WHERE id=? AND status=?",
                        (
                            BackgroundTaskStatus.CANCELLED,
                            child_payload,
                            now,
                            child["id"],
                            BackgroundTaskStatus.QUEUED,
                        ),
                    )
                    conn.execute(
                        "UPDATE plan_steps SET status=?, result_summary=? "
                        "WHERE work_plan_id=? AND background_task_id=? AND status=?",
                        (
                            PlanStepStatus.SKIPPED,
                            f"dependency {task_id} ended as {target_status.value}",
                            child["work_plan_id"],
                            child["id"],
                            PlanStepStatus.ACTIVE,
                        ),
                    )
                    conn.execute(
                        "INSERT INTO work_plan_events VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            str(uuid4()),
                            child["work_plan_id"],
                            "background_dependency_blocked",
                            f"dependency-blocked:{child['id']}:{task_id}",
                            child_payload,
                            now,
                        ),
                    )
            conn.execute(
                "INSERT OR IGNORE INTO background_task_notifications VALUES (?, ?, ?, ?, NULL)",
                (task_id, row["work_plan_id"], payload, now),
            )
            conn.commit()
            settled = conn.execute(
                "SELECT * FROM background_tasks WHERE id=?", (task_id,)
            ).fetchone()
            return self._task(settled)

    async def list_pending_background_notifications(self) -> list[BackgroundTask]:
        async with self._lock:
            conn = await self._open()
            rows = conn.execute(
                "SELECT t.* FROM background_tasks t JOIN background_task_notifications n ON n.background_task_id=t.id WHERE n.delivered_at IS NULL ORDER BY n.created_at"
            ).fetchall()
            return [self._task(row) for row in rows]

    async def list_background_tasks_with_pending_inbox(self) -> list[BackgroundTask]:
        async with self._lock:
            conn = await self._open()
            rows = conn.execute(
                "SELECT DISTINCT t.* FROM background_tasks t "
                "JOIN work_plan_inbox i ON i.background_task_id=t.id "
                "WHERE i.lease_id IS NULL OR i.lease_expires_at <= ? ORDER BY i.created_at",
                (time.time(),),
            ).fetchall()
            return [self._task(row) for row in rows]

    async def mark_background_notification_delivered(self, task_id: str) -> None:
        async with self._lock:
            conn = await self._open()
            conn.execute(
                "UPDATE background_task_notifications SET delivered_at=? WHERE background_task_id=?",
                (time.time(), task_id),
            )
            conn.commit()

    async def reconcile(self, *, waiting_user_timeout: float = 1800) -> dict[str, int]:
        async with self._lock:
            conn = await self._open()
            now = time.time()
            conn.execute("DELETE FROM planner_leases WHERE expires_at <= ?", (now,))
            interrupted = conn.execute(
                "UPDATE background_tasks SET status=?, updated_at=? WHERE status=?",
                (BackgroundTaskStatus.INTERRUPTED, now, BackgroundTaskStatus.RUNNING),
            ).rowcount
            paused = conn.execute(
                "UPDATE work_plans SET status=?, revision=revision+1, updated_at=? "
                "WHERE id IN (SELECT DISTINCT work_plan_id FROM background_tasks WHERE status=?) "
                "AND status NOT IN (?, ?, ?)",
                (
                    WorkPlanStatus.PAUSED,
                    now,
                    BackgroundTaskStatus.INTERRUPTED,
                    WorkPlanStatus.COMPLETED,
                    WorkPlanStatus.FAILED,
                    WorkPlanStatus.CANCELLED,
                ),
            ).rowcount
            paused_waiting = conn.execute(
                "UPDATE work_plans SET status=?, revision=revision+1, updated_at=? "
                "WHERE status=? AND updated_at <= ?",
                (
                    WorkPlanStatus.PAUSED,
                    now,
                    WorkPlanStatus.WAITING_USER,
                    now - max(1, waiting_user_timeout),
                ),
            ).rowcount
            compactable = conn.execute(
                "SELECT id, status, created_at, updated_at FROM work_plans "
                "WHERE status IN (?, ?, ?) AND updated_at <= ? "
                "AND NOT EXISTS (SELECT 1 FROM work_plan_inbox i WHERE i.work_plan_id=work_plans.id)",
                (
                    WorkPlanStatus.COMPLETED,
                    WorkPlanStatus.FAILED,
                    WorkPlanStatus.CANCELLED,
                    now - self.terminal_retention_seconds,
                ),
            ).fetchall()
            compacted = 0
            for plan in compactable:
                event_counts = dict(
                    conn.execute(
                        "SELECT event_type, COUNT(*) FROM work_plan_events "
                        "WHERE work_plan_id=? GROUP BY event_type",
                        (plan["id"],),
                    ).fetchall()
                )
                task_counts = dict(
                    conn.execute(
                        "SELECT status, COUNT(*) FROM background_tasks "
                        "WHERE work_plan_id=? GROUP BY status",
                        (plan["id"],),
                    ).fetchall()
                )
                summary = json.dumps(
                    {
                        "status": plan["status"],
                        "created_at": plan["created_at"],
                        "terminal_at": plan["updated_at"],
                        "event_counts": event_counts,
                        "background_task_counts": task_counts,
                    },
                    ensure_ascii=False,
                )
                conn.execute(
                    "INSERT INTO work_plan_event_summaries VALUES (?, ?, ?) "
                    "ON CONFLICT(work_plan_id) DO UPDATE SET summary_json=excluded.summary_json, compacted_at=excluded.compacted_at",
                    (plan["id"], summary, now),
                )
                conn.execute(
                    "DELETE FROM work_plan_events WHERE work_plan_id=?", (plan["id"],)
                )
                compacted += 1
            conn.commit()
            queued = conn.execute(
                "SELECT COUNT(*) FROM background_tasks WHERE status=?",
                (BackgroundTaskStatus.QUEUED,),
            ).fetchone()[0]
            return {
                "interrupted": interrupted,
                "paused": paused,
                "paused_waiting": paused_waiting,
                "compacted": compacted,
                "queued": queued,
            }

    async def get_event_summary(self, work_plan_id: str) -> dict[str, Any] | None:
        async with self._lock:
            conn = await self._open()
            row = conn.execute(
                "SELECT summary_json FROM work_plan_event_summaries WHERE work_plan_id=?",
                (work_plan_id,),
            ).fetchone()
            return json.loads(row["summary_json"]) if row is not None else None

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
