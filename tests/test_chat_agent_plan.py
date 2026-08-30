import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.ai.protocol import AssistantMessage, AssistantToolCall
from core.engine.agent_engine import AgentEngine
from core.engine.assistant_output import decide_assistant_output
from core.engine.planner_control import PlannerAction, PlannerControl
from core.engine.reply_necessity import (
    ReplyAdmission,
    ReplyNecessityGate,
    ReplyNecessityInput,
)
from core.engine.turn_capabilities import TurnCapabilities
from core.engine.turn_planner import PlannerRequest, PlannerResultKind, TurnPlanner
from core.managers.session_manager import (
    AdmissionOrigin,
    InboundIntent,
    PendingInbound,
)
from core.message import InputMessage
from core.orchestration.background_task_runner import BackgroundTaskRunner
from core.orchestration.work_plan_service import (
    PlanPrincipal,
    WorkPlanPermissionError,
    WorkPlanService,
)
from core.orchestration.work_plan_store import WorkPlanConflict, WorkPlanStore
from core.tasks.wake_runner import WakeRunner
from core.tasks.wake_coalescer import PendingWake, SOURCE_TASK


class Item:
    def __init__(self, content):
        self.content = content


def test_reply_gate_is_fail_closed_and_auditable():
    gate = ReplyNecessityGate()
    skipped = gate.evaluate(
        ReplyNecessityInput(
            source="ambient", chat_id="g", batch=[Item("hi")], pending_count=1
        )
    )
    assert skipped.admission is ReplyAdmission.SKIP
    assert skipped.score is None

    admitted = gate.evaluate(
        ReplyNecessityInput(
            source="ambient",
            chat_id="g",
            batch=[
                Item(
                    (
                        "请问你怎么看这个问题？"
                        + " 这是一个需要详细分析的上下文，请结合最近的情况给出完整建议。"
                    )
                    * 4
                ),
                Item("请给我建议"),
            ],
            pending_count=4,
            active_chat=True,
            mode="active",
        )
    )
    assert admitted.admission is ReplyAdmission.ADMIT
    assert dict(admitted.score_breakdown)["question"] == 15
    assert "frequency_factor" in dict(admitted.score_breakdown)


def test_chat_does_not_treat_plain_no_reply_text_as_control():
    decision = decide_assistant_output(
        "NO_REPLY",
        [],
        capabilities=TurnCapabilities.for_mode(
            mode="chat",
            capability_profile="private_chat",
            intent=InboundIntent.PRIVATE_CONVERSATION,
        ),
        explicit_delivery_already_sent=False,
        suppress_reply=False,
    )
    assert decision.should_deliver is True


@pytest.mark.asyncio
async def test_turn_planner_consumes_control_without_visible_draft():
    async def provider(**kwargs):
        return AssistantMessage(
            content=None,
            tool_calls=[
                AssistantToolCall("c", "planner_control", '{"action":"no_reply"}')
            ],
        )

    planner = TurnPlanner(provider=provider)
    result = await planner.run(PlannerRequest("turn", "chat", "private", [], []))
    assert result.kind is PlannerResultKind.NO_REPLY


@pytest.mark.asyncio
async def test_turn_planner_consumes_production_control_actions():
    planner = TurnPlanner()
    private = PlannerRequest("turn", "chat", "private", [], [], max_waits=1)
    handoff = await planner.consume_control(
        private,
        PlannerControl(
            PlannerAction.REQUEST_AGENT,
            task_summary="inspect repository",
            reason="requires workspace access",
        ),
    )
    assert handoff.kind is PlannerResultKind.HANDED_OFF
    assert handoff.task_summary == "inspect repository"

    waiting = await planner.consume_control(
        private, PlannerControl(PlannerAction.WAIT, wait_seconds=1)
    )
    assert waiting.kind is PlannerResultKind.WAITING
    assert waiting.wait_seconds == 1

    ambient = await planner.consume_control(
        PlannerRequest("ambient", "chat", "ambient", [], []),
        PlannerControl(
            PlannerAction.REQUEST_AGENT,
            task_summary="inspect repository",
            reason="requires workspace access",
        ),
    )
    assert ambient.kind is PlannerResultKind.FAILED
    assert ambient.reason == "request_agent_not_allowed"


@pytest.mark.asyncio
async def test_work_plan_store_cas_and_idempotent_background_task(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    principal = PlanPrincipal("chat", "owner")
    service = WorkPlanService(store)
    plan = await service.create(principal, "build")
    task1 = await service.delegate_background(
        principal,
        plan.id,
        expected_revision=0,
        brief={"title": "inspect"},
        idempotency_key="once",
    )
    task2 = await service.delegate_background(
        principal,
        plan.id,
        expected_revision=0,
        brief={"title": "inspect"},
        idempotency_key="once",
    )
    assert task1.id == task2.id
    with pytest.raises(WorkPlanConflict):
        await service.transition(
            principal, plan.id, expected_revision=0, status="ACTIVE"
        )
    current = await store.get_plan(plan.id)
    assert current.status == "WAITING_BACKGROUND"
    assert current.revision == 1
    assert await service.share(
        principal,
        plan.id,
        principal_id="peer",
        acl_role="viewer",
        collaboration_enabled=True,
    )
    peer_plans = await service.list(PlanPrincipal("chat", "peer"))
    assert [visible.id for visible in peer_plans] == [plan.id]
    await store.close()


@pytest.mark.asyncio
async def test_background_runner_records_structured_result_and_event(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    plan = await store.create_plan("chat", "owner", "build")
    task = await store.create_background_task(
        plan.id, {"brief": "inspect"}, idempotency_key="background"
    )

    async def executor(task):
        assert "inspect" in task.brief_json
        return {"status": "completed", "result": "done"}

    notifications = []

    async def on_result(completed_task, completed_result):
        notifications.append((completed_task.id, completed_result.status))

    result = await BackgroundTaskRunner(store, executor, on_result=on_result).run(task)
    assert result.status == "completed"
    assert notifications == [(task.id, "completed")]
    events = await store.list_events(plan.id)
    assert events[0]["payload"]["background_task_id"] == task.id
    await store.close()


@pytest.mark.asyncio
async def test_background_runner_atomically_claims_task_before_execution(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    plan = await store.create_plan("chat", "owner", "build")
    task = await store.create_background_task(
        plan.id, {"brief": "inspect"}, idempotency_key="concurrent"
    )
    executions = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def executor(_task):
        nonlocal executions
        executions += 1
        entered.set()
        await release.wait()
        return {"status": "completed", "result": "done"}

    runner = BackgroundTaskRunner(store, executor)
    first = asyncio.create_task(runner.run(task))
    await entered.wait()
    second = asyncio.create_task(runner.run(task))
    release.set()
    results = await asyncio.gather(first, second)

    assert executions == 1
    assert sorted(result.status for result in results) == ["completed", "deferred"]
    current = await store.get_background_task(task.id)
    assert current is not None
    assert current.attempts == 1
    await store.close()


@pytest.mark.asyncio
async def test_runner_does_not_notify_after_external_cancellation(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    plan = await store.create_plan("chat", "owner", "build")
    task = await store.create_background_task(
        plan.id, {"brief": "work"}, idempotency_key="cancel-during-run"
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    notifications = []

    async def executor(_task):
        entered.set()
        await release.wait()
        return {"status": "completed", "result": "should be discarded"}

    async def on_result(completed_task, result):
        notifications.append((completed_task.id, result.status))

    runner = BackgroundTaskRunner(store, executor, on_result=on_result)
    execution = asyncio.create_task(runner.run(task))
    await entered.wait()
    await store.update_background_task(task.id, "CANCELLED")
    release.set()

    result = await execution

    assert result.status == "cancelled"
    assert notifications == []
    assert (await store.get_background_task(task.id)).status == "CANCELLED"
    await store.close()


@pytest.mark.asyncio
async def test_runner_cancellation_does_not_emit_terminal_result_or_wake(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    plan = await store.create_plan("chat", "owner", "build")
    task = await store.create_background_task(
        plan.id, {"brief": "work"}, idempotency_key="runner-cancel"
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    notifications = []

    async def executor(_task):
        entered.set()
        await release.wait()
        return {"status": "completed", "result": "must not publish"}

    async def on_result(completed_task, result):
        notifications.append((completed_task.id, result.status))

    runner = BackgroundTaskRunner(store, executor, on_result=on_result)
    execution = runner.start(task)
    await entered.wait()
    await store.update_background_task(task.id, "CANCELLED")
    execution.cancel()
    await execution

    assert notifications == []
    assert (await store.get_background_task(task.id)).status == "CANCELLED"
    assert await store.list_events(plan.id) == []
    await store.close()


@pytest.mark.asyncio
async def test_failed_background_dependency_cancels_queued_children(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    plan = await store.create_plan("chat", "owner", "build")
    parent = await store.delegate_background(
        plan.id,
        expected_revision=0,
        brief={"task": "parent"},
        idempotency_key="parent-failure",
    )
    child = await store.delegate_background(
        plan.id,
        expected_revision=1,
        brief={"task": "child"},
        idempotency_key="child-blocked",
        depends_on=[parent.id],
    )
    await store.update_background_task(parent.id, "RUNNING")

    settled = await store.settle_background_task(
        parent.id, "FAILED", {"status": "failed", "error": "parent failed"}
    )

    assert settled.status == "FAILED"
    blocked = await store.get_background_task(child.id)
    assert blocked is not None
    assert blocked.status == "CANCELLED"
    assert await store.list_runnable_background_tasks() == []
    assert any(
        event["event_type"] == "background_dependency_blocked"
        for event in await store.list_events(plan.id)
    )
    await store.close()


@pytest.mark.asyncio
async def test_work_plan_owner_limit_applies_across_chats(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    service = WorkPlanService(store, max_open_per_chat=8, max_open_per_owner=1)
    await service.create(PlanPrincipal("chat-a", "owner"), "first")

    with pytest.raises(WorkPlanPermissionError, match="owner limit"):
        await service.create(PlanPrincipal("chat-b", "owner"), "second")

    chat_limited = WorkPlanService(store, max_open_per_chat=1, max_open_per_owner=8)
    await chat_limited.create(PlanPrincipal("shared-chat", "first-owner"), "first")
    with pytest.raises(WorkPlanPermissionError, match="open WorkPlan limit"):
        await chat_limited.create(
            PlanPrincipal("shared-chat", "second-owner"), "second"
        )
    await store.close()


@pytest.mark.asyncio
async def test_failed_background_task_can_be_retried_explicitly(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    service = WorkPlanService(store)
    principal = PlanPrincipal("chat", "owner")
    plan = await service.create(principal, "build")
    task = await store.create_background_task(
        plan.id, {"task_summary": "work"}, idempotency_key="failed"
    )
    await store.update_background_task(task.id, "FAILED", {"error": "bad"})

    retried = await service.retry_background(
        principal,
        plan.id,
        expected_revision=plan.revision,
        previous_task_id=task.id,
        brief={"task_summary": "try again"},
        idempotency_key="retry-1",
    )

    assert retried.id != task.id
    assert json.loads(retried.brief_json)["retry_of"] == task.id
    assert (await store.get_background_task(task.id)).status == "FAILED"
    assert (await store.list_events(plan.id))[-1]["event_type"] == "background_retry"
    await store.close()


@pytest.mark.asyncio
async def test_cancelled_background_task_cannot_be_revived_by_runner_write(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    plan = await store.create_plan("chat", "owner", "build")
    task = await store.create_background_task(
        plan.id, {"brief": "work"}, idempotency_key="cancel-race"
    )
    await store.update_background_task(task.id, "CANCELLED")

    after = await store.update_background_task(task.id, "RUNNING")

    assert after.status == "CANCELLED"
    assert after.attempts == 0
    await store.close()


@pytest.mark.asyncio
async def test_work_plan_details_projects_only_own_steps_and_task_summaries(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    service = WorkPlanService(store)
    principal = PlanPrincipal("chat", "owner")
    plan = await service.create(principal, "build")
    step = await service.add_step(
        principal, plan.id, expected_revision=plan.revision, title="inspect"
    )
    task = await store.create_background_task(
        plan.id, {"brief": "inspect"}, idempotency_key="detail"
    )

    details = await service.details(principal, plan.id)

    assert details["plan"]["id"] == plan.id
    assert details["steps"][0]["id"] == step.id
    assert details["background_tasks"][0]["id"] == task.id
    assert "brief_json" not in details["background_tasks"][0]
    await store.close()


@pytest.mark.asyncio
async def test_work_plan_foreground_slot_is_exclusive_and_waiting_user_expires(
    tmp_path,
):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    service = WorkPlanService(store)
    principal = PlanPrincipal("chat", "owner")
    first = await service.create(principal, "first")
    second = await service.create(principal, "second")
    active = await service.transition(
        principal, first.id, expected_revision=first.revision, status="ACTIVE"
    )
    with pytest.raises(WorkPlanConflict, match="foreground"):
        await service.transition(
            principal, second.id, expected_revision=second.revision, status="ACTIVE"
        )
    waiting = await service.transition(
        principal,
        first.id,
        expected_revision=active.revision,
        status="WAITING_USER",
    )
    async with store._lock:
        conn = await store._open()
        conn.execute("UPDATE work_plans SET updated_at=0 WHERE id=?", (waiting.id,))
        conn.commit()
    reconciled = await store.reconcile(waiting_user_timeout=1)
    assert reconciled["paused_waiting"] == 1
    assert (await store.get_plan(waiting.id)).status == "PAUSED"
    await store.close()


@pytest.mark.asyncio
async def test_background_delegation_is_serial_unless_parallel_is_explicit(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    service = WorkPlanService(store, max_running_background_per_plan=2)
    principal = PlanPrincipal("chat", "owner")
    plan = await service.create(principal, "build")
    first = await service.delegate_background(
        principal,
        plan.id,
        expected_revision=plan.revision,
        brief={"title": "first"},
        idempotency_key="first",
    )
    current = await store.get_plan(plan.id)
    with pytest.raises(WorkPlanConflict, match="concurrency"):
        await service.delegate_background(
            principal,
            plan.id,
            expected_revision=current.revision,
            brief={"title": "second"},
            idempotency_key="second",
        )
    second = await service.delegate_background(
        principal,
        plan.id,
        expected_revision=current.revision,
        brief={"title": "second"},
        idempotency_key="second-parallel",
        parallel=True,
    )
    assert first.id != second.id
    await store.close()


@pytest.mark.asyncio
async def test_work_plan_writes_require_and_renew_the_turn_planner_lease(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    service = WorkPlanService(store)
    created = await service.create(PlanPrincipal("chat", "owner"), "build")
    derived = service.principal_factory(
        SimpleNamespace(chat_id="chat", sender_id="owner", turn_id="turn-1")
    )
    assert derived.planner_lease_id == "turn-1"
    assert derived.planner_plan_id == ""
    planner = PlanPrincipal(
        "chat", "owner", planner_lease_id="turn-1", planner_plan_id=created.id
    )
    renamed = await service.update_title(
        planner,
        created.id,
        expected_revision=created.revision,
        title="renamed",
    )
    assert renamed.title == "renamed"
    renewed = await service.update_title(
        planner,
        created.id,
        expected_revision=renamed.revision,
        title="renamed again",
    )
    assert renewed.title == "renamed again"
    other = await service.create(PlanPrincipal("chat", "owner"), "other")
    with pytest.raises(WorkPlanPermissionError, match="another WorkPlan"):
        await service.update_title(
            planner,
            other.id,
            expected_revision=other.revision,
            title="must reject",
        )
    with pytest.raises(WorkPlanConflict, match="active planner lease"):
        await service.update_title(
            PlanPrincipal("chat", "owner", planner_lease_id="turn-2"),
            created.id,
            expected_revision=renewed.revision,
            title="contended",
        )
    assert await store.release_leases_by_id("turn-1") == 1
    released = await service.update_title(
        PlanPrincipal("chat", "owner", planner_lease_id="turn-2"),
        created.id,
        expected_revision=renewed.revision,
        title="after release",
    )
    assert released.title == "after release"
    await store.close()


@pytest.mark.asyncio
async def test_background_result_creates_durable_leased_work_plan_inbox_item(
    tmp_path,
):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    plan = await store.create_plan("chat", "owner", "build")
    task = await store.create_background_task(
        plan.id, {"brief": "inspect"}, idempotency_key="inbox"
    )
    await store.settle_background_task(
        task.id, "COMPLETED", {"status": "completed", "result": "done"}
    )

    claimed = await store.claim_inbox(plan.id, "lease-1")

    assert len(claimed) == 1
    assert claimed[0].coalesce_key == f"background:{task.id}"
    assert await store.acknowledge_inbox("lease-1", [claimed[0].event_id]) == 1
    assert await store.claim_inbox(plan.id, "lease-2") == []
    await store.close()


@pytest.mark.asyncio
async def test_work_plan_inbox_claim_requires_authorized_uncontended_planner_lease(
    tmp_path,
):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    service = WorkPlanService(store)
    principal = PlanPrincipal("chat", "owner")
    plan = await service.create(principal, "build")
    task = await store.create_background_task(
        plan.id, {"brief": "inspect"}, idempotency_key="lease-gate"
    )
    await store.settle_background_task(
        task.id, "COMPLETED", {"status": "completed", "result": "done"}
    )
    held_lease = await service.acquire_lease(principal, plan.id, "foreground")

    with pytest.raises(WorkPlanConflict, match="active planner lease"):
        await service.claim_inbox(principal, plan.id, lease_id="background")

    assert await service.release_lease(principal, plan.id, held_lease)
    claimed = await service.claim_inbox(principal, plan.id, lease_id="background")
    assert [item.background_task_id for item in claimed] == [task.id]
    assert (
        await service.acknowledge_inbox(
            principal,
            plan.id,
            lease_id="background",
            event_ids=[item.event_id for item in claimed],
        )
        == 1
    )
    assert await service.release_lease(principal, plan.id, "background")
    await store.close()


@pytest.mark.asyncio
async def test_background_task_dependency_runs_only_after_parent_completes(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    service = WorkPlanService(store, max_running_background_per_plan=2)
    principal = PlanPrincipal("chat", "owner")
    plan = await service.create(principal, "build")
    parent = await service.delegate_background(
        principal,
        plan.id,
        expected_revision=plan.revision,
        brief={"title": "parent"},
        idempotency_key="parent",
    )
    current = await store.get_plan(plan.id)
    child = await service.delegate_background(
        principal,
        plan.id,
        expected_revision=current.revision,
        brief={"title": "child"},
        idempotency_key="child",
        parallel=True,
        depends_on_tasks=[parent.id],
    )
    ran = []

    async def executor(task):
        ran.append(task.id)
        return {"status": "completed"}

    runner = BackgroundTaskRunner(store, executor)
    assert await runner.resume() == 1
    for _ in range(20):
        if len(ran) == 2:
            break
        await asyncio.sleep(0.01)
    await runner.stop()

    assert ran == [parent.id, child.id]
    assert (await store.get_background_task(child.id)).status == "COMPLETED"
    await store.close()


@pytest.mark.asyncio
async def test_background_claim_enforces_default_serial_and_chat_wide_limit(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    first_plan = await store.create_plan("chat", "first-owner", "first")
    second_plan = await store.create_plan("chat", "second-owner", "second")
    first = await store.create_background_task(
        first_plan.id, {"brief": "first"}, idempotency_key="serial-first"
    )
    same_plan = await store.create_background_task(
        first_plan.id, {"brief": "same"}, idempotency_key="serial-second"
    )
    other_plan = await store.create_background_task(
        second_plan.id, {"brief": "other"}, idempotency_key="chat-second"
    )

    assert await store.claim_background_task(
        first.id, max_retries=2, max_running_per_plan=2, max_running_per_chat=1
    )
    assert (
        await store.claim_background_task(
            same_plan.id,
            max_retries=2,
            max_running_per_plan=2,
            max_running_per_chat=1,
        )
        is None
    )
    assert (
        await store.claim_background_task(
            other_plan.id,
            max_retries=2,
            max_running_per_plan=2,
            max_running_per_chat=1,
        )
        is None
    )
    await store.close()


@pytest.mark.asyncio
async def test_work_plan_update_and_select(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    service = WorkPlanService(store)
    principal = PlanPrincipal("chat", "owner")
    plan = await service.create(principal, "before")

    updated = await service.update_title(
        principal, plan.id, expected_revision=plan.revision, title="after"
    )
    selected = await service.select(
        principal, plan.id, expected_revision=updated.revision
    )

    assert selected.title == "after"
    assert selected.status == "ACTIVE"
    await store.close()


@pytest.mark.asyncio
async def test_work_plan_cannot_complete_while_optional_task_is_active(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    service = WorkPlanService(store)
    principal = PlanPrincipal("chat", "owner")
    plan = await service.create(principal, "build")
    active = await service.transition(
        principal, plan.id, expected_revision=plan.revision, status="ACTIVE"
    )
    await store.create_background_task(
        plan.id, {"brief": "optional"}, idempotency_key="optional", required=False
    )

    with pytest.raises(WorkPlanConflict, match="settle"):
        await service.transition(
            principal, plan.id, expected_revision=active.revision, status="COMPLETED"
        )
    await store.close()


@pytest.mark.asyncio
async def test_work_plan_lifecycle_requires_tasks_and_cascades_cancellation(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    service = WorkPlanService(store)
    principal = PlanPrincipal("chat", "owner")
    plan = await service.create(principal, "build")
    active = await service.transition(
        principal, plan.id, expected_revision=plan.revision, status="ACTIVE"
    )
    task = await store.create_background_task(
        plan.id, {"brief": "work"}, idempotency_key="required"
    )
    with pytest.raises(WorkPlanConflict, match="required"):
        await service.transition(
            principal, plan.id, expected_revision=active.revision, status="COMPLETED"
        )
    cancelled = await service.cancel(
        principal, plan.id, expected_revision=active.revision
    )
    assert cancelled.status == "CANCELLED"
    assert (await store.get_background_task(task.id)).status == "CANCELLED"
    reopened = await service.reopen(
        principal, plan.id, expected_revision=cancelled.revision
    )
    assert reopened.status == "QUEUED"
    await store.close()


@pytest.mark.asyncio
async def test_reconcile_pauses_interrupted_work_and_never_replays_it(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    plan = await store.create_plan("chat", "owner", "build")
    task = await store.create_background_task(
        plan.id, {"brief": "side effect"}, idempotency_key="interrupted"
    )
    await store.update_background_task(task.id, "RUNNING")

    result = await store.reconcile()

    assert result["interrupted"] == 1
    assert result["paused"] == 1
    assert (await store.get_plan(plan.id)).status == "PAUSED"
    assert (await store.list_background_tasks(plan.id))[0].status == "INTERRUPTED"
    assert await store.list_runnable_background_tasks() == []
    await store.close()


@pytest.mark.asyncio
async def test_background_result_notification_recovers_without_replaying_task(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    plan = await store.create_plan("chat", "owner", "build")
    task = await store.create_background_task(
        plan.id, {"brief": "inspect"}, idempotency_key="notification-retry"
    )
    executions = 0

    async def executor(_task):
        nonlocal executions
        executions += 1
        return {"status": "completed", "result": "done"}

    async def failing_notification(_task, _result):
        raise RuntimeError("wake unavailable")

    await BackgroundTaskRunner(store, executor, on_result=failing_notification).run(
        task
    )
    assert executions == 1
    assert len(await store.list_pending_background_notifications()) == 1

    notifications = []

    async def recovered_notification(completed_task, completed_result):
        notifications.append((completed_task.id, completed_result.result))

    recovered = BackgroundTaskRunner(store, executor, on_result=recovered_notification)
    assert await recovered.resume() == 0
    assert executions == 1
    assert notifications == [(task.id, "done")]
    assert await store.list_pending_background_notifications() == []
    await store.close()


@pytest.mark.asyncio
async def test_needs_input_result_returns_work_plan_to_waiting_user(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    plan = await store.create_plan("chat", "owner", "build")
    task = await store.create_background_task(
        plan.id, {"brief": "needs context"}, idempotency_key="needs-input"
    )

    await store.settle_background_task(
        task.id, "NEEDS_INPUT", {"status": "needs_input", "error": "missing path"}
    )

    assert (await store.get_plan(plan.id)).status == "WAITING_USER"
    assert (await store.get_background_task(task.id)).status == "NEEDS_INPUT"
    await store.close()


@pytest.mark.asyncio
async def test_runner_recovers_unacknowledged_inbox_after_notification_was_delivered(
    tmp_path,
):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    plan = await store.create_plan("chat", "owner", "build")
    task = await store.create_background_task(
        plan.id, {"brief": "work"}, idempotency_key="inbox-recovery"
    )
    await store.settle_background_task(
        task.id, "COMPLETED", {"status": "completed", "result": "done"}
    )
    await store.mark_background_notification_delivered(task.id)
    recovered = []

    async def callback(recovered_task, result):
        recovered.append((recovered_task.id, result.status))

    runner = BackgroundTaskRunner(store, lambda _: None, on_result=callback)
    assert await runner.resume() == 0
    assert recovered == [(task.id, "completed")]
    await store.close()


@pytest.mark.asyncio
async def test_work_plan_background_result_wakes_owning_session(tmp_path, monkeypatch):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    plan = await store.create_plan("chat", "owner", "build")
    task = await store.create_background_task(
        plan.id, {"brief": "inspect"}, idempotency_key="wake"
    )
    await store.settle_background_task(
        task.id, "COMPLETED", {"status": "completed", "result": "done", "error": ""}
    )
    events = []
    wakes = []
    engine = AgentEngine.__new__(AgentEngine)
    engine.work_plan_store = store
    engine.work_plan_service = WorkPlanService(store)
    engine._system_events = SimpleNamespace(
        enqueue=lambda **kwargs: events.append(kwargs) or True
    )
    monkeypatch.setattr(
        "core.tasks.wake_coalescer.request_wake", lambda **kwargs: wakes.append(kwargs)
    )

    await engine._on_work_plan_background_result(
        task,
        SimpleNamespace(status="completed", result="done", error=""),
    )

    pending_inbox = await store.list_background_tasks_with_pending_inbox()
    assert [item.id for item in pending_inbox] == [task.id]
    assert wakes == [
        {
            "source": "background-task",
            "intent": "event",
            "session_key": "chat",
            "delivery_target": "chat",
            "work_plan_id": plan.id,
            "reason": f"workplan:{plan.short_handle}:background-result",
        }
    ]
    await store.close()


@pytest.mark.asyncio
async def test_work_plan_wake_scopes_inbox_to_requested_plan(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    service = WorkPlanService(store)
    principal = PlanPrincipal("chat", "owner")
    first = await service.create(principal, "first")
    second = await service.create(principal, "second")
    await service.add_step(
        principal, first.id, expected_revision=first.revision, title="inspect"
    )
    first_task = await store.create_background_task(
        first.id, {"brief": "first"}, idempotency_key="first-wake-scope"
    )
    second_task = await store.create_background_task(
        second.id, {"brief": "second"}, idempotency_key="second-wake-scope"
    )
    for task in (first_task, second_task):
        await store.settle_background_task(
            task.id, "COMPLETED", {"status": "completed", "result": task.id}
        )
    leases = await service.claim_wake_inbox("chat", work_plan_id=first.id)
    assert len(leases) == 1
    assert leases[0].plan_id == first.id
    assert first_task.id[:12] in leases[0].prompt
    assert second_task.id[:12] not in leases[0].prompt
    assert "inspect" in leases[0].prompt
    assert "revision=1" in leases[0].prompt
    with pytest.raises(WorkPlanConflict, match="evidence"):
        await service.settle_wake_inbox(leases, success=True)
    assert await store.list_background_tasks_with_pending_inbox()
    leases = await service.claim_wake_inbox("chat", work_plan_id=first.id)
    assert len(leases) == 1
    await service.record_consumer_evidence(leases[0], "acknowledge")
    await service.settle_wake_inbox(leases, success=True)
    pending = await store.list_background_tasks_with_pending_inbox()
    assert [task.id for task in pending] == [second_task.id]
    assert await service.claim_wake_inbox("other-chat", work_plan_id=second.id) == []
    await store.close()


@pytest.mark.asyncio
async def test_work_plan_wake_acks_inbox_only_after_ai_and_delivery_success(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    service = WorkPlanService(store)
    principal = PlanPrincipal("chat", "owner")
    plan = await service.create(principal, "build")
    task = await store.create_background_task(
        plan.id, {"brief": "inspect"}, idempotency_key="wake-consume"
    )
    await store.settle_background_task(
        task.id, "COMPLETED", {"status": "completed", "result": "done"}
    )
    prompts = []
    wake_calls = []

    class PromptBuilder:
        async def build_system_event_messages(
            self, *, prompt, system_event_key, **kwargs
        ):
            prompts.append(prompt)
            return [], []

    class Agent:
        work_plan_service = service
        prompt_builder = PromptBuilder()
        context_manager = None
        cost_tracker = None

        async def run_wake_turn(self, **kwargs):
            wake_calls.append(kwargs)
            await kwargs["consumer_evidence_callback"]("acknowledge")
            return SimpleNamespace(error="", captured_replies=["done"])

    from core.tasks.delivery_strategy import ChatReplyDeliveryStrategy

    async def send_reply(**_kwargs):
        return None

    runner = WakeRunner(
        Agent(),
        system_events=None,
        cooldown=SimpleNamespace(
            should_defer=lambda **_: SimpleNamespace(defer=False),
            record_run_start=lambda: None,
        ),
        delivery_strategies={"work-plan": ChatReplyDeliveryStrategy(send_reply)},
    )
    result = await runner(PendingWake(source=SOURCE_TASK, session_key="chat"))
    assert result.status == "skipped"
    assert result.skip_reason == "missing-work-plan-scope"
    assert await store.list_background_tasks_with_pending_inbox()

    result = await runner(
        PendingWake(source=SOURCE_TASK, session_key="chat", work_plan_id=plan.id)
    )
    assert result.status == "ran"
    assert wake_calls[0]["planner_sender_id"] == "owner"
    assert wake_calls[0]["planner_lease_id"].startswith(f"wake:{plan.id}:")
    assert wake_calls[0]["planner_plan_id"] == plan.id
    assert wake_calls[0]["work_plan_consumer"] is True
    assert "task=" in prompts[0]
    assert await store.list_background_tasks_with_pending_inbox() == []

    empty = await runner(
        PendingWake(source=SOURCE_TASK, session_key="chat", work_plan_id=plan.id)
    )
    assert empty.status == "skipped"
    assert empty.skip_reason == "no-pending-work-plan-events"
    assert len(wake_calls) == 1

    task = await store.create_background_task(
        plan.id, {"brief": "second"}, idempotency_key="wake-fail"
    )
    await store.settle_background_task(
        task.id, "COMPLETED", {"status": "completed", "result": "keep"}
    )
    Agent.run_wake_turn = AsyncMock(return_value=SimpleNamespace(error="failed"))
    with pytest.raises(RuntimeError, match="failed"):
        await runner(
            PendingWake(source=SOURCE_TASK, session_key="chat", work_plan_id=plan.id)
        )
    assert [
        item.id for item in await store.list_background_tasks_with_pending_inbox()
    ] == [task.id]
    await store.close()


@pytest.mark.asyncio
async def test_work_plan_inbox_coalesces_unleased_overflow_without_losing_task_ids(
    tmp_path,
):
    store = WorkPlanStore(
        str(tmp_path / "orchestration.sqlite"), max_pending_events_per_plan=1
    )
    plan = await store.create_plan("chat", "owner", "build")
    first = await store.create_background_task(
        plan.id, {"brief": "first"}, idempotency_key="overflow-first"
    )
    second = await store.create_background_task(
        plan.id, {"brief": "second"}, idempotency_key="overflow-second"
    )
    await store.settle_background_task(first.id, "COMPLETED", {"status": "completed"})
    await store.settle_background_task(second.id, "COMPLETED", {"status": "completed"})

    claimed = await store.claim_inbox(plan.id, "overflow-lease")
    assert len(claimed) == 1
    payload = json.loads(claimed[0].payload_json)
    assert payload["overflow"] is True
    assert set(payload["overflow_task_ids"]) == {first.id, second.id}
    assert any(
        event["event_type"] == "inbox_overflow"
        for event in await store.list_events(plan.id)
    )
    await store.close()


@pytest.mark.asyncio
async def test_reconcile_compacts_old_terminal_plan_events_but_keeps_tasks(tmp_path):
    store = WorkPlanStore(
        str(tmp_path / "orchestration.sqlite"), terminal_retention_seconds=0
    )
    service = WorkPlanService(store)
    principal = PlanPrincipal("chat", "owner")
    plan = await service.create(principal, "build")
    active = await service.transition(
        principal, plan.id, expected_revision=plan.revision, status="ACTIVE"
    )
    await service.transition(
        principal, plan.id, expected_revision=active.revision, status="COMPLETED"
    )

    report = await store.reconcile()

    assert report["compacted"] == 1
    assert await store.list_events(plan.id) == []
    summary = await store.get_event_summary(plan.id)
    assert summary is not None
    assert summary["status"] == "COMPLETED"
    assert await store.list_background_tasks(plan.id) == []
    await store.close()


@pytest.mark.asyncio
async def test_background_runner_resumes_queued_tasks(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    plan = await store.create_plan("chat", "owner", "build")
    task = await store.create_background_task(
        plan.id, {"brief": "inspect"}, idempotency_key="resume"
    )
    ran = asyncio.Event()

    async def executor(candidate):
        assert candidate.id == task.id
        ran.set()
        return {"status": "completed"}

    runner = BackgroundTaskRunner(store, executor)
    assert await runner.resume() == 1
    await asyncio.wait_for(ran.wait(), timeout=1)
    await runner.stop()
    assert (await store.list_background_tasks(plan.id))[0].status == "COMPLETED"
    await store.close()


@pytest.mark.asyncio
async def test_plan_steps_require_same_plan_dependencies_and_report_ready(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    plan = await store.create_plan("chat", "owner", "build")
    first = await store.create_step(plan.id, 0, title="prepare")
    refreshed = await store.get_plan(plan.id)
    second = await store.create_step(
        plan.id, refreshed.revision, title="finish", depends_on=[first.id]
    )
    with pytest.raises(WorkPlanConflict, match="dependencies"):
        await store.update_step(
            plan.id, refreshed.revision + 1, second.id, status="ACTIVE"
        )
    current = await store.get_plan(plan.id)
    await store.update_step(plan.id, current.revision, first.id, status="DONE")
    current = await store.get_plan(plan.id)
    assert [step.id for step in await store.list_ready_steps(plan.id)] == [second.id]
    assert (
        await store.update_step(plan.id, current.revision, second.id, status="ACTIVE")
    ).status == "ACTIVE"
    other = await store.create_plan("chat", "owner", "other")
    with pytest.raises(ValueError, match="must belong"):
        await store.create_step(other.id, 0, title="invalid", depends_on=[first.id])
    assert second.depends_on == [first.id]
    task_plan = await store.create_plan("chat", "owner", "task")
    task_step = await store.create_step(task_plan.id, 0, title="background")
    task = await store.delegate_background(
        task_plan.id,
        expected_revision=1,
        brief={"title": "inspect"},
        idempotency_key="step-task",
        step_id=task_step.id,
    )
    assert (await store.list_steps(task_plan.id))[0].background_task_id == task.id
    assert (await store.list_steps(task_plan.id))[0].status == "ACTIVE"
    current = await store.get_plan(task_plan.id)
    with pytest.raises(WorkPlanConflict, match="background step"):
        await store.update_step(
            task_plan.id, current.revision, task_step.id, status="DONE"
        )
    await store.update_background_task(task.id, "COMPLETED")
    current = await store.get_plan(task_plan.id)
    assert (
        await store.update_step(
            task_plan.id, current.revision, task_step.id, status="DONE"
        )
    ).status == "DONE"
    await store.close()


@pytest.mark.asyncio
async def test_delegate_background_stale_revision_creates_no_orphan_task(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    principal = PlanPrincipal("chat", "owner")
    service = WorkPlanService(store)
    plan = await service.create(principal, "build")
    with pytest.raises(WorkPlanConflict):
        await service.delegate_background(
            principal,
            plan.id,
            expected_revision=1,
            brief={"title": "inspect"},
            idempotency_key="stale",
        )
    assert await store.list_background_tasks(plan.id) == []
    assert (await store.get_plan(plan.id)).status == "QUEUED"
    await store.close()


@pytest.mark.asyncio
async def test_chat_handoff_is_durable_and_enters_agent_once(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    engine = AgentEngine.__new__(AgentEngine)
    engine.work_plan_store = store
    engine.scheduler = SimpleNamespace(revision=lambda _: 7)
    engine._process_message = AsyncMock()
    pending = PendingInbound(
        InputMessage("message", "user", "chat", "please work", False),
        "please work",
        InboundIntent.PRIVATE_CONVERSATION,
        AdmissionOrigin.USER_MESSAGE,
    )
    control = SimpleNamespace(task_summary="inspect", reason="needs files")

    await engine._handoff_chat_to_agent(
        pending,
        batch=(),
        control=control,
        reply_callback=AsyncMock(),
        get_user_nickname=lambda _: "user",
        text_committed=False,
    )
    await engine._handoff_chat_to_agent(
        pending,
        batch=(),
        control=control,
        reply_callback=AsyncMock(),
        get_user_nickname=lambda _: "user",
        text_committed=False,
    )

    engine._process_message.assert_awaited_once()
    handed_off = engine._process_message.await_args.args[0]
    assert handed_off.mode_routing is not None


@pytest.mark.asyncio
async def test_failed_chat_handoff_releases_reservation_for_retry(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    engine = AgentEngine.__new__(AgentEngine)
    engine.work_plan_store = store
    engine.scheduler = SimpleNamespace(revision=lambda _: 7)
    engine._process_message = AsyncMock(side_effect=[RuntimeError("transient"), None])
    pending = PendingInbound(
        InputMessage("message", "user", "chat", "please work", False),
        "please work",
        InboundIntent.PRIVATE_CONVERSATION,
        AdmissionOrigin.USER_MESSAGE,
    )
    control = SimpleNamespace(task_summary="inspect", reason="needs files")
    with pytest.raises(RuntimeError, match="transient"):
        await engine._handoff_chat_to_agent(
            pending,
            batch=(),
            control=control,
            reply_callback=AsyncMock(),
            get_user_nickname=lambda _: "user",
            text_committed=False,
        )
    await engine._handoff_chat_to_agent(
        pending,
        batch=(),
        control=control,
        reply_callback=AsyncMock(),
        get_user_nickname=lambda _: "user",
        text_committed=False,
    )
    assert engine._process_message.await_count == 2
    await store.close()


@pytest.mark.asyncio
async def test_concurrent_chat_handoff_retries_after_reservation_owner_fails(tmp_path):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    engines = [AgentEngine.__new__(AgentEngine) for _ in range(2)]
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def process(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
            raise RuntimeError("transient")

    for engine in engines:
        engine.work_plan_store = store
        engine.scheduler = SimpleNamespace(revision=lambda _: 7)
        engine._process_message = process
    pending = PendingInbound(
        InputMessage("message", "user", "chat", "please work", False),
        "please work",
        InboundIntent.PRIVATE_CONVERSATION,
        AdmissionOrigin.USER_MESSAGE,
    )
    control = SimpleNamespace(task_summary="inspect", reason="needs files")

    first = asyncio.create_task(
        engines[0]._handoff_chat_to_agent(
            pending,
            batch=(),
            control=control,
            reply_callback=AsyncMock(),
            get_user_nickname=lambda _: "user",
            text_committed=False,
        )
    )
    await started.wait()
    second = asyncio.create_task(
        engines[1]._handoff_chat_to_agent(
            pending,
            batch=(),
            control=control,
            reply_callback=AsyncMock(),
            get_user_nickname=lambda _: "user",
            text_committed=False,
        )
    )
    await asyncio.sleep(0.1)
    release.set()
    with pytest.raises(RuntimeError, match="transient"):
        await first
    await second

    assert calls == 2
    assert await store.get_handoff_status("chat:message:request_agent") == "COMPLETED"
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task_status", "expected_step", "expected_plan"),
    [
        ("COMPLETED", "DONE", "WAITING_BACKGROUND"),
        ("NEEDS_INPUT", "BLOCKED", "WAITING_USER"),
        ("FAILED", "BLOCKED", "PAUSED"),
    ],
)
async def test_background_terminal_result_projects_bound_step_and_plan_state(
    tmp_path, task_status, expected_step, expected_plan
):
    store = WorkPlanStore(str(tmp_path / "orchestration.sqlite"))
    service = WorkPlanService(store)
    principal = PlanPrincipal("chat", "owner")
    plan = await service.create(principal, "build")
    step = await service.add_step(
        principal,
        plan.id,
        expected_revision=plan.revision,
        title="inspect",
        execution_mode="background",
    )
    current = await store.get_plan(plan.id)
    task = await service.delegate_background(
        principal,
        plan.id,
        expected_revision=current.revision,
        brief={"title": "inspect"},
        step_id=step.id,
        idempotency_key=f"projection-{task_status}",
    )

    await store.settle_background_task(
        task.id,
        task_status,
        {"status": task_status.lower(), "result": "projection result"},
    )

    projected = (await store.list_steps(plan.id))[0]
    refreshed_plan = await store.get_plan(plan.id)
    assert projected.status == expected_step
    assert projected.result_summary == "projection result"
    assert refreshed_plan.status == expected_plan
    await store.close()
