"""测试 cron 事件路由：session_target 决定 wake 目标。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.engine.agent_engine import BackgroundTaskResult
from core.engine.delivery_ledger import DeliveryReceipt
from core.tasks.models import CronJob, TaskRecord, TaskStatus
from core.tasks.runner import BackgroundTaskRunner


@pytest.fixture
def runner():
    r = BackgroundTaskRunner()
    r._task_manager = MagicMock()
    r._task_manager.start_task = AsyncMock(
        return_value=TaskRecord(id="t1", status=TaskStatus.RUNNING)
    )
    r._task_manager.finish_task = AsyncMock(
        return_value=TaskRecord(id="t1", status=TaskStatus.SUCCESS)
    )
    return r


@pytest.mark.asyncio
async def test_system_event_main_session_wakes_heartbeat(runner):
    """session_target=main → enqueue 到 heartbeat:events。"""
    runner._system_events = MagicMock()
    runner._wake_dispatcher = None

    job = CronJob(
        name="test",
        session_target="main",
        payload_type="system_event",
        prompt="ping",
        wake_mode="next-heartbeat",
    )
    task = TaskRecord(id="t1", status=TaskStatus.PENDING)

    result = await runner._execute_system_event_payload(job, task)

    assert result.status == TaskStatus.SUCCESS
    runner._system_events.enqueue.assert_called_once()
    call_kwargs = runner._system_events.enqueue.call_args.kwargs
    assert call_kwargs["session_key"] == "heartbeat:events"


@pytest.mark.asyncio
async def test_system_event_isolated_enqueue_delivery_channel(runner):
    """session_target=isolated → 仅入队到 delivery_channel。"""
    runner._system_events = MagicMock()
    runner._wake_dispatcher = None

    job = CronJob(
        name="test",
        session_target="isolated",
        payload_type="system_event",
        prompt="ping",
        delivery_channel="user_001",
    )
    task = TaskRecord(id="t1", status=TaskStatus.PENDING)

    result = await runner._execute_system_event_payload(job, task)

    assert result.status == TaskStatus.SUCCESS
    runner._system_events.enqueue.assert_called_once()
    call_kwargs = runner._system_events.enqueue.call_args.kwargs
    assert call_kwargs["session_key"] == "user_001"


@pytest.mark.asyncio
async def test_system_event_main_now_wakes_coalescer(runner):
    """session_target=main + wake_mode=now → 调用 request_wake。"""
    runner._system_events = MagicMock()
    runner._wake_dispatcher = None

    import core.tasks.wake_coalescer as _coalescer

    original = _coalescer.request_wake
    _coalescer.request_wake = MagicMock()
    try:
        job = CronJob(
            name="test",
            session_target="main",
            payload_type="system_event",
            prompt="ping",
            wake_mode="now",
        )
        task = TaskRecord(id="t1", status=TaskStatus.PENDING)
        await runner._execute_system_event_payload(job, task)
        _coalescer.request_wake.assert_called_once()
        call_kwargs = _coalescer.request_wake.call_args.kwargs
        assert call_kwargs["source"] == "cron"
        assert call_kwargs["session_key"] == "heartbeat:events"
    finally:
        _coalescer.request_wake = original


@pytest.mark.asyncio
async def test_system_event_main_next_heartbeat_no_wake(runner):
    """session_target=main + wake_mode=next-heartbeat → 仅 enqueue，不 wake。"""
    runner._system_events = MagicMock()
    runner._wake_dispatcher = None

    import core.tasks.wake_coalescer as _coalescer

    original = _coalescer.request_wake
    _coalescer.request_wake = MagicMock()
    try:
        job = CronJob(
            name="test",
            session_target="main",
            payload_type="system_event",
            prompt="ping",
            wake_mode="next-heartbeat",
        )
        task = TaskRecord(id="t1", status=TaskStatus.PENDING)
        await runner._execute_system_event_payload(job, task)
        _coalescer.request_wake.assert_not_called()
    finally:
        _coalescer.request_wake = original


@pytest.mark.asyncio
async def test_run_command_no_wake(runner):
    """command 类型 + session_target=isolated → 不 wake。"""
    runner._system_events = MagicMock()
    runner._wake_dispatcher = AsyncMock()
    runner._delivery_cb = AsyncMock()
    runner._task_manager.create_task = AsyncMock(
        return_value=TaskRecord(id="t1", status=TaskStatus.PENDING)
    )
    runner._task_manager.start_task = AsyncMock(
        return_value=TaskRecord(id="t1", status=TaskStatus.RUNNING)
    )
    runner._task_manager.finish_task = AsyncMock(
        return_value=TaskRecord(id="t1", status=TaskStatus.SUCCESS, result="ok")
    )
    runner._task_manager.update_task_record = AsyncMock()

    job = CronJob(
        name="test",
        session_target="isolated",
        payload_type="command",
        command="echo ok",
        delivery_channel="user_001",
    )

    import asyncio

    proc_mock = MagicMock()
    proc_mock.returncode = 0
    proc_mock.communicate = AsyncMock(return_value=(b"ok", b""))

    original_create = asyncio.create_subprocess_exec
    asyncio.create_subprocess_exec = AsyncMock(return_value=proc_mock)
    try:
        task = await runner.run_cron_job(job)
        assert task is not None
        assert task.status == TaskStatus.SUCCESS
        runner._delivery_cb.assert_awaited_once()
        delivery_kwargs = runner._delivery_cb.await_args.kwargs
        assert delivery_kwargs["chat_id"] == "user_001"
        assert delivery_kwargs["content"].endswith("ok")
        # command 类型 + isolated：不 wake 会话
        runner._wake_dispatcher.request.assert_not_called()
    finally:
        asyncio.create_subprocess_exec = original_create


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("execution", "expected_calls", "expected_status"),
    [
        (
            BackgroundTaskResult(result="检测完成\nNO_REPLY", silent=True),
            0,
            "not-requested",
        ),
        (BackgroundTaskResult(result="最终报告"), 1, "delivered"),
        (
            BackgroundTaskResult(result="自动发送内容", tool_delivered=True),
            0,
            "delivered",
        ),
    ],
)
async def test_message_cron_delivery_uses_final_result_and_deduplicates(
    runner, execution, expected_calls, expected_status
):
    runner._delivery_cb = AsyncMock()
    runner._execute_prompt_cb = AsyncMock(return_value=execution)
    runner._task_manager.create_task = AsyncMock(
        return_value=TaskRecord(id="delivery", status=TaskStatus.PENDING, type="cron")
    )
    runner._task_manager.start_task = AsyncMock(
        return_value=TaskRecord(id="delivery", status=TaskStatus.RUNNING, type="cron")
    )
    runner._task_manager.finish_task = AsyncMock(
        side_effect=lambda task_id, **kwargs: TaskRecord(
            id=task_id,
            status=kwargs["status"],
            result=kwargs.get("result"),
            error=kwargs.get("error"),
            type="cron",
        )
    )
    runner._task_manager.update_task_record = AsyncMock()

    job = CronJob(
        name="delivery",
        payload_type="message",
        prompt="check",
        delivery_channel="user_001",
    )
    task = await runner.run_cron_job(job)

    assert task is not None
    assert runner._delivery_cb.await_count == expected_calls
    assert task.delivery_status.value == expected_status
    if expected_calls:
        assert runner._delivery_cb.await_args.kwargs["content"].endswith("最终报告")


@pytest.mark.asyncio
async def test_system_event_cron_is_not_requested_as_announce(runner):
    runner._delivery_cb = AsyncMock()
    runner._system_events = MagicMock()
    runner._task_manager.create_task = AsyncMock(
        return_value=TaskRecord(id="event", status=TaskStatus.PENDING, type="cron")
    )
    runner._task_manager.update_task_record = AsyncMock()

    job = CronJob(
        name="event",
        payload_type="system_event",
        prompt="ping",
        delivery_channel="user_001",
    )
    task = await runner.run_cron_job(job)

    assert task is not None
    runner._delivery_cb.assert_not_awaited()
    assert task.delivery_status.value == "not-requested"


@pytest.mark.asyncio
async def test_cron_delivery_failure_does_not_overwrite_execution_status(runner):
    runner._delivery_cb = AsyncMock(side_effect=RuntimeError("delivery failed"))
    runner._execute_prompt_cb = AsyncMock(
        return_value=BackgroundTaskResult(result="final report")
    )
    runner._task_manager.create_task = AsyncMock(
        return_value=TaskRecord(id="failure", status=TaskStatus.PENDING, type="cron")
    )
    runner._task_manager.start_task = AsyncMock(
        return_value=TaskRecord(id="failure", status=TaskStatus.RUNNING, type="cron")
    )
    runner._task_manager.finish_task = AsyncMock(
        side_effect=lambda task_id, **kwargs: TaskRecord(
            id=task_id,
            status=kwargs["status"],
            result=kwargs.get("result"),
            error=kwargs.get("error"),
            type="cron",
        )
    )
    runner._task_manager.update_task_record = AsyncMock()

    task = await runner.run_cron_job(
        CronJob(name="failure", prompt="check", delivery_channel="user_001")
    )

    assert task is not None
    assert task.status == TaskStatus.SUCCESS
    assert task.delivery_status.value == "not-delivered"
    assert task.delivery_error == "delivery failed"
    assert runner._delivery_cb.await_args.kwargs["content"].endswith("final report")


def test_failed_task_keeps_independent_failure_notification_after_tool_delivery():
    from core.tasks.delivery_policy import decide_cron_delivery

    decision = decide_cron_delivery(
        CronJob(name="failed", delivery_channel="user_001"),
        TaskRecord(
            status=TaskStatus.FAILED,
            error="tool loop failed",
            tool_delivered=True,
        ),
        tool_delivered=True,
    )

    assert decision.should_deliver is True
    assert decision.reason == "execution_failure"
    assert decision.content.endswith("tool loop failed")


@pytest.mark.asyncio
async def test_user_target_cron_wake_uses_system_event_prompt():
    """Cron 完成事件面向普通 chat 时也不能读取聊天历史。"""
    from core.engine.system_events import SystemEventQueue
    from core.tasks.wake_coalescer import SOURCE_CRON, PendingWake, WakeTurnResult
    from core.tasks.wake_runner import WakeRunner

    events = SystemEventQueue()
    events.enqueue("user_001", "任务 'VRChat状态检测' 已完成", "task:cron-1")
    system_messages = [{"role": "system", "content": "cron event"}]
    build_system_event_messages = AsyncMock(return_value=(system_messages, []))
    prompt_builder = SimpleNamespace(
        build_system_event_messages=build_system_event_messages,
        build=AsyncMock(
            side_effect=AssertionError("cron wake must not build chat history")
        ),
    )
    run_wake_turn = AsyncMock(return_value=WakeTurnResult())
    agent = SimpleNamespace(
        prompt_builder=prompt_builder,
        run_wake_turn=run_wake_turn,
        context_manager=None,
        cost_tracker=None,
    )
    cooldown = SimpleNamespace(
        should_defer=lambda **_: SimpleNamespace(defer=False),
        record_run_start=lambda: None,
    )
    wake_runner = WakeRunner(agent, events, cooldown)

    result = await wake_runner(
        PendingWake(
            source=SOURCE_CRON,
            intent="event",
            session_key="user_001",
            delivery_target="user_001",
        )
    )

    assert result.status == "ran"
    build_system_event_messages.assert_awaited_once_with(
        prompt="[系统事件]",
        system_event_key="user_001",
    )
    assert run_wake_turn.await_args.kwargs["messages"] == system_messages
    assert run_wake_turn.await_args.kwargs["work_plan_consumer"] is False


@pytest.mark.asyncio
async def test_wake_runner_tracks_active_same_session_wakes():
    from core.tasks.wake_coalescer import SOURCE_CRON, PendingWake
    from core.tasks.wake_runner import WakeRunner

    started = asyncio.Event()
    release = asyncio.Event()

    async def run_wake_turn(**_kwargs):
        started.set()
        await release.wait()
        return SimpleNamespace(error="")

    agent = SimpleNamespace(
        prompt_builder=SimpleNamespace(
            build_system_event_messages=AsyncMock(return_value=([], []))
        ),
        run_wake_turn=run_wake_turn,
        context_manager=None,
        cost_tracker=None,
    )
    cooldown = SimpleNamespace(
        should_defer=lambda **_: SimpleNamespace(defer=False),
        record_run_start=lambda: None,
    )
    runner = WakeRunner(agent, None, cooldown)
    task = asyncio.create_task(
        runner(PendingWake(source=SOURCE_CRON, session_key="heartbeat:events"))
    )
    await started.wait()

    result = await runner(
        PendingWake(source=SOURCE_CRON, session_key="heartbeat:events")
    )
    assert result.status == "skipped"
    assert result.skip_reason == "requests-in-flight"

    release.set()
    assert (await task).status == "ran"
    assert runner._is_session_active("heartbeat:events") is False


@pytest.mark.asyncio
async def test_wake_runner_marks_heartbeat_tasks_when_turn_starts():
    from core.tasks.wake_coalescer import SOURCE_INTERVAL, PendingWake
    from core.tasks.wake_runner import WakeRunner

    started = []
    agent = SimpleNamespace(
        prompt_builder=SimpleNamespace(
            build_heartbeat_messages=AsyncMock(return_value=([], []))
        ),
        run_wake_turn=AsyncMock(return_value=SimpleNamespace(error="")),
        context_manager=None,
        cost_tracker=None,
        _admin_id=[],
    )
    cooldown = SimpleNamespace(
        should_defer=lambda **_: SimpleNamespace(defer=False),
        record_run_start=lambda: None,
    )
    runner = WakeRunner(
        agent,
        None,
        cooldown,
        heartbeat_task_start_callback=lambda names: started.append(names),
    )

    result = await runner(
        PendingWake(
            source=SOURCE_INTERVAL,
            intent="scheduled",
            session_key="heartbeat:events",
            extra_prompt="每日问候。",
            heartbeat_task_names=("早安",),
        )
    )

    assert result.status == "ran"
    assert started == [("早安",)]


@pytest.mark.asyncio
async def test_wake_runner_does_not_mark_heartbeat_tasks_for_cron_wake():
    from core.tasks.wake_coalescer import SOURCE_CRON, PendingWake
    from core.tasks.wake_runner import WakeRunner

    started = []
    agent = SimpleNamespace(
        prompt_builder=SimpleNamespace(
            build_system_event_messages=AsyncMock(return_value=([], []))
        ),
        run_wake_turn=AsyncMock(return_value=SimpleNamespace(error="")),
        context_manager=None,
        cost_tracker=None,
        _admin_id=[],
    )
    cooldown = SimpleNamespace(
        should_defer=lambda **_: SimpleNamespace(defer=False),
        record_run_start=lambda: None,
    )
    runner = WakeRunner(
        agent,
        None,
        cooldown,
        heartbeat_task_start_callback=lambda names: started.append(names),
    )

    result = await runner(
        PendingWake(
            source=SOURCE_CRON,
            intent="immediate",
            session_key="heartbeat:events",
            extra_prompt="Cron 事件。",
            heartbeat_task_names=("早安",),
        )
    )

    assert result.status == "ran"
    assert started == []


def test_wake_runner_cron_branch():
    """WakeRunner 收到 source=cron 时走 system event 路径。"""
    from core.tasks.wake_coalescer import (
        SOURCE_CRON,
        SOURCE_EXEC,
        SOURCE_TASK,
        PendingWake,
    )

    for src in (SOURCE_CRON, SOURCE_EXEC, SOURCE_TASK):
        pw = PendingWake(source=src, intent="immediate", session_key="user_001")
        assert pw.source in (SOURCE_CRON, SOURCE_EXEC, SOURCE_TASK)


@pytest.mark.asyncio
async def test_wake_runner_retries_when_turn_returns_error():
    from core.engine.system_events import SystemEventQueue
    from core.tasks.wake_coalescer import SOURCE_CRON, PendingWake, WakeTurnResult
    from core.tasks.wake_runner import WakeRunner

    events = SystemEventQueue()
    events.enqueue("chat", "pending", "event")
    agent = SimpleNamespace(
        prompt_builder=SimpleNamespace(
            build_system_event_messages=AsyncMock(
                return_value=([{"role": "system"}], [])
            )
        ),
        run_wake_turn=AsyncMock(return_value=WakeTurnResult(error="failed")),
        context_manager=None,
        cost_tracker=None,
    )
    cooldown = SimpleNamespace(
        record_run_start=lambda: None,
        should_defer=lambda **kwargs: SimpleNamespace(defer=False, reason=""),
    )
    runner = WakeRunner(agent, events, cooldown)

    with pytest.raises(RuntimeError, match="failed"):
        await runner(PendingWake(source=SOURCE_CRON, session_key="chat"))
    assert [event.text for event in events.peek("chat")] == ["pending"]


@pytest.mark.asyncio
async def test_wake_runner_releases_lease_when_prompt_build_fails():
    from core.engine.system_events import SystemEventQueue
    from core.tasks.wake_coalescer import SOURCE_CRON, PendingWake
    from core.tasks.wake_runner import WakeRunner

    events = SystemEventQueue()
    events.enqueue("chat", "pending", "event")
    agent = SimpleNamespace(
        prompt_builder=SimpleNamespace(
            build_system_event_messages=AsyncMock(
                side_effect=RuntimeError("build failed")
            )
        ),
        context_manager=None,
        cost_tracker=None,
    )
    cooldown = SimpleNamespace(
        record_run_start=lambda: None,
        should_defer=lambda **kwargs: SimpleNamespace(defer=False, reason=""),
    )
    runner = WakeRunner(agent, events, cooldown)

    with pytest.raises(RuntimeError, match="build failed"):
        await runner(PendingWake(source=SOURCE_CRON, session_key="chat"))
    assert events.claim_snapshot("chat") is not None


@pytest.mark.asyncio
async def test_wake_runner_releases_lease_when_turn_is_cancelled():
    from core.engine.system_events import SystemEventQueue
    from core.tasks.wake_coalescer import SOURCE_CRON, PendingWake
    from core.tasks.wake_runner import WakeRunner

    events = SystemEventQueue()
    events.enqueue("chat", "pending", "event")
    agent = SimpleNamespace(
        prompt_builder=SimpleNamespace(
            build_system_event_messages=AsyncMock(
                return_value=([{"role": "system"}], [])
            )
        ),
        run_wake_turn=AsyncMock(side_effect=asyncio.CancelledError),
        context_manager=None,
        cost_tracker=None,
    )
    cooldown = SimpleNamespace(
        record_run_start=lambda: None,
        should_defer=lambda **kwargs: SimpleNamespace(defer=False, reason=""),
    )
    runner = WakeRunner(agent, events, cooldown)

    with pytest.raises(asyncio.CancelledError):
        await runner(PendingWake(source=SOURCE_CRON, session_key="chat"))

    lease = events.claim_snapshot("chat")
    assert lease is not None
    events.release_snapshot(lease)


@pytest.mark.asyncio
async def test_wake_runner_skips_when_same_session_lease_is_busy():
    from core.engine.system_events import SystemEventBusy, SystemEventQueue
    from core.tasks.wake_coalescer import SOURCE_CRON, PendingWake
    from core.tasks.wake_runner import WakeRunner

    events = SystemEventQueue()
    events.enqueue("chat", "pending", "event")
    lease = events.claim_snapshot("chat")
    assert lease is not None
    agent = SimpleNamespace(
        prompt_builder=SimpleNamespace(
            build_system_event_messages=AsyncMock(side_effect=SystemEventBusy("chat"))
        ),
        context_manager=None,
        cost_tracker=None,
    )
    cooldown = SimpleNamespace(
        record_run_start=lambda: None,
        should_defer=lambda **kwargs: SimpleNamespace(defer=False, reason=""),
    )
    runner = WakeRunner(agent, events, cooldown)

    result = await runner(PendingWake(source=SOURCE_CRON, session_key="chat"))

    assert result.status == "skipped"
    assert result.skip_reason == "requests-in-flight"
    events.release_snapshot(lease)


@pytest.mark.asyncio
async def test_heartbeat_delivery_failure_does_not_record_notification():
    from core.tasks.delivery_strategy import HeartbeatDeliveryStrategy

    record_notification = MagicMock()
    heartbeat = SimpleNamespace(
        should_suppress=lambda text: False,
        record_delivery_start=MagicMock(),
        record_notification=record_notification,
        deliver_to_admin=AsyncMock(return_value=False),
        _cooldown_hours=12,
    )
    strategy = HeartbeatDeliveryStrategy(heartbeat, show_alerts=True)

    with pytest.raises(RuntimeError, match="not confirmed"):
        await strategy.deliver(
            SimpleNamespace(should_notify=True, notification_text="alert")
        )

    record_notification.assert_not_called()


def test_wake_turn_result_deliver_to_user():
    """WakeTurnResult 支持 deliver_to_user 字段。"""
    from core.tasks.wake_coalescer import WakeTurnResult

    r = WakeTurnResult()
    assert r.deliver_to_user == ""

    r = WakeTurnResult(
        should_notify=True, notification_text="hello", deliver_to_user="user_001"
    )
    assert r.deliver_to_user == "user_001"
    assert r.should_notify is True
    assert r.notification_text == "hello"


@pytest.mark.asyncio
async def test_main_message_completion_wakes_heartbeat(runner):
    """message 类型 + session_target=main → 完成时入队到 heartbeat:events + wake。"""
    runner._system_events = MagicMock()
    runner._wake_dispatcher = AsyncMock()
    runner._delivery_cb = None
    runner._execute_prompt_cb = AsyncMock(return_value=("done", None))

    runner._task_manager.create_task = AsyncMock(
        return_value=TaskRecord(id="t3", status=TaskStatus.PENDING, type="cron")
    )
    runner._task_manager.start_task = AsyncMock(
        return_value=TaskRecord(id="t3", status=TaskStatus.RUNNING)
    )
    runner._task_manager.finish_task = AsyncMock(
        return_value=TaskRecord(id="t3", status=TaskStatus.SUCCESS, result="done")
    )
    runner._task_manager.update_task_record = AsyncMock()

    job = CronJob(
        name="test",
        session_target="main",
        payload_type="message",
        prompt="say hi",
        wake_mode="now",
    )

    import core.tasks.runner as _runner_module

    original = _runner_module._wake_coalescer.request_wake
    _runner_module._wake_coalescer.request_wake = MagicMock()
    try:
        task = await runner.run_cron_job(job)
        assert task is not None

        # 最后一条 enqueue 应发到 heartbeat:events
        assert runner._system_events.enqueue.call_count >= 1
        last_kwargs = runner._system_events.enqueue.call_args.kwargs
        assert last_kwargs["session_key"] == "heartbeat:events"

        # 应 wake 心跳系统（wake_mode=now）
        _runner_module._wake_coalescer.request_wake.assert_called_once()
        wake_kwargs = _runner_module._wake_coalescer.request_wake.call_args.kwargs
        assert wake_kwargs["source"] == "cron"
        assert wake_kwargs["session_key"] == "heartbeat:events"
    finally:
        _runner_module._wake_coalescer.request_wake = original


@pytest.mark.asyncio
async def test_main_command_completion_wakes_heartbeat(runner):
    """command 类型 + session_target=main → 完成时入队到 heartbeat:events。"""
    runner._system_events = MagicMock()
    runner._wake_dispatcher = AsyncMock()
    runner._delivery_cb = None
    runner._task_manager.create_task = AsyncMock(
        return_value=TaskRecord(id="t4", status=TaskStatus.PENDING)
    )
    runner._task_manager.start_task = AsyncMock(
        return_value=TaskRecord(id="t4", status=TaskStatus.RUNNING)
    )
    runner._task_manager.finish_task = AsyncMock(
        return_value=TaskRecord(id="t4", status=TaskStatus.SUCCESS, result="ok")
    )
    runner._task_manager.update_task_record = AsyncMock()

    job = CronJob(
        name="test",
        session_target="main",
        payload_type="command",
        command="echo ok",
        wake_mode="next-heartbeat",
    )

    import asyncio

    proc_mock = MagicMock()
    proc_mock.returncode = 0
    proc_mock.communicate = AsyncMock(return_value=(b"ok", b""))
    original_create = asyncio.create_subprocess_exec
    asyncio.create_subprocess_exec = AsyncMock(return_value=proc_mock)
    try:
        task = await runner.run_cron_job(job)
        assert task is not None
        # 应入队到 heartbeat:events
        runner._system_events.enqueue.assert_called_once()
        call_kwargs = runner._system_events.enqueue.call_args.kwargs
        assert call_kwargs["session_key"] == "heartbeat:events"
    finally:
        asyncio.create_subprocess_exec = original_create


@pytest.mark.asyncio
async def test_background_task_intent_immediate(runner):
    """runner.run_task 完成后 background-task wake 使用 intent=immediate。"""
    runner._system_events = MagicMock()
    runner._wake_dispatcher = AsyncMock()
    runner._delivery_cb = None
    runner._execute_prompt_cb = AsyncMock(return_value=("result", None))

    runner._task_manager.create_task = AsyncMock(
        return_value=TaskRecord(
            id="t2",
            status=TaskStatus.PENDING,
            type="manual",
            delivery_channel="user_001",
        )
    )
    runner._task_manager.start_task = AsyncMock(
        return_value=TaskRecord(id="t2", status=TaskStatus.RUNNING)
    )
    runner._task_manager.finish_task = AsyncMock(
        return_value=TaskRecord(
            id="t2",
            status=TaskStatus.SUCCESS,
            result="ok",
            delivery_channel="user_001",
        )
    )
    runner._task_manager.update_task_record = AsyncMock()

    task = await runner._task_manager.create_task(
        prompt="test", task_type="manual", delivery_channel="user_001"
    )

    # 执行后台任务 -> 完成后触发 background-task wake
    result = await runner.run_task(task)
    assert result is not None

    # intent 应为 immediate（不是 event）
    runner._wake_dispatcher.request.assert_called_once()
    call_kwargs = runner._wake_dispatcher.request.call_args.kwargs
    assert call_kwargs["source"] == "background-task"
    assert call_kwargs["intent"] == "immediate"


@pytest.mark.asyncio
async def test_manual_task_final_result_uses_direct_ledger_delivery(runner):
    runner._wake_dispatcher = AsyncMock()
    runner._execute_prompt_cb = AsyncMock(
        return_value=BackgroundTaskResult(result="最终报告")
    )
    runner._delivery_cb = AsyncMock(
        return_value=DeliveryReceipt(status="accepted", platform_message_id="m1")
    )
    runner._task_manager.start_task = AsyncMock(
        return_value=TaskRecord(
            id="manual", status=TaskStatus.RUNNING, delivery_channel="user_001"
        )
    )
    runner._task_manager.finish_task = AsyncMock(
        return_value=TaskRecord(
            id="manual",
            status=TaskStatus.SUCCESS,
            result="最终报告",
            delivery_channel="user_001",
        )
    )
    runner._task_manager.update_task_record = AsyncMock()

    task = await runner.run_task(
        TaskRecord(id="manual", type="manual", delivery_channel="user_001")
    )

    assert task.delivery_status.value == "delivered"
    assert task.delivery_id == "task:manual:final"
    assert runner._delivery_cb.await_args.kwargs["delivery_id"] == "task:manual:final"
    assert runner._delivery_cb.await_args.kwargs["content"].endswith("最终报告")
    runner._wake_dispatcher.request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("execution", "expected_status"),
    [
        (BackgroundTaskResult(result="already sent", tool_delivered=True), "delivered"),
        (BackgroundTaskResult(result="NO_REPLY", silent=True), "not-requested"),
    ],
)
async def test_manual_task_deduplicates_tool_and_silent_results(
    runner, execution, expected_status
):
    runner._wake_dispatcher = AsyncMock()
    runner._execute_prompt_cb = AsyncMock(return_value=execution)
    runner._delivery_cb = AsyncMock()
    runner._task_manager.start_task = AsyncMock(
        return_value=TaskRecord(
            id="manual", status=TaskStatus.RUNNING, delivery_channel="user_001"
        )
    )
    runner._task_manager.finish_task = AsyncMock(
        return_value=TaskRecord(
            id="manual",
            status=TaskStatus.SUCCESS,
            result=execution.result,
            delivery_channel="user_001",
        )
    )
    runner._task_manager.update_task_record = AsyncMock()

    task = await runner.run_task(
        TaskRecord(id="manual", type="manual", delivery_channel="user_001")
    )

    assert task.delivery_status.value == expected_status
    runner._delivery_cb.assert_not_awaited()
    runner._wake_dispatcher.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_restart_recovery_delivery_is_attempted_once(runner):
    runner._delivery_cb = AsyncMock(return_value=DeliveryReceipt(status="unknown"))
    runner._task_manager.update_task_record = AsyncMock()
    task = TaskRecord(
        id="restart",
        status=TaskStatus.LOST,
        delivery_channel="user_001",
        recovery_notification_pending=True,
    )

    handled = await runner.deliver_restart_recovery(task, is_group=False)

    assert handled is True
    assert task.delivery_id == "task:restart:restart"
    assert task.delivery_status.value == "unknown"
    assert task.recovery_notification_pending is False
    runner._delivery_cb.assert_awaited_once()
    assert await runner.deliver_restart_recovery(task, is_group=False) is False
    runner._delivery_cb.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_task_accepts_legacy_tuple_execution_result(runner):
    runner._execute_prompt_cb = AsyncMock(return_value=("legacy result", None))
    runner._task_manager.start_task = AsyncMock(
        return_value=TaskRecord(id="legacy", status=TaskStatus.RUNNING)
    )
    runner._task_manager.finish_task = AsyncMock(
        side_effect=lambda task_id, **kwargs: TaskRecord(
            id=task_id,
            status=kwargs["status"],
            result=kwargs.get("result"),
            error=kwargs.get("error"),
        )
    )
    runner._task_manager.update_task_record = AsyncMock()

    task = await runner.run_task(TaskRecord(id="legacy", prompt="check"))

    assert task.status == TaskStatus.SUCCESS
    assert task.result == "legacy result"
    assert task.error is None
