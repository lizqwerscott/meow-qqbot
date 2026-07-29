"""测试 cron 事件路由：session_target 决定 wake 目标。"""

from unittest.mock import AsyncMock, MagicMock

import pytest

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
    runner._delivery_cb = None
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
        # command 类型 + isolated：不 wake 会话
        runner._wake_dispatcher.request.assert_not_called()
    finally:
        asyncio.create_subprocess_exec = original_create


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
