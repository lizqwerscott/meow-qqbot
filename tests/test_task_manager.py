import time
from unittest.mock import MagicMock

import pytest

from core.tasks.manager import TaskManager
from core.tasks.models import TaskRecord, TaskStatus


@pytest.fixture
def mock_store():
    store = MagicMock()
    store._tasks = {}
    store.all_tasks.return_value = store._tasks

    def get_task(task_id):
        return store._tasks.get(task_id)

    async def update_task(task):
        store._tasks[task.id] = task

    async def add_task(task):
        store._tasks[task.id] = task

    store.get_task.side_effect = get_task
    store.update_task.side_effect = update_task
    store.add_task.side_effect = add_task
    return store


@pytest.fixture
def mgr(mock_store):
    return TaskManager(store=mock_store)


@pytest.mark.asyncio
async def test_create_task(mgr, mock_store):
    task = await mgr.create_task(prompt="测试任务")
    assert task.status == TaskStatus.PENDING
    assert task.id in mock_store._tasks


@pytest.mark.asyncio
async def test_start_task(mgr, mock_store):
    task = await mgr.create_task(prompt="测试任务")
    started = await mgr.start_task(task.id)
    assert started.status == TaskStatus.RUNNING
    assert started.started_at is not None


@pytest.mark.asyncio
async def test_finish_task(mgr, mock_store):
    task = await mgr.create_task(prompt="测试任务")
    await mgr.start_task(task.id)
    finished = await mgr.finish_task(task.id, TaskStatus.SUCCESS, result="ok")
    assert finished.status == TaskStatus.SUCCESS
    assert finished.result == "ok"


@pytest.mark.asyncio
async def test_finish_already_cancelled_task(mgr, mock_store):
    """回归测试：已取消的任务不应被 finish_task 覆写。"""
    task = await mgr.create_task(prompt="测试任务")
    await mgr.start_task(task.id)
    await mgr.cancel_task(task.id)
    result = await mgr.finish_task(
        task.id, TaskStatus.SUCCESS, result="should_not_override"
    )
    assert result is not None
    assert result.status == TaskStatus.CANCELLED
    assert result.result is None or result.result != "should_not_override"


@pytest.mark.asyncio
async def test_cancel_pending_task(mgr, mock_store):
    task = await mgr.create_task(prompt="测试任务")
    cancelled = await mgr.cancel_task(task.id)
    assert cancelled is True
    assert mock_store._tasks[task.id].status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_already_finished_task(mgr, mock_store):
    task = await mgr.create_task(prompt="测试任务")
    await mgr.start_task(task.id)
    await mgr.finish_task(task.id, TaskStatus.SUCCESS, result="ok")
    cancelled = await mgr.cancel_task(task.id)
    assert cancelled is False


@pytest.mark.asyncio
async def test_interrupt_active_tasks_on_restart_marks_pending_and_running(
    mgr, mock_store
):
    pending = await mgr.create_task(prompt="pending")
    running = await mgr.create_task(prompt="running")
    await mgr.start_task(running.id)

    interrupted = await mgr.interrupt_active_tasks_on_restart()

    assert {task.id for task in interrupted} == {pending.id, running.id}
    for task in (pending, running):
        assert mock_store._tasks[task.id].status == TaskStatus.LOST
        assert mock_store._tasks[task.id].error == "任务因进程重启中断，未自动重放"
        assert mock_store._tasks[task.id].finished_at is not None
        assert mock_store._tasks[task.id].recovery_notification_pending

    assert {task.id for task in mgr.list_restart_recovery_tasks()} == {
        pending.id,
        running.id,
    }
    assert await mgr.interrupt_active_tasks_on_restart() == []

    # pending 任务，很久以前创建
    old_task = TaskRecord(
        prompt="old",
        type="manual",
        created_at=time.time() - 7200,
    )
    old_task.status = TaskStatus.PENDING
    await mock_store.add_task(old_task)

    count = await mgr.detect_lost_tasks(lost_detection_minutes=30)
    assert count == 1
    assert mock_store._tasks[old_task.id].status == TaskStatus.LOST


@pytest.mark.asyncio
async def test_detect_lost_skips_running(mgr, mock_store):
    task = await mgr.create_task(prompt="running")
    await mgr.start_task(task.id)
    count = await mgr.detect_lost_tasks(lost_detection_minutes=30)
    assert count == 0


@pytest.mark.asyncio
async def test_detect_lost_skips_terminal(mgr, mock_store):
    task = await mgr.create_task(prompt="finished")
    await mgr.start_task(task.id)
    await mgr.finish_task(task.id, TaskStatus.SUCCESS, result="ok")
    count = await mgr.detect_lost_tasks(lost_detection_minutes=30)
    assert count == 0
