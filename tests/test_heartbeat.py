from core.tasks.heartbeat import HeartbeatManager


def test_heartbeat_task_is_marked_only_when_execution_starts(monkeypatch):
    manager = HeartbeatManager(
        {"enabled": True},
    )

    async def read_heartbeat_file():
        return (
            "---\n"
            "tasks:\n"
            "  - name: 早安\n"
            "    interval_seconds: 86400\n"
            "    prompt: 发送早安问候\n"
            "---\n"
            "每日问候。\n"
        )

    monkeypatch.setattr(manager, "_read_heartbeat_file", read_heartbeat_file)

    import asyncio

    prompt, task_names = asyncio.run(manager._load_heartbeat_content())

    assert "早安" in prompt
    assert task_names == ("早安",)
    assert manager._task_last_run == {}

    _, pending_task_names = asyncio.run(manager._load_heartbeat_content())
    assert pending_task_names == ("早安",)

    manager.mark_tasks_started(task_names)

    assert "早安" in manager._task_last_run

    _, completed_task_names = asyncio.run(manager._load_heartbeat_content())
    assert completed_task_names == ()
