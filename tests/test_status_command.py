from types import SimpleNamespace

import pytest

from core.command_handlers.status import StatusCommand
from core.message import InputMessage


@pytest.mark.asyncio
async def test_status_displays_global_history_migration_summary(monkeypatch):
    import core.command_handlers.status as status_module

    monkeypatch.setattr(
        status_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=20, used=2 * 1024**3, total=10 * 1024**3),
    )
    monkeypatch.setattr(status_module.psutil, "cpu_percent", lambda interval=0: 3)
    monkeypatch.setattr(
        status_module.psutil,
        "disk_usage",
        lambda path: SimpleNamespace(percent=30, used=3 * 1024**3, total=10 * 1024**3),
    )

    class Process:
        def memory_info(self):
            return SimpleNamespace(rss=100 * 1024**2)

        def cpu_percent(self, interval=0):
            return 1

    monkeypatch.setattr(status_module.psutil, "Process", Process)

    class Engine:
        _skill_managers = None

        async def get_stats(self):
            return {
                "queue_sizes": {},
                "active_chats": 0,
                "hindsight_health": {"status": "disabled"},
                "learners": {},
            }

        async def get_engagement_status(self):
            return {}

        async def get_history_migration_status(self, chat_id):
            return {}

        async def get_history_migration_summary(self):
            return {
                "session_count": 4,
                "sessions_ready_for_legacy_read_removal": 2,
                "sessions_with_missing_legacy_visible": 1,
                "sessions_with_legacy_protocol": 1,
            }

    replies = await StatusCommand(Engine()).execute(
        InputMessage("status", "admin", "chat", "", False), ""
    )

    content = replies[0]["content"]
    assert "全局会话: `4`" in content
    assert "可退出 fallback: `2`" in content
    assert "全局缺口会话: `1`，协议残留会话: `1`" in content
