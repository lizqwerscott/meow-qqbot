from types import SimpleNamespace

import pytest

from core.command_handlers.status import StatusCommand
from core.message import InputMessage


@pytest.mark.asyncio
async def test_status_displays_ledger_integrity_summary(monkeypatch):
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

        async def get_ledger_integrity_async(self, chat_id):
            return {
                "legacy_conflict_count": 1,
                "event_integrity": {
                    "turn_count": 5,
                    "invalid_turn_count": 1,
                    "incomplete_turn_count": 0,
                    "open_turn_count": 0,
                    "waiting_tool_turn_count": 0,
                },
            }

    replies = await StatusCommand(Engine()).execute(
        InputMessage("status", "admin", "chat", "", False), ""
    )

    content = replies[0]["content"]
    assert "账本完整性" in content
    assert "identity 冲突: `1`" in content
    assert "total=5" in content
    assert "invalid=1" in content
