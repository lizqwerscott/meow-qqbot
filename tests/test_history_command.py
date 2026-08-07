from types import SimpleNamespace

import pytest

from core.command_handlers.history import HistoryCommand
from core.message import InputMessage


@pytest.mark.asyncio
async def test_history_command_compacts_through_context_manager():
    calls = []
    context = SimpleNamespace(get_history_count=lambda: 2)

    class ContextManager:
        compaction_threshold_tokens = 100

        async def get_chat_history_async(self, chat_id):
            assert chat_id == "chat_001"
            return [{"role": "user", "content": "one"}] * 2

        async def compact_history_if_needed(self, chat_id, force=False):
            calls.append((chat_id, force))
            return True, None, 1

    command = HistoryCommand(ContextManager())
    input_message = InputMessage(
        id="message_001",
        sender_id="admin_001",
        chat_id="chat_001",
        content="",
        is_group=False,
    )

    replies = await command._compact(input_message, "")

    assert calls == [("chat_001", True)]
    assert "压缩完成" in replies[0]["content"]
