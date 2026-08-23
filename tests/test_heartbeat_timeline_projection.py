from types import SimpleNamespace

import pytest

from core.engine.conversation_timeline import ConversationTimeline
from core.engine.prompt_builder import PromptBuilder


@pytest.mark.asyncio
async def test_heartbeat_main_prompt_repairs_legacy_gap_in_nonempty_timeline(tmp_path):
    class ContextManager:
        async def get_chat_history_async(self, chat_id, max_messages=None):
            return [
                {
                    "role": "user",
                    "content": "hello",
                    "message_id": "u1",
                    "timestamp": 1,
                },
                {"role": "assistant", "content": "legacy answer", "timestamp": 2},
            ]

    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    await timeline.append_user_message(
        chat_id="admin-chat",
        message_id="u1",
        content="hello",
        sender_id="admin",
        timestamp=1,
    )

    builder = PromptBuilder.__new__(PromptBuilder)
    builder.context_manager = ContextManager()
    builder.timeline = timeline
    builder.hindsight = None
    builder._workspace_manager = None
    builder._has_tasks = False
    builder._system_events = None
    builder._deps = None

    messages, _ = await builder.build_heartbeat_messages(
        "check",
        session_mode="main",
        admin_chat_id="admin-chat",
    )

    assert {message["content"] for message in messages} >= {"hello", "legacy answer"}
    await timeline.close()
