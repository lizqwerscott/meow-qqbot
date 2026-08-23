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


@pytest.mark.asyncio
async def test_history_command_reads_visible_timeline_projection(tmp_path):
    from core.engine.conversation_timeline import ConversationTimeline

    class ContextManager:
        async def get_chat_history_async(self, chat_id):
            return [{"role": "tool", "content": "hidden"}]

    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    await timeline.append_user_message(
        chat_id="chat_001",
        message_id="user-1",
        content="hello",
        sender_id="user-1",
        timestamp=1.0,
    )
    await timeline.append_accepted_delivery(
        chat_id="chat_001",
        delivery_id="reply-1",
        content="world",
        delivery_kind="response",
        timestamp=2.0,
    )

    command = HistoryCommand(ContextManager(), timeline=timeline)
    history = await command._get_visible_history("chat_001")

    assert [item["content"] for item in history] == ["hello", "world"]
    assert [item["role"] for item in history] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_history_command_repairs_legacy_gap_in_nonempty_timeline(tmp_path):
    from core.engine.conversation_timeline import ConversationTimeline

    class ContextManager:
        async def get_chat_history_async(self, chat_id):
            return [
                {
                    "role": "user",
                    "content": "hello",
                    "message_id": "user-1",
                    "timestamp": 1.0,
                },
                {"role": "assistant", "content": "legacy answer", "timestamp": 2.0},
            ]

    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    await timeline.append_user_message(
        chat_id="chat_001",
        message_id="user-1",
        content="hello",
        sender_id="user-1",
        timestamp=1.0,
    )
    command = HistoryCommand(ContextManager(), timeline=timeline)

    history = await command._get_visible_history("chat_001")

    assert [item["content"] for item in history] == ["hello", "legacy answer"]
    await timeline.close()


@pytest.mark.asyncio
async def test_history_command_repairs_legacy_visible_history_before_fallback(tmp_path):
    from core.engine.conversation_timeline import ConversationTimeline

    class ContextManager:
        async def get_chat_history_async(self, chat_id):
            return [
                {
                    "role": "user",
                    "content": "legacy question",
                    "message_id": "legacy-1",
                    "timestamp": 1.0,
                },
                {
                    "role": "assistant",
                    "content": "legacy answer",
                    "timestamp": 2.0,
                },
                {"role": "tool", "content": "hidden protocol"},
            ]

    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    command = HistoryCommand(ContextManager(), timeline=timeline)

    history = await command._get_visible_history("chat_001")

    assert [item["content"] for item in history] == [
        "legacy question",
        "legacy answer",
    ]
    assert [item["role"] for item in history] == ["user", "assistant"]
    await timeline.close()


@pytest.mark.asyncio
async def test_history_command_does_not_expose_legacy_protocol_only_history(tmp_path):
    from core.engine.conversation_timeline import ConversationTimeline

    class ContextManager:
        async def get_chat_history_async(self, chat_id):
            return [
                {
                    "role": "assistant",
                    "content": "internal tool plan",
                    "tool_calls": [{"id": "call-1"}],
                },
                {"role": "tool", "content": "secret tool result"},
            ]

    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    command = HistoryCommand(ContextManager(), timeline=timeline)

    assert await command._get_visible_history("chat_001") == []
    await timeline.close()


@pytest.mark.asyncio
async def test_history_command_filters_legacy_protocol_when_timeline_unavailable():
    class ContextManager:
        async def get_chat_history_async(self, chat_id):
            return [
                {"role": "user", "content": "visible"},
                {
                    "role": "assistant",
                    "content": "internal",
                    "tool_calls": [{"id": "call-1"}],
                },
                {"role": "tool", "content": "secret"},
            ]

    command = HistoryCommand(ContextManager())

    assert await command._get_visible_history("chat_001") == [
        {"role": "user", "content": "visible"}
    ]


@pytest.mark.asyncio
async def test_history_command_clear_removes_timeline_projection(tmp_path):
    from core.engine.conversation_timeline import ConversationTimeline

    class ContextManager:
        async def clear_chat_history_async(self, chat_id):
            assert chat_id == "chat_001"

    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    await timeline.append_user_message(
        chat_id="chat_001",
        message_id="user-1",
        content="hello",
        sender_id="user-1",
        timestamp=1.0,
    )
    command = HistoryCommand(ContextManager(), timeline=timeline)
    await command._clear(InputMessage("command", "admin", "chat_001", "", False), "")

    assert await timeline.history("chat_001") == []


@pytest.mark.asyncio
async def test_history_command_lists_timeline_summary_and_falls_back(tmp_path):
    class ContextManager:
        async def get_all_chat_ids_async(self):
            return ["legacy-chat"]

        async def get_chat_history_async(self, chat_id):
            if chat_id == "legacy-chat":
                return [{"role": "user", "content": "legacy", "timestamp": 60}]
            raise AssertionError("timeline session should not read legacy history")

    from core.engine.conversation_timeline import ConversationTimeline

    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    await timeline.append_user_message(
        chat_id="timeline-chat",
        message_id="user-1",
        content="hello",
        sender_id="user-1",
        timestamp=120.0,
    )
    command = HistoryCommand(ContextManager(), timeline=timeline)
    replies = await command._list_sessions(
        InputMessage("command", "admin", "timeline-chat", "", False)
    )

    assert "timeline-chat (1 条" in replies[0]["content"]
    assert "legacy-chat (1 条" in replies[0]["content"]
