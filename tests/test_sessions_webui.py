from unittest.mock import AsyncMock

import httpx
import pytest

from core.engine.archive_index import ArchiveIndex, ArchiveTurnRecord
from core.engine.conversation_event_log import (
    ConversationEvent,
    ConversationEventLog,
    TurnKind,
)
from core.engine.prompt_history_projection import PromptHistoryProjection
from core.engine.turn_protocol_history import TurnProtocolHistory
from core.webui.app import create_app
from core.webui.routers.sessions import _redact_message


class _ContextManager:
    async def get_all_disk_chat_ids_async(self):
        return ["group-1", "private-1"]

    async def get_archived_sessions_summary_async(self):
        return {}

    async def get_session_summary_async(self, chat_id):
        return {"message_count": 0, "last_activity": 0, "estimated_tokens": 0}

    async def get_chat_history_async(self, chat_id, max_messages=None):
        if chat_id == "legacy-1":
            return [
                {"role": "assistant", "content": "旧回复", "timestamp": 1},
                {
                    "role": "tool",
                    "tool_name": "read_file",
                    "tool_call_id": "call-legacy",
                    "content": "旧工具结果",
                    "timestamp": 2,
                },
            ]
        return []

    async def clear_chat_history_async(self, chat_id):
        return None

    async def get_archived_files_async(self, chat_id):
        return []

    def get_chat_type(self, chat_id):
        return chat_id.startswith("group")


def test_webui_history_redaction_covers_nested_tool_arguments():
    safe = _redact_message(
        {
            "role": "assistant",
            "content": "Authorization: Bearer top-secret",
            "tool_calls": [
                {
                    "function": {
                        "name": "request",
                        "arguments": '{"api_key":"top-secret","q":"hello"}',
                    }
                }
            ],
        }
    )

    assert "top-secret" not in str(safe)
    assert "[已脱敏]" in str(safe)


def test_webui_redaction_normalizes_legacy_user_display_prefix():
    safe = _redact_message(
        {
            "role": "user",
            "content": (
                "[用户 在 2026-07-13 17:14:22]: "
                "[用户 在 2026-07-13 17:14:22]: 原始内容"
            ),
        }
    )

    assert safe["content"] == "原始内容"


@pytest.mark.asyncio
async def test_sessions_protocol_view_and_kind_filter(tmp_path):
    protocol = TurnProtocolHistory(str(tmp_path / "protocol.sqlite3"))
    await protocol.append_assistant(
        chat_id="heartbeat:events",
        turn_id="turn-1",
        event_id="assistant:0",
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "function": {"name": "heartbeat_respond", "arguments": "{}"},
            }
        ],
    )
    await protocol.append_tool_result(
        chat_id="heartbeat:events",
        turn_id="turn-1",
        event_id="tool:call-1",
        tool_call_id="call-1",
        tool_name="heartbeat_respond",
        content="done",
    )
    app = create_app(
        {
            "context_manager": _ContextManager(),
            "protocol_history": protocol,
        },
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        protocol_page = await client.get("/sessions/heartbeat:events/protocol")
        heartbeat_page = await client.get("/sessions?kind=heartbeat")
        group_page = await client.get("/sessions?kind=group")

    assert protocol_page.status_code == 200
    assert "heartbeat_respond" in protocol_page.text
    assert "done" in protocol_page.text
    assert heartbeat_page.status_code == 200
    assert "heartbeat:events" in heartbeat_page.text
    assert "group-1" not in heartbeat_page.text
    assert group_page.status_code == 200
    assert "group-1" in group_page.text
    assert "private-1" not in group_page.text
    await protocol.close()


@pytest.mark.asyncio
async def test_session_list_paginates_sessions(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    for chat_id in ("chat-a", "chat-b"):
        await event_log.append_user_message(
            chat_id=chat_id,
            turn_id=f"turn-{chat_id}",
            message_id=f"message-{chat_id}",
            content=chat_id,
        )
    app = create_app(
        {"context_manager": _ContextManager(), "conversation_event_log": event_log},
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/sessions?page=1&page_size=1")

    assert response.status_code == 200
    assert "共 2 个会话" in response.text
    assert ("chat-a" in response.text) != ("chat-b" in response.text)
    await event_log.close()


@pytest.mark.asyncio
async def test_session_list_uses_batched_ledger_summaries(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await event_log.append_user_message(
        chat_id="batched-chat",
        turn_id="turn-1",
        message_id="message-1",
        content="批量统计",
    )
    event_log.session_summary = AsyncMock(
        side_effect=AssertionError("session list must use batched summaries")
    )
    app = create_app(
        {"context_manager": _ContextManager(), "conversation_event_log": event_log},
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/sessions")

    assert response.status_code == 200
    assert "batched-chat" in response.text
    await event_log.close()


@pytest.mark.asyncio
async def test_protocol_view_falls_back_to_legacy_context_history(tmp_path):
    protocol = TurnProtocolHistory(str(tmp_path / "protocol.sqlite3"))
    app = create_app(
        {
            "context_manager": _ContextManager(),
            "protocol_history": protocol,
        },
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/sessions/legacy-1/protocol")

    assert response.status_code == 200
    assert "旧回复" in response.text
    assert "旧工具结果" in response.text
    await protocol.close()


@pytest.mark.asyncio
async def test_session_clear_removes_protocol_history(tmp_path):
    protocol = TurnProtocolHistory(str(tmp_path / "protocol.sqlite3"))
    await protocol.append_assistant(
        chat_id="clear-me",
        turn_id="turn-clear",
        event_id="assistant:0",
        content="will be deleted",
    )
    app = create_app(
        {
            "context_manager": _ContextManager(),
            "protocol_history": protocol,
        },
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/sessions/clear-me/clear")

    assert response.status_code == 303
    assert await protocol.history("clear-me") == []
    await protocol.close()


@pytest.mark.asyncio
async def test_session_detail_keeps_core_ledger_history_over_stale_timeline(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await event_log.append_user_message(
        chat_id="ledger-chat",
        turn_id="turn-1",
        message_id="user-1",
        content="账本问题",
        timestamp=1,
    )
    await event_log.append_accepted_delivery(
        chat_id="ledger-chat",
        turn_id="turn-1",
        delivery_id="delivery-1",
        content="账本回复",
        timestamp=2,
    )

    class ContextManager:
        async def get_archived_files_async(self, chat_id):
            return []

    stale_timeline = AsyncMock()
    stale_timeline.history = AsyncMock(return_value=[])
    app = create_app(
        {
            "context_manager": ContextManager(),
            "conversation_event_log": event_log,
            "conversation_timeline": stale_timeline,
        },
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/sessions/ledger-chat")

    assert response.status_code == 200
    assert "账本回复" in response.text
    stale_timeline.history.assert_not_awaited()
    await event_log.close()


@pytest.mark.asyncio
async def test_session_detail_pages_complete_turns_and_keeps_tools_collapsed(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    for index in range(1, 4):
        turn_id = f"turn-{index}"
        await event_log.append_user_message(
            chat_id="paged-chat",
            turn_id=turn_id,
            message_id=f"user-{index}",
            content=f"问题 {index}",
            timestamp=index * 10,
            turn_kind=TurnKind.AI if index == 3 else TurnKind.AMBIENT,
        )
        if index == 3:
            await event_log.append_event(
                ConversationEvent(
                    chat_id="paged-chat",
                    turn_id=turn_id,
                    event_id="assistant:call-3",
                    role="assistant",
                    kind="assistant_tool_call",
                    tool_calls=({"id": "call-3", "function": {"name": "task"}},),
                )
            )
            await event_log.append_event(
                ConversationEvent(
                    chat_id="paged-chat",
                    turn_id=turn_id,
                    event_id="tool:call-3",
                    role="tool",
                    kind="tool_result",
                    tool_call_id="call-3",
                    tool_name="task",
                    content="完成",
                )
            )
        await event_log.append_accepted_delivery(
            chat_id="paged-chat",
            turn_id=turn_id,
            delivery_id=f"delivery-{index}",
            content=f"回答 {index}",
            timestamp=index * 10 + 1,
        )
        await event_log.append_turn_terminal(chat_id="paged-chat", turn_id=turn_id)

    page = await event_log.snapshot_turn_page(
        "paged-chat", page=1, page_size=2, include_internal=True
    )
    assert [turn.turn_id for turn in page.turns] == ["turn-3", "turn-2"]
    assert page.total_turns == 3
    assert page.total_pages == 2
    assert {event.turn_id for event in page.events} == {"turn-3", "turn-2"}

    app = create_app(
        {
            "context_manager": _ContextManager(),
            "conversation_event_log": event_log,
        },
        {},
    )
    event_log.history = AsyncMock(side_effect=AssertionError("full history read"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/sessions/paged-chat?page=1&page_size=2")

    assert response.status_code == 200
    assert "问题 3" in response.text
    assert "问题 1" not in response.text
    assert "AI 对话" in response.text
    assert "查看工具链" in response.text
    assert '<details class="conversation-tool-chain">' in response.text
    assert '<details class="conversation-tool-chain" open>' not in response.text
    await event_log.close()


@pytest.mark.asyncio
async def test_archive_views_use_ledger_index_without_reading_jsonl(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    archive_index = ArchiveIndex(str(tmp_path / "archive-index.sqlite3"))
    await event_log.append_user_message(
        chat_id="ledger-archive",
        turn_id="turn-1",
        message_id="message-1",
        content="归档问题",
        timestamp=1,
    )
    await event_log.append_accepted_delivery(
        chat_id="ledger-archive",
        turn_id="turn-1",
        delivery_id="delivery-1",
        content="归档回答",
        timestamp=2,
    )
    await event_log.append_turn_terminal(
        chat_id="ledger-archive", turn_id="turn-1", timestamp=3
    )
    await projection.apply_archive_retention(
        "ledger-archive",
        operation_id="archive-op-1",
        hidden_event_ids=("user:message-1", "delivery:delivery-1", "terminal:turn-1"),
        captured_cutoff_seq=3,
    )
    batch = await archive_index.prepare_batch(
        batch_id="batch:archive-op-1",
        operation_id="archive-op-1",
        chat_id="ledger-archive",
        captured_cutoff_seq=3,
        turn_records=[ArchiveTurnRecord("turn-1", 1, "1970-01-01", 3, 3)],
        event_ids=[
            ("user:message-1", "turn-1"),
            ("delivery:delivery-1", "turn-1"),
            ("terminal:turn-1", "turn-1"),
        ],
    )
    await archive_index.mark_state(batch.batch_id, "committed")

    class LedgerOnlyContext(_ContextManager):
        async def get_archived_sessions_summary_async(self):
            raise AssertionError("JSONL archive summary must not be read")

        async def get_archived_files_async(self, chat_id):
            raise AssertionError("JSONL archive files must not be read")

    app = create_app(
        {
            "context_manager": LedgerOnlyContext(),
            "conversation_event_log": event_log,
            "prompt_history_projection": projection,
            "archive_index": archive_index,
        },
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        list_response = await client.get("/sessions/archived")
        detail_response = await client.get("/sessions/archived/ledger-archive")
        history_response = await client.get("/sessions/ledger-archive/archive-history")
        messages_response = await client.get(
            "/sessions/archived/ledger-archive/messages/batch:archive-op-1"
        )

    assert list_response.status_code == 200
    assert "ledger-archive" in list_response.text
    assert detail_response.status_code == 200
    assert "batch:archive-op-1" in detail_response.text
    assert history_response.status_code == 200
    assert "归档回答" in history_response.text
    assert messages_response.status_code == 200
    assert "归档回答" in messages_response.text
    await event_log.close()
    await projection.close()
    await archive_index.close()


@pytest.mark.asyncio
async def test_timeline_view_includes_internal_wire_events(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await event_log.append_user_message(
        chat_id="timeline-chat",
        turn_id="turn-1",
        message_id="message-1",
        content="执行任务",
    )
    await event_log.append_event(
        ConversationEvent(
            chat_id="timeline-chat",
            turn_id="turn-1",
            event_id="assistant:call-1",
            role="assistant",
            kind="assistant_tool_call",
            tool_calls=({"id": "call-1", "function": {"name": "task"}},),
        )
    )
    await event_log.append_event(
        ConversationEvent(
            chat_id="timeline-chat",
            turn_id="turn-1",
            event_id="tool:call-1",
            role="tool",
            kind="tool_result",
            tool_call_id="call-1",
            tool_name="task",
            content="完成",
        )
    )
    await event_log.append_accepted_delivery(
        chat_id="timeline-chat",
        turn_id="turn-1",
        delivery_id="delivery-1",
        content="已经完成",
    )
    await event_log.append_turn_terminal(chat_id="timeline-chat", turn_id="turn-1")

    app = create_app(
        {
            "context_manager": _ContextManager(),
            "conversation_event_log": event_log,
        },
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/sessions/timeline-chat/timeline")
        protocol_response = await client.get("/sessions/timeline-chat/protocol")

    assert response.status_code == 200
    assert "查看工具链" in response.text
    assert "task" in response.text
    assert "turn_terminal" not in response.text
    assert "已经完成" in response.text
    assert protocol_response.status_code == 200
    assert "工具调用:" in protocol_response.text
    assert "完成" in protocol_response.text
    await event_log.close()


@pytest.mark.asyncio
async def test_active_view_excludes_archived_turns_from_core_page(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    for index, content in enumerate(("旧问题", "新问题"), start=1):
        turn_id = f"turn-{index}"
        await event_log.append_user_message(
            chat_id="active-chat",
            turn_id=turn_id,
            message_id=f"message-{index}",
            content=content,
        )
        await event_log.append_accepted_delivery(
            chat_id="active-chat",
            turn_id=turn_id,
            delivery_id=f"delivery-{index}",
            content=f"{content}的回答",
        )
        await event_log.append_turn_terminal(chat_id="active-chat", turn_id=turn_id)
    await projection.apply_archive_retention(
        "active-chat",
        operation_id="archive-old",
        hidden_event_ids=(
            "user:message-1",
            "delivery:delivery-1",
            "terminal:turn-1",
        ),
        captured_cutoff_seq=3,
    )

    app = create_app(
        {
            "context_manager": _ContextManager(),
            "conversation_event_log": event_log,
            "prompt_history_projection": projection,
        },
        {},
    )
    projection.snapshot_for_active = AsyncMock(
        side_effect=AssertionError("active view must use paged ledger reads")
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/sessions/active-chat/active")

    assert response.status_code == 200
    assert "新问题" in response.text
    assert "旧问题" not in response.text
    await event_log.close()
    await projection.close()


@pytest.mark.asyncio
async def test_timeline_view_uses_database_turn_page(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await event_log.append_user_message(
        chat_id="paged-timeline",
        turn_id="turn-1",
        message_id="message-1",
        content="分页时间线",
    )
    await event_log.append_accepted_delivery(
        chat_id="paged-timeline",
        turn_id="turn-1",
        delivery_id="delivery-1",
        content="已分页",
    )
    event_log.snapshot_turns = AsyncMock(
        side_effect=AssertionError("timeline must not materialize all turns")
    )
    app = create_app(
        {
            "context_manager": _ContextManager(),
            "conversation_event_log": event_log,
        },
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/sessions/paged-timeline/timeline")

    assert response.status_code == 200
    assert "分页时间线" in response.text
    await event_log.close()


@pytest.mark.asyncio
async def test_prompt_view_pages_selected_turns_without_materializing_turn_index(
    tmp_path,
):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    projection = PromptHistoryProjection(
        event_log, metadata_path=str(tmp_path / "projection.sqlite3")
    )
    for index in range(1, 4):
        await event_log.append_user_message(
            chat_id="paged-prompt",
            turn_id=f"turn-{index}",
            message_id=f"message-{index}",
            content=f"Prompt 问题 {index}",
        )
        await event_log.append_accepted_delivery(
            chat_id="paged-prompt",
            turn_id=f"turn-{index}",
            delivery_id=f"delivery-{index}",
            content=f"Prompt 回答 {index}",
        )
        await event_log.append_turn_terminal(
            chat_id="paged-prompt", turn_id=f"turn-{index}"
        )
    event_log.snapshot_turns = AsyncMock(
        side_effect=AssertionError("prompt view must not materialize all turns")
    )
    app = create_app(
        {
            "context_manager": _ContextManager(),
            "conversation_event_log": event_log,
            "prompt_history_projection": projection,
        },
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/sessions/paged-prompt/prompt?page=1&page_size=2")

    assert response.status_code == 200
    assert "Prompt 问题 3" in response.text
    assert "Prompt 问题 1" not in response.text
    await event_log.close()
    await projection.close()
